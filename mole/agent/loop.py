"""Stateless chat-completion + tool-dispatch loop.

Drives a chat-completion API directly via our `LLMClient` Protocol so the
same loop works with any backend — OpenRouter, Anthropic, self-hosted vLLM,
or any OpenAI-compatible server. Reproducibility: the conversation state is a list of
messages we own, with no session affinity to any specific provider.

Each tool call routes through a `dispatcher` callable of shape
`async (ToolCall) -> ToolResult`. A bare dispatcher just executes the
requested action; the gating layer (when active) wraps the dispatcher so
every call passes through a monitor + protocol decision first. The loop
itself stays oblivious — its only contract is "give me a callable that
turns a ToolCall into a ToolResult."

The core is deliberately small. Persona conditioning, audit collection, etc.
are not this module's responsibility — they're upstream (persona prompts
arrive in `messages`; audit events are emitted by the dispatcher).
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable

from ..llm import ChatMessage, ChatResponse, LLMClient, ToolCall, ToolSchema

logger = logging.getLogger(__name__)

# Optional per-turn output cap. Default None = unbounded (legacy behavior: kimi/gpt
# terminate their own reasoning + tool-call within a few K tokens). Set FOCAL_MAX_TOKENS
# for models whose reasoning RUNS AWAY to the context limit — e.g. GLM-5.2 emits
# non-terminating reasoning, so a single uncapped turn generates ~65K tokens (~15 min at
# ~70 tok/s) and never returns, stalling the session. Env-gated so a running job that
# doesn't set it (kimi 30d) is completely unaffected.
_FOCAL_MAX_TOKENS = (
    int(os.environ["FOCAL_MAX_TOKENS"]) if os.environ.get("FOCAL_MAX_TOKENS") else None
)


@dataclass
class ToolResult:
    """Output of a tool call. Becomes a role=tool message back into the LLM."""
    tool_call_id: str
    name: str
    content: str   # serialized result the model will read
    is_error: bool = False


# Dispatcher: receives one ToolCall, returns its ToolResult.
# A bare dispatcher is a thin wrapper that looks up the requested tool by
# name and invokes the matching state-backend method. The gating middleware
# (when active) wraps the dispatcher so every call passes through a monitor
# + protocol decision first.
Dispatcher = Callable[[ToolCall], Awaitable[ToolResult]]


@dataclass
class AgentLoopResult:
    """Outcome of one agent run (one stage's worth of agent activity)."""
    final_message: str                              # last assistant content
    messages: list[ChatMessage] = field(default_factory=list)   # full conversation
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    turn_count: int = 0
    finish_reason: str = "stop"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    aborted_reason: str | None = None               # e.g. "max_turns", "abort_signal"


# A single unbounded tool result (e.g. plane.list_issues returning every issue
# after days of accumulation — observed at 1.2 MB / ~300K tokens) single-handedly
# overflows the model's context window -> a context_length_exceeded BadRequest
# that aborts the whole session. Cap any one result before it enters the LLM
# context. Keep the HEAD (structure / first items) AND the TAIL (most-recent
# items) with a marker between — so neither the start nor the latest rows are
# lost — and only bound the MODEL's view (the full result still reaches the
# audit layer). Model-agnostic: protects gpt-5.x / kimi / DeepSeek alike.
# Override with $MAX_TOOL_RESULT_CHARS.
_MAX_TOOL_RESULT_CHARS = int(os.environ.get("MAX_TOOL_RESULT_CHARS", "16000"))


def _cap_tool_result(content: str, limit: int = _MAX_TOOL_RESULT_CHARS) -> str:
    if not content or limit <= 0 or len(content) <= limit:
        return content
    head = (limit * 7) // 10
    tail = limit - head
    omitted = len(content) - head - tail
    return (content[:head]
            + f"\n\n...[{omitted} chars truncated to fit context; showing first "
              f"{head} + last {tail} chars]...\n\n"
            + content[-tail:])


# The per-result cap bounds any single result, but a long agentic session still
# RE-SENDS every prior tool result on every turn — so input tokens accumulate as
# ~turns x context, and at kimi's 850K-tokens/min cap that throttles throughput
# to a crawl. Windowing fixes the accumulation: once the conversation grows past
# CONTEXT_BUDGET_CHARS, elide OLD tool-result bodies to a placeholder, keeping the
# task (system+user), the agent's action history (assistant tool_calls), and the
# most-recent `keep_recent` messages in full — so the agent still completes its
# task (no turns dropped) but each call's payload stays bounded. The full,
# untrimmed history is still persisted; only the MODEL's view is windowed.
_CONTEXT_BUDGET_CHARS = int(os.environ.get("CONTEXT_BUDGET_CHARS", "100000"))  # ~25K tok


def _window_context(msgs: "list[ChatMessage]", budget: int = _CONTEXT_BUDGET_CHARS,
                    keep_recent: int = 6) -> "list[ChatMessage]":
    total = sum(len(m.content) for m in msgs if isinstance(m.content, str))
    if budget <= 0 or total <= budget:
        return msgs
    n = len(msgs)
    out: list[ChatMessage] = []
    for i, m in enumerate(msgs):
        protected = m.role == "system" or i <= 1 or i >= n - keep_recent
        if (not protected) and m.role == "tool" and isinstance(m.content, str) \
                and len(m.content) > 160:
            out.append(replace(
                m, content=f"[older {m.name or 'tool'} result elided to bound "
                           f"context — was {len(m.content)} chars]"))
        else:
            out.append(m)
    return out


async def run_agent_loop(
    *,
    llm: LLMClient,
    initial_messages: list[ChatMessage],
    tools: list[ToolSchema],
    dispatcher: Dispatcher,
    max_turns: int = 50,
    temperature: float = 0.7,
    seed: int | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    next_task: Callable[[], str | None] | None = None,
    post_turn: Callable[[int], None] | None = None,
) -> AgentLoopResult:
    """Drive the agent until it stops calling tools or hits max_turns.

    Args:
        llm: any backend implementing the LLMClient Protocol.
        initial_messages: system + user messages to seed the conversation.
            Persona conditioning + cover-task delivery happens upstream.
        tools: tool schemas the agent can call.
        dispatcher: async callable that executes a ToolCall and returns a ToolResult.
            The gating layer integrates by wrapping this callable.
        max_turns: hard cap on assistant turns (prevents runaway loops).
        temperature: sampling temperature passed to the LLM each call.
        seed: deterministic-sampling seed (where the backend supports it).
        on_event: optional hook called with structured event dicts for tracing.

    Returns:
        AgentLoopResult capturing the full conversation and per-turn artefacts.
    """
    msgs: list[ChatMessage] = list(initial_messages)
    all_tool_calls: list[ToolCall] = []
    all_tool_results: list[ToolResult] = []
    in_tokens = 0
    out_tokens = 0
    turn = 0
    finish_reason = "stop"
    aborted: str | None = None

    def _emit(evt: dict[str, Any]) -> None:
        if on_event is not None:
            try:
                on_event(evt)
            except Exception:
                logger.exception("on_event hook raised; continuing loop")

    while turn < max_turns:
        turn += 1
        _emit({"type": "llm.request", "turn": turn, "n_messages": len(msgs)})
        try:
            resp: ChatResponse = await llm.complete(
                _window_context(msgs), tools=tools, temperature=temperature, seed=seed,
                max_tokens=_FOCAL_MAX_TOKENS,
            )
        except Exception as exc:
            logger.exception("LLM call failed on turn %d", turn)
            aborted = f"llm_error: {type(exc).__name__}: {exc}"
            break

        in_tokens += resp.usage.input_tokens
        out_tokens += resp.usage.output_tokens
        finish_reason = resp.finish_reason
        _emit({"type": "llm.response", "turn": turn,
               "finish_reason": resp.finish_reason,
               "tool_calls": len(resp.tool_calls),
               "content_len": len(resp.content)})

        # Record the assistant message — include tool_calls when present so the
        # following tool-result message is a valid response to a preceding
        # `tool_calls` (required by the chat-completions wire format).
        msgs.append(ChatMessage(
            role="assistant",
            content=resp.content,
            tool_calls=resp.tool_calls or None,
        ))

        # A reasoning model can end a turn having only THOUGHT — no answer, no tool
        # call (its chain-of-thought arrives in `reasoning_content`, which some
        # backends surface as content). That is not "done with the item", it is an
        # unfinished turn, and treating it as done silently truncates the session:
        # gpt-oss-120b averaged 5.6 turns / 3.0 tool calls per session (vs 16-32 /
        # 14-53 for models that answer normally) and scored a spurious 0/40 executed.
        # Nudge it to act instead. Bounded: one nudge per turn, and a model that
        # only ever thinks still terminates via the max_turns cap.
        if not resp.tool_calls and resp.finish_reason == "reasoning_only":
            _emit({"type": "llm.reasoning_only", "turn": turn})
            msgs.append(ChatMessage(
                role="user",
                content="Continue. Use the tools to carry out the work; "
                        "reply without a tool call only when the task is finished.",
            ))
            continue

        # No tools requested → agent is done with the current item. In rich-session mode
        # `next_task` may hand it the next agenda item to keep working (returns the
        # continuation message, else None); otherwise the session ends here (default).
        if not resp.tool_calls:
            if next_task is not None:
                nxt = next_task()
                if nxt is not None:
                    msgs.append(ChatMessage(role="user", content=nxt))
                    continue
            break

        # Dispatch tool calls. When the model emits MULTIPLE tool calls in one turn,
        # run them concurrently via asyncio.gather — saves latency proportional to the
        # number of parallel calls. Errors become tool messages so the agent can react
        # (e.g., adjust args and retry) rather than crashing the loop.
        for tc in resp.tool_calls:
            all_tool_calls.append(tc)
            _emit({"type": "tool.request", "turn": turn,
                   "tool_call_id": tc.id, "name": tc.name})

        async def _one(tc):
            try:
                return await dispatcher(tc)
            except Exception as exc:                            # noqa: BLE001
                logger.exception("Dispatcher raised for tool %s", tc.name)
                return ToolResult(
                    tool_call_id=tc.id, name=tc.name,
                    content=f"DispatcherError: {type(exc).__name__}: {exc}",
                    is_error=True,
                )

        results = await asyncio.gather(*[_one(tc) for tc in resp.tool_calls])
        for result in results:
            all_tool_results.append(result)
            _emit({"type": "tool.result", "turn": turn,
                   "tool_call_id": result.tool_call_id, "name": result.name,
                   "is_error": result.is_error,
                   "content_len": len(result.content)})
            msgs.append(ChatMessage(
                role="tool", content=_cap_tool_result(result.content),
                tool_call_id=result.tool_call_id, name=result.name,
            ))

        # Rich-session mode: advance the simulated clock so the next turn's events get a
        # later timestamp (intra-session time spread). No-op by default.
        if post_turn is not None:
            post_turn(turn)
    else:
        # Loop exited via max_turns (not via break).
        aborted = "max_turns"

    final = next(
        (m.content for m in reversed(msgs) if m.role == "assistant"),
        "",
    )
    return AgentLoopResult(
        final_message=final,
        messages=msgs,
        tool_calls_made=all_tool_calls,
        tool_results=all_tool_results,
        turn_count=turn,
        finish_reason=finish_reason,
        total_input_tokens=in_tokens,
        total_output_tokens=out_tokens,
        aborted_reason=aborted,
    )
