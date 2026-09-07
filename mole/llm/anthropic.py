"""Anthropic direct backend.

Targets the official `anthropic` SDK rather than going through a proxy.
Use cases:
  - paper-grade runs on Claude frontier models (Opus 4.x, Sonnet 4.x)
  - cross-family monitor / agent pairing without a shared proxy
  - environments where only a direct API key is available

Auth: `ANTHROPIC_API_KEY` env var. The SDK refuses to construct without it.

Tool-calling note. Anthropic's Messages API encodes tool calls as content
blocks of type `tool_use` inside `message.content`. The corresponding
tool result must be sent back as a `user` message with a `tool_result`
content block (id-matched). We translate to/from our provider-agnostic
ChatMessage / ToolCall shape in this layer so the agent loop and
unit tests don't need to know about the differences.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from anthropic import Anthropic

from .base import (
    ChatMessage, ChatResponse, LLMClient, TokenUsage, ToolCall, ToolSchema,
)

logger = logging.getLogger(__name__)


class AnthropicClient:
    """LLMClient backed by https://api.anthropic.com.

    Default model: claude-opus-4-7. Override via $ANTHROPIC_MODEL or the constructor.
    """

    backend = "anthropic"
    DEFAULT_MODEL = "claude-opus-4-7"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens_default: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model_id = model
        self._model = model
        self._max_tokens_default = int(max_tokens_default)
        # The SDK reads ANTHROPIC_API_KEY itself when api_key is None.
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,            # Anthropic Messages API doesn't expose seed
    ) -> ChatResponse:
        # Anthropic requires tool names to match ^[a-zA-Z0-9_-]{1,128}$
        # but our universal convention is "<service>.<action>" (dot-separated).
        # Sanitize on the way out, reverse on the way back. Names everywhere
        # else in the codebase keep the dot form.
        sanitized_map: dict[str, str] = {}
        if tools:
            for t in tools:
                sanitized = _sanitize_tool_name(t.name)
                sanitized_map[sanitized] = t.name
        messages_for_call = _rename_tool_uses_in_messages(messages, sanitized_map)
        system_text, anthropic_messages = _split_system_and_convert(messages_for_call)

        def _call() -> ChatResponse:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": anthropic_messages,
                "max_tokens": int(max_tokens or self._max_tokens_default),
            }
            # Newer Claude families (opus-4.x, sonnet-4.x) reject the
            # `temperature` kwarg as deprecated. Only forward it when the
            # caller has explicitly set a non-default value AND opted in
            # via $ANTHROPIC_FORWARD_TEMPERATURE=1 (e.g., for an older
            # claude-3.5 model).
            if (
                temperature != 0.0
                and os.environ.get("ANTHROPIC_FORWARD_TEMPERATURE") == "1"
            ):
                kwargs["temperature"] = temperature
            # Prompt caching: system prompt and the tool catalog are large and identical
            # across all turns of a session — mark them as cache breakpoints so subsequent
            # turns pay only for the (tiny) incremental message history. Cuts cost and
            # latency dramatically on long agent sessions. Disable with $ANTHROPIC_CACHE=0.
            cache_enabled = os.environ.get("ANTHROPIC_CACHE", "1") != "0"
            if system_text:
                if cache_enabled:
                    kwargs["system"] = [{
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }]
                else:
                    kwargs["system"] = system_text
            if tools:
                tool_blocks = [
                    _tool_to_anthropic(t, name_override=_sanitize_tool_name(t.name))
                    for t in tools
                ]
                # Setting cache_control on the LAST tool caches the whole tools block.
                if cache_enabled and tool_blocks:
                    tool_blocks[-1] = {**tool_blocks[-1],
                                       "cache_control": {"type": "ephemeral"}}
                kwargs["tools"] = tool_blocks
            resp = self._client.messages.create(**kwargs)

            content_text = ""
            tool_calls: list[ToolCall] = []
            for block in resp.content or []:
                btype = getattr(block, "type", None)
                if btype == "text":
                    content_text += getattr(block, "text", "") or ""
                elif btype == "tool_use":
                    args = getattr(block, "input", None) or {}
                    sani = getattr(block, "name", "") or ""
                    # Reverse the sanitisation so the dispatcher sees the
                    # original dot-form (e.g., "eval_server.submit_eval").
                    original = sanitized_map.get(sani, sani)
                    tool_calls.append(ToolCall(
                        id=getattr(block, "id", "") or "",
                        name=original,
                        arguments=args if isinstance(args, dict) else {},
                    ))

            usage = getattr(resp, "usage", None)
            return ChatResponse(
                content=content_text,
                tool_calls=tool_calls,
                usage=TokenUsage(
                    input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                ),
                backend=self.backend,
                model_id=self._model,
                finish_reason=_map_stop_reason(getattr(resp, "stop_reason", None)),
            )

        return await asyncio.to_thread(_call)


# ── conversion helpers ───────────────────────────────────────────────


def _split_system_and_convert(
    messages: list[ChatMessage],
) -> tuple[str, list[dict[str, Any]]]:
    """Anthropic's API puts the system prompt in a top-level `system` field,
    not in the messages array. Split it out, then convert the rest.

    Tool-result messages (role="tool") are folded into the preceding
    user/assistant content as `tool_result` content blocks under a
    fresh user message, matching the Anthropic protocol.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
            continue
        if m.role == "tool":
            # Tool results are sent back as user messages with a tool_result
            # content block referencing the original tool_use id.
            tool_id = getattr(m, "tool_call_id", "") or ""
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": m.content or "",
                }],
            })
            continue
        if m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in (m.tool_calls or []):
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments or {},
                })
            out.append({"role": "assistant", "content": blocks or [
                {"type": "text", "text": ""},
            ]})
            continue
        # role == "user"
        out.append({"role": "user", "content": m.content or ""})
    return ("\n".join(system_parts).strip(), out)


def _tool_to_anthropic(
    t: ToolSchema, *, name_override: str | None = None,
) -> dict[str, Any]:
    """Anthropic uses `input_schema` (not `parameters`) for tool JSON Schemas
    and requires names to match ^[a-zA-Z0-9_-]{1,128}$ (no dots)."""
    return {
        "name": name_override or t.name,
        "description": t.description,
        "input_schema": t.parameters,
    }


def _sanitize_tool_name(name: str) -> str:
    """Make a tool name Anthropic-legal: dots → underscores.

    Our cross-backend convention is `<service>.<action>` (e.g.,
    `eval_server.submit_eval`). Anthropic rejects dots. Sanitisation
    is reversible via the per-call lookup map in complete().
    """
    return name.replace(".", "_")


def _rename_tool_uses_in_messages(
    messages: list[ChatMessage], sanitized_map: dict[str, str],
) -> list[ChatMessage]:
    """When replaying prior turns, rename tool_calls inside assistant
    messages so they match the names we'll send in this call.

    A trace recorded with the dot-form name needs to be rewritten to
    the sanitised name before being sent back to Anthropic, or the
    `tool_use_id`/name pairing on the tool_result blocks won't line
    up with what the model expects.
    """
    if not sanitized_map:
        return messages
    # Build reverse: original (dot) → sanitised (underscore).
    reverse = {orig: sani for sani, orig in sanitized_map.items()}
    out: list[ChatMessage] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            renamed = [
                ToolCall(
                    id=tc.id,
                    name=reverse.get(tc.name, tc.name),
                    arguments=tc.arguments,
                )
                for tc in m.tool_calls
            ]
            out.append(ChatMessage(
                role=m.role, content=m.content,
                tool_call_id=m.tool_call_id, name=m.name,
                tool_calls=renamed,
            ))
        else:
            out.append(m)
    return out


def _map_stop_reason(reason: str | None) -> str:
    """Translate Anthropic stop_reason → our cross-backend finish_reason."""
    if reason in (None, "end_turn", "stop_sequence"):
        return "stop"
    if reason == "tool_use":
        return "tool_calls"
    if reason == "max_tokens":
        return "length"
    return str(reason)


def from_env() -> AnthropicClient:
    """Construct from environment variables."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set; set it before constructing the "
            "Anthropic backend."
        )
    return AnthropicClient(
        model=os.environ.get("ANTHROPIC_MODEL", AnthropicClient.DEFAULT_MODEL),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens_default=int(os.environ.get(
            "ANTHROPIC_MAX_TOKENS", AnthropicClient.DEFAULT_MAX_TOKENS,
        )),
    )
