"""Local vLLM (or any OpenAI-compatible server) backend.

Points the OpenAI SDK at a self-hosted vLLM `/v1` endpoint — e.g. Kimi-K2.6
served with `--tool-call-parser kimi_k2 --enable-auto-tool-choice
--enable-prefix-caching`. Prefix caching is a big throughput win here: the static
tool-schema + persona prefix is re-sent every turn (input:output ≈ 72:1), so
reusing it skips prefill and raises the concurrent-request ceiling. Keep that
prefix byte-stable on the client side — see the PREFIX-CACHING CONTRACT note in
generator/agentic_member._run_session.

Two kimi tool-call workarounds are applied here because they are *model*
behaviors (kimi), not client-specific:
  1. Dotted tool-name sanitization — kimi rejects names containing `.`; our
     convention is `<service>.<action>`, so dots are swapped to `_` on the way
     out and reversed when parsing tool_calls back.
  2. Claude-style XML `<invoke …>` fallback — kimi occasionally emits tool calls
     as XML inside `content` instead of structured `tool_calls`. vLLM's
     `kimi_k2` parser usually handles this server-side, but the fallback is kept
     as a cheap safety net.

Env:
  VLLM_BASE_URL   required, e.g. http://localhost:8000/v1
  VLLM_MODEL      required, the --served-model-name, e.g. kimi-k2.6
  VLLM_API_KEY    optional; vLLM ignores auth by default (default "EMPTY")
  VLLM_DISABLE_THINKING  optional; "1" forwards chat_template_kwargs.enable_thinking=False
                  (for Qwen3-style hybrid reasoning models). Unset for non-thinking models.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import threading

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .base import ChatMessage, ChatResponse, TokenUsage, ToolCall, ToolSchema
from .toolcalls import _parse_xml_function_calls, _strip_xml_function_calls


class VLLMClient:
    """LLMClient for a local OpenAI-compatible vLLM endpoint."""

    backend = "vllm"

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        self.model_id = model
        self._model = model
        # base_url may be a comma-separated list of OpenAI-compatible endpoints; we
        # round-robin requests across them so the sim can drive multiple vLLM servers at
        # once (the LLM server is the throughput bottleneck, so N servers ~= Nx). A single
        # URL is a no-op. Pair with a higher --concurrency (~ servers x per-server cap).
        urls = [u.strip() for u in base_url.split(",") if u.strip()]
        self._clients = [OpenAI(base_url=u, api_key=api_key or "EMPTY") for u in urls]
        self._rr = itertools.cycle(range(len(self._clients)))
        self._rr_lock = threading.Lock()

    def _next_client(self) -> OpenAI:
        if len(self._clients) == 1:
            return self._clients[0]
        with self._rr_lock:
            return self._clients[next(self._rr)]

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        # kimi rejects dotted tool names — sanitize out, reverse on parse.
        sanitized_map: dict[str, str] = {}
        if tools:
            for t in tools:
                if "." in t.name:
                    sanitized_map[t.name.replace(".", "_")] = t.name

        def _call() -> ChatResponse:
            # Build the request as a raw JSON body and POST it directly, bypassing
            # the OpenAI SDK's per-call maybe_transform. That transform is a recursive
            # pure-Python walk of the ENTIRE request (all messages + tool schemas) on
            # every turn, so it scales with context length and re-runs each turn
            # (~O(n^2) over a session). Because it runs under the GIL in the worker
            # thread, it serializes Python execution and starves the event loop, so the
            # sim can only keep a handful of requests in flight and the vLLM server runs
            # a tiny batch despite --concurrency. `messages`/`tools` are already in
            # API-JSON shape, so the body we POST is byte-identical to what .create()
            # would have sent (proven by test); the thread-pool concurrency model is
            # unchanged, we just drop the wasted CPU.
            body: dict = {
                "model": self._model,
                "messages": [m.to_openai() for m in messages],
                "temperature": temperature,
            }
            if tools:
                tool_payload = []
                for t in tools:
                    raw = t.to_openai()
                    if "function" in raw:
                        raw = dict(raw)
                        raw["function"] = dict(raw["function"])
                        raw["function"]["name"] = raw["function"]["name"].replace(".", "_")
                    tool_payload.append(raw)
                body["tools"] = tool_payload
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            if seed is not None:
                body["seed"] = seed
            if os.environ.get("VLLM_DISABLE_THINKING") == "1":
                # Hybrid reasoning models (e.g. Qwen3) default to "thinking": they burn the
                # per-call token budget on a parsed-away <think> block (empty content at small
                # budgets, ~10-60x more tokens otherwise). Forward enable_thinking=False via the
                # chat template. .create() merges extra_body into the top-level request body, so
                # we set it directly on `body` to keep the sent JSON identical. Env-gated so
                # non-thinking backends (kimi) are untouched.
                body["chat_template_kwargs"] = {"enable_thinking": False}

            resp = self._next_client().post(
                "/chat/completions", cast_to=ChatCompletion, body=body,
            )
            choice = resp.choices[0]
            tool_calls: list[ToolCall] = []
            for tc in choice.message.tool_calls or []:
                args = tc.function.arguments
                returned_name = tc.function.name
                original_name = sanitized_map.get(returned_name, returned_name)
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=original_name,
                        arguments=json.loads(args) if isinstance(args, str) else (args or {}),
                    )
                )

            content_text = choice.message.content or ""
            # Reasoning-leak guard. Some served reasoning models (observed: DeepSeek-V4) pre-seed the
            # assistant turn INSIDE a <think> block via the chat template, so the model's stream emits a
            # bare closing </think> with no opening tag and vLLM's reasoning parser fails to split it —
            # the raw chain-of-thought then lands in `content`. Drop everything up to and including a
            # leaked </think> so the transcript (and every downstream labeler/judge/affordance that reads
            # message content) carries the model's actual answer, not its monologue. Qwen-family runs
            # avoid this upstream via VLLM_DISABLE_THINKING=1; this is the belt for models lacking a
            # working thinking-disable. See TODO in llm_harm_label.py for pre-fix corpora already on disk.
            if "</think>" in content_text:
                content_text = content_text.split("</think>", 1)[1].lstrip()
            if not tool_calls and "<invoke" in content_text:
                tool_calls = _parse_xml_function_calls(content_text, sanitized_map)
                if tool_calls:
                    content_text = _strip_xml_function_calls(content_text)

            usage = resp.usage
            return ChatResponse(
                content=content_text,
                tool_calls=tool_calls,
                usage=TokenUsage(
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                ),
                backend=self.backend,
                model_id=self._model,
                finish_reason=choice.finish_reason or "stop",
            )

        return await asyncio.to_thread(_call)


def from_env() -> VLLMClient:
    """Construct from `VLLM_BASE_URL` / `VLLM_MODEL` / `VLLM_API_KEY`."""
    base_url = os.environ.get("VLLM_BASE_URL")
    model = os.environ.get("VLLM_MODEL")
    if not base_url or not model:
        raise ValueError(
            "vllm backend requires VLLM_BASE_URL (e.g. http://host:8000/v1) "
            "and VLLM_MODEL (the --served-model-name, e.g. kimi-k2.6)"
        )
    return VLLMClient(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
