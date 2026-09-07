"""OpenRouter backend — hosted OpenAI-v1 chat-completions via openrouter.ai.

The default reproduction backend: hosted models behind an OpenAI-v1 API
(base_url https://openrouter.ai/api/v1) with plain API-key auth. The key is a
static secret read once from OPENROUTER_API_KEY (process env only, never a file
on a shared disk).

Robustness for reasoning / DeepSeek-style models that emit malformed
`{}`-prefixed tool-call args: raw `.post()` body bypass (skips the OpenAI SDK's
O(n^2) per-call maybe_transform that serializes the event loop on long sessions),
malformed-arg-tolerant `_parse_tool_args`, empty-200 degrade, dotted tool-name
sanitization, and the `</think>` / `<invoke>` reasoning-leak guards.

Env:
  OPENROUTER_API_KEY   required — the OpenRouter API key (process env ONLY).
  OPENROUTER_MODEL     model slug, e.g. `deepseek/deepseek-v4-pro` (default) or
                       `deepseek/deepseek-v4-flash`.
  OPENROUTER_BASE_URL  default https://openrouter.ai/api/v1.
  OPENROUTER_REFERER / OPENROUTER_TITLE  optional attribution headers OpenRouter
                       uses for its public rankings (harmless if unset).
  OPENROUTER_SEND_SEED=1  opt into sending `seed` (off by default; temp-0 moots it,
                       and an unknown-param 400 would kill sessions).
"""
from __future__ import annotations

import asyncio
import json
import os

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .base import ChatMessage, ChatResponse, TokenUsage, ToolCall, ToolSchema
from .toolcalls import _parse_xml_function_calls, _strip_xml_function_calls

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"


def _parse_tool_args(args) -> dict:
    """Tolerantly parse tool-call arguments into a dict.

    Some DeepSeek deployments emit an empty `{}` prefix before the real object,
    so a bare `json.loads` raises "Extra data" and, being non-retryable, kills the
    whole session at turn 1. Walk successive JSON values and take the first
    non-empty dict; never crash a session over one bad tool call.
    """
    if not isinstance(args, str):
        return args or {}
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError:
        parsed, dec, i = {}, json.JSONDecoder(), 0
        while i < len(args):
            while i < len(args) and args[i].isspace():
                i += 1
            if i >= len(args):
                break
            try:
                obj, i = dec.raw_decode(args, i)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict) and obj:
                parsed = obj
                break
    return parsed if isinstance(parsed, dict) else {}


class OpenRouterClient:
    """LLMClient for an OpenRouter OpenAI-v1 chat-completions model."""

    backend = "openrouter"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for LLM_BACKEND=openrouter "
                "(set it in the process environment; never a file on a shared disk)."
            )
        self.model_id = model
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._send_seed = os.environ.get("OPENROUTER_SEND_SEED") == "1"
        # Optional OpenRouter attribution headers (used for its public rankings).
        headers: dict[str, str] = {}
        if os.environ.get("OPENROUTER_REFERER"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_REFERER"]
        if os.environ.get("OPENROUTER_TITLE"):
            headers["X-Title"] = os.environ["OPENROUTER_TITLE"]
        # Static API key (OpenRouter keys don't expire hourly like short-lived cloud tokens),
        # so set once at construction. One client; the connection pool is reused.
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=key,
            default_headers=headers or None,
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        # Some deployments reject dotted tool names (our convention is
        # `<service>.<action>`); sanitize on the way out, reverse on parse.
        sanitized_map: dict[str, str] = {}
        if tools:
            for t in tools:
                if "." in t.name:
                    sanitized_map[t.name.replace(".", "_")] = t.name

        def _call() -> ChatResponse:
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
            if seed is not None and self._send_seed:
                body["seed"] = seed

            resp = self._client.post(
                "/chat/completions", cast_to=ChatCompletion, body=body,
            )
            # OpenAI-compatible 200 with empty `choices` (oversized prompt / flap):
            # degrade to an empty completion so the retry layer's empty-200 path
            # handles it instead of an IndexError surfacing.
            if not resp.choices:
                u = getattr(resp, "usage", None)
                return ChatResponse(
                    content="", tool_calls=[],
                    usage=TokenUsage(
                        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(u, "completion_tokens", 0) or 0,
                    ),
                    backend=self.backend, model_id=self._model,
                    finish_reason="empty_choices",
                )

            choice = resp.choices[0]
            tool_calls: list[ToolCall] = []
            for tc in choice.message.tool_calls or []:
                returned_name = tc.function.name
                original_name = sanitized_map.get(returned_name, returned_name)
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=original_name,
                        arguments=_parse_tool_args(tc.function.arguments),
                    )
                )

            content_text = choice.message.content or ""
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


def from_env() -> OpenRouterClient:
    """Construct from OPENROUTER_* environment variables."""
    return OpenRouterClient(
        model=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    )
