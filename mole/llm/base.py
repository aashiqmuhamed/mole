"""LLMClient Protocol — provider-agnostic chat completion + tool calling.

All backends (OpenRouter, local vLLM, Anthropic direct) implement this Protocol
so the agent loop, monitors, and generator can be swapped between providers
without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# -- Messages ---------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    """One message in a chat conversation.

    For an assistant message that requested tool calls, set `tool_calls`. For a
    tool-result message, set `tool_call_id` and `name`.
    """
    role: Role
    content: str
    tool_call_id: str | None = None
    name: str | None = None             # tool name when role=="tool"
    tool_calls: list["ToolCall"] | None = None   # set when assistant requested tools

    def to_openai(self) -> dict[str, Any]:
        """Render in the OpenAI chat.completions wire format."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "tool":
            assert self.tool_call_id, "tool message must have tool_call_id"
            msg["tool_call_id"] = self.tool_call_id
            if self.name:
                msg["name"] = self.name
        if self.role == "assistant" and self.tool_calls:
            import json as _json
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


@dataclass
class ToolCall:
    """A tool invocation requested by the assistant."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass
class ChatResponse:
    """Result of an LLMClient.complete() call."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    backend: str = ""           # "openrouter" | "vllm" | "anthropic"
    model_id: str = ""
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length" | ...


# -- Tool schemas (subset of OpenAI function-calling format) ---------------


@dataclass
class ToolSchema:
    """Tool/function specification advertised to the LLM."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# -- The Protocol ----------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat client. Every backend implements this."""

    backend: str
    model_id: str

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        """Single non-streaming chat completion."""
        ...
