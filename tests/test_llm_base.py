"""Unit tests for the LLM source-layer data shapes.

These tests catch JSON / argument-shape regressions in the wire format,
without needing network access. They verify that messages and tool schemas
serialize to exactly the shape the OpenAI-compatible chat completions API
expects (which is what OpenRouter, vLLM, and other OpenAI-compatible servers
all consume).
"""
from __future__ import annotations

import pytest

from mole.llm import (
    ChatMessage,
    ChatResponse,
    TokenUsage,
    ToolCall,
    ToolSchema,
)


# ---- ChatMessage.to_openai ------------------------------------------------


def test_system_message_to_openai_has_role_and_content():
    m = ChatMessage(role="system", content="You are terse.")
    out = m.to_openai()
    assert out == {"role": "system", "content": "You are terse."}


def test_user_message_to_openai_has_role_and_content():
    m = ChatMessage(role="user", content="hello")
    assert m.to_openai() == {"role": "user", "content": "hello"}


def test_assistant_message_to_openai_omits_tool_fields():
    m = ChatMessage(role="assistant", content="hi")
    out = m.to_openai()
    assert "tool_call_id" not in out
    assert "name" not in out
    assert "tool_calls" not in out


def test_assistant_message_with_tool_calls_serializes_them():
    """Assistant messages that requested tools must round-trip the tool_calls
    field, otherwise the next tool-result message fails the wire-format contract."""
    import json
    m = ChatMessage(
        role="assistant",
        content="I'll use a tool.",
        tool_calls=[
            ToolCall(id="tc-1", name="read_inbox", arguments={"max_count": 5}),
            ToolCall(id="tc-2", name="send_email", arguments={"to": "a@b.c"}),
        ],
    )
    out = m.to_openai()
    assert "tool_calls" in out
    assert len(out["tool_calls"]) == 2
    tc0 = out["tool_calls"][0]
    assert tc0["id"] == "tc-1"
    assert tc0["type"] == "function"
    assert tc0["function"]["name"] == "read_inbox"
    # arguments must be a JSON string per the wire format, not a dict.
    assert isinstance(tc0["function"]["arguments"], str)
    assert json.loads(tc0["function"]["arguments"]) == {"max_count": 5}


def test_tool_message_to_openai_includes_tool_call_id():
    m = ChatMessage(
        role="tool", content="[]", tool_call_id="tc-1", name="read_inbox",
    )
    out = m.to_openai()
    assert out["role"] == "tool"
    assert out["tool_call_id"] == "tc-1"
    assert out["name"] == "read_inbox"
    assert out["content"] == "[]"


def test_tool_message_without_tool_call_id_raises():
    m = ChatMessage(role="tool", content="[]")
    with pytest.raises(AssertionError):
        m.to_openai()


# ---- ToolSchema.to_openai -------------------------------------------------


def test_tool_schema_wraps_function_in_openai_envelope():
    s = ToolSchema(
        name="send_email",
        description="Send an email.",
        parameters={
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    )
    out = s.to_openai()
    assert out["type"] == "function"
    assert out["function"]["name"] == "send_email"
    assert out["function"]["description"] == "Send an email."
    assert out["function"]["parameters"]["type"] == "object"
    assert "to" in out["function"]["parameters"]["properties"]


# ---- ToolCall + ChatResponse ---------------------------------------------


def test_tool_call_carries_id_name_args():
    tc = ToolCall(id="tc-9", name="commit", arguments={"path": "a.py", "message": "x"})
    assert tc.id == "tc-9"
    assert tc.name == "commit"
    assert tc.arguments == {"path": "a.py", "message": "x"}


def test_chat_response_default_fields_empty():
    r = ChatResponse(content="hi")
    assert r.tool_calls == []
    assert r.usage.input_tokens == 0
    assert r.usage.output_tokens == 0
    assert r.finish_reason == "stop"
    assert r.backend == ""


def test_chat_response_with_usage_and_tool_calls():
    r = ChatResponse(
        content="will use a tool",
        tool_calls=[ToolCall(id="tc-1", name="x", arguments={})],
        usage=TokenUsage(input_tokens=50, output_tokens=10),
        backend="openrouter",
        model_id="gpt-x",
        finish_reason="tool_calls",
    )
    assert len(r.tool_calls) == 1
    assert r.usage.input_tokens == 50
    assert r.finish_reason == "tool_calls"


# ---- Runtime Protocol check -----------------------------------------------


def test_concrete_clients_satisfy_protocol_signature():
    """Each concrete client must implement complete() with the right shape.

    We import lazily inside the test so a missing SDK doesn't break collection.
    """
    import inspect

    from mole.llm import LLMClient
    from mole.llm.openrouter import OpenRouterClient
    from mole.llm.vllm import VLLMClient
    from mole.llm.anthropic import AnthropicClient

    for cls in (OpenRouterClient, VLLMClient, AnthropicClient):
        assert hasattr(cls, "backend")
        assert hasattr(cls, "complete")
        sig = inspect.signature(cls.complete)
        # `messages` is the only positional after self; the rest are keyword-only.
        assert "messages" in sig.parameters
        for kw in ("tools", "temperature", "max_tokens", "seed"):
            assert kw in sig.parameters, f"{cls.__name__}.complete missing {kw}"
