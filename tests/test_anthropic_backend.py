"""Unit tests for the Anthropic LLM backend.

Mocks the anthropic SDK so no API key or network call is needed at test
time. Covers: system-prompt extraction, tool-use translation in both
directions, tool-result threading, stop-reason mapping.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mole.llm.anthropic import (
    AnthropicClient,
    _map_stop_reason,
    _rename_tool_uses_in_messages,
    _sanitize_tool_name,
    _split_system_and_convert,
    _tool_to_anthropic,
)
from mole.llm.base import (
    ChatMessage, ToolCall, ToolSchema,
)


# ── conversion helpers ────────────────────────────────────────────────


def test_system_message_is_lifted_out():
    msgs = [
        ChatMessage(role="system", content="you are kara.p"),
        ChatMessage(role="user", content="hi"),
    ]
    system, out = _split_system_and_convert(msgs)
    assert system == "you are kara.p"
    assert out == [{"role": "user", "content": "hi"}]


def test_multiple_system_messages_are_concatenated():
    msgs = [
        ChatMessage(role="system", content="part 1"),
        ChatMessage(role="system", content="part 2"),
        ChatMessage(role="user", content="hello"),
    ]
    system, out = _split_system_and_convert(msgs)
    assert system == "part 1\npart 2"


def test_tool_result_becomes_user_message_with_tool_result_block():
    msgs = [
        ChatMessage(role="tool", content='{"ok": true}',
                    tool_call_id="tu_abc123", name="org_whoami"),
    ]
    _, out = _split_system_and_convert(msgs)
    assert out == [{
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "tu_abc123",
            "content": '{"ok": true}',
        }],
    }]


def test_assistant_with_tool_calls_emits_text_plus_tool_use_blocks():
    msgs = [
        ChatMessage(
            role="assistant",
            content="Let me check.",
            tool_calls=[
                ToolCall(id="tu_1", name="org_whoami", arguments={"account": "kara.p"}),
            ],
        ),
    ]
    _, out = _split_system_and_convert(msgs)
    assert out == [{
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu_1", "name": "org_whoami",
             "input": {"account": "kara.p"}},
        ],
    }]


def test_assistant_with_only_tool_call_still_gets_a_block():
    msgs = [
        ChatMessage(
            role="assistant", content="",
            tool_calls=[ToolCall(id="tu_x", name="ping", arguments={})],
        ),
    ]
    _, out = _split_system_and_convert(msgs)
    assert out[0]["role"] == "assistant"
    assert any(b["type"] == "tool_use" for b in out[0]["content"])


def test_user_message_passes_through_verbatim():
    msgs = [ChatMessage(role="user", content="hello world")]
    _, out = _split_system_and_convert(msgs)
    assert out == [{"role": "user", "content": "hello world"}]


# ── tool-schema conversion ───────────────────────────────────────────


def test_tool_to_anthropic_uses_input_schema_not_parameters():
    t = ToolSchema(
        name="org_whoami",
        description="Look up the calling account's profile.",
        parameters={"type": "object", "properties": {"account": {"type": "string"}}},
    )
    out = _tool_to_anthropic(t)
    assert out == {
        "name": "org_whoami",
        "description": "Look up the calling account's profile.",
        "input_schema": {"type": "object", "properties": {"account": {"type": "string"}}},
    }


def test_sanitize_tool_name_replaces_dots():
    """Anthropic rejects dots in tool names. Our convention uses
    `service.action`. The sanitiser maps dots to underscores."""
    assert _sanitize_tool_name("eval_server.submit_eval") == "eval_server_submit_eval"
    assert _sanitize_tool_name("gitlab.commit") == "gitlab_commit"
    assert _sanitize_tool_name("plain_name") == "plain_name"     # no-op when no dot


def test_tool_to_anthropic_honors_name_override():
    t = ToolSchema(
        name="eval_server.submit_eval", description="...",
        parameters={"type": "object"},
    )
    out = _tool_to_anthropic(t, name_override="eval_server_submit_eval")
    assert out["name"] == "eval_server_submit_eval"


def test_rename_tool_uses_rewrites_dot_names_in_assistant_messages():
    """When replaying prior turns through the SDK, the dot-form tool
    names in assistant messages must be rewritten to the sanitised
    form so the tool_use_id ↔ name pairing matches the freshly-sent
    `tools` array on this call."""
    sanitized_map = {"eval_server_submit_eval": "eval_server.submit_eval"}
    msgs = [
        ChatMessage(role="user", content="run the eval"),
        ChatMessage(
            role="assistant", content="ok",
            tool_calls=[ToolCall(
                id="tu_1", name="eval_server.submit_eval",
                arguments={"eval_config": {}, "account": "kara.p"},
            )],
        ),
    ]
    renamed = _rename_tool_uses_in_messages(msgs, sanitized_map)
    assert renamed[1].tool_calls[0].name == "eval_server_submit_eval"
    # User message is left alone.
    assert renamed[0].content == "run the eval"


def test_rename_tool_uses_is_noop_when_map_empty():
    msgs = [
        ChatMessage(
            role="assistant", content="",
            tool_calls=[ToolCall(id="tu_1", name="foo.bar", arguments={})],
        ),
    ]
    out = _rename_tool_uses_in_messages(msgs, {})
    assert out[0].tool_calls[0].name == "foo.bar"


@patch("mole.llm.anthropic.Anthropic")
def test_complete_reverses_sanitized_tool_name_in_response(mock_anthropic_class):
    """Round-trip: dot-form goes out as underscore, comes back as dot."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(
        text="",
        tool_uses=[("tu_1", "eval_server_submit_eval",
                    {"eval_config": {}, "account": "kara.p"})],
        stop_reason="tool_use",
    )
    mock_anthropic_class.return_value = mock_client

    client = AnthropicClient(api_key="sk-test")
    resp = asyncio.run(client.complete(
        [ChatMessage(role="user", content="go")],
        tools=[ToolSchema(name="eval_server.submit_eval", description="d",
                          parameters={"type": "object"})],
    ))
    # The dispatcher sees the original dot-form name, not the sanitised one.
    assert resp.tool_calls[0].name == "eval_server.submit_eval"
    # And we did send the sanitised name to Anthropic.
    sent_tools = mock_client.messages.create.call_args.kwargs["tools"]
    assert sent_tools[0]["name"] == "eval_server_submit_eval"


# ── stop-reason mapping ──────────────────────────────────────────────


def test_stop_reason_map():
    assert _map_stop_reason(None) == "stop"
    assert _map_stop_reason("end_turn") == "stop"
    assert _map_stop_reason("stop_sequence") == "stop"
    assert _map_stop_reason("tool_use") == "tool_calls"
    assert _map_stop_reason("max_tokens") == "length"
    assert _map_stop_reason("weird_new_reason") == "weird_new_reason"


# ── complete() with mocked SDK ───────────────────────────────────────


class _FakeContentBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_response(
    *,
    text: str = "",
    tool_uses: list[tuple[str, str, dict]] | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
):
    """Construct a SimpleNamespace that mimics anthropic.types.Message."""
    content: list[_FakeContentBlock] = []
    if text:
        content.append(_FakeContentBlock(type="text", text=text))
    for tid, tname, targs in (tool_uses or []):
        content.append(_FakeContentBlock(
            type="tool_use", id=tid, name=tname, input=targs,
        ))
    return SimpleNamespace(
        content=content, stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@patch("mole.llm.anthropic.Anthropic")
def test_complete_returns_text_response(mock_anthropic_class):
    """Plain text reply maps into ChatResponse.content."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(
        text="Hello, Kara.", stop_reason="end_turn",
        input_tokens=42, output_tokens=7,
    )
    mock_anthropic_class.return_value = mock_client

    client = AnthropicClient(model="claude-opus-4-7", api_key="sk-test")
    resp = asyncio.run(client.complete(
        [ChatMessage(role="user", content="hi")],
    ))
    assert resp.content == "Hello, Kara."
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.backend == "anthropic"
    assert resp.model_id == "claude-opus-4-7"
    assert resp.usage.input_tokens == 42
    assert resp.usage.output_tokens == 7


@patch("mole.llm.anthropic.Anthropic")
def test_complete_translates_tool_use_to_tool_calls(mock_anthropic_class):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(
        text="Looking up.",
        tool_uses=[("tu_1", "org_whoami", {"account": "kara.p"})],
        stop_reason="tool_use",
    )
    mock_anthropic_class.return_value = mock_client

    client = AnthropicClient(api_key="sk-test")
    resp = asyncio.run(client.complete(
        [ChatMessage(role="user", content="who am i?")],
        tools=[ToolSchema(name="org_whoami", description="...",
                          parameters={"type": "object"})],
    ))
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "tu_1"
    assert tc.name == "org_whoami"
    assert tc.arguments == {"account": "kara.p"}


@patch("mole.llm.anthropic.Anthropic")
def test_complete_passes_system_as_top_level_field(mock_anthropic_class):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(text="ok")
    mock_anthropic_class.return_value = mock_client

    client = AnthropicClient(api_key="sk-test")
    asyncio.run(client.complete([
        ChatMessage(role="system", content="you are kara.p"),
        ChatMessage(role="user", content="hi"),
    ]))
    kwargs = mock_client.messages.create.call_args.kwargs
    # System is now sent as a cache_control'd block (prompt caching enabled by default).
    assert kwargs["system"] == [{
        "type": "text", "text": "you are kara.p",
        "cache_control": {"type": "ephemeral"},
    }]
    assert all(m["role"] != "system" for m in kwargs["messages"])


@patch("mole.llm.anthropic.Anthropic")
def test_complete_threads_tools_with_input_schema(mock_anthropic_class):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(text="done")
    mock_anthropic_class.return_value = mock_client

    client = AnthropicClient(api_key="sk-test")
    asyncio.run(client.complete(
        [ChatMessage(role="user", content="hi")],
        tools=[ToolSchema(name="ping", description="d",
                          parameters={"type": "object", "properties": {}})],
    ))
    kwargs = mock_client.messages.create.call_args.kwargs
    # Last tool now carries cache_control so the tool catalog is cached.
    assert kwargs["tools"][0] == {
        "name": "ping", "description": "d",
        "input_schema": {"type": "object", "properties": {}},
        "cache_control": {"type": "ephemeral"},
    }


@patch("mole.llm.anthropic.Anthropic")
def test_complete_uses_default_max_tokens_when_caller_omits(mock_anthropic_class):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(text="ok")
    mock_anthropic_class.return_value = mock_client

    client = AnthropicClient(api_key="sk-test", max_tokens_default=2048)
    asyncio.run(client.complete([ChatMessage(role="user", content="hi")]))
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 2048


# ── from_env ──────────────────────────────────────────────────────────


def test_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from mole.llm.anthropic import from_env
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        from_env()


def test_from_env_picks_up_model_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    with patch("mole.llm.anthropic.Anthropic") as mock_cls:
        from mole.llm.anthropic import from_env
        c = from_env()
        assert c.model_id == "claude-sonnet-4-6"
        assert mock_cls.called


# ── factory wiring ────────────────────────────────────────────────────


def test_factory_routes_anthropic_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("LLM_RETRY", "0")          # don't wrap with retry for this check
    with patch("mole.llm.anthropic.Anthropic"):
        from mole.llm.factory import build_llm
        client = build_llm("anthropic")
        assert client.backend == "anthropic"


def test_factory_routes_claude_alias(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("LLM_RETRY", "0")
    with patch("mole.llm.anthropic.Anthropic"):
        from mole.llm.factory import build_llm
        client = build_llm("claude")
        assert client.backend == "anthropic"
