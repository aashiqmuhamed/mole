"""Unit tests for the agent stateless tool-dispatch loop.

These exercise the loop's behavior across success / error / edge paths using
fake LLM clients and fake dispatchers — no network, no real models.
"""
from __future__ import annotations

import asyncio

import pytest

from mole.agent.loop import (
    AgentLoopResult,
    ToolResult,
    run_agent_loop,
)
from mole.llm import (
    ChatMessage,
    ChatResponse,
    TokenUsage,
    ToolCall,
    ToolSchema,
)


def _msg(role, content) -> ChatMessage:
    return ChatMessage(role=role, content=content)


def _tool() -> ToolSchema:
    return ToolSchema(name="echo", description="", parameters={"type": "object"})


class _ScriptedLLM:
    """Returns the next ChatResponse from a pre-loaded script."""
    backend = "scripted"
    model_id = "scripted-1"

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    async def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=None, seed=None):
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("Scripted LLM ran out of responses")
        return self._responses.pop(0)


def _resp(content="", tool_calls=None, finish_reason="stop", usage=(0, 0)) -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(input_tokens=usage[0], output_tokens=usage[1]),
        backend="scripted",
        model_id="scripted-1",
        finish_reason=finish_reason,
    )


# ---- Basic flow -----------------------------------------------------------


def test_loop_returns_immediately_when_no_tool_calls():
    llm = _ScriptedLLM([_resp(content="hello", finish_reason="stop", usage=(10, 5))])

    async def dispatcher(_tc):
        pytest.fail("dispatcher should not be called")

    result = asyncio.run(run_agent_loop(
        llm=llm,
        initial_messages=[_msg("system", "be terse"), _msg("user", "hi")],
        tools=[],
        dispatcher=dispatcher,
        max_turns=5,
    ))
    assert result.turn_count == 1
    assert result.final_message == "hello"
    assert result.tool_calls_made == []
    assert result.aborted_reason is None
    assert result.total_input_tokens == 10
    assert result.total_output_tokens == 5


def test_loop_dispatches_tool_then_finishes():
    tc = ToolCall(id="tc-1", name="echo", arguments={"x": 1})
    llm = _ScriptedLLM([
        _resp(content="calling tool", tool_calls=[tc], finish_reason="tool_calls", usage=(20, 5)),
        _resp(content="done", finish_reason="stop", usage=(30, 4)),
    ])

    async def dispatcher(call):
        assert call.id == "tc-1"
        return ToolResult(tool_call_id=call.id, name=call.name, content='{"ok":true}')

    result = asyncio.run(run_agent_loop(
        llm=llm,
        initial_messages=[_msg("user", "do it")],
        tools=[_tool()],
        dispatcher=dispatcher,
        max_turns=5,
    ))
    assert result.turn_count == 2
    assert result.final_message == "done"
    assert [c.name for c in result.tool_calls_made] == ["echo"]
    assert result.total_input_tokens == 50
    assert result.total_output_tokens == 9


def test_tool_message_is_threaded_back_into_conversation():
    tc = ToolCall(id="tc-1", name="echo", arguments={})
    llm = _ScriptedLLM([
        _resp(content="t1", tool_calls=[tc], finish_reason="tool_calls"),
        _resp(content="final", finish_reason="stop"),
    ])

    async def dispatcher(call):
        return ToolResult(tool_call_id=call.id, name=call.name, content="RESULT")

    result = asyncio.run(run_agent_loop(
        llm=llm,
        initial_messages=[_msg("user", "do it")],
        tools=[_tool()],
        dispatcher=dispatcher,
        max_turns=5,
    ))
    # Second LLM call must have seen the tool-result message we appended.
    tool_msgs = [m for m in llm.calls[1] if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "tc-1"
    assert tool_msgs[0].content == "RESULT"


def test_assistant_message_carries_tool_calls_after_tool_request():
    """When the LLM returns tool_calls, the assistant message we append back
    into the conversation must include those tool_calls — otherwise the
    chat-completions wire format rejects the subsequent tool-result message."""
    tc = ToolCall(id="tc-1", name="echo", arguments={"x": 1})
    llm = _ScriptedLLM([
        _resp(content="calling", tool_calls=[tc], finish_reason="tool_calls"),
        _resp(content="done", finish_reason="stop"),
    ])

    async def dispatcher(call):
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")

    asyncio.run(run_agent_loop(
        llm=llm,
        initial_messages=[_msg("user", "go")],
        tools=[_tool()],
        dispatcher=dispatcher,
        max_turns=5,
    ))
    # On the second LLM call, the assistant message preceding the tool result
    # must carry tool_calls.
    assistant_msgs = [m for m in llm.calls[1] if m.role == "assistant"]
    assert assistant_msgs, "no assistant message in turn 2 input"
    last_assistant = assistant_msgs[-1]
    assert last_assistant.tool_calls is not None, (
        "assistant message before tool result must carry tool_calls"
    )
    assert last_assistant.tool_calls[0].id == "tc-1"


# ---- Error paths ----------------------------------------------------------


def test_dispatcher_exception_becomes_tool_error_message():
    tc = ToolCall(id="tc-1", name="echo", arguments={})
    llm = _ScriptedLLM([
        _resp(tool_calls=[tc], finish_reason="tool_calls"),
        _resp(content="recovered", finish_reason="stop"),
    ])

    async def dispatcher(_call):
        raise ValueError("nope")

    result = asyncio.run(run_agent_loop(
        llm=llm,
        initial_messages=[_msg("user", "do it")],
        tools=[_tool()],
        dispatcher=dispatcher,
        max_turns=5,
    ))
    # Loop survives the exception and gives the model a chance to recover.
    assert result.turn_count == 2
    assert result.final_message == "recovered"
    assert any(tr.is_error for tr in result.tool_results)


def test_llm_exception_aborts_loop_with_reason():
    class BrokenLLM:
        backend = "broken"
        model_id = "x"
        async def complete(self, *a, **kw):
            raise RuntimeError("network down")

    async def dispatcher(_):
        pytest.fail("should not be called")

    result = asyncio.run(run_agent_loop(
        llm=BrokenLLM(),
        initial_messages=[_msg("user", "do it")],
        tools=[],
        dispatcher=dispatcher,
        max_turns=5,
    ))
    assert result.aborted_reason is not None
    assert "llm_error" in result.aborted_reason


def test_max_turns_aborts_when_agent_loops_forever():
    # An LLM that always requests another tool call → loop hits the cap.
    forever_tc = lambda: ToolCall(id="x", name="echo", arguments={})  # noqa: E731
    llm = _ScriptedLLM([
        _resp(tool_calls=[forever_tc()], finish_reason="tool_calls"),
        _resp(tool_calls=[forever_tc()], finish_reason="tool_calls"),
        _resp(tool_calls=[forever_tc()], finish_reason="tool_calls"),
    ])

    async def dispatcher(call):
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")

    result = asyncio.run(run_agent_loop(
        llm=llm,
        initial_messages=[_msg("user", "go")],
        tools=[_tool()],
        dispatcher=dispatcher,
        max_turns=3,
    ))
    assert result.aborted_reason == "max_turns"
    assert result.turn_count == 3


# ---- Event hook -----------------------------------------------------------


def test_event_hook_fires_for_each_phase():
    tc = ToolCall(id="tc-1", name="echo", arguments={})
    llm = _ScriptedLLM([
        _resp(tool_calls=[tc], finish_reason="tool_calls"),
        _resp(content="done", finish_reason="stop"),
    ])

    async def dispatcher(call):
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")

    events = []
    asyncio.run(run_agent_loop(
        llm=llm,
        initial_messages=[_msg("user", "go")],
        tools=[_tool()],
        dispatcher=dispatcher,
        max_turns=5,
        on_event=events.append,
    ))
    types_seen = {e["type"] for e in events}
    assert types_seen == {"llm.request", "llm.response", "tool.request", "tool.result"}


def test_event_hook_exception_does_not_break_loop():
    llm = _ScriptedLLM([_resp(content="done", finish_reason="stop")])

    async def dispatcher(_):
        return ToolResult(tool_call_id="", name="", content="")

    def bad_hook(_evt):
        raise RuntimeError("hook crashed")

    result = asyncio.run(run_agent_loop(
        llm=llm,
        initial_messages=[_msg("user", "go")],
        tools=[],
        dispatcher=dispatcher,
        max_turns=5,
        on_event=bad_hook,
    ))
    assert result.final_message == "done"
    assert result.aborted_reason is None
