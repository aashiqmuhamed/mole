"""A reasoning-only turn must not end the agent's task.

Regression test for the gpt-oss-120b measurement bug (2026-07-15). Reasoning
models can end a turn having only thought: no answer, no tool call, chain-of-
thought in `reasoning_content`. Some backends surface that as `content`, and
the loop used to end the task on ANY turn without a tool call -- so an unfinished
thought read as "done". gpt-oss-120b was crushed to 5.6 turns / 3.0 tool calls per
session (vs 16-32 / 14-53 for models that answer normally) and scored a spurious
0/40 executed, which nearly reached the paper as "the safest model measured".
"""
import asyncio

from mole.llm.base import ChatResponse, TokenUsage, ToolCall


def _resp(content="", tool_calls=None, finish_reason="stop"):
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        backend="test",
        model_id="test",
        finish_reason=finish_reason,
    )


class _ScriptedClient:
    """Replays a fixed list of responses and records the messages it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def complete(self, messages, tools=None, **kw):
        self.calls.append(list(messages))
        return self._responses.pop(0)


class _NoopDispatcher:
    async def dispatch(self, name, arguments):
        return {"ok": True}


def _run(responses):
    from mole.agent.loop import run_agent_loop
    from mole.llm.base import ChatMessage

    client = _ScriptedClient(responses)
    result = asyncio.run(run_agent_loop(
        llm=client,
        initial_messages=[ChatMessage(role="user", content="do the work")],
        tools=[],
        dispatcher=_NoopDispatcher(),
        max_turns=6,
    ))
    return client, result


def test_reasoning_only_turn_is_nudged_not_treated_as_done():
    """A think-only turn must NOT end the task; the agent gets to act afterwards."""
    client, result = _run([
        # turn 1: the model only thinks -- this used to end the session outright
        _resp(content="Let's list the files first.", finish_reason="reasoning_only"),
        # turn 2: after the nudge it acts
        _resp(tool_calls=[ToolCall(id="1", name="noop", arguments={})],
              finish_reason="tool_calls"),
        # turn 3: a genuine final answer -> now the task may end
        _resp(content="Done.", finish_reason="stop"),
    ])

    # it did not stop at the thought: all three responses were consumed
    assert len(client.calls) == 3, f"loop stopped early after {len(client.calls)} turn(s)"
    # and the tool call after the nudge actually ran
    assert len(result.tool_calls_made) == 1
    # the nudge was delivered as a user message
    nudged = [m for m in client.calls[1] if m.role == "user" and "Continue" in (m.content or "")]
    assert nudged, "no nudge sent after the reasoning-only turn"


def test_normal_stop_still_ends_the_task():
    """The fix must not make ordinary final answers loop forever."""
    client, _ = _run([_resp(content="All done.", finish_reason="stop")])
    assert len(client.calls) == 1, "a real final answer should end the task immediately"


def test_reasoning_only_is_distinct_from_stop():
    """The backend must not label a think-only turn as a normal 'stop'.

    `stop` means the model gave a final answer and the loop may end the task.
    A think-only turn is unfinished; conflating them is the bug.
    """
    thinking = _resp(content="I should check the repo.", finish_reason="reasoning_only")
    answered = _resp(content="Here is the summary.", finish_reason="stop")

    assert thinking.finish_reason == "reasoning_only"
    assert answered.finish_reason == "stop"
    assert thinking.finish_reason != answered.finish_reason
    # both carry text, so content alone cannot distinguish them -- which is exactly
    # why the loop needs the finish_reason and not a content heuristic
    assert thinking.content and answered.content
    assert not thinking.tool_calls and not answered.tool_calls
