"""Sim-clock plumbing: TaskContext.advance_clock + orchestrator per-stage advances.

The bug this layer fixes: rollup features that depend on time-of-day
(after_hours_rate, etc.) were keyed on wall-clock at event-emission time,
so the same benign threat produced different feature values depending
on when the test was run. After the fix: each stage's `time` field
authoritatively drives the timestamp on every event emitted during it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mole.audit import AuditCollector
from mole.state.context import TaskContext


def _find_audit_jsonl(results_base: Path, task_id: str) -> Path:
    """Locate audit.jsonl under results_base/<task_id>/<any_run_id>/.

    The orchestrator now stamps each run with its own run_id-named
    subdir so concurrent / sequential runs don't overwrite each
    other's traces. Tests should find the most-recent run rather
    than assume a flat layout.
    """
    task_dir = results_base / task_id
    candidates = sorted(task_dir.glob("*/audit.jsonl"))
    assert candidates, f"no audit.jsonl found under {task_dir}"
    return candidates[-1]


# ── TaskContext.advance_clock ─────────────────────────────────────────


def test_task_context_advance_clock_updates_sim_now():
    c = AuditCollector()
    ctx = TaskContext(managers={}, sandbox=None, task_dir=Path("."),
                      audit=c, sim_start="2026-04-06T09:00:00Z")
    assert ctx.sim_now == "2026-04-06T09:00:00Z"
    ctx.advance_clock("2026-04-07T10:00:00Z")
    assert ctx.sim_now == "2026-04-07T10:00:00Z"


def test_task_context_advance_clock_propagates_to_collector():
    c = AuditCollector()
    ctx = TaskContext(managers={}, sandbox=None, task_dir=Path("."),
                      audit=c, sim_start="2026-04-06T09:00:00Z")
    ctx.advance_clock("2026-04-06T22:00:00Z")
    assert c._clock_fn() == "2026-04-06T22:00:00Z"


def test_task_context_advance_clock_safe_without_audit():
    """A ctx with audit=None should not crash on advance_clock."""
    ctx = TaskContext(managers={}, sandbox=None, task_dir=Path("."), audit=None)
    ctx.advance_clock("2026-04-06T09:00:00Z")
    assert ctx.sim_now == "2026-04-06T09:00:00Z"


# ── Orchestrator advances clock between stages ────────────────────────


def test_orchestrator_advances_clock_from_stage_return(tmp_path: Path):
    """A stage's `time` return becomes the ts on all subsequent events."""
    # Compose a tiny task with one stage that returns a specific sim time +
    # a dummy backend that emits an event we can inspect.
    from mole.llm import ChatResponse, TokenUsage
    from mole.orchestrator import run_task_full
    from mole.state.base import StateManager

    if "clock_test" not in StateManager._registry:
        @StateManager.register("clock_test")
        class _M(StateManager):
            NEEDS_SANDBOX = False
            async def setup(self, *, sandbox): pass
            async def cleanup(self): pass
            async def ping(self, *, who: str) -> str:
                return f"pong:{who}"

    task_py = tmp_path / "task.py"
    task_py.write_text(
        "METADATA = {\n"
        '    "id": "clock_test_task",\n'
        '    "name": "clock test",\n'
        '    "category": "test",\n'
        '    "environments": ["clock_test"],\n'
        '    "focal_account": "bob.li",\n'
        "}\n"
        'PROMPT = "Test."\n'
        "async def stage0(ctx):\n"
        '    return {"notification": "x", "time": "2026-04-06T22:00:00Z"}\n'
        "async def _always_passes(ctx): return True\n"
        'RUBRIC = {"final": [{"id": "u.ok", "checker": _always_passes, "weight": 1.0}]}\n',
        encoding="utf-8",
    )

    class _ToolingLLM:
        backend = "fake"
        model_id = "fake-1"
        def __init__(self): self.calls = 0
        async def complete(self, messages, *, tools=None, temperature=0.0,
                           max_tokens=None, seed=None):
            self.calls += 1
            if self.calls == 1:
                from mole.llm import ToolCall
                return ChatResponse(
                    content="calling", tool_calls=[ToolCall(
                        id="t1", name="clock_test.ping", arguments={"who": "x"},
                    )],
                    usage=TokenUsage(0, 0), backend="fake", model_id="fake-1",
                    finish_reason="tool_calls",
                )
            return ChatResponse(
                content="done", tool_calls=[], usage=TokenUsage(0, 0),
                backend="fake", model_id="fake-1", finish_reason="stop",
            )

    result = asyncio.run(run_task_full(
        task_dir=tmp_path, llm=_ToolingLLM(),
        results_dir=tmp_path / "results",
    ))
    assert result.error == ""

    audit_path = _find_audit_jsonl(tmp_path / "results", result.task_id)
    assert audit_path.exists()
    recs = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    ping_events = [r for r in recs if r["action"] == "ping"]
    assert ping_events
    # ALL ping events have the simulated stage time, not the wall-clock time
    # the test happened to run at.
    for r in ping_events:
        assert r["ts"] == "2026-04-06T22:00:00Z"


def test_orchestrator_stage_advance_propagates_to_later_stages(tmp_path: Path):
    """Stage 0 advances to T1; stage 1 events should still use T1 unless
    stage 1 returns its own `time` field."""
    from mole.llm import ChatResponse, ToolCall, TokenUsage
    from mole.orchestrator import run_task_full
    from mole.state.base import StateManager

    if "clock_propagate" not in StateManager._registry:
        @StateManager.register("clock_propagate")
        class _M2(StateManager):
            NEEDS_SANDBOX = False
            async def setup(self, *, sandbox): pass
            async def cleanup(self): pass
            async def beep(self) -> str: return "beep"

    task_py = tmp_path / "task.py"
    task_py.write_text(
        "METADATA = {\n"
        '    "id": "clock_propagate_task",\n'
        '    "name": "clock propagate",\n'
        '    "category": "test",\n'
        '    "environments": ["clock_propagate"],\n'
        '    "focal_account": "bob.li",\n'
        "}\n"
        'PROMPT = "Test."\n'
        "async def stage0(ctx):\n"
        '    return {"notification": "s0", "time": "2026-04-06T09:00:00Z"}\n'
        "async def stage1(ctx):\n"
        '    return {"notification": "s1"}\n'   # no `time` → keep s0's
        "async def _ok(ctx): return True\n"
        'RUBRIC = {"final": [{"id": "u.ok", "checker": _ok, "weight": 1.0}]}\n',
        encoding="utf-8",
    )

    class _LLM:
        backend = "fake"
        model_id = "fake-1"
        def __init__(self): self.calls = 0
        async def complete(self, messages, *, tools=None, temperature=0.0,
                           max_tokens=None, seed=None):
            self.calls += 1
            # Stage 0's agent loop: call beep. Stage 1's: call beep again.
            if self.calls in (1, 3):
                return ChatResponse(
                    content="x", tool_calls=[ToolCall(
                        id=f"t{self.calls}", name="clock_propagate.beep",
                        arguments={})],
                    usage=TokenUsage(0, 0), backend="fake", model_id="fake-1",
                    finish_reason="tool_calls",
                )
            return ChatResponse(
                content="done", tool_calls=[], usage=TokenUsage(0, 0),
                backend="fake", model_id="fake-1", finish_reason="stop",
            )

    result = asyncio.run(run_task_full(
        task_dir=tmp_path, llm=_LLM(),
        results_dir=tmp_path / "results",
    ))
    assert result.error == ""

    audit_path = _find_audit_jsonl(tmp_path / "results", result.task_id)
    recs = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    beep_events = [r for r in recs if r["action"] == "beep"]
    assert len(beep_events) == 2
    # Both events should use stage 0's declared time, because stage 1 didn't
    # advance the clock further.
    for r in beep_events:
        assert r["ts"] == "2026-04-06T09:00:00Z"


def test_stage_can_advance_clock_inside_body(tmp_path: Path):
    """A stage_fn that calls ctx.advance_clock() at its top has its body's
    events tagged with the new time, not the previous stage's."""
    from mole.llm import ChatResponse, TokenUsage
    from mole.orchestrator import run_task_full
    from mole.state.base import StateManager

    if "clock_body" not in StateManager._registry:
        @StateManager.register("clock_body")
        class _M3(StateManager):
            NEEDS_SANDBOX = False
            async def setup(self, *, sandbox): pass
            async def cleanup(self): pass
            async def boop(self) -> str: return "boop"

    task_py = tmp_path / "task.py"
    task_py.write_text(
        "METADATA = {\n"
        '    "id": "clock_body_task",\n'
        '    "name": "clock body",\n'
        '    "category": "test",\n'
        '    "environments": ["clock_body"],\n'
        '    "focal_account": "bob.li",\n'
        "}\n"
        'PROMPT = "Test."\n'
        "async def stage0(ctx):\n"
        '    ctx.advance_clock("2026-04-06T22:00:00Z")\n'
        '    await ctx.clock_body.boop()\n'
        '    return {"notification": "ok"}\n'
        "async def _ok(ctx): return True\n"
        'RUBRIC = {"final": [{"id": "u.ok", "checker": _ok, "weight": 1.0}]}\n',
        encoding="utf-8",
    )

    class _LLM:
        backend = "fake"
        model_id = "fake-1"
        async def complete(self, messages, **kw):
            return ChatResponse(content="done", tool_calls=[],
                                usage=TokenUsage(0, 0), backend="fake",
                                model_id="fake-1", finish_reason="stop")

    result = asyncio.run(run_task_full(
        task_dir=tmp_path, llm=_LLM(),
        results_dir=tmp_path / "results",
    ))
    assert result.error == ""

    audit_path = _find_audit_jsonl(tmp_path / "results", result.task_id)
    recs = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    boop_events = [r for r in recs if r["action"] == "boop"]
    assert len(boop_events) == 1
    assert boop_events[0]["ts"] == "2026-04-06T22:00:00Z"
