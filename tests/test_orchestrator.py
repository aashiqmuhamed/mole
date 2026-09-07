"""Integration tests for the orchestrator.

Uses a fake LLM + DryRunSandbox + an in-memory task definition to drive the
orchestrator end-to-end without spinning up Docker. Verifies stage iteration,
rubric scoring, error handling, and conversation persistence across stages.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mole.llm import ChatMessage, ChatResponse, TokenUsage
from mole.orchestrator import run_task_full
from mole.state.base import StateManager


# Register a no-op backend named "dummy" so tests that exercise the
# environments path can use it without depending on any real services.
@StateManager.register("dummy")
class _DummyManager(StateManager):
    async def setup(self, *, sandbox):
        pass

    async def cleanup(self):
        pass


class _FakeLLM:
    """Returns a canned plain-text response for every call."""
    backend = "fake"
    model_id = "fake-1"

    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies or ["ok"])
        self.calls = 0

    async def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=None, seed=None):
        self.calls += 1
        idx = min(self.calls - 1, len(self._replies) - 1)
        return ChatResponse(
            content=self._replies[idx],
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            backend="fake", model_id="fake-1", finish_reason="stop",
        )


def _write_task(
    dir_: Path,
    *,
    stages: int = 1,
    env: list[str] | None = None,
    focal_account: str | None = None,
    adversary_level: str | None = None,
) -> None:
    """Write a minimal task.py into dir_."""
    env_list = env if env is not None else []
    metadata_lines = [
        "METADATA = {",
        '    "id": "fixture",',
        '    "name": "Fixture",',
        '    "category": "test",',
        f'    "environments": {env_list!r},',
    ]
    if focal_account is not None:
        metadata_lines.append(f'    "focal_account": {focal_account!r},')
    if adversary_level is not None:
        metadata_lines.append(f'    "adversary_level": {adversary_level!r},')
    metadata_lines.append("}")
    lines: list[str] = metadata_lines + [
        "",
        'PROMPT = "You are a terse assistant."',
        "",
    ]
    for i in range(stages):
        lines += [
            f"async def stage{i}(ctx):",
            f'    return {{"notification": "Day {i}: please summarize.",',
            f'            "time": "2026-04-0{i + 6}T09:00:00"}}',
            "",
        ]
    lines += [
        "async def _always_passes(ctx):",
        "    return True",
        "",
        "async def _always_fails(ctx):",
        "    return False",
        "",
        "RUBRIC = {",
        '    "final": [',
        '        {"id": "utility.ok",  "checker": _always_passes, "weight": 2.0},',
        '        {"id": "harm.failed", "checker": _always_fails,  "weight": 1.0},',
        "    ],",
        "}",
        "",
    ]
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "task.py").write_text("\n".join(lines), encoding="utf-8")


def test_orchestrator_runs_empty_envs_task(tmp_path: Path):
    """Task with no environments: no sandbox needed, agent runs with empty tools."""
    _write_task(tmp_path, stages=1, env=[])
    llm = _FakeLLM(replies=["Sure, here's a summary."])

    result = asyncio.run(run_task_full(task_dir=tmp_path, llm=llm))

    assert result.task_id == "fixture"
    assert result.error == ""
    assert len(result.stage_results) == 2  # 1 stage + 1 "final"
    assert result.stage_results[0].stage_id == "stage0"
    assert result.stage_results[1].stage_id == "final"
    # Weighted: utility.ok (w=2, pass) + harm.failed (w=1, fail) → 2/3
    assert result.score == pytest.approx(2 / 3)
    assert llm.calls == 1  # one stage → one LLM call


def test_orchestrator_iterates_multiple_stages(tmp_path: Path):
    _write_task(tmp_path, stages=3, env=[])
    llm = _FakeLLM(replies=["s0 done", "s1 done", "s2 done"])

    result = asyncio.run(run_task_full(task_dir=tmp_path, llm=llm))

    stage_ids = [s.stage_id for s in result.stage_results]
    assert stage_ids == ["stage0", "stage1", "stage2", "final"]
    assert llm.calls == 3


def test_orchestrator_persists_conversation_history_across_stages(tmp_path: Path):
    """The agent should see prior assistant + user messages on the second stage."""
    _write_task(tmp_path, stages=2, env=[])

    class _RecordingLLM(_FakeLLM):
        def __init__(self):
            super().__init__(replies=["first reply", "second reply"])
            self.message_counts: list[int] = []

        async def complete(self, messages, **kw):
            self.message_counts.append(len(messages))
            return await super().complete(messages, **kw)

    llm = _RecordingLLM()
    asyncio.run(run_task_full(task_dir=tmp_path, llm=llm))

    # First stage: system + user (notification) = 2 messages.
    # Second stage: system + user + assistant (from stage 0) + user (notification) = 4.
    assert llm.message_counts == [2, 4]


def test_orchestrator_rubric_score_is_weighted_pass_rate(tmp_path: Path):
    _write_task(tmp_path, stages=1, env=[])
    llm = _FakeLLM()
    result = asyncio.run(run_task_full(task_dir=tmp_path, llm=llm))
    # Same rubric (w=2 pass, w=1 fail) → 2/3.
    assert result.score == pytest.approx(2 / 3)
    # The aggregated items list includes both rubric checks (in the "final" stage).
    final_items = [it for it in result.rubric_results if it.item_id.startswith(("utility.", "harm."))]
    assert {it.item_id for it in final_items} == {"utility.ok", "harm.failed"}


def test_orchestrator_demands_lab_compose_when_envs_nonempty(tmp_path: Path):
    """If a task declares environments but no lab_compose is given, fail loudly.

    The fixture references an environment we haven't registered, so loader
    succeeds (it doesn't check the registry) and the orchestrator only fails
    when it tries to spin up a sandbox without a compose file. The error is
    surfaced in result.error; no stages run.
    """
    _write_task(tmp_path, stages=1, env=["dummy"], focal_account="bob.li")
    llm = _FakeLLM()
    result = asyncio.run(run_task_full(task_dir=tmp_path, llm=llm))
    assert "no lab_compose provided" in result.error.lower()
    assert result.stage_results == []   # crashed before any stage ran
    assert result.score == 0.0


def test_orchestrator_requires_focal_account_when_envs_nonempty(tmp_path: Path):
    """Default 'agent' account would leak a meta-label into audit
    events. Force task authors to set METADATA['focal_account']."""
    _write_task(tmp_path, stages=1, env=["dummy"])   # no focal_account
    llm = _FakeLLM()
    result = asyncio.run(run_task_full(task_dir=tmp_path, llm=llm))
    assert "focal_account" in result.error.lower()
    assert result.score == 0.0


def test_orchestrator_no_focal_account_ok_when_envs_empty(tmp_path: Path):
    """Smoke tests with environments=[] don't emit events, so the account
    requirement doesn't apply."""
    _write_task(tmp_path, stages=1, env=[])    # no envs, no focal_account
    llm = _FakeLLM(replies=["done"])
    result = asyncio.run(run_task_full(task_dir=tmp_path, llm=llm))
    assert result.error == ""
    assert result.score >= 0.0


def test_orchestrator_derives_is_malicious_from_adversary_level(tmp_path: Path):
    """When METADATA['adversary_level'] is not 'benign', audit events get
    is_malicious=True automatically — no manual flag plumbing needed."""
    from mole.state.base import StateManager
    # Register a tiny dummy backend that emits one wrapped call so we have
    # something to inspect in the audit log.
    if "ev_test" not in StateManager._registry:
        @StateManager.register("ev_test")
        class _EvTest(StateManager):
            NEEDS_SANDBOX = False
            async def setup(self, *, sandbox): pass
            async def cleanup(self): pass
            async def ping(self, *, who: str) -> str:
                return f"pong:{who}"

    _write_task(tmp_path, stages=1, env=["ev_test"],
                focal_account="bob.li", adversary_level="A0")

    # Use a fake LLM that actually calls a tool so we exercise the wrapper.
    from mole.llm import ToolCall, ChatResponse, TokenUsage

    class _ToolingLLM:
        backend = "fake"
        model_id = "fake-1"
        def __init__(self):
            self.calls = 0
        async def complete(self, messages, *, tools=None, temperature=0.0,
                           max_tokens=None, seed=None):
            self.calls += 1
            if self.calls == 1:
                return ChatResponse(
                    content="calling",
                    tool_calls=[ToolCall(id="tc-1", name="ev_test.ping",
                                          arguments={"who": "bob"})],
                    usage=TokenUsage(0, 0), backend="fake", model_id="fake-1",
                    finish_reason="tool_calls",
                )
            return ChatResponse(
                content="done", tool_calls=[], usage=TokenUsage(0, 0),
                backend="fake", model_id="fake-1", finish_reason="stop",
            )

    result = asyncio.run(run_task_full(task_dir=tmp_path, llm=_ToolingLLM()))
    assert result.error == ""

    # Read the audit.jsonl that the orchestrator wrote.
    audit_path = Path("results") / result.task_id / "audit.jsonl"
    if audit_path.exists():
        import json
        recs = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        ping_events = [r for r in recs if r.get("action") == "ping"]
        assert ping_events, "expected ev_test.ping event in audit log"
        assert all(r["is_malicious"] is True for r in ping_events)
        assert all(r["account"] == "bob.li" for r in ping_events)
        # And NOT the leaky default fallback account.
        assert all(r["account"] != "agent" for r in ping_events)
