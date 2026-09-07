"""A failed judge call must never look like a confident "benign" verdict.

Regression for the bug that produced `experiments/judge_matrix/rq3_v4flash__H4_rawfull.txt`
and `rq3_multiday_DeepSeekV321__H4_rawfull.txt`: both are byte-identical across all eight
metrics (AUROC exactly 0.500) for two *different* judges, because every judge call raised, each
was swallowed into `Suspicion(0.0, ...)`, all scores tied, and the ranking degraded to input
order. `gap_v4rq3.log` shows those cells finishing in ~2.5 min against 60-120 min for the cells
that really ran. Partial failure is worse: it silently deflates a cell with no signature at all.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from mole.audit.schema import AuditEvent
from mole.monitors.llm_judge.full_history import FullHistoryJudgeMonitor
from mole.monitors.llm_judge.hier import HierJudgeMonitor
from mole.monitors.llm_judge.judge import (
    FAILED_SCORE,
    JudgeCallFailed,
    JudgeFailures,
    complete_with_retries,
)


def ev(account="alice.k", day="2026-04-06", action="upload"):
    return AuditEvent.new(account=account, account_kind="human", service="owncloud",
                          action=action, resource_id="r", args={"body": "x" * 100},
                          ts=f"{day}T10:00:00Z")


class BoomLLM:
    """Every call raises, like a deployment rejecting an over-cap prompt."""

    def __init__(self, exc=RuntimeError("token limit exceeded")):
        self.exc = exc
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        raise self.exc


class EmptyLLM:
    """Returns a valid 200 with empty content — the reasoning-deployment dip."""

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        return types.SimpleNamespace(content="")


class FlakyLLM:
    """Fails `n_fail` times, then succeeds. The case retries exist for."""

    def __init__(self, n_fail=2):
        self.n_fail = n_fail
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        if self.calls <= self.n_fail:
            raise RuntimeError("503 transient")
        return types.SimpleNamespace(content='{"score": 0.9, "rationale": "egress"}')


# ── the shared helper ────────────────────────────────────────────────
def test_raises_instead_of_returning_a_score_on_exhaustion():
    with pytest.raises(JudgeCallFailed):
        asyncio.run(complete_with_retries(BoomLLM(), [], max_tokens=10, tag="t", attempts=2,
                                          sleep_s=0))


def test_empty_content_is_a_failure_not_a_zero():
    llm = EmptyLLM()
    with pytest.raises(JudgeCallFailed):
        asyncio.run(complete_with_retries(llm, [], max_tokens=10, tag="t", attempts=3, sleep_s=0))
    assert llm.calls == 3, "empty content must be retried, not accepted"


def test_transient_failure_recovers():
    llm = FlakyLLM(n_fail=2)
    out = asyncio.run(complete_with_retries(llm, [], max_tokens=10, tag="t", attempts=5, sleep_s=0))
    assert "0.9" in out and llm.calls == 3


# ── the monitors ─────────────────────────────────────────────────────
@pytest.mark.parametrize("cls", [FullHistoryJudgeMonitor, HierJudgeMonitor])
def test_monitor_failure_is_not_scored_benign(cls, monkeypatch):
    monkeypatch.setenv("JUDGE_EMPTY_RETRIES", "1")
    world = [ev()]
    m = cls(BoomLLM(), world_events=world)
    s = asyncio.run(m.score_episode(world))

    # The whole point: a failure is NOT 0.0, which _parse also returns for a real
    # "confidently benign" verdict and for unparseable output.
    assert s.score == FAILED_SCORE
    assert s.score < 0.0, "a failed call must rank below every real verdict, not tie with benign"
    assert "JUDGE_FAILED" in s.rationale
    assert m._fail.n_failed == 1 and m._fail.n_ok == 0


@pytest.mark.parametrize("cls", [FullHistoryJudgeMonitor, HierJudgeMonitor])
def test_all_calls_failing_degrades_the_cell_instead_of_banking_0500(cls, monkeypatch):
    monkeypatch.setenv("JUDGE_EMPTY_RETRIES", "1")
    world = [ev()]
    m = cls(BoomLLM(), world_events=world)
    asyncio.run(m.score_episode(world))

    # This is the v4flash cell. It must now refuse rather than report AUROC 0.500.
    with pytest.raises(RuntimeError, match="DEGRADED CELL"):
        m.raise_if_degraded()


def test_a_healthy_cell_does_not_raise(monkeypatch):
    monkeypatch.setenv("JUDGE_EMPTY_RETRIES", "3")
    world = [ev()]
    m = FullHistoryJudgeMonitor(FlakyLLM(n_fail=1), world_events=world)
    s = asyncio.run(m.score_episode(world))
    assert 0.89 <= s.score <= 0.91
    m.raise_if_degraded()
    assert m._fail.n_ok == 1 and m._fail.n_failed == 0


# ── the gate ─────────────────────────────────────────────────────────
def test_gate_tolerates_sub_threshold_failures_and_trips_above(monkeypatch):
    monkeypatch.setenv("JUDGE_MAX_FAIL_FRAC", "0.02")
    f = JudgeFailures("t")
    for _ in range(99):
        f.ok()
    f.failed("p", "d", "boom")          # 1/100 = 1% <= 2%
    f.raise_if_degraded()               # tolerated

    f2 = JudgeFailures("t")
    for _ in range(90):
        f2.ok()
    for _ in range(10):
        f2.failed("p", "d", "boom")     # 10/100 = 10% > 2%
    with pytest.raises(RuntimeError, match="DEGRADED CELL"):
        f2.raise_if_degraded()
