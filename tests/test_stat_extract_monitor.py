"""Mechanics for the two-stage extract-then-judge monitor. Scripted fakes, no network."""
from __future__ import annotations

import asyncio
import types

import pytest

from mole.audit.schema import AuditEvent
from mole.monitors.llm_judge.judge import FAILED_SCORE
from mole.monitors.llm_judge.stat_extract import StatExtractJudgeMonitor


def ev(account="alice.k", day="2026-04-06", action="upload"):
    return AuditEvent.new(account=account, account_kind="human", service="owncloud",
                          action=action, resource_id="r", args={"b": "x" * 50},
                          ts=f"{day}T10:00:00Z")


class RoleFake:
    """Routes by system prompt: extractor vs judge, and counts each."""
    model_id = "fake"

    def __init__(self, verdict='{"score": 0.7, "rationale": "spike vs baseline"}'):
        self.verdict = verdict
        self.n_extract = 0
        self.n_judge = 0

    async def complete(self, messages, **kw):
        sys_p = messages[0].content if messages else ""
        if "behavioural baseline" in sys_p:
            self.n_extract += 1
            return types.SimpleNamespace(
                content='{"profile": {"file_writes_per_day": 3}, "tells": ["bulk egress"]}')
        self.n_judge += 1
        return types.SimpleNamespace(content=self.verdict)


class BoomExtractor:
    model_id = "boom"

    async def complete(self, messages, **kw):
        raise RuntimeError("extractor down")


def _mon(judge, extractor=None, train=None, world=None):
    m = StatExtractJudgeMonitor(judge, extractor=extractor, world_events=world or [])
    m.fit_events(train or [])
    return m


def test_profile_is_built_from_train_and_reaches_the_judge():
    train = [ev("alice.k", "2026-04-01"), ev("alice.k", "2026-04-02")]
    today = [ev("alice.k", "2026-04-10")]
    f = RoleFake()
    m = _mon(f, train=train, world=today)
    s = asyncio.run(m.score_episode(today))
    assert 0.69 <= s.score <= 0.71
    assert f.n_extract == 1 and f.n_judge == 1


def test_extraction_is_amortised_one_per_account_not_per_day():
    """The whole cost argument: a statistic is per-ACCOUNT, so one strong extraction serves
    every day that account appears."""
    train = [ev("alice.k", "2026-04-01")]
    f = RoleFake()
    m = _mon(f, train=train, world=[ev("alice.k", d) for d in ("2026-04-10", "2026-04-11", "2026-04-12")])
    for d in ("2026-04-10", "2026-04-11", "2026-04-12"):
        asyncio.run(m.score_episode([ev("alice.k", d)]))
    assert f.n_judge == 3
    assert f.n_extract == 1, "extraction must be cached per account"
    assert m.n_extract == 1


def test_strong_extractor_and_weak_judge_are_different_clients():
    train = [ev("alice.k", "2026-04-01")]
    strong, weak = RoleFake(), RoleFake()
    m = _mon(weak, extractor=strong, train=train, world=[ev("alice.k", "2026-04-10")])
    asyncio.run(m.score_episode([ev("alice.k", "2026-04-10")]))
    assert strong.n_extract == 1 and strong.n_judge == 0, "extractor must only extract"
    assert weak.n_judge == 1 and weak.n_extract == 0, "judge must only judge"


def test_failed_extraction_is_not_silently_downgraded_to_the_flat_judge(monkeypatch):
    """A dead extractor must fail loudly. Silently judging with no baseline would make a BROKEN
    cell look like a weak-judge result -- the same class of bug as scoring failures 0.0."""
    monkeypatch.setenv("JUDGE_EMPTY_RETRIES", "1")
    train = [ev("alice.k", "2026-04-01")]
    m = _mon(RoleFake(), extractor=BoomExtractor(), train=train, world=[ev("alice.k", "2026-04-10")])
    s = asyncio.run(m.score_episode([ev("alice.k", "2026-04-10")]))
    assert s.score == FAILED_SCORE
    assert "JUDGE_FAILED" in s.rationale
    with pytest.raises(RuntimeError, match="DEGRADED CELL"):
        m.raise_if_degraded()


def test_extractor_prompt_does_not_name_our_hand_built_features():
    """Naming our 20 features would turn the model into an expensive calculator for a statistic we
    already have, and the discovery question would go untested."""
    from mole.monitors.classical.anomaly import NUMERIC_FEATURES
    from mole.monitors.llm_judge.stat_extract import _EXTRACT_SYS
    leaked = [f for f in NUMERIC_FEATURES if f in _EXTRACT_SYS]
    assert not leaked, f"extractor prompt leaks our hand-built features: {leaked}"


def test_extraction_resumes_from_cache_across_runs(tmp_path, monkeypatch):
    """Extraction is the slow front-loaded phase (one call per account). A crash mid-run
    must not throw it away: a fresh monitor (the next retry attempt) reuses the cached profiles."""
    monkeypatch.setenv("JUDGE_SCORE_CACHE", str(tmp_path / "c.jsonl"))
    train = [ev("alice.k", "2026-04-01"), ev("bob.j", "2026-04-01")]
    world = [ev("alice.k", "2026-04-10"), ev("bob.j", "2026-04-10")]

    strong1 = RoleFake()
    m1 = _mon(RoleFake(), extractor=strong1, train=train, world=world)
    for p in ("alice.k", "bob.j"):
        asyncio.run(m1.score_episode([ev(p, "2026-04-10")]))
    assert strong1.n_extract == 2

    strong2 = RoleFake()                                  # the retry after a wedge
    m2 = _mon(RoleFake(), extractor=strong2, train=train, world=world)
    for p in ("alice.k", "bob.j"):
        asyncio.run(m2.score_episode([ev(p, "2026-04-10")]))
    assert strong2.n_extract == 0, "extraction did not resume; a wedge would restart 3.5h from zero"
