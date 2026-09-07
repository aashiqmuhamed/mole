"""Tests for monitors/metrics.py — the monitor-agnostic ranking-metric framework."""
from __future__ import annotations

import pytest

from mole.audit.schema import AuditEvent
from mole.monitors.base import Suspicion
from mole.monitors.metrics import (
    compute_ranking_metrics,
    malicious_account_days,
)


def _ev(account: str, ts: str, *, malicious: bool = False) -> AuditEvent:
    return AuditEvent.new(
        account=account, account_kind="background_llm_agent",
        service="x", action="y", ts=ts, is_malicious=malicious,
    )


def test_recall_at_pct_perfect_ranking():
    # Top 2 of 10 are malicious — perfect monitor scores both first.
    ranked = [
        (("attacker", "2026-03-02"), 1.0),
        (("attacker", "2026-03-03"), 0.99),
    ] + [((f"p{i}", "2026-03-02"), 0.1 + i * 0.01) for i in range(8)]
    truth = {("attacker", "2026-03-02"), ("attacker", "2026-03-03")}
    m = compute_ranking_metrics("perfect", ranked, truth, pcts=(0.1, 0.2, 0.5))
    assert m.n_total == 10
    assert m.n_malicious == 2
    assert m.recall_at_pct[0.2] == 1.0   # top 20% = top 2 → catches both
    assert m.recall_at_pct[0.1] == 0.5   # top 10% = top 1 → catches 1 of 2
    assert m.auroc == 1.0
    assert m.pr_auc == 1.0


def test_recall_at_k_lopsided():
    # 1 malicious among 10, monitor ranks it 3rd.
    ranked = [
        (("noise1", "d"), 0.9),
        (("noise2", "d"), 0.8),
        (("attacker", "d"), 0.7),
    ] + [((f"p{i}", "d"), 0.1) for i in range(7)]
    truth = {("attacker", "d")}
    m = compute_ranking_metrics("ok", ranked, truth, ks=(1, 3, 5))
    assert m.recall_at_k[1] == 0.0
    assert m.recall_at_k[3] == 1.0
    assert m.recall_at_k[5] == 1.0


def test_no_malicious_keys():
    ranked = [(("p", "d"), 0.5)]
    m = compute_ranking_metrics("nomal", ranked, set(), pcts=(0.1,))
    assert m.n_malicious == 0
    assert m.auroc == 0.0
    assert m.pr_auc == 0.0
    assert m.recall_at_pct[0.1] == 0.0


def test_empty_ranking():
    m = compute_ranking_metrics("empty", [], {("a", "b")})
    assert m.n_total == 0 and m.n_malicious == 0


def test_accepts_suspicion_objects():
    ranked = [
        (("a", "d"), Suspicion(score=0.9, rationale="", tokens_inspected=0)),
        (("b", "d"), Suspicion(score=0.1, rationale="", tokens_inspected=0)),
    ]
    m = compute_ranking_metrics("susp", ranked, {("a", "d")})
    assert m.recall_at_k[1] == 1.0


def test_malicious_account_days_from_events():
    events = [
        _ev("alice", "2026-03-02T09:00:00Z", malicious=False),
        _ev("bob",   "2026-03-02T10:00:00Z", malicious=True),
        _ev("bob",   "2026-03-02T10:01:00Z", malicious=False),   # same day still malicious
        _ev("bob",   "2026-03-03T09:00:00Z", malicious=False),   # different day, NOT malicious
        _ev("kara",  "2026-03-04T11:00:00Z", malicious=True),
    ]
    keys = malicious_account_days(events)
    assert keys == {("bob", "2026-03-02"), ("kara", "2026-03-04")}
