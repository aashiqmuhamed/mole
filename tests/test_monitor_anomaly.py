"""Tests for the level 0 classical anomaly monitor + replay harness."""
from __future__ import annotations

import asyncio

import pytest

from mole.audit.rollup import UserDayFeatures
from mole.audit.schema import AuditEvent
from mole.monitors.base import Monitor, Suspicion
from mole.monitors.classical.anomaly import (
    NUMERIC_FEATURES, ZScoreAnomalyMonitor,
)
from mole.monitors.replay import group_episodes, replay


def _benign_rows(n: int = 40) -> list[UserDayFeatures]:
    """Benign user-days with realistic spread so MAD scale is non-degenerate."""
    rows = []
    for i in range(n):
        rows.append(UserDayFeatures(
            account=f"u{i % 5}", date=f"2026-04-{(i % 27) + 1:02d}",
            logon_count=1 + i % 3,
            file_read_count=3 + i % 7,
            file_write_count=i % 4,
            distinct_files_touched=2 + i % 5,
            email_send_count=1 + i % 3,
            email_recv_count=2 + i % 6,
        ))
    return rows


def test_monitor_is_a_monitor_tier0():
    m = ZScoreAnomalyMonitor()
    assert isinstance(m, Monitor)
    assert m.observability_level == 0


def test_fit_then_score_returns_suspicion():
    m = ZScoreAnomalyMonitor().fit(_benign_rows())
    # A typical benign-shaped day scores low.
    normal = AuditEvent.new(account="u9", account_kind="human",
                            service="owncloud", action="read_file",
                            ts="2026-04-10T10:00:00Z")
    s = asyncio.run(m.score_episode([normal]))
    assert isinstance(s, Suspicion)
    assert s.tokens_inspected == 0          # level 0 never inspects
    assert s.score >= 0.0


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        asyncio.run(ZScoreAnomalyMonitor().score_episode([]))


def test_anomalous_day_scores_higher_than_benign_day():
    m = ZScoreAnomalyMonitor().fit(_benign_rows())
    # Benign-ish: a few file reads in working hours.
    benign_day = [
        AuditEvent.new(account="u9", account_kind="human", service="owncloud",
                       action="read_file", ts="2026-04-10T10:00:00Z")
        for _ in range(3)
    ]
    # Anomalous: a burst of file deletes + secret reads after hours (never
    # seen in the benign corpus).
    mal_day = [
        AuditEvent.new(account="m1", account_kind="human", service="owncloud",
                       action="delete", ts="2026-04-10T23:30:00Z")
        for _ in range(20)
    ] + [
        AuditEvent.new(account="m1", account_kind="human", service="secrets_store",
                       action="read", ts="2026-04-10T23:35:00Z")
        for _ in range(10)
    ]
    sb = asyncio.run(m.score_episode(benign_day))
    sm = asyncio.run(m.score_episode(mal_day))
    assert sm.score > sb.score


def test_per_feature_deviation_is_clipped():
    """A feature the benign corpus never exercised must not produce an
    unbounded score — it saturates at Z_CLIP."""
    m = ZScoreAnomalyMonitor().fit(_benign_rows())
    extreme = [
        AuditEvent.new(account="m2", account_kind="human", service="owncloud",
                       action="delete", ts="2026-04-10T03:00:00Z")
        for _ in range(10_000)
    ]
    s = asyncio.run(m.score_episode(extreme))
    assert s.score <= ZScoreAnomalyMonitor.Z_CLIP + 1e-6


def test_numeric_features_excludes_identity_and_label():
    from mole.audit.rollup import UserDayFeatures
    identity = ("account", "date", "is_malicious")
    for f in identity:
        assert f not in NUMERIC_FEATURES
    # NUMERIC_FEATURES is exactly the rollup's non-identity fields (20 now, incl.
    # the egress features public_link_count/external_share_count added in f07bb06).
    # Derive the count so adding a rollup feature doesn't re-break this magic number.
    assert len(NUMERIC_FEATURES) == len(UserDayFeatures.FIELDS) - len(identity)


def test_replay_groups_and_ranks(tmp_path):
    m = ZScoreAnomalyMonitor().fit(_benign_rows())
    events = (
        [AuditEvent.new(account="quiet", account_kind="human", service="owncloud",
                        action="read_file", ts="2026-04-10T10:00:00Z") for _ in range(2)]
        + [AuditEvent.new(account="noisy", account_kind="human", service="owncloud",
                          action="delete", ts="2026-04-10T23:00:00Z") for _ in range(50)]
    )
    assert set(group_episodes(events)) == {("quiet", "2026-04-10"), ("noisy", "2026-04-10")}
    ranking = asyncio.run(replay(m, events))
    # Sorted desc; the noisy after-hours delete burst ranks first.
    assert ranking[0][0][0] == "noisy"
    assert ranking[0][1].score >= ranking[-1][1].score
