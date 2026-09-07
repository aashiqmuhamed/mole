"""Tests for the FACADE-inspired peer-fit monitor."""
from __future__ import annotations

import asyncio

import pytest

from mole.audit.rollup import UserDayFeatures
from mole.audit.schema import AuditEvent
from mole.monitors.base import Monitor
from mole.monitors.classical.anomaly import ZScoreAnomalyMonitor
from mole.monitors.facade.peer_fit import (
    PeerFitMonitor, peers_from_org_yaml,
)


def _benign_rows() -> tuple[list[UserDayFeatures], dict[str, str]]:
    """Two peer groups with distinct benign profiles:
    - infra: heavy gitlab commits, ~no email
    - finance: heavy email/file, ~no gitlab
    """
    rows: list[UserDayFeatures] = []
    peer_of: dict[str, str] = {}
    for i in range(20):
        u = f"infra{i}"
        peer_of[u] = "infra"
        rows.append(UserDayFeatures(account=u, date=f"2026-04-{i % 27 + 1:02d}",
                                    gitlab_commit_count=8 + i % 5, gitlab_mr_open_count=1 + i % 3,
                                    email_send_count=i % 2))
    for i in range(20):
        u = f"fin{i}"
        peer_of[u] = "finance"
        rows.append(UserDayFeatures(account=u, date=f"2026-04-{i % 27 + 1:02d}",
                                    email_send_count=6 + i % 4, file_read_count=10 + i % 6,
                                    gitlab_commit_count=i % 2))
    return rows, peer_of


def test_is_tier0_monitor():
    assert isinstance(PeerFitMonitor(), Monitor)
    assert PeerFitMonitor().observability_level == 0


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        asyncio.run(PeerFitMonitor().score_episode([]))


def test_peer_conditioning_flags_cross_role_behavior():
    """A finance user doing heavy gitlab commits is anomalous *vs finance
    peers* — even though infra users commit all day. This is the property
    a global anomaly detector misses and peer-fit catches."""
    rows, peer_of = _benign_rows()
    peer = PeerFitMonitor().fit(rows, peer_of=peer_of)
    glob = ZScoreAnomalyMonitor().fit(rows)

    # A finance account suddenly doing 12 gitlab commits.
    fin_commits = [
        AuditEvent.new(account="fin0", account_kind="human", service="gitlab",
                       action="commit", ts="2026-04-15T11:00:00Z")
        for _ in range(12)
    ]
    s_peer = asyncio.run(peer.score_episode(fin_commits))
    s_glob = asyncio.run(glob.score_episode(fin_commits))

    # Peer-fit flags it harder than the global detector, because commits are
    # normal *globally* (infra does them) but abnormal for finance peers.
    assert s_peer.score > s_glob.score
    assert "finance" in s_peer.rationale


def test_in_group_behavior_scores_low():
    """An infra user committing code (their norm) scores low under peer-fit."""
    rows, peer_of = _benign_rows()
    peer = PeerFitMonitor().fit(rows, peer_of=peer_of)
    infra_commits = [
        AuditEvent.new(account="infra0", account_kind="human", service="gitlab",
                       action="commit", ts="2026-04-15T11:00:00Z")
        for _ in range(9)
    ]
    s = asyncio.run(peer.score_episode(infra_commits))
    # Well within infra's benign profile → low.
    assert s.score < 2.0


def test_unknown_account_falls_back_to_global():
    rows, peer_of = _benign_rows()
    peer = PeerFitMonitor().fit(rows, peer_of=peer_of)
    ev = [AuditEvent.new(account="stranger", account_kind="human", service="gitlab",
                         action="commit", ts="2026-04-15T11:00:00Z")]
    s = asyncio.run(peer.score_episode(ev))     # no peer group → global profile, no crash
    assert s.score >= 0.0


def test_peers_from_org_yaml_maps_accounts_to_groups():
    peer_of = peers_from_org_yaml()
    # alice.kim's first group in org_template is alignment-leads.
    assert peer_of.get("alice.kim") == "alignment-leads"
    assert all(isinstance(v, str) for v in peer_of.values())
