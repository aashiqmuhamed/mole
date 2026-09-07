"""Event-level FACADE (two-tower contrastive) — torch-gated.

Skips cleanly when torch isn't installed (the bench is numpy-only by default;
install the `facade` extra to run this). The module itself imports without torch
(lazy import), so the rest of the suite is unaffected.
"""
from __future__ import annotations

import asyncio

import pytest

from mole.audit.schema import AuditEvent
# module imports fine without torch (lazy import inside train/score):
from mole.monitors.facade import snn as facade_snn

pytest.importorskip("torch", reason="facade extra (torch) not installed")


def _ev(account, resource, kind="background_rules_agent"):
    return AuditEvent.new(account=account, account_kind=kind, service="oc",
                          action="read", resource_id=resource, ts="2026-05-01T09:00:00Z")


def test_facade_imports_without_torch_at_module_level():
    # The featurizer + peer fold work without torch (no model yet).
    feat = facade_snn.FacadeFeaturizer(peer_of={"a": {"b"}, "b": {"a"}})
    feat.fit([_ev("a", "/x"), _ev("b", "/x")])
    assert feat.context_tokens("a")            # b is a's peer -> non-empty
    assert "oc.read" in feat.atype_vocab


def test_trained_facade_flags_out_of_peer_action():
    """An account acting on a resource used only by NON-peers should score as
    anomalous; a peer of the resource's normal users should not."""
    peer = {"alice": {"bob"}, "bob": {"alice"}, "carol": {"dave"}, "dave": {"carol"}}
    benign = []
    for _ in range(60):
        benign += [_ev("alice", "/reports"), _ev("bob", "/reports"), _ev("carol", "/hr")]
    mon = facade_snn.train_facade(benign, peer_of=peer, epochs=15, dim=16, batch_size=32, seed=0)

    in_peer = asyncio.run(mon.score_episode([_ev("alice", "/reports", "agent")]))
    out_peer = asyncio.run(mon.score_episode([_ev("carol", "/reports", "agent")]))
    assert out_peer.score > in_peer.score                 # out-of-peer is more anomalous
    assert out_peer.score > 0.5
    assert mon.observability_level == 0 and out_peer.tokens_inspected == 0


def test_facade_empty_episode():
    mon = facade_snn.train_facade([_ev("alice", "/x"), _ev("bob", "/x")],
                                  peer_of={"alice": {"bob"}, "bob": {"alice"}},
                                  epochs=2, dim=8, batch_size=8)
    s = asyncio.run(mon.score_episode([]))
    assert s.score == 0.0


def _ev_ts(account, resource, ts, kind="background_rules_agent"):
    return AuditEvent.new(account=account, account_kind=kind, service="oc",
                          action="read", resource_id=resource, ts=ts)


def test_facade_window_days_filters_by_recency():
    """window_days keeps only co-actors who touched the resource within the trailing
    window (recency), relative to the scored event's day; None = all-time (original)."""
    fit_evs = [_ev_ts("old", "/shared", "2026-05-01T09:00:00Z"),
               _ev_ts("recent", "/shared", "2026-05-10T09:00:00Z")]
    score_ev = _ev_ts("actor", "/shared", "2026-05-11T09:00:00Z")

    def coactors(feat):
        feat.fit(fit_evs)
        ids, _ = feat.action_tokens(score_ev)
        rev = {v: k for k, v in feat.vocab.items()}
        return {rev[i] for i in ids if rev.get(i) not in (None, "<pad>", "<oov>")}

    assert coactors(facade_snn.FacadeFeaturizer(peer_of={})) == {"old", "recent"}
    assert coactors(facade_snn.FacadeFeaturizer(peer_of={}, window_days=3)) == {"recent"}
    assert coactors(facade_snn.FacadeFeaturizer(peer_of={}, window_days=30)) == {"old", "recent"}
    # None path builds no resource_times; windowed path does
    f0 = facade_snn.FacadeFeaturizer(peer_of={}); f0.fit(fit_evs)
    assert f0.resource_times == {}
