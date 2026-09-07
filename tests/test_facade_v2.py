"""FACADE v2 (faithful port): featurizer-cache correctness + prefit-featurizer injection.

The cache memoizes action/context features (pure functions of the event/account given
the fitted state), so a sweep can reuse one featurizer across many models. These tests
assert the cache is FAITHFUL (identical outputs to the uncached path) and that a prefit,
cache-enabled featurizer trains + scores. torch-gated like the v1 test.
"""
from __future__ import annotations

import asyncio

import pytest

from mole.audit.schema import AuditEvent

pytest.importorskip("torch", reason="facade extra (torch) not installed")

from mole.monitors.facade.v2.featurizer import FacadeV2Featurizer, _ts_seconds
from mole.monitors.facade.v2.monitor import train_facade_v2


def _ev(account, resource, ts, kind="background_rules_agent"):
    return AuditEvent.new(account=account, account_kind=kind, service="oc",
                          action="read", resource_id=resource, ts=ts)


def _corpus():
    evs = []
    for d in range(1, 8):
        ts = f"2026-05-0{d}T09:00:00Z"
        evs += [_ev("alice", "/reports", ts), _ev("bob", "/reports", ts), _ev("carol", "/hr", ts)]
    return evs


def test_v2_featurizer_cache_matches_uncached():
    """Cache on == cache off, for both feature functions (non-empty action bags via
    /reports co-access; non-empty context via the coaccess fold)."""
    corpus = _corpus()
    later_ts = max(_ts_seconds(e) for e in corpus) + 3600.0
    probes = corpus + [_ev("dave", "/reports", "2026-05-08T09:00:00Z")]

    for source in ("org", "coaccess"):
        f_no = FacadeV2Featurizer(peer_source=source).fit(corpus)
        f_ca = FacadeV2Featurizer(peer_source=source).enable_cache().fit(corpus)
        seen_nonempty_action = False
        for e in probes:
            a1 = f_ca.action_features(e)
            a2 = f_ca.action_features(e)          # 2nd call exercises the cache-hit path
            assert a1 == a2 == f_no.action_features(e)
            seen_nonempty_action |= bool(a1[0])
        assert seen_nonempty_action                # the test actually exercised non-empty bags
        for p in ("alice", "bob", "carol", "dave"):
            assert f_ca.context_features(p, later_ts) == f_no.context_features(p, later_ts)
        if source == "coaccess":
            assert f_ca.context_features("alice", later_ts)[0]   # alice has co-access peers


def test_v2_prefit_featurizer_trains_and_scores():
    corpus = _corpus()
    feat = FacadeV2Featurizer(peer_source="coaccess").enable_cache().fit(corpus)
    mon = train_facade_v2(corpus, epochs=2, dim=8, token_dim=4, snn_layers=(8,),
                          batch_size=8, prefit_featurizer=feat)
    s = asyncio.run(mon.score_episode([_ev("alice", "/reports", "2026-05-08T09:00:00Z")]))
    assert s.score == s.score and s.score >= 0.0    # finite, non-negative
    assert mon.observability_level == 0 and feat._afeat_cache      # cache was populated during train+score
