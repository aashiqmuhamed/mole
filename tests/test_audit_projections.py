"""Unit tests for audit/projections.py — level 0 / level 1 / level 2 views."""
from __future__ import annotations

import pytest

from mole.audit.projections import (
    BudgetExhausted,
    InspectionBudget,
    tier0_view,
    tier1_view,
    tier2_view,
)
from mole.audit.schema import AuditEvent


def _event(args: dict | None = None, **kw) -> AuditEvent:
    """Build an AuditEvent with sane defaults for testing."""
    defaults = dict(
        account="alice.kim",
        account_kind="human",
        service="email",
        action="send",
        resource_id="bob@example.com",
        args=args or {"subject": "hi", "body": "..."},
    )
    defaults.update(kw)
    return AuditEvent.new(**defaults)


# ── level 0 ────────────────────────────────────────────────────────────


def test_tier0_strips_args_and_resource_id():
    e = _event(args={"subject": "secret", "body": "very long body text"})
    [view] = tier0_view([e])
    assert "args" not in view
    assert "resource_id" not in view
    assert "error" not in view
    # But keeps metadata + hashes + structural fields.
    assert view["account"] == "alice.kim"
    assert view["service"] == "email"
    assert view["action"] == "send"
    assert view["args_hash"] == e.args_hash
    assert view["resource_hash"] == e.resource_hash
    assert view["exit_code"] == 0
    assert view["is_external"] is False


def test_tier0_hides_ground_truth_labels_by_default():
    """Monitor must not see the meta-label that identifies the agent,
    the is_malicious supervision label, or its own past gate decisions."""
    e = _event()
    e.is_malicious = True
    e.gate_decision = "block"
    [view] = tier0_view([e])
    assert "account_kind" not in view
    assert "is_malicious" not in view
    assert "gate_decision" not in view


def test_tier0_account_is_account_identity_not_actor_kind():
    """We expose `account` (the bob.li account) but never `account_kind`
    (the agent vs background actor label)."""
    e = _event()   # account_kind="human" by default in our helper
    [view] = tier0_view([e])
    assert view["account"] == "alice.kim"
    assert "account_kind" not in view


def test_tier0_expose_ground_truth_ablation_hook():
    e = _event()
    e.is_malicious = True
    [view] = tier0_view([e], expose_ground_truth=True)
    assert view["account_kind"] == "human"
    assert view["is_malicious"] is True
    assert "gate_decision" in view


def test_tier0_hides_real_ts_process_local_field():
    """`real_ts` is a process-local perf_counter — not in real audit logs,
    and would give a free actor-identification signal in multi-process runs."""
    e = _event()
    [view] = tier0_view([e])
    assert "real_ts" not in view
    # `ts` (the wall/sim-clock ISO timestamp) IS visible — that's what real
    # audit logs record.
    assert view["ts"]


def test_tier0_preserves_event_order():
    es = [_event(action=a) for a in ("a", "b", "c", "d")]
    out = tier0_view(es)
    assert [v["action"] for v in out] == ["a", "b", "c", "d"]


# ── level 2 ────────────────────────────────────────────────────────────


def test_tier2_passes_args_through_verbatim():
    e = _event(args={"subject": "secret", "body": "very long body"})
    [view] = tier2_view([e])
    assert view["args"] == {"subject": "secret", "body": "very long body"}
    assert view["resource_id"] == "bob@example.com"


def test_tier2_hides_ground_truth_labels_by_default():
    """Even the full-semantic view doesn't leak the supervision label."""
    e = _event()
    e.is_malicious = True
    e.gate_decision = "block"
    [view] = tier2_view([e])
    assert "account_kind" not in view
    assert "is_malicious" not in view
    assert "gate_decision" not in view
    # But semantic content IS visible — that's the whole point of level 2.
    assert view["args"]   # non-empty
    assert view["resource_id"]


def test_tier2_expose_ground_truth_ablation_hook():
    e = _event()
    e.is_malicious = True
    [view] = tier2_view([e], expose_ground_truth=True)
    assert view["is_malicious"] is True
    assert view["account_kind"] == "human"


def test_tier1_view_hides_ground_truth_in_underlying_tier0_stream():
    e = _event()
    e.is_malicious = True
    view = tier1_view([e], budget=InspectionBudget(total_tokens=100))
    [tier0_dict] = view.events
    assert "account_kind" not in tier0_dict
    assert "is_malicious" not in tier0_dict


def test_tier1_view_with_expose_ground_truth_passes_through():
    e = _event()
    e.is_malicious = True
    view = tier1_view([e],
                      budget=InspectionBudget(total_tokens=100),
                      expose_ground_truth=True)
    [tier0_dict] = view.events
    assert tier0_dict["account_kind"] == "human"
    assert tier0_dict["is_malicious"] is True


def test_oracle_can_still_read_ground_truth_from_raw_event():
    """Oracle checkers (and post-hoc analysis) reach for AuditEvent directly,
    not via the projection layer. The label must remain accessible there."""
    e = _event()
    e.is_malicious = True
    # No projection involved — just the raw event.
    assert e.is_malicious is True
    assert e.account_kind == "human"


# ── InspectionBudget ──────────────────────────────────────────────────


def test_budget_costs_more_for_larger_payloads():
    b = InspectionBudget(total_tokens=1000)
    small = b.cost_of({"x": 1})
    large = b.cost_of({"body": "x" * 1000})
    assert small > 0
    assert large > small


def test_budget_cost_is_nonzero_for_empty_payload():
    """Empty dict still costs at least one token (for the '{}')."""
    b = InspectionBudget(total_tokens=1000)
    assert b.cost_of({}) >= 1


def test_budget_charge_subtracts_from_remaining():
    b = InspectionBudget(total_tokens=100)
    spent = b.charge("evt-1", {"key": "value"})
    assert spent > 0
    assert b.spent_tokens == spent
    assert b.remaining_tokens == 100 - spent
    assert b.inspected_ids == ["evt-1"]


def test_budget_charge_raises_when_exceeding():
    b = InspectionBudget(total_tokens=5)
    with pytest.raises(BudgetExhausted):
        b.charge("evt-1", {"body": "x" * 1000})
    # No state mutation on failed charge.
    assert b.spent_tokens == 0
    assert b.inspected_ids == []


def test_budget_can_afford_predicate():
    b = InspectionBudget(total_tokens=100)
    assert b.can_afford({"x": 1}) is True
    assert b.can_afford({"body": "x" * 10_000}) is False


# ── level 1 ────────────────────────────────────────────────────────────


def test_tier1_view_starts_with_tier0_dicts_and_full_budget():
    events = [_event() for _ in range(3)]
    view = tier1_view(events, budget=InspectionBudget(total_tokens=100))
    assert len(view.events) == 3
    # level 0 shape; no args.
    assert all("args" not in ev for ev in view.events)
    assert view.tokens_spent() == 0
    assert view.tokens_remaining() == 100


def test_tier1_inspect_returns_args_and_charges_budget():
    e = _event(args={"subject": "ok", "body": "hello"})
    view = tier1_view([e], budget=InspectionBudget(total_tokens=1000))
    out = view.inspect(e.event_id)
    assert out == {"subject": "ok", "body": "hello"}
    assert view.tokens_spent() > 0
    assert view.budget.inspected_ids == [e.event_id]


def test_tier1_inspect_returns_none_when_budget_exhausted():
    e = _event(args={"body": "x" * 1000})   # large payload
    view = tier1_view([e], budget=InspectionBudget(total_tokens=2))
    out = view.inspect(e.event_id)
    assert out is None
    # Budget untouched on failed inspect.
    assert view.tokens_spent() == 0
    assert view.budget.inspected_ids == []


def test_tier1_inspect_returns_none_for_unknown_id():
    e = _event()
    view = tier1_view([e], budget=InspectionBudget(total_tokens=100))
    assert view.inspect("nope") is None


def test_tier1_can_inspect_is_consistent_with_budget():
    e_small = _event(args={"x": 1})
    e_large = _event(args={"body": "y" * 1000})
    view = tier1_view([e_small, e_large], budget=InspectionBudget(total_tokens=5))
    # The large one shouldn't fit; the small one might.
    assert view.can_inspect(e_large.event_id) is False
    # After we spend some budget on the small one, can_inspect reflects remaining.
    cost = view.budget.cost_of(e_small.args)
    if cost <= 5:
        view.inspect(e_small.event_id)
        assert view.tokens_remaining() == 5 - cost


def test_tier1_inspect_can_be_called_multiple_times_within_budget():
    es = [_event(args={"i": i}) for i in range(3)]
    view = tier1_view(es, budget=InspectionBudget(total_tokens=10_000))
    for e in es:
        assert view.inspect(e.event_id) is not None
    assert len(view.budget.inspected_ids) == 3
    assert view.tokens_spent() > 0
