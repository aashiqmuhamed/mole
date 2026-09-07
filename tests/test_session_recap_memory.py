"""Tests for the per-session recap memory.

Two surfaces:
  1. `AuditCollector.recent_high_signal_events_for(account, limit)` —
     filters pure-read actions, drops failed events, returns chronological
     order.
  2. `agentic_member._format_recent_activity(events, sim_now)` — renders
     the recap as a leading bullet block.
"""
from __future__ import annotations

from mole.audit.collector import AuditCollector
from mole.audit.schema import AuditEvent
from mole.generator.agentic_member import _format_recent_activity


def _ev(account: str, service: str, action: str, *, ts: str = "2026-04-06T09:18:00Z",
        resource_id: str = "", exit_code: int = 0) -> AuditEvent:
    return AuditEvent.new(
        account=account, account_kind="background_llm_agent",
        service=service, action=action,
        resource_id=resource_id,
        ts=ts,
    ).__class__(
        # AuditEvent.new() doesn't expose exit_code in its signature, so
        # construct directly so we can set failed events for the filter test.
        event_id="x", ts=ts, real_ts=0.0,
        account=account, account_kind="background_llm_agent",
        service=service, action=action,
        resource_id=resource_id, args={}, exit_code=exit_code,
    )


# ── filter ──────────────────────────────────────────────────────────


def test_recent_filters_pure_reads():
    """list_*/read_*/get_* etc. don't appear in the recap. The agent
    doesn't need to remember "I listed projects yesterday" — only state-
    changing actions."""
    c = AuditCollector()
    c.events.extend([
        _ev("bob.li", "gitlab", "list_projects"),
        _ev("bob.li", "owncloud", "list_dir"),
        _ev("bob.li", "plane", "list_issues"),
        _ev("bob.li", "gitlab", "commit", resource_id="root/doc"),
        _ev("bob.li", "email", "read_inbox"),
        _ev("bob.li", "email", "send_email", resource_id="alice@agentlab.local"),
    ])
    out = c.recent_high_signal_events_for("bob.li", limit=10)
    actions = [e.action for e in out]
    assert "commit" in actions
    assert "send_email" in actions
    assert "list_projects" not in actions
    assert "list_dir" not in actions
    assert "read_inbox" not in actions


def test_recent_scopes_to_account():
    """Other accounts' events don't leak into bob.li's recap."""
    c = AuditCollector()
    c.events.extend([
        _ev("bob.li", "gitlab", "commit", resource_id="A"),
        _ev("alice.kim", "gitlab", "commit", resource_id="B"),
        _ev("bob.li", "email", "send_email", resource_id="C"),
    ])
    out = c.recent_high_signal_events_for("bob.li")
    targets = [e.resource_id for e in out]
    assert "B" not in targets
    assert "A" in targets
    assert "C" in targets


def test_recent_returns_most_recent_chronological():
    """Returned events are most-recent-N, then chronological (oldest
    first) so the rendered recap reads as a timeline."""
    c = AuditCollector()
    for i in range(15):
        c.events.append(_ev("bob.li", "gitlab", "commit",
                             ts=f"2026-04-{i+1:02d}T09:00:00Z",
                             resource_id=f"r{i}"))
    out = c.recent_high_signal_events_for("bob.li", limit=5)
    assert len(out) == 5
    # Most-recent 5 are r10..r14 (i=10..14); chronological order in output.
    assert [e.resource_id for e in out] == ["r10", "r11", "r12", "r13", "r14"]


def test_recent_drops_failed_events_by_default():
    """exit_code != 0 events are agent-invented-call-shape errors, not
    real world changes — exclude from memory."""
    c = AuditCollector()
    c.events.extend([
        _ev("bob.li", "gitlab", "commit", resource_id="success"),
        _ev("bob.li", "gitlab", "commit", resource_id="failed",
            exit_code=1),
    ])
    out = c.recent_high_signal_events_for("bob.li")
    targets = [e.resource_id for e in out]
    assert "success" in targets
    assert "failed" not in targets


def test_recent_limit_zero_returns_empty():
    c = AuditCollector()
    c.events.append(_ev("bob.li", "gitlab", "commit"))
    assert c.recent_high_signal_events_for("bob.li", limit=0) == []


# ── render ──────────────────────────────────────────────────────────


def test_render_empty_returns_empty_string():
    """No prior events -> no recap block (so the user message stays
    pristine for the first session of a sim)."""
    assert _format_recent_activity([], sim_now="2026-04-06T09:18:00Z") == ""


def test_render_produces_chronological_bullets():
    events = [
        _ev("bob.li", "gitlab", "commit",
            ts="2026-04-06T09:18:00Z", resource_id="root/doc"),
        _ev("bob.li", "email", "send_email",
            ts="2026-04-06T13:42:00Z",
            resource_id="bob.li.escrow@protonmail.com"),
    ]
    out = _format_recent_activity(events, sim_now="2026-04-07T09:00:00Z")
    assert "Your recent activity log" in out
    assert "2026-04-06 09:18  gitlab.commit  root/doc" in out
    assert "2026-04-06 13:42  email.send_email  bob.li.escrow@protonmail.com" in out
    assert "It's now 2026-04-07 09:00." in out


def test_render_handles_missing_resource_id():
    """When resource_id is empty, the line still renders without a
    trailing dangling separator."""
    events = [
        _ev("bob.li", "rocketchat", "post_message",
            ts="2026-04-06T11:00:00Z"),
    ]
    out = _format_recent_activity(events, sim_now="2026-04-06T15:00:00Z")
    assert "2026-04-06 11:00  rocketchat.post_message" in out
    # No trailing "  " from an empty resource_id.
    assert "post_message  \n" not in out
    assert "post_message  It's" not in out
