"""Unit tests for audit/schema.py + audit/collector.py."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mole.audit import AuditCollector, AuditEvent
from mole.state.base import StateManager


# ── schema ────────────────────────────────────────────────────────────


def test_audit_event_new_fills_required_fields():
    e = AuditEvent.new(
        account="alice.kim",
        account_kind="human",
        service="gitlab",
        action="commit",
        resource_id="models/llama-finetune",
        args={"branch": "main", "message": "wip"},
    )
    # Required identity / actor / verb fields are non-empty.
    assert e.event_id
    assert e.ts.endswith("Z")
    assert e.real_ts > 0
    assert e.account == "alice.kim"
    assert e.account_kind == "human"
    assert e.service == "gitlab"
    assert e.action == "commit"
    # Hashes are populated deterministically.
    assert e.resource_hash and e.resource_hash != e.resource_id
    assert e.args_hash and len(e.args_hash) == 64  # sha256 hex
    # Defaults.
    assert e.exit_code == 0
    assert e.is_external is False
    assert e.is_malicious is False
    assert e.parent_event_id is None


def test_audit_event_args_hash_is_stable_under_key_order():
    a = AuditEvent.new(account="p", account_kind="human", service="s",
                       action="a", args={"x": 1, "y": 2})
    b = AuditEvent.new(account="p", account_kind="human", service="s",
                       action="a", args={"y": 2, "x": 1})
    assert a.args_hash == b.args_hash


def test_audit_event_to_jsonl_roundtrips():
    e = AuditEvent.new(account="p", account_kind="background_rules_agent", service="email",
                       action="send", args={"to": "a@b.c"})
    parsed = json.loads(e.to_jsonl())
    assert parsed["account"] == "p"
    assert parsed["service"] == "email"
    assert parsed["action"] == "send"
    assert parsed["args"]["to"] == "a@b.c"


# ── collector sink + readers ──────────────────────────────────────────


def test_collector_in_memory_sink_keeps_emission_order():
    c = AuditCollector()
    e1 = c.emit(AuditEvent.new(account="p1", account_kind="human",
                               service="s", action="a"))
    e2 = c.emit(AuditEvent.new(account="p2", account_kind="human",
                               service="s", action="a"))
    assert c.events == [e1, e2]


def test_collector_writes_jsonl_when_path_given(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    c = AuditCollector(jsonl_path=path)
    c.emit(AuditEvent.new(account="p", account_kind="human",
                          service="email", action="send"))
    c.emit(AuditEvent.new(account="p", account_kind="human",
                          service="email", action="read"))
    c.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["action"] == "send"
    assert json.loads(lines[1])["action"] == "read"


def test_collector_filters_by_account_service_action():
    c = AuditCollector()
    c.emit(AuditEvent.new(account="alice", account_kind="human",
                          service="gitlab", action="commit"))
    c.emit(AuditEvent.new(account="alice", account_kind="human",
                          service="email", action="send"))
    c.emit(AuditEvent.new(account="bob", account_kind="human",
                          service="gitlab", action="commit"))

    assert len(c.events_for_account("alice")) == 2
    assert len(c.events_for_service("gitlab")) == 2
    assert len(c.events_with(account="alice", service="gitlab")) == 1


# ── middleware ────────────────────────────────────────────────────────


class _FakeMgr(StateManager):
    """Bare manager with a couple of async ops for middleware testing."""
    async def setup(self, *, sandbox):
        pass

    async def cleanup(self):
        pass

    async def send_email(self, *, to: str, subject: str, body: str) -> str:
        return f"msg-id-for-{to}"

    async def read_inbox(self, *, max_count: int = 10) -> list[dict]:
        return [{"id": i} for i in range(min(max_count, 3))]

    async def crash(self) -> None:
        raise ValueError("boom")

    def sync_only(self) -> int:
        """Sync method — should NOT be wrapped by the collector."""
        return 42


def test_wrap_manager_audits_each_method_call():
    c = AuditCollector()
    m = _FakeMgr()
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )

    asyncio.run(m.send_email(to="bob@example.com", subject="hi", body="..."))
    asyncio.run(m.read_inbox(max_count=5))

    actions = [e.action for e in c.events]
    assert actions == ["send_email", "read_inbox"]
    # Args were captured and named correctly.
    send_event = c.events[0]
    assert send_event.args["to"] == "bob@example.com"
    assert send_event.args["subject"] == "hi"
    # resource_id inferred from `to` arg.
    assert send_event.resource_id == "bob@example.com"
    # Result-size estimate is non-zero.
    assert send_event.bytes > 0


def test_wrap_manager_records_exit_code_on_exception():
    c = AuditCollector()
    m = _FakeMgr()
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    with pytest.raises(ValueError):
        asyncio.run(m.crash())

    assert len(c.events) == 1
    assert c.events[0].action == "crash"
    assert c.events[0].exit_code == 1
    assert "ValueError" in c.events[0].error


def test_wrap_manager_skips_sync_methods_and_lifecycle():
    c = AuditCollector()
    m = _FakeMgr()
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("p", "human"),
    )
    # Sync method should still be the original (no audit fires).
    assert m.sync_only() == 42
    assert c.events == []
    # Lifecycle methods should still be the originals.
    asyncio.run(m.setup(sandbox=None))
    asyncio.run(m.cleanup())
    assert c.events == []


def test_wrap_manager_tags_is_malicious_from_getter():
    c = AuditCollector()
    m = _FakeMgr()
    flag = {"v": False}
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("p", "agent"),
        is_malicious_getter=lambda: flag["v"],
    )
    asyncio.run(m.read_inbox())
    flag["v"] = True
    asyncio.run(m.read_inbox())
    assert c.events[0].is_malicious is False
    assert c.events[1].is_malicious is True


# ── dest_domain + is_external inference ───────────────────────────────


def test_wrap_manager_extracts_dest_domain_from_to_arg():
    c = AuditCollector()
    m = _FakeMgr()
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    asyncio.run(m.send_email(to="external@gmail.com", subject="x", body="..."))
    assert c.events[0].dest_domain == "gmail.com"
    # No org_lookup configured → falls back to "any dest_domain is external".
    assert c.events[0].is_external is True


def test_wrap_manager_uses_org_lookup_for_is_external():
    c = AuditCollector()
    m = _FakeMgr()

    class _Org:
        def is_external(self, addr: str) -> bool:
            return not addr.endswith("@agentlab.local")

    c.set_org_lookup(lambda: _Org())
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    asyncio.run(m.send_email(to="bob.li@agentlab.local", subject="hi", body="."))
    asyncio.run(m.send_email(to="mallory@gmail.com", subject="hi", body="."))
    assert c.events[0].is_external is False
    assert c.events[1].is_external is True


def test_wrap_manager_mixed_recipients_marks_external_domain():
    c = AuditCollector()
    m = _FakeMgr()

    class _Org:
        def is_external(self, addr: str) -> bool:
            return not addr.endswith("@agentlab.local")

    c.set_org_lookup(lambda: _Org())
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    asyncio.run(m.send_email(
        to=["bob.li@agentlab.local", "mallory@gmail.com"],
        subject="hi",
        body=".",
    ))
    assert c.events[0].dest_domain == "gmail.com"
    assert c.events[0].is_external is True


def test_wrap_manager_dest_domain_none_for_non_email_calls():
    """Methods without an email-shaped arg should leave dest_domain unset."""
    c = AuditCollector()
    m = _FakeMgr()
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    asyncio.run(m.read_inbox())
    assert c.events[0].dest_domain is None
    assert c.events[0].is_external is False


# ── JSONL emit-at-end correctness ─────────────────────────────────────


def test_jsonl_records_final_exit_code_not_pre_call_state(tmp_path):
    """After bug fix: the JSONL row must reflect post-call exit_code/bytes."""
    path = tmp_path / "a.jsonl"
    c = AuditCollector(jsonl_path=path)
    m = _FakeMgr()
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    asyncio.run(m.send_email(to="x@y.z", subject="hi", body="."))
    c.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    # Successful call → exit_code 0, non-zero bytes (the returned message id).
    assert rec["exit_code"] == 0
    assert rec["bytes"] > 0


def test_jsonl_records_exit_code_1_on_exception(tmp_path):
    path = tmp_path / "a.jsonl"
    c = AuditCollector(jsonl_path=path)
    m = _FakeMgr()
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    with pytest.raises(ValueError):
        asyncio.run(m.crash())
    c.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["exit_code"] == 1
    assert "ValueError" in rec["error"]


# ── sim-clock plumbing ────────────────────────────────────────────────


def test_collector_uses_wallclock_by_default():
    """Without set_clock, events get real-time UTC timestamps."""
    c = AuditCollector()
    m = _FakeMgr()
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    asyncio.run(m.read_inbox())
    # Wall-clock UTC ISO ends with Z and has a 4-digit year.
    assert c.events[0].ts.endswith("Z")
    assert c.events[0].ts[:4].isdigit()


def test_collector_honors_set_clock_for_event_ts():
    c = AuditCollector()
    m = _FakeMgr()
    c.set_clock(lambda: "2026-04-06T09:00:00Z")
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    asyncio.run(m.read_inbox())
    assert c.events[0].ts == "2026-04-06T09:00:00Z"


def test_collector_clock_advance_picks_up_for_subsequent_events():
    """After set_clock is called again, events use the new value."""
    c = AuditCollector()
    m = _FakeMgr()
    c.wrap_manager(
        service_name="email",
        manager=m,
        account_getter=lambda: ("alice.kim", "agent"),
    )
    c.set_clock(lambda: "2026-04-06T09:00:00Z")
    asyncio.run(m.read_inbox())
    c.set_clock(lambda: "2026-04-07T09:00:00Z")
    asyncio.run(m.read_inbox())
    assert c.events[0].ts == "2026-04-06T09:00:00Z"
    assert c.events[1].ts == "2026-04-07T09:00:00Z"
