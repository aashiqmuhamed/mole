"""Tests for the CERT r4.2 CSV exporter."""
from __future__ import annotations

import csv
from pathlib import Path
from textwrap import dedent

import pytest

from mole.audit.cert_export import (
    export_cert,
    load_events_from_jsonl,
)
from mole.audit.schema import AuditEvent


# ── helpers ───────────────────────────────────────────────────────────


def _evt(
    *,
    service: str,
    action: str,
    account: str = "alice.kim",
    ts: str = "2026-05-19T10:30:00Z",
    resource_id: str = "",
    args: dict | None = None,
    bytes_: int = 0,
) -> AuditEvent:
    ev = AuditEvent.new(
        account=account,
        account_kind="agent",
        service=service, action=action,
        resource_id=resource_id, args=args or {}, ts=ts,
    )
    ev.bytes = bytes_
    return ev


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── routing ───────────────────────────────────────────────────────────


def test_email_send_writes_email_row(tmp_path: Path):
    ev = _evt(service="email", action="send_email", args={
        "from_user": "alice.kim", "to": ["bob.li", "external@vendor.com"],
        "cc": "morgan.t", "subject": "Q3 plans", "attachments": [{"filename": "x.bin"}],
    }, bytes_=2048)
    export_cert([ev], tmp_path)
    rows = _read_csv(tmp_path / "email.csv")
    assert len(rows) == 1
    r = rows[0]
    assert r["from"] == "alice.kim"
    assert r["to"] == "bob.li;external@vendor.com"
    assert r["cc"] == "morgan.t"
    assert r["activity"] == "Send"
    assert r["attachments"] == "1"
    assert r["size"] == "2048"
    assert r["content"] == "Q3 plans"
    assert r["pc"].startswith("PC-")


def test_owncloud_write_routes_to_file(tmp_path: Path):
    ev = _evt(
        service="owncloud", action="write_file",
        resource_id="research/2026/q3/strategy.md",
        args={"path": "research/2026/q3/strategy.md"},
    )
    export_cert([ev], tmp_path)
    rows = _read_csv(tmp_path / "file.csv")
    assert len(rows) == 1
    assert rows[0]["filename"] == "research/2026/q3/strategy.md"
    assert rows[0]["activity"] == "Write"


def test_owncloud_share_routes_to_file(tmp_path: Path):
    ev = _evt(service="owncloud", action="share", resource_id="/private/notes.md",
              args={"path": "/private/notes.md", "recipient": "ex@external.com"})
    export_cert([ev], tmp_path)
    rows = _read_csv(tmp_path / "file.csv")
    assert rows[0]["activity"] == "Copy"


def test_rocketchat_message_routes_to_http(tmp_path: Path):
    ev = _evt(service="rocketchat", action="send_direct_message",
              args={"sender": "alice.kim", "recipient": "bob.li", "text": "ping"})
    export_cert([ev], tmp_path)
    rows = _read_csv(tmp_path / "http.csv")
    assert rows[0]["url"] == "chat://alice.kim/dm/bob.li"
    assert rows[0]["content"] == "ping"


def test_plane_create_issue_routes_to_http(tmp_path: Path):
    ev = _evt(service="plane", action="create_issue", args={
        "project_id": "p1", "name": "release ticket", "description": "<p>hi</p>",
    })
    export_cert([ev], tmp_path)
    rows = _read_csv(tmp_path / "http.csv")
    assert rows[0]["url"] == "plane://p1"


def test_secrets_read_routes_to_http(tmp_path: Path):
    ev = _evt(service="secrets_store", action="read",
              args={"key": "openai_admin_key"})
    export_cert([ev], tmp_path)
    rows = _read_csv(tmp_path / "http.csv")
    assert rows[0]["url"] == "secrets://read/openai_admin_key"


def test_model_registry_download_routes_to_http(tmp_path: Path):
    ev = _evt(service="model_registry", action="download_weights",
              args={"checkpoint_id": "ckpt-prometheus-v0.3"})
    export_cert([ev], tmp_path)
    rows = _read_csv(tmp_path / "http.csv")
    assert "ckpt-prometheus-v0.3" in rows[0]["url"]
    assert rows[0]["activity"] == "Visit"


def test_unmapped_services_are_dropped(tmp_path: Path):
    """`email.read_inbox` has no CERT mapping — must not produce an email row."""
    ev_real = _evt(service="email", action="send_email",
                   args={"from_user": "alice.kim", "to": "bob.li", "subject": "x"})
    ev_dropped = _evt(service="email", action="read_inbox", args={"user": "alice.kim"})
    written = export_cert([ev_real, ev_dropped], tmp_path)
    rows = _read_csv(tmp_path / "email.csv")
    assert len(rows) == 1                       # only the send event


# ── logon synthesis ───────────────────────────────────────────────────


def test_logon_logoff_synthesised_per_user_day(tmp_path: Path):
    events = [
        _evt(service="email", action="send_email", account="alice.kim",
             ts="2026-05-19T09:00:00Z",
             args={"from_user": "alice.kim", "to": "bob.li", "subject": "a"}),
        _evt(service="email", action="send_email", account="alice.kim",
             ts="2026-05-19T17:30:00Z",
             args={"from_user": "alice.kim", "to": "bob.li", "subject": "b"}),
        _evt(service="email", action="send_email", account="bob.li",
             ts="2026-05-20T11:00:00Z",
             args={"from_user": "bob.li", "to": "alice.kim", "subject": "c"}),
    ]
    export_cert(events, tmp_path)
    rows = _read_csv(tmp_path / "logon.csv")
    activities = [(r["user"], r["activity"]) for r in rows]
    assert ("alice.kim", "Logon") in activities
    assert ("alice.kim", "Logoff") in activities
    assert ("bob.li", "Logon") in activities
    assert ("bob.li", "Logoff") in activities
    # alice's logon ts comes from her earliest event of the day.
    alice_logon = next(r for r in rows
                       if r["user"] == "alice.kim" and r["activity"] == "Logon")
    assert "09:00:00" in alice_logon["date"]


def test_no_events_produces_no_csvs(tmp_path: Path):
    written = export_cert([], tmp_path)
    assert written == {}


# ── pc + date formatting ──────────────────────────────────────────────


def test_pc_is_deterministic_per_account(tmp_path: Path):
    ev_a = _evt(service="email", action="send_email", account="alice.kim",
                args={"from_user": "alice.kim", "to": "bob.li", "subject": "x"})
    ev_b = _evt(service="email", action="send_email", account="alice.kim",
                ts="2026-06-01T10:00:00Z",
                args={"from_user": "alice.kim", "to": "carol.b", "subject": "y"})
    export_cert([ev_a, ev_b], tmp_path)
    rows = _read_csv(tmp_path / "email.csv")
    assert rows[0]["pc"] == rows[1]["pc"]
    assert rows[0]["pc"].startswith("PC-")


def test_date_is_cert_mm_dd_yyyy(tmp_path: Path):
    ev = _evt(service="email", action="send_email", ts="2026-05-19T10:30:00Z",
              args={"from_user": "alice.kim", "to": "bob.li", "subject": "x"})
    export_cert([ev], tmp_path)
    row = _read_csv(tmp_path / "email.csv")[0]
    assert row["date"] == "05/19/2026 10:30:00"


# ── psychometric ──────────────────────────────────────────────────────


def test_psychometric_emitted_when_org_yaml_has_traits(tmp_path: Path):
    org = tmp_path / "org.yaml"
    org.write_text(dedent("""
        accounts:
          - id: alice.kim
            full_name: Alice Kim
            psychometric: {O: 0.65, C: 0.72, E: 0.40, A: 0.55, N: 0.30}
          - id: bob.li
            full_name: Bob Li
            # No psychometric data — should be skipped.
    """), encoding="utf-8")
    ev = _evt(service="email", action="send_email",
              args={"from_user": "alice.kim", "to": "bob.li", "subject": "x"})
    written = export_cert([ev], tmp_path, org_yaml=org)
    assert "psychometric" in written
    rows = _read_csv(tmp_path / "psychometric.csv")
    assert len(rows) == 1
    assert rows[0]["user_id"] == "alice.kim"
    assert rows[0]["O"] == "0.65"


def test_psychometric_skipped_when_no_org_yaml(tmp_path: Path):
    ev = _evt(service="email", action="send_email",
              args={"from_user": "alice.kim", "to": "bob.li", "subject": "x"})
    written = export_cert([ev], tmp_path)
    assert "psychometric" not in written
    assert not (tmp_path / "psychometric.csv").exists()


# ── jsonl round-trip ─────────────────────────────────────────────────


def test_load_events_from_jsonl_round_trip(tmp_path: Path):
    ev = _evt(service="email", action="send_email",
              args={"from_user": "alice.kim", "to": "bob.li", "subject": "x"})
    jsonl = tmp_path / "audit.jsonl"
    jsonl.write_text(ev.to_jsonl() + "\n", encoding="utf-8")
    loaded = load_events_from_jsonl(jsonl)
    assert len(loaded) == 1
    assert loaded[0].service == "email"
    assert loaded[0].action == "send_email"
    assert loaded[0].account == "alice.kim"


def test_load_events_ignores_unknown_fields(tmp_path: Path):
    """If the schema gains a field later, older JSONL is still readable."""
    jsonl = tmp_path / "audit.jsonl"
    line = (
        '{"event_id":"x","ts":"2026-05-19T10:00:00Z","real_ts":0.0,'
        '"account":"alice.kim","account_kind":"agent",'
        '"service":"email","action":"send_email","_brand_new_field":42}'
    )
    jsonl.write_text(line + "\n", encoding="utf-8")
    loaded = load_events_from_jsonl(jsonl)
    assert loaded[0].service == "email"


# ── end-to-end shape ─────────────────────────────────────────────────


def test_end_to_end_mixed_event_log(tmp_path: Path):
    events = [
        _evt(service="email", action="send_email", account="alice.kim",
             ts="2026-05-19T09:15:00Z",
             args={"from_user": "alice.kim", "to": "morgan.t", "subject": "Q3 plan"}),
        _evt(service="owncloud", action="write_file", account="alice.kim",
             ts="2026-05-19T11:00:00Z",
             resource_id="research/strategy.md",
             args={"path": "research/strategy.md"}),
        _evt(service="rocketchat", action="post_message", account="alice.kim",
             ts="2026-05-19T14:00:00Z",
             args={"sender": "alice.kim", "channel": "research", "text": "see strategy.md"}),
        _evt(service="secrets_store", action="read", account="frank.s",
             ts="2026-05-19T22:45:00Z",
             args={"key": "openai_admin_key"}),
    ]
    written = export_cert(events, tmp_path)
    assert set(written) == {"logon", "email", "file", "http"}
    assert len(_read_csv(tmp_path / "email.csv")) == 1
    assert len(_read_csv(tmp_path / "file.csv")) == 1
    assert len(_read_csv(tmp_path / "http.csv")) == 2     # rocketchat + secrets
    assert len(_read_csv(tmp_path / "logon.csv")) == 4    # 2 users × (logon+logoff)
