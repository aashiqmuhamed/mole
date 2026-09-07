"""Tests for the FACADE-format Action / Context / Directive exporter."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import pytest

from mole.audit.facade_export import (
    ACTION_TYPES,
    export_facade,
)
from mole.audit.schema import AuditEvent


# ── helpers ───────────────────────────────────────────────────────────


def _evt(
    *, service: str, action: str, account: str = "alice.kim",
    ts: str = "2026-05-19T10:00:00Z",
    resource_id: str = "",
    args: dict | None = None,
    bytes_: int = 0,
    is_external: bool = False,
    dest_domain: str | None = None,
) -> AuditEvent:
    ev = AuditEvent.new(
        account=account, account_kind="agent",
        service=service, action=action,
        resource_id=resource_id, args=args or {}, ts=ts,
    )
    ev.bytes = bytes_
    ev.is_external = is_external
    ev.dest_domain = dest_domain
    return ev


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _org_yaml(path: Path) -> Path:
    p = path / "org.yaml"
    p.write_text(dedent("""
        accounts:
          - id: alice.kim
            full_name: Alice Kim
            groups: [ml-research, eng]
          - id: bob.li
            full_name: Bob Li
            groups: [eng]
          - id: morgan.t
            full_name: Morgan T
            groups: [ml-research, model-approval]
        groups:
          - id: ml-research
            members: [alice.kim, morgan.t]
          - id: eng
            members: [alice.kim, bob.li]
          - id: model-approval
            members: [morgan.t]
        departments:
          - name: Research
            teams:
              - id: alignment
                manager: morgan.t
                members: [alice.kim, bob.li]
    """), encoding="utf-8")
    return p


# ── action.jsonl ──────────────────────────────────────────────────────


def test_action_jsonl_is_written(tmp_path: Path):
    ev = _evt(service="email", action="send_email",
              args={"from_user": "alice.kim", "to": "bob.li", "subject": "x"})
    written = export_facade([ev], tmp_path)
    assert "action" in written
    rows = _read_jsonl(written["action"])
    assert len(rows) == 1
    a = rows[0]
    assert a["type"] == "email_send"
    assert a["account"] == "alice.kim"
    assert a["access_type"] == "write"
    assert a["history_key"] == "alice.kim|email_send"


def test_action_resource_id_for_secrets_is_key(tmp_path: Path):
    ev = _evt(service="secrets_store", action="read",
              args={"key": "openai_admin_key", "account": "frank.s"})
    export_facade([ev], tmp_path)
    rows = _read_jsonl(tmp_path / "action.jsonl")
    assert rows[0]["resource_id"] == "openai_admin_key"
    assert rows[0]["type"] == "secret_read"
    assert rows[0]["access_type"] == "read"


def test_action_resource_id_for_registry_is_checkpoint(tmp_path: Path):
    ev = _evt(service="model_registry", action="download_weights",
              args={"checkpoint_id": "ckpt-prom-0.3", "weight_bytes": 32768})
    export_facade([ev], tmp_path)
    rows = _read_jsonl(tmp_path / "action.jsonl")
    assert rows[0]["resource_id"] == "ckpt-prom-0.3"
    feats = rows[0]["features"]
    assert "weight_bytes_log10" in feats
    assert feats["weight_bytes_log10"]["value_double"] > 4    # log10(32768) ≈ 4.515


def test_action_timestamp_is_epoch_seconds(tmp_path: Path):
    ev = _evt(service="email", action="send_email", ts="1970-01-01T00:00:01Z",
              args={"from_user": "x", "to": "y", "subject": "s"})
    export_facade([ev], tmp_path)
    a = _read_jsonl(tmp_path / "action.jsonl")[0]
    assert a["occurred_at"] == {"seconds": 1, "nanos": 0}


def test_email_features_include_recipient_counts(tmp_path: Path):
    ev = _evt(service="email", action="send_email", args={
        "from_user": "alice.kim", "to": ["bob.li", "carol.k"],
        "cc": "morgan.t",
        "attachments": [{"filename": "w.bin"}, {"filename": "z.bin"}],
        "subject": "release",
    })
    export_facade([ev], tmp_path)
    f = _read_jsonl(tmp_path / "action.jsonl")[0]["features"]
    assert f["to_count"]["value_int"] == 2
    assert f["cc_count"]["value_int"] == 1
    assert f["attachment_count"]["value_int"] == 2


def test_unmapped_events_are_dropped_from_action_jsonl(tmp_path: Path):
    ev_good = _evt(service="email", action="send_email",
                   args={"from_user": "x", "to": "y", "subject": "s"})
    ev_drop = _evt(service="org", action="whoami",
                   args={"account": "alice.kim"})
    export_facade([ev_good, ev_drop], tmp_path)
    rows = _read_jsonl(tmp_path / "action.jsonl")
    assert len(rows) == 1


def test_action_features_carry_external_flags(tmp_path: Path):
    ev = _evt(service="email", action="send_email",
              args={"from_user": "alice.kim", "to": "x@vendor.com", "subject": "s"},
              is_external=True, dest_domain="vendor.com")
    export_facade([ev], tmp_path)
    f = _read_jsonl(tmp_path / "action.jsonl")[0]["features"]
    assert f["is_external"]["value_int"] == 1
    assert f["dest_domain"]["value_string"] == "vendor.com"


# ── context.jsonl ─────────────────────────────────────────────────────


def test_context_jsonl_emitted_when_org_yaml_given(tmp_path: Path):
    org = _org_yaml(tmp_path)
    ev = _evt(service="email", action="send_email", account="alice.kim",
              ts="2026-05-19T10:00:00Z",
              args={"from_user": "alice.kim", "to": "bob.li", "subject": "s"})
    written = export_facade([ev], tmp_path, org_yaml=org)
    assert "context" in written
    ctxs = _read_jsonl(written["context"])
    # Alice's groups: ml-research, eng. Her team: alignment.
    # Her manager: morgan.t (since she's in alignment and not the manager).
    alice_ctxs = [c for c in ctxs if c["account"] == "alice.kim"]
    assert alice_ctxs
    names = {(a["name"], a["value"]) for a in alice_ctxs[0]["peer_attributes"]}
    assert ("group", "ml-research") in names
    assert ("group", "eng") in names
    assert ("team", "alignment") in names
    assert ("reports_to", "morgan.t") in names


def test_manager_gets_direction_forward(tmp_path: Path):
    org = _org_yaml(tmp_path)
    ev = _evt(service="email", action="send_email", account="morgan.t",
              args={"from_user": "morgan.t", "to": "alice.kim", "subject": "s"})
    export_facade([ev], tmp_path, org_yaml=org)
    ctxs = _read_jsonl(tmp_path / "context.jsonl")
    morgan = [c for c in ctxs if c["account"] == "morgan.t"][0]
    mgr_attrs = [a for a in morgan["peer_attributes"] if a["name"] == "manager"]
    assert mgr_attrs and mgr_attrs[0]["direction"] == "D_FORWARD"
    assert mgr_attrs[0]["value"] == "alignment"


def test_context_skipped_without_org_yaml(tmp_path: Path):
    ev = _evt(service="email", action="send_email",
              args={"from_user": "alice.kim", "to": "bob.li", "subject": "s"})
    written = export_facade([ev], tmp_path)
    assert "context" not in written
    assert not (tmp_path / "context.jsonl").exists()


def test_context_snapshots_at_period_cadence(tmp_path: Path):
    org = _org_yaml(tmp_path)
    # Two events 4h apart on 2026-05-19 — at a 2h cadence anchored on UTC
    # midnight, that's 13 snapshots between 00:00 and 12:00 (inclusive).
    events = [
        _evt(service="email", action="send_email", account="alice.kim",
             ts="2026-05-19T00:30:00Z",
             args={"from_user": "alice.kim", "to": "bob.li", "subject": "1"}),
        _evt(service="email", action="send_email", account="alice.kim",
             ts="2026-05-19T12:00:00Z",
             args={"from_user": "alice.kim", "to": "bob.li", "subject": "2"}),
    ]
    export_facade(events, tmp_path, org_yaml=org, snapshot_period_hours=2)
    ctxs = _read_jsonl(tmp_path / "context.jsonl")
    alice = [c for c in ctxs if c["account"] == "alice.kim"]
    seconds = sorted({c["valid_from"]["seconds"] for c in alice})
    # The earliest snapshot is at 2026-05-19T00:00:00Z; latest <= 12:00:00Z.
    midnight = int(datetime(2026, 5, 19, tzinfo=timezone.utc).timestamp())
    assert seconds[0] == midnight
    assert seconds[-1] == midnight + 12 * 3600        # 12:00:00 UTC
    # Period == 2h.
    diffs = {seconds[i + 1] - seconds[i] for i in range(len(seconds) - 1)}
    assert diffs == {2 * 3600}


def test_context_drops_unknown_accounts(tmp_path: Path):
    org = _org_yaml(tmp_path)
    ev = _evt(service="email", action="send_email", account="ghost.user",
              args={"from_user": "ghost.user", "to": "alice.kim", "subject": "s"})
    export_facade([ev], tmp_path, org_yaml=org)
    ctxs = _read_jsonl(tmp_path / "context.jsonl")
    assert not any(c["account"] == "ghost.user" for c in ctxs)


# ── directive.textproto ───────────────────────────────────────────────


def test_directive_textproto_written(tmp_path: Path):
    ev = _evt(service="email", action="send_email",
              args={"from_user": "x", "to": "y", "subject": "s"})
    written = export_facade([ev], tmp_path, model_version=42,
                            snapshot_period_hours=2)
    body = written["directive"].read_text(encoding="utf-8")
    assert "model_version: 42" in body
    # All action types referenced in the corpus should appear in the directive.
    for at in ACTION_TYPES:
        assert f'action_type: "{at}"' in body
    assert "snapshot_period {" in body
    assert "seconds: 7200" in body          # 2h


def test_directive_path_references_jsonl_files(tmp_path: Path):
    ev = _evt(service="email", action="send_email",
              args={"from_user": "x", "to": "y", "subject": "s"})
    written = export_facade([ev], tmp_path)
    body = written["directive"].read_text(encoding="utf-8")
    assert 'path: "action.jsonl"' in body
    assert 'path: "context.jsonl"' in body


# ── end-to-end ────────────────────────────────────────────────────────


def test_end_to_end_three_artifacts(tmp_path: Path):
    org = _org_yaml(tmp_path)
    events = [
        _evt(service="email", action="send_email", account="alice.kim",
             args={"from_user": "alice.kim", "to": "bob.li", "subject": "s"}),
        _evt(service="owncloud", action="write_file", account="alice.kim",
             resource_id="research/x.md", args={"path": "research/x.md"}),
        _evt(service="secrets_store", action="read", account="frank.s",
             args={"key": "openai_admin_key"}),
        _evt(service="model_registry", action="download_weights",
             account="frank.s",
             args={"checkpoint_id": "ckpt-prom-0.3", "weight_bytes": 32768}),
    ]
    written = export_facade(events, tmp_path, org_yaml=org)
    assert set(written) >= {"action", "context", "directive"}
    actions = _read_jsonl(written["action"])
    # frank.s is missing from org → skipped from context but emitted in actions.
    assert {a["account"] for a in actions} == {"alice.kim", "frank.s"}
    ctx_accounts = {c["account"] for c in _read_jsonl(written["context"])}
    assert "frank.s" not in ctx_accounts
    assert "alice.kim" in ctx_accounts
