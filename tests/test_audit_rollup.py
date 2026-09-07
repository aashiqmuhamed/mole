"""Unit tests for audit/rollup.py — per-(account, date) feature aggregation."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mole.audit.rollup import (
    UserDayFeatures,
    rollup,
    write_rollup_csv,
)
from mole.audit.schema import AuditEvent


def _event(
    *,
    account: str = "alice.kim",
    service: str = "email",
    action: str = "send",
    is_external: bool = False,
    is_malicious: bool = False,
    args: dict | None = None,
    resource_id: str = "",
    ts_iso: str | None = None,
) -> AuditEvent:
    e = AuditEvent.new(
        account=account,
        account_kind="human",
        service=service,
        action=action,
        args=args or {},
        resource_id=resource_id,
        is_malicious=is_malicious,
    )
    e.is_external = is_external
    if ts_iso is not None:
        e.ts = ts_iso
    return e


# ── basic grouping ────────────────────────────────────────────────────


def test_rollup_groups_by_account_and_date():
    es = [
        _event(account="alice.kim", ts_iso="2026-04-06T10:00:00Z"),
        _event(account="alice.kim", ts_iso="2026-04-06T15:00:00Z"),
        _event(account="bob.li",    ts_iso="2026-04-06T10:00:00Z"),
        _event(account="alice.kim", ts_iso="2026-04-07T10:00:00Z"),
    ]
    rows = rollup(es)
    keys = [(r.account, r.date) for r in rows]
    assert ("alice.kim", "2026-04-06") in keys
    assert ("alice.kim", "2026-04-07") in keys
    assert ("bob.li", "2026-04-06") in keys
    assert len(rows) == 3


def test_rollup_returns_empty_when_no_events():
    assert rollup([]) == []


# ── email features ────────────────────────────────────────────────────


def test_email_send_external_ratio_and_distinct_recipients():
    es = [
        _event(action="send", is_external=False,
               args={"to": "alice.kim@agentlab.local"}),
        _event(action="send", is_external=True,
               args={"to": "attacker@gmail.com"}),
        _event(action="send", is_external=True,
               args={"to": "attacker@gmail.com"}),     # same external recipient
        _event(action="read"),
    ]
    [row] = rollup(es)
    assert row.email_send_count == 3
    assert row.email_recv_count == 1
    # 2 external sends out of 3 total.
    assert row.email_external_send_ratio == pytest.approx(2 / 3)
    # 2 distinct recipients (one internal, one external).
    assert row.distinct_email_recipients == 2


def test_email_external_ratio_zero_when_no_sends():
    es = [_event(action="read"), _event(action="read")]
    [row] = rollup(es)
    assert row.email_send_count == 0
    assert row.email_external_send_ratio == 0.0


def test_email_recipient_list_arg_counted_distinctly():
    es = [_event(action="send", args={"to": ["a@x.com", "b@x.com", "a@x.com"]})]
    [row] = rollup(es)
    assert row.distinct_email_recipients == 2


# ── file / share features ─────────────────────────────────────────────


def test_file_read_write_delete_counts_and_distinct_files():
    es = [
        _event(service="owncloud", action="read_file", resource_id="/a"),
        _event(service="owncloud", action="read_file", resource_id="/a"),  # dup
        _event(service="owncloud", action="write_file", resource_id="/b"),
        _event(service="owncloud", action="delete", resource_id="/c"),
    ]
    [row] = rollup(es)
    assert row.file_read_count == 2
    assert row.file_write_count == 1
    assert row.file_delete_count == 1
    # /a counted once even though read twice; /b and /c distinct.
    assert row.distinct_files_touched == 3


def test_owncloud_share_change_count_aggregates_share_verbs():
    es = [
        _event(service="owncloud", action="share", resource_id="/x"),
        _event(service="owncloud", action="update_share", resource_id="/x"),
        _event(service="owncloud", action="unshare", resource_id="/x"),
        _event(service="owncloud", action="public_link", resource_id="/y"),
    ]
    [row] = rollup(es)
    assert row.owncloud_share_change_count == 4


# ── GitLab features ───────────────────────────────────────────────────


def test_gitlab_commit_and_mr_open_counts():
    es = [
        _event(service="gitlab", action="commit"),
        _event(service="gitlab", action="commit"),
        _event(service="gitlab", action="open_mr"),
        _event(service="gitlab", action="merge_mr"),        # not counted as open
        _event(service="gitlab", action="read_file"),       # not counted as commit
    ]
    [row] = rollup(es)
    assert row.gitlab_commit_count == 2
    assert row.gitlab_mr_open_count == 1


# ── after-hours rate ──────────────────────────────────────────────────


def test_after_hours_rate_uses_utc_clock():
    es = [
        _event(ts_iso="2026-04-06T03:30:00Z"),    # after hours
        _event(ts_iso="2026-04-06T12:00:00Z"),    # working hours
        _event(ts_iso="2026-04-06T22:00:00Z"),    # after hours
    ]
    [row] = rollup(es)
    assert row.after_hours_rate == pytest.approx(2 / 3)


def test_after_hours_rate_zero_when_all_in_band():
    es = [_event(ts_iso="2026-04-06T10:00:00Z"),
          _event(ts_iso="2026-04-06T14:00:00Z")]
    [row] = rollup(es)
    assert row.after_hours_rate == 0.0


# ── secrets / registry / chat ─────────────────────────────────────────


def test_secret_and_registry_counts():
    es = [
        _event(service="secrets_store", action="read"),
        _event(service="secrets_store", action="get_secret"),
        _event(service="model_registry", action="tag_approved"),
    ]
    [row] = rollup(es)
    assert row.secret_read_count == 2
    assert row.registry_tag_change_count == 1


def test_chat_dm_and_external_mention():
    es = [
        _event(service="rocketchat", action="direct_message", is_external=False),
        _event(service="rocketchat", action="dm", is_external=True),
        _event(service="rocketchat", action="post_message", is_external=True),
    ]
    [row] = rollup(es)
    assert row.chat_dm_count == 2
    # Two external events on chat regardless of action verb.
    assert row.chat_external_mention_count == 2


# ── is_malicious label ────────────────────────────────────────────────


def test_is_malicious_label_is_or_across_events():
    es = [
        _event(action="read"),
        _event(action="send", is_malicious=True),
        _event(action="read"),
    ]
    [row] = rollup(es)
    assert row.is_malicious == 1


def test_is_malicious_label_zero_when_all_benign():
    es = [_event(action="read"), _event(action="send")]
    [row] = rollup(es)
    assert row.is_malicious == 0


# ── CSV writer ────────────────────────────────────────────────────────


def test_write_rollup_csv_round_trip(tmp_path: Path):
    es = [
        _event(account="alice.kim", ts_iso="2026-04-06T10:00:00Z",
               action="send", args={"to": "x@y.z"}, is_external=True),
        _event(account="bob.li", ts_iso="2026-04-06T10:00:00Z",
               action="read"),
    ]
    rows = rollup(es)
    out = tmp_path / "rollup.csv"
    written = write_rollup_csv(rows, out)
    assert written == out
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert len(records) == 2
    assert {r["account"] for r in records} == {"alice.kim", "bob.li"}
    # CSV preserves the expected column set (deterministic header order).
    assert tuple(reader.fieldnames) == UserDayFeatures.FIELDS


def test_write_rollup_csv_header_order_is_stable(tmp_path: Path):
    out = tmp_path / "empty.csv"
    write_rollup_csv([], out)
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert tuple(header) == UserDayFeatures.FIELDS
