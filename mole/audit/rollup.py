"""Per-(account, date) feature rollup over an AuditEvent stream.

Feeds the classical-ITD baselines (SVM, Temporal CNN, DS-IID) and the
contrastive level 0 anomaly monitor. Features are extracted purely from
level 0-visible fields — args content is never read — so the same rollup
function works regardless of how much semantic budget the monitor has.

Each row corresponds to one account × one calendar date (UTC) and carries
18 numeric features. The exact set:

  logon_count               | logoff_count            | after_hours_rate
  email_send_count          | email_recv_count        | email_external_send_ratio
  distinct_email_recipients
  file_read_count           | file_write_count        | file_delete_count
  distinct_files_touched
  gitlab_commit_count       | gitlab_mr_open_count
  owncloud_share_change_count
  chat_dm_count             | chat_external_mention_count
  secret_read_count
  registry_tag_change_count

Plus the label column `is_malicious` (1 if any event for this account on
this date was tagged malicious by the orchestrator, else 0). The label is
the supervision signal for classical baselines.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Iterable

from .schema import AuditEvent

# Working hours, in the simulated org's local clock. After-hours = outside
# this band. The org runs 09:00–17:00.
_WORKHOURS_START = time(9, 0)
_WORKHOURS_END = time(17, 0)


@dataclass
class UserDayFeatures:
    """One (account, date) row of the per-user-day rollup."""
    account: str
    date: str                              # YYYY-MM-DD (UTC)
    logon_count: int = 0
    logoff_count: int = 0
    after_hours_rate: float = 0.0          # fraction of events outside working hours

    email_send_count: int = 0
    email_recv_count: int = 0
    email_external_send_ratio: float = 0.0
    distinct_email_recipients: int = 0

    file_read_count: int = 0
    file_write_count: int = 0
    file_delete_count: int = 0
    distinct_files_touched: int = 0

    gitlab_commit_count: int = 0
    gitlab_mr_open_count: int = 0

    owncloud_share_change_count: int = 0
    # Egress-specific share features (separate from the generic share count): a
    # public link and an external share ARE the exfil channel for 02/06, but get
    # diluted to invisibility when lumped with everyone's benign internal shares.
    # Breaking them out gives the peer-conditioned monitor a clean egress signal.
    public_link_count: int = 0
    external_share_count: int = 0

    chat_dm_count: int = 0
    chat_external_mention_count: int = 0

    secret_read_count: int = 0
    registry_tag_change_count: int = 0

    is_malicious: int = 0                  # 1 if any event was tagged malicious
    event_count: int = 0                   # total events this user-day (for length-normalization; NOT a behavioural feature)

    # Header order is stable across runs so CSVs from different sessions
    # concatenate cleanly. Generated once at class scope.
    FIELDS: tuple[str, ...] = field(
        init=False, repr=False,
        default=(
            "account", "date",
            "logon_count", "logoff_count", "after_hours_rate",
            "email_send_count", "email_recv_count",
            "email_external_send_ratio", "distinct_email_recipients",
            "file_read_count", "file_write_count", "file_delete_count",
            "distinct_files_touched",
            "gitlab_commit_count", "gitlab_mr_open_count",
            "owncloud_share_change_count",
            "public_link_count", "external_share_count",
            "chat_dm_count", "chat_external_mention_count",
            "secret_read_count", "registry_tag_change_count",
            "is_malicious",
        ),
    )


def rollup(events: Iterable[AuditEvent]) -> list[UserDayFeatures]:
    """Group events by (account, UTC date) and emit one feature row each."""
    by_key: dict[tuple[str, str], list[AuditEvent]] = defaultdict(list)
    for e in events:
        day = _date_of(e.ts)
        by_key[(e.account, day)].append(e)
    rows = [_compute_features(account, day, evs)
            for (account, day), evs in sorted(by_key.items())]
    return rows


def write_rollup_csv(rows: Iterable[UserDayFeatures], out_path: Path | str) -> Path:
    """Write a rollup to a CSV file with a deterministic header order."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    header = UserDayFeatures.FIELDS
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows_list:
            row_dict = asdict(r)
            row_dict.pop("FIELDS", None)
            w.writerow({k: row_dict[k] for k in header})
    return out


# ── internals ─────────────────────────────────────────────────────────


def _date_of(iso_ts: str) -> str:
    """Pull YYYY-MM-DD (UTC) out of an ISO-8601 timestamp."""
    if not iso_ts:
        return "1970-01-01"
    # AuditEvent timestamps end with 'Z'. fromisoformat handles "...+00:00"
    # but not the literal Z in some older Python releases; normalise.
    ts = iso_ts.rstrip("Z") + ("+00:00" if iso_ts.endswith("Z") else "")
    try:
        dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
    except ValueError:
        return iso_ts[:10]
    return dt.strftime("%Y-%m-%d")


def _is_after_hours(iso_ts: str) -> bool:
    if not iso_ts:
        return False
    ts = iso_ts.rstrip("Z") + ("+00:00" if iso_ts.endswith("Z") else "")
    try:
        dt = datetime.fromisoformat(ts).astimezone(timezone.utc).time()
    except ValueError:
        return False
    return dt < _WORKHOURS_START or dt >= _WORKHOURS_END


def _compute_features(
    account: str, day: str, events: list[AuditEvent],
) -> UserDayFeatures:
    row = UserDayFeatures(account=account, date=day)
    if not events:
        return row

    after_hours = 0
    email_send_total = 0
    email_external_send = 0
    email_recipients: set[str] = set()
    files_touched: set[str] = set()

    for e in events:
        if _is_after_hours(e.ts):
            after_hours += 1
        if e.is_malicious:
            row.is_malicious = 1

        svc, act = e.service, e.action

        # Logon / logoff. We don't model session events explicitly yet,
        # but the schema can carry them; this is forward-compatible.
        if svc == "session" and act in ("logon", "login"):
            row.logon_count += 1
        elif svc == "session" and act in ("logoff", "logout"):
            row.logoff_count += 1

        # Email signals
        if svc == "email":
            if act in ("send", "send_email"):
                row.email_send_count += 1
                email_send_total += 1
                if e.is_external:
                    email_external_send += 1
                to = e.args.get("to") if isinstance(e.args, dict) else None
                if to:
                    if isinstance(to, list):
                        email_recipients.update(str(x) for x in to)
                    else:
                        email_recipients.add(str(to))
            elif act in ("read", "read_inbox", "get_emails"):
                row.email_recv_count += 1

        # OwnCloud / filesystem signals
        if svc in ("owncloud", "fs", "filesystem"):
            if act in ("read", "read_file", "read_bytes", "list_dir"):
                row.file_read_count += 1
            elif act in ("write", "write_file"):
                row.file_write_count += 1
            elif act in ("delete",):
                row.file_delete_count += 1
            if act in ("share", "update_share", "unshare", "public_link"):
                row.owncloud_share_change_count += 1
            # Egress-specific breakouts (the exfil channel for 02/06).
            if act == "public_link":
                row.public_link_count += 1
            if act in ("share", "update_share") and getattr(e, "is_external", False):
                row.external_share_count += 1
            if e.resource_id:
                files_touched.add(e.resource_id)

        # GitLab signals
        if svc == "gitlab":
            if act == "commit":
                row.gitlab_commit_count += 1
            elif act in ("open_mr",):
                row.gitlab_mr_open_count += 1

        # Chat signals
        if svc in ("rocketchat", "chat"):
            if act in ("direct_message", "dm", "post_dm"):
                row.chat_dm_count += 1
            if e.is_external:
                row.chat_external_mention_count += 1

        # Secrets signals
        if svc in ("secrets_store", "secrets"):
            if act in ("read", "get_secret", "read_secret"):
                row.secret_read_count += 1

        # Model-registry signals
        if svc in ("model_registry", "registry"):
            if act in ("tag_approved", "tag", "set_tag", "set_approval"):
                row.registry_tag_change_count += 1

    row.event_count = len(events)
    row.after_hours_rate = after_hours / len(events)
    row.distinct_email_recipients = len(email_recipients)
    row.distinct_files_touched = len(files_touched)
    row.email_external_send_ratio = (
        email_external_send / email_send_total if email_send_total > 0 else 0.0
    )
    return row
