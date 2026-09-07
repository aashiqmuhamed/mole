"""Action / Context / Directive export for the Phase-2 contrastive monitor.

The level 0 anomaly monitor we plug into in Phase 2 (Google's open-sourced
contrastive ITD framework) ingests three artifacts:

  - action.tfrecord    — one record per resource access (audit event)
  - context.tfrecord   — per-account snapshots at a fixed cadence,
                          carrying peer-graph edges
  - directive.textproto — config tying action types to featurization rules

We emit a JSONL intermediate of the same conceptual shape rather than
true TFRecord. This (a) keeps the benchmark free of a tensorflow runtime
dependency, (b) makes unit tests trivial, and (c) preserves all the
field information needed for a downstream converter to produce the
binary format at training time (`audit/facade_to_tfrecord.py` lives
separately and is the only place that needs the proto modules).

JSONL line shapes mirror the protos field-for-field:

  Action:
    type           : string ("email_send", "file_write", …)
    id             : hex string (event_id from AuditEvent)
    resource_id    : string
    account      : string
    occurred_at    : {seconds: int, nanos: int}
    history_key    : string
    access_type    : string ("read" | "write" | "share" | …)
    features       : {key: {value_string|value_int|value_double}}

  Context:
    type           : string ("org_directory")
    account      : string
    valid_from     : {seconds: int, nanos: int}
    peer_attributes: [{name: str, value: str, direction: str}]

Directive is emitted as plain-text proto (textproto) so a human or the
facade tooling can read it without any extra dependency.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import AuditEvent

logger = logging.getLogger(__name__)


# (service, action) → (action_type, access_type). Action types are
# the strings the directive's action_sources list — keep them stable.
_ACTION_KIND: dict[tuple[str, str], tuple[str, str]] = {
    ("email", "send_email"): ("email_send", "write"),
    ("email", "forward_email"): ("email_send", "write"),
    ("email", "read_inbox"): ("email_read", "read"),
    ("email", "find_emails"): ("email_read", "read"),
    ("owncloud", "write_file"): ("file_write", "write"),
    ("owncloud", "read_file"): ("file_read", "read"),
    ("owncloud", "share"): ("file_share", "share"),
    ("owncloud", "unshare"): ("file_share", "share"),
    ("owncloud", "update_share"): ("file_share", "share"),
    ("gitlab", "commit"): ("repo_write", "write"),
    ("gitlab", "push"): ("repo_write", "write"),
    ("gitlab", "read_file"): ("repo_read", "read"),
    ("gitlab", "open_mr"): ("repo_mr", "write"),
    ("gitlab", "merge"): ("repo_mr", "write"),
    ("rocketchat", "post_message"): ("chat_send", "write"),
    ("rocketchat", "send_direct_message"): ("chat_dm", "write"),
    ("rocketchat", "set_topic"): ("chat_admin", "write"),
    ("plane", "create_issue"): ("ticket_create", "write"),
    ("plane", "update_issue"): ("ticket_update", "write"),
    ("plane", "transition_issue"): ("ticket_update", "write"),
    ("plane", "add_comment"): ("ticket_comment", "write"),
    ("model_registry", "register_checkpoint"): ("registry_write", "write"),
    ("model_registry", "tag_approved"): ("registry_admin", "write"),
    ("model_registry", "download_weights"): ("registry_read", "read"),
    ("model_registry", "deploy"): ("registry_admin", "write"),
    ("secrets_store", "read"): ("secret_read", "read"),
    ("secrets_store", "set_access_policy"): ("secret_admin", "write"),
    ("eval_server", "submit_eval"): ("eval_submit", "write"),
    ("eval_server", "cancel_job"): ("eval_admin", "write"),
}


# All action types referenced above — kept distinct from the dict so the
# directive can iterate them without re-keying.
ACTION_TYPES = tuple(sorted({kind for kind, _ in _ACTION_KIND.values()}))


# ── public API ────────────────────────────────────────────────────────


def export_facade(
    events: Iterable[AuditEvent],
    out_dir: str | Path,
    *,
    org_yaml: str | Path | None = None,
    snapshot_period_hours: int = 2,
    model_version: int = 1,
) -> dict[str, Path]:
    """Project the audit stream + org into Action/Context/Directive artifacts.

    Returns a `{name: path}` map for every file written
    (always action + directive; context only if org_yaml is provided).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events_list = list(events)

    # ── Actions ─────────────────────────────────────────────────────
    action_path = out_dir / "action.jsonl"
    with action_path.open("w", encoding="utf-8") as f:
        for ev in events_list:
            action = _build_action(ev)
            if action is None:
                continue
            f.write(json.dumps(action, ensure_ascii=False) + "\n")
    written: dict[str, Path] = {"action": action_path}

    # ── Contexts ────────────────────────────────────────────────────
    if org_yaml is not None:
        ctx_path = out_dir / "context.jsonl"
        contexts = _build_contexts(
            events_list, Path(org_yaml),
            snapshot_period_hours=snapshot_period_hours,
        )
        with ctx_path.open("w", encoding="utf-8") as f:
            for ctx in contexts:
                f.write(json.dumps(ctx, ensure_ascii=False) + "\n")
        written["context"] = ctx_path

    # ── Directive (textproto) ───────────────────────────────────────
    directive_path = out_dir / "directive.textproto"
    directive_path.write_text(
        _build_directive(
            snapshot_period_hours=snapshot_period_hours,
            model_version=model_version,
        ),
        encoding="utf-8",
    )
    written["directive"] = directive_path

    logger.info(
        "facade_export: %d events → action.jsonl (%d records)%s + directive at %s",
        len(events_list),
        sum(1 for ev in events_list if _build_action(ev) is not None),
        (
            f" + context.jsonl ({sum(1 for _ in _build_contexts(events_list, Path(org_yaml), snapshot_period_hours=snapshot_period_hours))} records)"
            if org_yaml is not None else ""
        ),
        out_dir,
    )
    return written


# ── action builder ────────────────────────────────────────────────────


def _build_action(ev: AuditEvent) -> dict[str, Any] | None:
    kind = _ACTION_KIND.get((ev.service, ev.action))
    if kind is None:
        return None
    action_type, access_type = kind
    args = ev.args or {}
    return {
        "type": action_type,
        "id": ev.event_id,
        "resource_id": ev.resource_id or _resource_for(ev),
        "account": ev.account,
        "occurred_at": _to_ts(ev.ts),
        "history_key": f"{ev.account}|{action_type}",
        "access_type": access_type,
        "features": _build_features(ev, args),
    }


def _resource_for(ev: AuditEvent) -> str:
    args = ev.args or {}
    if ev.service == "secrets_store":
        return str(args.get("key", ""))
    if ev.service == "model_registry":
        return str(args.get("checkpoint_id", ""))
    if ev.service == "eval_server":
        return str(args.get("job_id", "") or args.get("eval_config", {}).get("model_id", ""))
    if ev.service == "rocketchat":
        return str(args.get("channel") or args.get("recipient") or "")
    if ev.service == "plane":
        return f"{args.get('project_id', '')}/{args.get('issue_id', '')}".rstrip("/")
    if ev.service == "gitlab":
        return str(args.get("project") or args.get("path") or "")
    if ev.service == "email":
        return ";".join(_as_list(args.get("to")))
    return ""


def _build_features(ev: AuditEvent, args: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A small fixed feature set per action. Keeps each line bounded."""
    feats: dict[str, dict[str, Any]] = {
        "exit_code": {"value_int": int(ev.exit_code)},
        "bytes": {"value_int": int(ev.bytes or 0)},
        "is_external": {"value_int": int(bool(ev.is_external))},
    }
    if ev.dest_domain:
        feats["dest_domain"] = {"value_string": str(ev.dest_domain)}
    if ev.service == "email":
        feats["to_count"] = {"value_int": len(_as_list(args.get("to")))}
        feats["cc_count"] = {"value_int": len(_as_list(args.get("cc")))}
        feats["attachment_count"] = {"value_int": len(args.get("attachments") or [])}
    if ev.service == "model_registry" and ev.action == "download_weights":
        feats["weight_bytes_log10"] = {
            "value_double": _safe_log10(args.get("weight_bytes", ev.bytes)),
        }
    return feats


# ── context builder ──────────────────────────────────────────────────


def _build_contexts(
    events: list[AuditEvent],
    org_yaml_path: Path,
    *,
    snapshot_period_hours: int,
) -> list[dict[str, Any]]:
    """Per-account snapshots at fixed cadence with peer_attributes.

    Snapshots are anchored to UTC midnight on the earliest event's date
    and emitted every `snapshot_period_hours` until the latest event's
    timestamp. Each account gets one snapshot per period.
    """
    if not org_yaml_path.exists():
        logger.warning("facade_export: org yaml not found at %s — skipping context",
                       org_yaml_path)
        return []
    with org_yaml_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    employees = {e["id"]: e for e in doc.get("accounts") or []}
    groups = doc.get("groups") or []
    departments = doc.get("departments") or []

    # Pre-compute peer attributes per account.
    peer_attrs_by_account: dict[str, list[dict[str, str]]] = {
        eid: _peer_attrs_for(eid, emp, groups, departments)
        for eid, emp in employees.items()
    }

    accounts = sorted({ev.account for ev in events
                         if ev.account in employees})
    if not accounts or not events:
        return []

    timestamps = [ _parse_iso(ev.ts) for ev in events if ev.ts ]
    if not timestamps:
        return []
    earliest = min(timestamps).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc,
    )
    latest = max(timestamps).replace(tzinfo=timezone.utc) if timestamps else earliest
    period = timedelta(hours=snapshot_period_hours)

    snapshots: list[dict[str, Any]] = []
    t = earliest
    while t <= latest:
        for p in accounts:
            snapshots.append({
                "type": "org_directory",
                "account": p,
                "valid_from": _to_ts_dt(t),
                "peer_attributes": peer_attrs_by_account[p],
            })
        t += period
    return snapshots


def _peer_attrs_for(
    eid: str, emp: dict[str, Any],
    groups: list[dict[str, Any]],
    departments: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Edges for FACADE's peer graph from {groups, teams, manager}."""
    attrs: list[dict[str, str]] = []
    for g in groups:
        if eid in (g.get("members") or []):
            attrs.append({
                "name": "group",
                "value": str(g.get("id", "")),
                "direction": "D_UNSPECIFIED",
            })
    for dept in departments:
        for team in dept.get("teams") or []:
            members = team.get("members") or []
            if eid in members:
                attrs.append({
                    "name": "team",
                    "value": str(team.get("id", "")),
                    "direction": "D_UNSPECIFIED",
                })
            mgr = team.get("manager")
            if mgr and mgr == eid:
                attrs.append({
                    "name": "manager",
                    "value": str(team.get("id", "")),
                    "direction": "D_FORWARD",            # this user manages the team
                })
            elif mgr and eid in members:
                attrs.append({
                    "name": "reports_to",
                    "value": str(mgr),
                    "direction": "D_BACKWARD",           # this user reports to mgr
                })
    return attrs


# ── directive builder ────────────────────────────────────────────────


def _build_directive(*, snapshot_period_hours: int, model_version: int) -> str:
    lines: list[str] = [
        f"model_version: {model_version}",
    ]
    for at in ACTION_TYPES:
        lines.append("action_sources {")
        lines.append(f'  action_type: "{at}"')
        lines.append('  format: ACTION_JSONL')
        lines.append('  path: "action.jsonl"')
        lines.append("}")
    lines.append("context_sources {")
    lines.append('  context_type: "org_directory"')
    lines.append('  format: CONTEXT_JSONL')
    lines.append('  path: "context.jsonl"')
    lines.append("}")
    lines.append("dataset_parameters {")
    lines.append("  snapshot_period {")
    lines.append(f"    seconds: {snapshot_period_hours * 3600}")
    lines.append("  }")
    lines.append("  snapshot_offset {")
    lines.append("    seconds: 0")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ── helpers ───────────────────────────────────────────────────────────


def _to_ts(iso_ts: str) -> dict[str, int]:
    if not iso_ts:
        return {"seconds": 0, "nanos": 0}
    dt = _parse_iso(iso_ts)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - epoch
    return {
        "seconds": int(delta.total_seconds()),
        "nanos": int(delta.microseconds * 1000),
    }


def _to_ts_dt(dt: datetime) -> dict[str, int]:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - epoch
    return {
        "seconds": int(delta.total_seconds()),
        "nanos": int(delta.microseconds * 1000),
    }


def _parse_iso(iso_ts: str) -> datetime:
    s = iso_ts.rstrip("Z") + ("+00:00" if iso_ts.endswith("Z") else "")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _safe_log10(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if x <= 0:
        return 0.0
    from math import log10
    return log10(x)
