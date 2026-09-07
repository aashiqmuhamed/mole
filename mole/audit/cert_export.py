"""CERT r4.2 CSV exporter.

The CMU CERT insider-threat dataset (r4.2) is the canonical benchmark for
classical ITD baselines (SVM/TCNN/DS-IID), and many published models
expect its specific CSV shape. This exporter projects our audit-event
stream into that shape so the same baselines train + evaluate on our
agentic data without modification.

Output is a directory containing up to five CSVs:

  logon.csv         — id, date, user, pc, activity
  email.csv         — id, date, user, pc, to, cc, bcc, from,
                       activity, size, attachments, content
  file.csv          — id, date, user, pc, filename, activity,
                       to_removable_media, from_removable_media, content
  http.csv          — id, date, user, pc, url, activity, content
  psychometric.csv  — employee_name, user_id, O, C, E, A, N

Mapping rules (audit service.action → CERT row):

  email.send_email                 → email.csv (activity="Send")
  email.read_inbox                 → email.csv per returned message
                                     (activity="View") — only if the
                                     returned payload metadata is in args
                                     (we don't depend on it; readers can
                                     also be derived from per-recipient
                                     events). Phase 1 emits Send rows only.
  email.forward_email              → email.csv (activity="Send",
                                     attachments inferred from args)
  owncloud.write_file              → file.csv (activity="Write")
  owncloud.read_file               → file.csv (activity="Open")
  owncloud.share                   → file.csv (activity="Copy") +
                                     http.csv if external
  gitlab.commit / push             → file.csv (activity="Write")
  rocketchat.post_message          → http.csv (url = chat://channel)
  rocketchat.send_direct_message   → http.csv (url = chat://dm/...)
  plane.create_issue / *           → http.csv (url = plane://project/issue)
  model_registry.* / secrets_store.read / eval_server.* → http.csv

Logon/logoff. CERT models the workday boundary explicitly. We synthesise
one Logon row per (account, date) at the account's earliest event
and one Logoff row at the latest, with synthetic PCs derived
deterministically from the account id.

Psychometric. CERT r4.2 includes per-employee OCEAN trait scores. We
read them from the same org_template.yaml that drives the org backend
(employees[].psychometric = {O, C, E, A, N}). If an account's record
lacks psychometric data, that user is skipped from psychometric.csv
(rather than fabricating values).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import AuditEvent

logger = logging.getLogger(__name__)


# Service.action → CERT csv name. None means "no CERT mapping; drop".
_SERVICE_ACTION_TO_CSV: dict[tuple[str, str], str] = {
    ("email", "send_email"): "email",
    ("email", "forward_email"): "email",
    # `read_inbox` and `find_emails` are reads of stored mail — CERT logs
    # message *receipts* not API reads. Drop unless the orchestrator
    # explicitly emits per-message receive events.
    ("owncloud", "write_file"): "file",
    ("owncloud", "read_file"): "file",
    ("owncloud", "share"): "file",
    ("owncloud", "unshare"): "file",
    ("owncloud", "update_share"): "file",
    ("gitlab", "commit"): "file",
    ("gitlab", "push"): "file",
    ("gitlab", "read_file"): "file",
    ("gitlab", "open_mr"): "http",
    ("gitlab", "merge"): "http",
    ("rocketchat", "post_message"): "http",
    ("rocketchat", "send_direct_message"): "http",
    ("rocketchat", "set_topic"): "http",
    ("plane", "create_issue"): "http",
    ("plane", "update_issue"): "http",
    ("plane", "transition_issue"): "http",
    ("plane", "add_comment"): "http",
    ("model_registry", "register_checkpoint"): "http",
    ("model_registry", "tag_approved"): "http",
    ("model_registry", "download_weights"): "http",
    ("model_registry", "deploy"): "http",
    ("secrets_store", "read"): "http",
    ("secrets_store", "set_access_policy"): "http",
    ("eval_server", "submit_eval"): "http",
    ("eval_server", "cancel_job"): "http",
}


# Activity strings CERT models recognise (keep verbatim).
_ACTIVITY_BY_ACTION: dict[tuple[str, str], str] = {
    ("email", "send_email"): "Send",
    ("email", "forward_email"): "Send",
    ("owncloud", "write_file"): "Write",
    ("owncloud", "read_file"): "Open",
    ("owncloud", "share"): "Copy",
    ("owncloud", "unshare"): "Copy",
    ("owncloud", "update_share"): "Copy",
    ("gitlab", "commit"): "Write",
    ("gitlab", "push"): "Write",
    ("gitlab", "read_file"): "Open",
    ("gitlab", "open_mr"): "View",
    ("gitlab", "merge"): "View",
    ("rocketchat", "post_message"): "Visit",
    ("rocketchat", "send_direct_message"): "Visit",
    ("rocketchat", "set_topic"): "Visit",
    ("plane", "create_issue"): "Visit",
    ("plane", "update_issue"): "Visit",
    ("plane", "transition_issue"): "Visit",
    ("plane", "add_comment"): "Visit",
    ("model_registry", "register_checkpoint"): "Visit",
    ("model_registry", "tag_approved"): "Visit",
    ("model_registry", "download_weights"): "Visit",
    ("model_registry", "deploy"): "Visit",
    ("secrets_store", "read"): "Visit",
    ("secrets_store", "set_access_policy"): "Visit",
    ("eval_server", "submit_eval"): "Visit",
    ("eval_server", "cancel_job"): "Visit",
}

_EMAIL_FIELDS = (
    "id", "date", "user", "pc",
    "to", "cc", "bcc", "from", "activity", "size", "attachments", "content",
)
_FILE_FIELDS = (
    "id", "date", "user", "pc", "filename", "activity",
    "to_removable_media", "from_removable_media", "content",
)
_HTTP_FIELDS = ("id", "date", "user", "pc", "url", "activity", "content")
_LOGON_FIELDS = ("id", "date", "user", "pc", "activity")
_PSYCH_FIELDS = ("employee_name", "user_id", "O", "C", "E", "A", "N")


# ── public API ────────────────────────────────────────────────────────


def export_cert(
    events: Iterable[AuditEvent],
    out_dir: str | Path,
    *,
    org_yaml: str | Path | None = None,
) -> dict[str, Path]:
    """Project the audit stream into a CERT r4.2 directory.

    Returns a {csv_name: path} dict for every CSV actually written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events_list = list(events)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counter = {"n": 0}

    def _next_id() -> str:
        counter["n"] += 1
        return f"{{C{counter['n']:08d}-A0A0-B1B1-C2C2-D3D3D3D3D3D3}}"

    for ev in events_list:
        csv_name = _SERVICE_ACTION_TO_CSV.get((ev.service, ev.action))
        if csv_name is None:
            continue
        if csv_name == "email":
            rows["email"].append(_email_row(ev, _next_id()))
        elif csv_name == "file":
            rows["file"].append(_file_row(ev, _next_id()))
        elif csv_name == "http":
            rows["http"].append(_http_row(ev, _next_id()))

    # Synthesise logon/logoff per (account, UTC date).
    logon_rows: list[dict[str, Any]] = []
    by_user_day: dict[tuple[str, str], list[AuditEvent]] = defaultdict(list)
    for ev in events_list:
        if not ev.ts:
            continue
        day = _date_of(ev.ts)
        by_user_day[(ev.account, day)].append(ev)
    for (account, day), evs in sorted(by_user_day.items()):
        evs.sort(key=lambda e: e.ts)
        logon_rows.append(_logon_row(account, evs[0].ts, _next_id(), "Logon"))
        logon_rows.append(_logon_row(account, evs[-1].ts, _next_id(), "Logoff"))

    written: dict[str, Path] = {}
    if logon_rows:
        written["logon"] = _write_csv(out_dir / "logon.csv", _LOGON_FIELDS, logon_rows)
    if rows["email"]:
        written["email"] = _write_csv(out_dir / "email.csv", _EMAIL_FIELDS, rows["email"])
    if rows["file"]:
        written["file"] = _write_csv(out_dir / "file.csv", _FILE_FIELDS, rows["file"])
    if rows["http"]:
        written["http"] = _write_csv(out_dir / "http.csv", _HTTP_FIELDS, rows["http"])

    # Psychometric is org-template-driven; only emit if data is available.
    if org_yaml is not None:
        psych_rows = _psychometric_rows(Path(org_yaml))
        if psych_rows:
            written["psychometric"] = _write_csv(
                out_dir / "psychometric.csv", _PSYCH_FIELDS, psych_rows,
            )

    logger.info(
        "cert_export: %d events → %d csvs in %s",
        len(events_list), len(written), out_dir,
    )
    return written


def load_events_from_jsonl(path: str | Path) -> list[AuditEvent]:
    """Re-hydrate AuditEvent objects from an audit.jsonl file."""
    events: list[AuditEvent] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # from_row drops any keys that aren't fields (forward-compatible).
            ev = AuditEvent.from_row(d)
            events.append(ev)
    return events


# ── row builders ──────────────────────────────────────────────────────


def _email_row(ev: AuditEvent, row_id: str) -> dict[str, Any]:
    args = ev.args or {}
    to = _as_str_list(args.get("to"))
    cc = _as_str_list(args.get("cc"))
    attachments = args.get("attachments") or []
    return {
        "id": row_id,
        "date": _cert_ts(ev.ts),
        "user": ev.account,
        "pc": _pc_for(ev.account),
        "to": ";".join(to),
        "cc": ";".join(cc),
        "bcc": "",
        "from": args.get("from_user", "") or ev.account,
        "activity": _ACTIVITY_BY_ACTION.get((ev.service, ev.action), "Send"),
        "size": int(ev.bytes or 0),
        "attachments": int(len(attachments)),
        "content": str(args.get("subject", "") or ""),
    }


def _file_row(ev: AuditEvent, row_id: str) -> dict[str, Any]:
    args = ev.args or {}
    filename = (
        args.get("path") or args.get("file") or args.get("filename")
        or ev.resource_id or ""
    )
    return {
        "id": row_id,
        "date": _cert_ts(ev.ts),
        "user": ev.account,
        "pc": _pc_for(ev.account),
        "filename": str(filename),
        "activity": _ACTIVITY_BY_ACTION.get((ev.service, ev.action), "Open"),
        "to_removable_media": False,
        "from_removable_media": False,
        # CERT 'content' is the file body sample; we emit a hash so monitors
        # can correlate without us shipping payload text.
        "content": ev.resource_hash[:16],
    }


def _http_row(ev: AuditEvent, row_id: str) -> dict[str, Any]:
    args = ev.args or {}
    url = _http_url_for(ev, args)
    content = str(args.get("text") or args.get("body") or args.get("subject") or "")
    return {
        "id": row_id,
        "date": _cert_ts(ev.ts),
        "user": ev.account,
        "pc": _pc_for(ev.account),
        "url": url,
        "activity": _ACTIVITY_BY_ACTION.get((ev.service, ev.action), "Visit"),
        "content": content[:200],         # keep CSV lines small
    }


def _logon_row(account: str, iso_ts: str, row_id: str, activity: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "date": _cert_ts(iso_ts),
        "user": account,
        "pc": _pc_for(account),
        "activity": activity,
    }


def _psychometric_rows(org_yaml_path: Path) -> list[dict[str, Any]]:
    if not org_yaml_path.exists():
        logger.warning("psychometric: org yaml not found at %s", org_yaml_path)
        return []
    with org_yaml_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    rows: list[dict[str, Any]] = []
    for emp in doc.get("accounts") or []:
        psy = emp.get("psychometric")
        if not psy:
            continue
        rows.append({
            "employee_name": emp.get("full_name", emp.get("id", "")),
            "user_id": emp.get("id", ""),
            "O": float(psy.get("O", 0.0)),
            "C": float(psy.get("C", 0.0)),
            "E": float(psy.get("E", 0.0)),
            "A": float(psy.get("A", 0.0)),
            "N": float(psy.get("N", 0.0)),
        })
    return rows


# ── helpers ───────────────────────────────────────────────────────────


def _http_url_for(ev: AuditEvent, args: dict[str, Any]) -> str:
    """Synthesise a stable URL string CERT readers can ingest."""
    svc = ev.service
    res = ev.resource_id or ""
    if svc == "rocketchat":
        if ev.action == "send_direct_message":
            peer = args.get("recipient", "")
            return f"chat://{ev.account}/dm/{peer}"
        return f"chat://channel/{args.get('channel', res) or 'unknown'}"
    if svc == "plane":
        proj = args.get("project_id", "")
        iss = args.get("issue_id", "")
        return f"plane://{proj}/{iss}" if iss else f"plane://{proj}"
    if svc == "gitlab":
        return f"gitlab://{args.get('project', res) or 'unknown'}"
    if svc == "model_registry":
        ck = args.get("checkpoint_id", res)
        return f"registry://{ev.action}/{ck}"
    if svc == "secrets_store":
        return f"secrets://{ev.action}/{args.get('key', res) or 'unknown'}"
    if svc == "eval_server":
        return f"eval://{ev.action}/{args.get('job_id', res) or ''}"
    return f"{svc}://{ev.action}/{res}"


def _pc_for(account: str) -> str:
    """Deterministic synthetic PC ID per account — matches CERT's PC-XXXX form."""
    h = hashlib.sha1(account.encode("utf-8")).hexdigest()[:4].upper()
    return f"PC-{h}"


def _cert_ts(iso_ts: str) -> str:
    """Convert AuditEvent ISO timestamp into CERT's 'MM/DD/YYYY HH:MM:SS' shape."""
    if not iso_ts:
        return ""
    s = iso_ts.rstrip("Z") + ("+00:00" if iso_ts.endswith("Z") else "")
    try:
        dt = datetime.fromisoformat(s).astimezone(timezone.utc)
    except ValueError:
        return iso_ts
    return dt.strftime("%m/%d/%Y %H:%M:%S")


def _date_of(iso_ts: str) -> str:
    if not iso_ts:
        return "1970-01-01"
    s = iso_ts.rstrip("Z") + ("+00:00" if iso_ts.endswith("Z") else "")
    try:
        dt = datetime.fromisoformat(s).astimezone(timezone.utc)
    except ValueError:
        return iso_ts[:10]
    return dt.strftime("%Y-%m-%d")


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path
