"""Audit-event schema — one record per state-manager method invocation.

Every event emitted by the collector lands in two places:
  - `audit.jsonl`: the per-task append-only event stream (level 2 source of truth)
  - in-memory `events` list on the AuditCollector (for fast oracle lookups)

The schema is a strict superset of CERT r4.2 fields and FACADE's per-action
shape, so post-processors can produce both views without losing information.
level 0 / level 1 / level 2 projections (see audit/projections.py) are functions
over this base record; the projection layer decides what semantic content
the monitor can see, not the collector.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _stable_hash(value: Any) -> str:
    """SHA-256 of a stable JSON encoding (sorted keys, no whitespace)."""
    try:
        encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        encoded = repr(value)
    return _sha256_hex(encoded.encode("utf-8"))


@dataclass
class AuditEvent:
    """One service-side event. Roughly: a tool call with side-effects."""

    # Identity + timing
    event_id: str
    ts: str                          # ISO-8601 UTC; deterministic order
    real_ts: float                   # monotonic perf counter for latency math

    # Actor + verb
    account: str                   # e.g. "alice.kim", "agent", "julian.x"
    account_kind: str              # "human" | "agent" | "background_llm_agent" | "background_rules_agent" | "system"
    service: str                     # "gitlab" | "owncloud" | "email" | "org" | ...
    action: str                      # method name on the state-manager

    # Target + payload
    resource_id: str = ""            # human-readable identifier (project/file/key)
    resource_hash: str = ""          # sha256(resource_id) — level 0 view shows this only
    args: dict[str, Any] = field(default_factory=dict)
    args_hash: str = ""              # sha256 of args (canonicalised)

    # Result
    exit_code: int = 0               # 0 = success, nonzero = error
    bytes: int = 0                   # response payload size (rough)
    error: str = ""                  # short error message if exit_code != 0

    # Cross-channel signals (relevant for email / chat / external transfers)
    dest_domain: str | None = None
    is_external: bool = False

    # Causality
    parent_event_id: str | None = None
    gate_decision: str | None = None  # "allow" | "block" | "escalate" (set by gating layer)

    # Ground-truth label injected by the orchestrator (Cell-3/4 vs Cell-1/2 runs)
    is_malicious: bool = False

    # Per-session task id (set by `account_context.set_account` at session
    # start, propagated through the asyncio Context). Lets the attribution
    # audit pair every event with the session that issued it without args-
    # fuzzy heuristics — 100% coverage vs the ~14% we get from args matching.
    task_id: str = ""

    @classmethod
    def new(
        cls,
        *,
        account: str,
        account_kind: str,
        service: str,
        action: str,
        resource_id: str = "",
        args: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
        is_malicious: bool = False,
        ts: str | None = None,
        task_id: str = "",
    ) -> "AuditEvent":
        """Construct a fresh event.

        `ts` is the simulated wall-clock timestamp the event should be tagged
        with (ISO-8601). If None, falls back to real wall-clock time. The
        AuditCollector sets `ts` via its clock_fn so per-stage simulated
        time advances are honored by the rollup features that depend on
        time-of-day (e.g. after_hours_rate).
        """
        args = args or {}
        return cls(
            event_id=uuid4().hex,
            ts=ts if ts is not None else _now_iso(),
            real_ts=time.perf_counter(),
            account=account,
            account_kind=account_kind,
            service=service,
            action=action,
            resource_id=resource_id,
            resource_hash=_sha256_hex(resource_id.encode("utf-8")) if resource_id else "",
            args=args,
            args_hash=_stable_hash(args),
            parent_event_id=parent_event_id,
            is_malicious=is_malicious,
            task_id=task_id,
        )

    @classmethod
    def from_row(cls, d: dict[str, Any]) -> "AuditEvent":
        """Build from a persisted JSONL row, ignoring any keys that are not
        fields (forward-compatible with extra columns)."""
        from dataclasses import fields as _fields
        names = {f.name for f in _fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)
