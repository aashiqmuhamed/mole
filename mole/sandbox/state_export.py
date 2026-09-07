"""Light-weight in-process manager state export at day boundaries.

The 5 in-process managers (`org`, `model_registry`, `eval_server`,
`secrets_store`, `plane`) hold mutable Python state that docker snapshots
don't capture (they live in the simulator's Python process, not in
containers). This module exports each manager's relevant `_`-prefixed
state to JSON files at sim-day boundaries.

Scope (this task is the lighter Stage-2 of resume support):
  - EXPORT only. We do not wire deserialize/resume back into the managers.
  - Provides post-hoc auditability of end-of-day state.
  - Provides a starting point if v1 results justify building full Stage-2
    resume (deserialize on `--resume` to reconstruct world state).

Usage from generator/run.py at each day boundary:

    from ..sandbox.state_export import export_manager_state
    export_manager_state(ctx, out_dir / "managers.json")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Which `_`-prefixed attributes to export for each manager type, keyed by
# the manager's registry name (the key under composite.managers). Keeping
# this whitelist makes the export robust to managers that grow internal
# caches we don't care about (e.g., a cached http client).
_STATE_FIELDS: dict[str, tuple[str, ...]] = {
    "org": ("_doc", "_accounts_by_id", "_groups_by_id", "_policies_by_action"),
    "model_registry": ("_checkpoints", "_deployments", "_required_approvers"),
    "eval_server": ("_baseline", "_scoring", "_jobs", "_deploy_log"),
    "secrets_store": ("_secrets",),
    "plane": ("_workspace_slug", "_state", "_next_issue_seq"),
}


def export_manager_state(ctx: Any, out_path: Path) -> dict[str, Any]:
    """Serialize the 5 in-process managers' state to JSON at `out_path`.

    Returns the exported dict (also written to disk). Unknown managers and
    missing fields are skipped silently — the export is best-effort and
    must not abort a sim that's already running.
    """
    managers = getattr(ctx, "_managers", None) or {}
    out: dict[str, Any] = {}
    for name, fields in _STATE_FIELDS.items():
        mgr = managers.get(name)
        if mgr is None:
            continue
        snapshot: dict[str, Any] = {}
        for field in fields:
            if not hasattr(mgr, field):
                continue
            try:
                value = getattr(mgr, field)
                # Round-trip through json to validate serializability and
                # produce a clean copy. Non-serializable values are skipped
                # (the helper would raise; we log and continue).
                snapshot[field] = json.loads(json.dumps(value, default=str))
            except (TypeError, ValueError) as exc:
                logger.warning("state_export: %s.%s not JSON-serializable: %s",
                               name, field, exc)
        if snapshot:
            out[name] = snapshot

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def import_manager_state(ctx: Any, in_path: Path) -> dict[str, list[str]]:
    """Re-apply a `managers.json` dump onto the live in-process managers.

    Symmetric inverse of `export_manager_state`, used by `--resume` to restore
    agent-mutated in-process state (model_registry / secrets_store / eval_server
    / plane / org) that re-seeding from YAML would otherwise reset to day-0.
    `setattr`s each whitelisted `_`-field back onto its manager.

    Why this is needed even with a "live" sandbox: the 5 managers live in the
    SIMULATOR's Python process, not in containers. Any python restart loses them
    regardless of whether the docker lab survived — so a correct resume must
    replay this dump.

    Best-effort: a missing file, unknown manager, or absent field is skipped.
    Returns `{manager_name: [restored_fields]}` for logging/verification.

    Caveat: values the export coerced via `json.dumps(default=str)` (e.g. a
    datetime) come back as strings. Every whitelisted field is a plain
    dict/list/str/int container today, so this is lossless for the current
    managers; revisit if a manager grows a non-JSON field.
    """
    in_path = Path(in_path)
    if not in_path.exists():
        logger.warning("import_manager_state: %s missing; nothing to restore", in_path)
        return {}
    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("import_manager_state: %s unreadable (%s); skipping", in_path, exc)
        return {}

    managers = getattr(ctx, "_managers", None) or {}
    applied: dict[str, list[str]] = {}
    for name, fields in _STATE_FIELDS.items():
        snapshot = data.get(name)
        mgr = managers.get(name)
        if not snapshot or mgr is None:
            continue
        restored: list[str] = []
        for field in fields:
            if field in snapshot and hasattr(mgr, field):
                setattr(mgr, field, snapshot[field])
                restored.append(field)
        if restored:
            applied[name] = restored
            logger.info("import_manager_state: restored %s.{%s}",
                        name, ",".join(restored))
    return applied
