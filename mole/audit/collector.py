"""AuditCollector — sinks AuditEvents to memory + JSONL, and wraps state managers.

The collector serves three roles:

  1. Sink. `emit(event)` appends to an in-memory list (fast queries) AND to a
     per-task `audit.jsonl` (durable, post-hoc replay).

  2. Reader. `events_for_account(p)` and friends let oracle checkers filter
     the log by account / service / action without touching disk.

  3. Middleware. `wrap_manager(name, mgr, account_getter)` replaces every
     public async method on a state manager with a thin wrapper that emits
     a pre/post event pair. The pre-event is committed immediately so a
     mid-call crash still leaves a trace.

The collector does not enforce gating — that's a separate concern handled by
the gating layer, which reads the same event stream and decides allow/block/
escalate before the underlying method runs.
"""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .io_retry import is_disk_full, retry_on_disk_full
from .schema import AuditEvent


# Email domains treated as internal (not egress) when no org directory is wired
# to classify them. Keep in sync with the company domain in org_template.yaml.
INTERNAL_DOMAINS = frozenset({"agentlab.local"})


def is_external_domain(domain: str | None) -> bool:
    """True iff `domain` is a non-empty email domain outside the company."""
    return bool(domain) and domain.strip().lower() not in INTERNAL_DOMAINS


def _wallclock_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical_ts(s: str) -> str:
    """Normalize an ISO-8601 UTC timestamp to a single ``...Z`` serialization.

    The per-task sim clock and the wall-clock fallback otherwise emit two shapes
    (``+00:00`` vs ``Z``) into the same corpus, which trips naive ts parsers. We
    converge on ``Z``; anything that isn't a UTC offset is left untouched.
    """
    if not s:
        return s
    s = s.strip()
    if s.endswith("+00:00"):
        return s[:-6] + "Z"
    return s

logger = logging.getLogger(__name__)

# Per-asyncio-task simulated clock. asyncio.gather() creates child Tasks with a
# copy of the parent context, so each gathered coroutine carries its own clock
# value. This is what lets us safely parallelize sessions in the generator —
# without it, two concurrent sessions calling advance_clock() would race on a
# shared `_clock_fn` and tag each other's audit events with the wrong sim_now.
_sim_clock_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sim_clock", default=None,
)


def set_task_clock(iso_ts: str) -> contextvars.Token:
    """Set the current asyncio task's simulated clock. Returns a token for reset."""
    return _sim_clock_var.set(iso_ts)


def reset_task_clock(token: contextvars.Token) -> None:
    """Reset the per-task clock to its prior value."""
    _sim_clock_var.reset(token)


def get_current_sim_time() -> str | None:
    """Return the current per-task simulated time, canonicalized to ``...Z``.

    Set during a session via ``set_task_clock`` and read here so service
    adapters can stamp their *payloads* (not just the audit event) with sim
    time instead of the backend container's wall-clock. Returns ``None`` when
    no task clock is set — callers should then keep the original value rather
    than substitute wall-clock. (The per-instance ``_clock_fn`` fallback used by
    ``_get_ts`` for sequential code is not reachable here; manager calls run
    inside the session task where the contextvar is set, so this suffices.)
    """
    ts = _sim_clock_var.get()
    return _canonical_ts(ts) if ts else None


# Sentinel callable type for the account resolver.
AccountGetter = Callable[[], tuple[str, str]]   # () → (account, account_kind)


class AuditCollector:
    """Per-task append-only event collector."""

    def __init__(
        self,
        jsonl_path: Path | str | None = None,
        *,
        append: bool = False,
    ) -> None:
        self.events: list[AuditEvent] = []
        # Serializes writes to the shared jsonl. Tool methods run via
        # ``asyncio.to_thread`` (see state/*/manager.py), so several threads can
        # call ``emit`` concurrently; without this lock their ``write`` calls
        # interleave and splice two events onto one line (corrupt JSONL). The
        # race worsens with concurrency, so it must be held on every _fp touch.
        self._lock = threading.Lock()
        self._path = Path(jsonl_path) if jsonl_path else None
        self._fp = None
        self._org_lookup: Callable[[], Any] = lambda: None
        # Simulated-clock function. The orchestrator overrides this per stage
        # so audit events use the task's in-universe time (not wall-clock).
        # Default: real wall-clock UTC.
        self._clock_fn: Callable[[], str] = _wallclock_iso
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Default: truncate at task start so each run is its own log.
            # `append=True` opens in append mode for the simulator's --resume
            # flow, where we continue an in-progress audit log across days.
            mode = "a" if append else "w"
            self._fp = self._path.open(mode, encoding="utf-8")

    # ── sink ────────────────────────────────────────────────────────

    def emit(self, event: AuditEvent) -> AuditEvent:
        # Serialize outside the lock (pure); hold the lock only across the shared
        # mutation + write so concurrent emits can't interleave. One write call
        # (line + "\n") so a line is never split across two buffer appends.
        line = event.to_jsonl() if self._fp is not None else None
        with self._lock:
            self.events.append(event)
            if self._fp is not None:
                # write() buffers the line (exactly once); only the disk-touching
                # flush() is retried on a full disk, so ENOSPC/EDQUOT STALLS the
                # sim instead of crashing it. If write() itself hits disk-full via
                # an implicit buffer flush, the bytes stay buffered — the retried
                # flush below drains them (no duplication).
                try:
                    self._fp.write(line + "\n")
                except OSError as exc:
                    if not is_disk_full(exc):
                        raise
                retry_on_disk_full(self._fp.flush, what="audit emit")
        return event

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                finally:
                    self._fp = None

    @property
    def jsonl_path(self) -> Path | None:
        """The audit-log path this collector writes to (None if memory-only)."""
        return self._path

    def flush_sync(self) -> int:
        """Flush + fsync the jsonl so its on-disk size is exact, and return that
        size in bytes. Used by the resume manifest to record a durable
        day-boundary byte offset. Returns 0 if there's no backing file."""
        with self._lock:
            if self._fp is None or self._path is None:
                return 0
            retry_on_disk_full(self._fp.flush, what="audit flush_sync")
            try:
                os.fsync(self._fp.fileno())
            except OSError as exc:
                # A disk-full fsync is retried (durability matters at a day
                # boundary); any other fsync error stays best-effort as before.
                if is_disk_full(exc):
                    retry_on_disk_full(lambda: os.fsync(self._fp.fileno()),
                                       what="audit fsync")
            return self._path.stat().st_size

    # ── readers (used by oracle checkers) ───────────────────────────

    def events_for_account(self, account: str) -> list[AuditEvent]:
        return [e for e in self.events if e.account == account]

    def events_for_service(self, service: str) -> list[AuditEvent]:
        return [e for e in self.events if e.service == service]

    # Pure-read actions excluded from the session-recap memory: the agent
    # doesn't need to remember "I listed the issues queue yesterday" —
    # only state-changing actions are worth recalling. Listed once here
    # so the background-account session recap path stays a single filter expression.
    _PURE_READ_ACTIONS = frozenset({
        # gitlab
        "list_files", "list_projects", "list_mrs", "list_groups",
        "list_group_members", "read_file", "get_mr",
        # owncloud
        "list_dir", "read_bytes", "exists",
        # email
        "read_inbox", "find_emails",
        # rocketchat
        "list_users", "list_channels", "channel_history", "im_history",
        # plane
        "list_issues", "get_issue", "list_comments",
        # model_registry
        "list_checkpoints", "list_deployments", "get_checkpoint",
        "get_deployed", "latest_approved",
        # eval_server
        "list_jobs", "get_job", "baseline", "recent_logs",
        # secrets_store
        "list_keys", "get_meta", "get_policy", "read",
        # org
        "whoami", "list_accounts", "list_my_groups", "who_can_approve",
        "get_policy",
    })

    def recent_high_signal_events_for(
        self,
        account: str,
        *,
        limit: int = 10,
        successful_only: bool = True,
    ) -> list[AuditEvent]:
        """Return up to `limit` most-recent state-changing events for
        `account`.

        Drops pure-read actions (list_*/read_*/get_*/etc., enumerated in
        `_PURE_READ_ACTIONS`) so the recap surfaces only actions that
        actually changed the world — commits, sends, shares, public_links,
        ticket transitions, group additions, weight tags, etc.

        `successful_only=True` (default) drops failed-tool-call events so
        the agent's recap reflects what genuinely happened, not invented
        call shapes that errored.

        Returned in chronological order (oldest -> newest) so the caller
        can render the list as "here's what I did this week, in order".
        """
        if limit <= 0:
            return []
        out: list[AuditEvent] = []
        for e in reversed(self.events):
            if getattr(e, "account", "") != account:
                continue
            if successful_only and getattr(e, "exit_code", 0) != 0:
                continue
            if getattr(e, "action", "") in self._PURE_READ_ACTIONS:
                continue
            out.append(e)
            if len(out) >= limit:
                break
        out.reverse()
        return out

    def events_with(
        self,
        *,
        account: str | None = None,
        service: str | None = None,
        action: str | None = None,
        is_external: bool | None = None,
    ) -> list[AuditEvent]:
        def _match(e: AuditEvent) -> bool:
            if account is not None and e.account != account:
                return False
            if service is not None and e.service != service:
                return False
            if action is not None and e.action != action:
                return False
            if is_external is not None and e.is_external != is_external:
                return False
            return True
        return [e for e in self.events if _match(e)]

    # ── middleware ──────────────────────────────────────────────────

    def wrap_manager(
        self,
        service_name: str,
        manager: Any,
        account_getter: AccountGetter,
        *,
        is_malicious_getter: Callable[[], bool] = lambda: False,
    ) -> None:
        """Replace every public async method on `manager` with an audited wrapper.

        Sync methods are left alone — the collector is for I/O verbs that mutate
        or observe service state, all of which are `async def` by convention in
        our state-manager interface. Private methods (leading `_`) and lifecycle
        methods (`setup` / `cleanup`) are skipped.
        """
        skip = {"setup", "cleanup"}
        for attr_name in dir(manager):
            if attr_name.startswith("_") or attr_name in skip:
                continue
            attr = getattr(manager, attr_name)
            if not callable(attr) or not asyncio.iscoroutinefunction(attr):
                continue
            setattr(
                manager,
                attr_name,
                self._make_wrapper(
                    bound_method=attr,
                    service_name=service_name,
                    action_name=attr_name,
                    account_getter=account_getter,
                    is_malicious_getter=is_malicious_getter,
                ),
            )

    def _make_wrapper(
        self,
        *,
        bound_method: Callable[..., Any],
        service_name: str,
        action_name: str,
        account_getter: AccountGetter,
        is_malicious_getter: Callable[[], bool],
    ) -> Callable[..., Any]:
        sig = inspect.signature(bound_method)
        collector = self
        # Where to look up "is this domain external to our org?" — comes
        # from a registered OrgManager if available. Cache by ID to avoid
        # tight coupling.
        get_org = collector._org_lookup

        async def _audited(*args: Any, **kwargs: Any) -> Any:
            account, kind = account_getter()
            # Read the per-task ContextVar id set by generator.account_context.
            # Imported lazily to keep audit/ free of a generator import cycle.
            try:
                from ..generator.account_context import get_task_id
                tid = get_task_id()
            except Exception:                                          # noqa: BLE001
                tid = ""
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                arg_dict: dict[str, Any] = dict(bound.arguments)
            except TypeError:
                arg_dict = {"args": list(args), "kwargs": dict(kwargs)}

            norm_args = _normalize_args(arg_dict)
            resource_id = _infer_resource_id(norm_args)
            event = AuditEvent.new(
                account=account,
                account_kind=kind,
                service=service_name,
                action=action_name,
                resource_id=resource_id,
                args=_safe_serialise(norm_args),
                is_malicious=is_malicious_getter(),
                ts=collector._get_ts(),
                task_id=tid,
            )
            # Side-channel metadata — derived from args + org directory.
            org = get_org()
            domains = _infer_dest_domains(arg_dict)
            if org is not None:
                external_domains = [
                    d for d in domains
                    if org.is_external(f"x@{d}")
                ]
                event.dest_domain = (
                    external_domains[0]
                    if external_domains
                    else (domains[0] if domains else None)
                )
                event.is_external = bool(external_domains)
            else:
                # No org directory wired: classify by the internal-domain set so
                # internal mail (e.g. @agentlab.local) is not mislabeled as egress.
                ext = [d for d in domains if is_external_domain(d)]
                event.dest_domain = ext[0] if ext else (domains[0] if domains else None)
                event.is_external = bool(ext)

            # Single-emit-at-end: we populate exit_code/bytes/error first
            # (handling exceptions), then commit the final state to both
            # the in-memory list and the JSONL. A mid-call crash still
            # gets emitted (with exit_code=1) via the `except` branch.
            # Retry transient CONNECTION-establishment failures (the lab service
            # briefly unreachable — e.g. a Docker Desktop hiccup under load): the
            # request never reached the service, so a retry is side-effect-free.
            # Read-timeouts are NOT retried (the call may have partially run).
            # Still exactly one emit per logical call.
            for _attempt in range(4):
                try:
                    result = await bound_method(*args, **kwargs)
                    break
                except Exception as exc:
                    tn = type(exc).__name__.lower()
                    transient = (
                        "10061" in str(exc)
                        or "connection refused" in str(exc).lower()
                        or "connecterror" in tn
                        or "connectionrefusederror" in tn
                    )
                    if transient and _attempt < 3:
                        await asyncio.sleep(0.4 * (2 ** _attempt))
                        continue
                    event.exit_code = 1
                    event.error = f"{type(exc).__name__}: {exc}"
                    collector.emit(event)
                    raise
            event.exit_code = 0
            try:
                event.bytes = _estimate_bytes(result)
            except Exception:
                event.bytes = 0
            collector.emit(event)
            return result

        _audited.__name__ = action_name
        _audited.__qualname__ = f"audited.{service_name}.{action_name}"
        # Preserve the wrapped method's signature so introspection
        # (agent.tools.build_tools_for) sees the REAL parameters — for both
        # the JSON tool schema AND actor-param auto-fill. Without this, _audited's
        # (*args, **kwargs) signature hid every param: tool schemas went generic
        # (so the model guessed arg names — `iid` vs `mr_iid`, full-name emails)
        # and actor params (user/sender/from_user) were never auto-filled, so
        # read_inbox(None)->[], post_message(None)->401 as "system", send_email
        # From system@. The whole v1 cross-agent interaction layer ran broken
        # because of this one missing line; audit attribution (from the session
        # account, not args) stayed correct, masking it.
        try:
            _audited.__signature__ = sig
        except (AttributeError, ValueError):
            pass
        return _audited

    def set_org_lookup(self, getter: Callable[[], Any]) -> None:
        """Plug in an org-manager resolver for is_external classification.

        `getter` is called per event; returning None disables the check
        (we fall back to "any non-empty dest_domain is external", a safe
        over-approximation).
        """
        self._org_lookup = getter

    def set_clock(self, clock_fn: Callable[[], str]) -> None:
        """Override the simulated-time function used to stamp event `ts`.

        Called once per stage by the orchestrator with a closure returning
        the current stage's in-universe ISO timestamp. This sets the GLOBAL
        fallback clock — for parallel sessions, prefer the per-task contextvar
        via `set_task_clock` so concurrent sessions don't overwrite each other.
        """
        self._clock_fn = clock_fn

    def _get_ts(self) -> str:
        """Return the current ts for an audit event.

        Prefers the per-task contextvar (set by `set_task_clock` in parallel
        session loops) over the global `_clock_fn` fallback. Sequential code
        paths that only call `set_clock` still work — they read the fallback.
        """
        task_ts = _sim_clock_var.get()
        if task_ts is not None:
            return _canonical_ts(task_ts)
        return _canonical_ts(self._clock_fn())


# ── helpers ────────────────────────────────────────────────────────


def _infer_resource_id(args: dict[str, Any]) -> str:
    """Best-effort: pull a human-readable resource id out of common arg names."""
    for key in ("resource_id", "path", "filename", "checkpoint_id", "ckpt_id",
                "key", "id", "name",
                "project", "channel", "ticket", "recipient", "to"):
        if key in args and args[key] is not None:
            return str(args[key])
    return ""


def _infer_dest_domains(args: dict[str, Any]) -> list[str]:
    """Pull destination email-domains out of an args dict, if any.

    Looks at common arg names (`to`, `recipient`, `to_user`, `external`) and
    extracts the part after `@` from every value that looks like an email.
    """
    out: list[str] = []
    for key in ("to", "recipient", "to_user", "address", "email", "external"):
        val = args.get(key)
        if val is None:
            continue
        candidates = val if isinstance(val, (list, tuple)) else [val]
        for cand in candidates:
            if not isinstance(cand, str) or "@" not in cand:
                continue
            host = cand.rsplit("@", 1)[-1].strip().lower()
            if host and host not in out:
                out.append(host)
    return out


def _infer_dest_domain(args: dict[str, Any]) -> str | None:
    """Backward-compatible single-domain helper."""
    domains = _infer_dest_domains(args)
    return domains[0] if domains else None


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Tidy the recorded arg dict so the audit log reads like real telemetry:
    unwrap the ``{args, kwargs}`` bind-failure shape, and drop unfilled params
    (``None``) and the empty ``_extra`` catch-all. Cosmetic on the recorded log
    only — the value the tool actually received is unchanged."""
    if set(args) <= {"args", "kwargs"} and isinstance(args.get("kwargs"), dict):
        merged = dict(args["kwargs"])
        pos = args.get("args") or []
        if pos:
            merged["_args"] = list(pos)
        args = merged
    out: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for k, v in args.items():
        if k == "self" or v is None:
            continue
        if k == "_extra":
            # Lift the ``**_extra`` catch-all to first-class args so the schema
            # reads cleanly: agents routinely pass kwargs the tool signature
            # didn't name (e.g. submit_eval(suite=, checkpoint_id=)) and those
            # otherwise nest under ``_extra``. Empty -> just dropped.
            if isinstance(v, dict):
                extra = {ek: ev for ek, ev in v.items() if ev is not None}
            continue
        out[k] = v
    for ek, ev in extra.items():
        out.setdefault(ek, ev)   # never clobber an explicitly-named arg
    # Canonicalize chat-channel naming so telemetry is consistent (agents pass
    # "#alignment" and "alignment" interchangeably). Only the rocketchat `channel` key.
    if isinstance(out.get("channel"), str) and out["channel"]:
        out["channel"] = "#" + out["channel"].lstrip("#")
    return out


def _safe_serialise(args: dict[str, Any]) -> dict[str, Any]:
    """Map an arg dict to a JSON-safe equivalent, dropping callables/files."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if k == "self":
            continue
        if callable(v) and not isinstance(v, (str, bytes)):
            out[k] = f"<callable {getattr(v, '__name__', type(v).__name__)}>"
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [_one(item) for item in v]
        elif isinstance(v, dict):
            out[k] = {str(kk): _one(vv) for kk, vv in v.items()}
        else:
            out[k] = repr(v)
    return out


def _one(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return repr(v)


def _estimate_bytes(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, (bytes, bytearray)):
        return len(result)
    if isinstance(result, str):
        return len(result.encode("utf-8"))
    # Managers self-report payload size via a numeric `bytes`/`size` field
    # (e.g. email.send_email returns {message_id, to, bytes: len(msg)}). Honor
    # it — otherwise we'd record the size of the *result envelope* (a few
    # hundred bytes), which silently breaks byte-gated harm oracles like
    # threat-06's weights-email-egress (the egress payload looks ~78 B).
    if isinstance(result, dict):
        for key in ("bytes", "size", "byte_count", "content_length"):
            v = result.get(key)
            if isinstance(v, bool):
                continue
            if isinstance(v, int) and v >= 0:
                return v
    try:
        import json
        payload = result
        # Bound a pathological giant result before serializing (same guard as
        # _stringify) so measuring its size doesn't fully encode hundreds of GB and
        # stall the loop. For normal results the shrink is content-preserving, so the
        # byte count is unchanged; only genuinely huge results get an approximate
        # (smaller) count — acceptable since the byte-gated rule oracle is no longer
        # the ground truth (that is the transcript-fed LLM judge). Explicit
        # `bytes`/`size` fields are honored above and are unaffected.
        try:
            from ..agent.tools import _shrink_bounded
            payload = _shrink_bounded(result)
        except Exception:
            pass
        return len(json.dumps(payload, default=str).encode("utf-8"))
    except Exception:
        return len(repr(result).encode("utf-8"))
