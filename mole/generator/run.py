"""Generator runner.

    python -m mole.generator.run --personas 3 --days 5 --out npc_audit.jsonl

Produces a benign-only audit log: N personas act over D simulated days
against the in-process data managers (org / model_registry / eval_server
/ secrets_store), with every call captured by the AuditCollector and
attributed to the acting background account (account_kind="background_rules_agent"). Deterministic for a
given --seed, so FACADE pretraining data is reproducible.

No containers are required — the data managers self-load from the
bootstrap YAML fixtures. This is the benign baseline Phase-2 monitors
train on (real agent traces from threat runs are the contrast).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from ..audit.collector import AuditCollector
from ..audit.io_retry import is_disk_full, retry_on_disk_full
from ..sandbox.base import DryRunSandbox
from ..state.composite import CompositeStateManager
from . import resume_state
from .member import Member, NPCConfig
from .persona_loader import load_personas


class AuthCircuitBreakerError(RuntimeError):
    """Raised to halt a run after too many consecutive auth failures.

    Guards against the failure mode that silently destroyed an overnight kimi
    corpus: the token daemon died, every subsequent session got a 401, but the
    per-session `except` swallowed it and the sim marched on, accumulating
    thousands of empty auth-failed sessions. We'd rather stop and resume than
    burn the run writing garbage.
    """


# ── session-level (bucket-granular) resume: day-plan (de)serialization ──────
# For --session-resume, the interrupted day's plan is persisted so a warm resume
# reproduces the EXACT slots + attack schedule without re-running plan_day (which,
# on a fresh process whose per-member RNG is back at its day-0 state, would produce
# a different schedule). Statistically-equivalent fidelity: the schedule is pinned;
# within-session RNG draws for the re-run buckets are fresh (accepted).


def _serialize_day_plan(day_plan: list[tuple[Any, Any, str]]) -> list[list[str]]:
    """[(slot_datetime, member, key)] -> JSON-safe [[iso_ts, persona_id, key], ...]."""
    return [[t.isoformat(), m.persona.id, key] for (t, m, key) in day_plan]


def _session_index(key: str) -> int:
    """Slot index i from a "session_<i>" key. plan_day emits _today_keys in this
    order and the attack-slot pick indexes into it, so restore must match it."""
    try:
        return int(key.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 0


def _restore_day_plan(saved: list[list[str]], members: list[Any]) -> list[tuple[Any, Any, str]]:
    """Rebuild [(slot_datetime, member, key)] from a serialized plan and re-set each
    member's `_today_keys` (which plan_day would have set) so the deterministic
    attack-slot selection matches the pre-crash run. Members absent from the plan get
    an empty `_today_keys` (PTO / fully loafed that day)."""
    by_id = {getattr(m, "persona", m).id: m for m in members}
    plan: list[tuple[Any, Any, str]] = []
    keys_by_member: dict[int, list[str]] = {}
    for t_iso, member_id, key in saved:
        m = by_id.get(member_id)
        if m is None:
            continue                       # roster changed — shouldn't happen on resume
        plan.append((datetime.fromisoformat(t_iso), m, key))
        keys_by_member.setdefault(id(m), []).append(key)
    plan.sort(key=lambda x: x[0])
    for m in members:
        keys = keys_by_member.get(id(m), [])
        m._today_keys = sorted(keys, key=_session_index)   # plan_day order (by slot index)
    return plan


def _is_auth_error(exc: BaseException) -> bool:
    """True if `exc` (or a cause in its chain) looks like an auth/credential
    failure — an expired/absent token, a 401/403, or a rejected credential —
    as opposed to a model/tool/transport error worth continuing past."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = type(cur).__name__
        # NB: a per-session complete() 403 (PermissionDeniedError) is normally
        # absorbed by the retry layer now — for some providers a 403 is a *flapping* grant,
        # so retry.py treats it as transient and waits it out instead of letting
        # it surface here. This breaker arm is the backstop for 403s that bypass
        # that path (capped/disabled retry, or auth failures outside complete()):
        # halting beats accumulating empty sessions. 401 still fails fast — a dead
        # token needs a re-mint, not a wait.
        if name in ("AuthenticationError", "PermissionDeniedError"):
            return True
        if getattr(cur, "status_code", None) in (401, 403):
            return True
        msg = str(cur)
        if any(tok in msg for tok in (
            "invalid_token", "invalid_api_key", "Access token has expired",
            "401 Unauthorized", "Unauthorized",
        )):
            return True
        cur = cur.__cause__ or cur.__context__
    return False

logger = logging.getLogger(__name__)

# Rule-based workflows only touch in-process data managers (fast, no containers).
NPC_ENVS: tuple[str, ...] = ("org", "model_registry", "eval_server", "secrets_store")
# Agentic background accounts exercise the full service surface (like the agent), so they
# need the container-backed managers too — same env set the threats use.
# Override with $NPC_AGENTIC_ENVS (comma-separated) to run against a reduced set —
# e.g. dropping gitlab/rocketchat for a fast validation lab. Must match the
# services present in $LAB_COMPOSE (a manager whose container is absent fails setup).
_DEFAULT_AGENTIC_ENVS: tuple[str, ...] = (
    "org", "gitlab", "owncloud", "email", "rocketchat",
    "plane", "model_registry", "eval_server", "secrets_store",
)
AGENTIC_ENVS: tuple[str, ...] = tuple(
    e.strip() for e in os.environ["NPC_AGENTIC_ENVS"].split(",") if e.strip()
) if os.environ.get("NPC_AGENTIC_ENVS") else _DEFAULT_AGENTIC_ENVS


async def build_context(
    collector: AuditCollector,
    *,
    tmp_dir: Path,
    envs: tuple[str, ...] = NPC_ENVS,
    sandbox: Any | None = None,
    seed_world: bool = False,
) -> Any:
    """Build the canonical TaskContext (same path the orchestrator + generator tests
    use): audit-wrapped managers + a mutable account box the Member/AgenticMember
    writes before each session so events attribute to the background account. With `sandbox=None`
    a DryRunSandbox is used (in-process); pass a LabSandbox for full-service
    agentic runs. `seed_world=True` runs seed_org so agentic background accounts act on a
    populated world."""
    composite = CompositeStateManager(environments=list(envs))
    sandbox = sandbox or DryRunSandbox()
    await composite.setup(sandbox=sandbox)

    if seed_world:
        from types import SimpleNamespace
        from ..seeders.seed_company import seed_company
        try:
            # Seed the ONE shared company world (org + every threat's resources) so
            # benign background accounts act on the same repos/files the attacks use — not an isolated
            # org-only world where they'd invent fictional names.
            counts = await seed_company(SimpleNamespace(**composite.managers))
            logger.info("generator build_context: seeded shared company world: %s", counts)
        except Exception:                                       # noqa: BLE001
            logger.exception("generator build_context: seed_company failed; continuing")

    # Per-task account via ContextVar (see generator/account_context.py) — the
    # earlier mutable-dict box race-corrupted attribution under concurrency.
    from .account_context import get_account
    for svc_name, mgr in composite.managers.items():
        collector.wrap_manager(
            service_name=svc_name,
            manager=mgr,
            account_getter=get_account,
        )

    # Wire the org directory so is_external is classified by the company domain
    # (mirrors orchestrator.py). Without it the collector uses its internal-domain
    # fallback; either way internal mail is no longer mislabeled as egress.
    collector.set_org_lookup(lambda: composite.managers.get("org"))

    ctx = composite.create_context(
        task_dir=tmp_dir,
        sandbox=sandbox,
        audit=collector,
        sim_start="2026-04-06T00:00:00Z",
    )
    # Backward-compat shim — older code paths read `ctx._account_box["account"]`
    # directly. Expose a get-only proxy that resolves to the contextvar value;
    # writes are dropped (the canonical write path is set_account()).
    class _AccountProxy:
        def get(self, k, default=None):
            p, kind = get_account()
            return {"account": p, "kind": kind}.get(k, default)
        def __getitem__(self, k):
            return self.get(k)
        def __setitem__(self, k, v):
            # Quiet no-op; callers should use set_account() directly.
            pass
    ctx._account_box = _AccountProxy()
    ctx.collector = collector
    return ctx


def _weekday_dates(start: date, n_days: int) -> list[str]:
    """`n_days` consecutive working days (skip Sat/Sun) from `start`."""
    out: list[str] = []
    d = start
    while len(out) < n_days:
        if d.weekday() < 5:                # Mon–Fri
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _last_completed_day(audit_jsonl: Path) -> str | None:
    """Read an existing audit jsonl and return the max date prefix (YYYY-MM-DD)
    found in any event's `ts`, or None if the file is empty / missing.

    Used by --resume to skip days already in the corpus. We treat any event on
    day D as evidence that D was "in progress"; the conservative choice is to
    restart from day D+1, accepting that the very last partial day may be
    re-run. Combined with the day-boundary snapshot logic, this gives a
    coarse-grained resume: at most one re-run day, no day permanently lost.
    """
    if not audit_jsonl.exists():
        return None
    last_date: str | None = None
    with audit_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("ts", "")
            if len(ts) >= 10:
                day = ts[:10]
                if last_date is None or day > last_date:
                    last_date = day
    return last_date


async def simulate(
    *,
    n_personas: int,
    n_days: int,
    out_path: str | Path,
    start_date: str = "2026-04-06",
    seed: int | None = 0,
    sessions_per_day: int = 8,        # generation default; ~8 × 0.95 loaf ≈ 7.6/day
    config: NPCConfig | None = None,
    mode: str = "rules",
    llm: Any | None = None,
    include_holdouts: bool = False,
    concurrency: int = 1,
    bucket_minutes: int = 30,
    resume: bool = False,
    restore_world: bool = False,
    session_resume: bool = False,
    session_id: str | None = None,
    insider_persona_map: dict[str, str] | None = None,
    insider_threat_map: dict[str, str] | None = None,
    insider_role_map: dict[str, str] | None = None,
    max_turns: int = 30,
    elicit: bool = True,
    allowed_persona_ids: set[str] | None = None,
    attack_session_rate: float = 0.20,
    multi_day: bool = False,
    benign_arcs: bool = False,
    planning: str | None = None,
    campaign_days: int = 4,   # K; calibrated to the v1 ar05 base rate (~3.2%)
    auth_failure_limit: int = 25,
    absent_probability: float = 0.01,
) -> AuditCollector:
    """Generate a benign background-account audit corpus.

    mode="agentic" (the standard baseline): each background-account session is driven by an LLM
    persona-agent over the full service surface (needs a lab sandbox + llm), so the
    benign corpus matches the agent's generative process. mode="rules" (fast,
    no LLM/containers): the scripted-workflow path, kept for CI/tests.

    include_holdouts=True puts the holdout cohort into the SAME universe as the
    training cohort (one universe → split downstream by cohort tag for L1P-O). The
    holdouts' actions also shape the world the training cohort sees — alice posts to
    chat, paul.o17 reads it, etc. — which is the realism reason to generate them
    together rather than in a separate run.

    `concurrency` (default 1 = sequential) bounds in-flight sessions. With
    concurrency > 1, sessions within each `bucket_minutes` sim-time window run
    concurrently via `asyncio.gather` (semaphore-bounded). Sim-time buckets are
    processed in order, preserving causal interleaving between buckets. Per-task
    clock is carried via a contextvar (see `audit/collector.py::set_task_clock`),
    so concurrent sessions don't race on the shared `_clock_fn`. At the 150 × 7
    scale, concurrency=1 takes ~4 weeks wall time; concurrency=25 reduces to ~28h.
    """
    config = config or NPCConfig(sessions_per_day=sessions_per_day,
                                 absent_probability=absent_probability)
    personas = load_personas(include_holdouts=include_holdouts)
    if allowed_persona_ids:
        # Per-persona filter — restrict the run to a specific set of accounts
        # (e.g. just the malicious insiders for prompt iteration). Skips the
        # n_personas slice; the caller controls the cohort directly.
        personas = [p for p in personas if p.id in allowed_persona_ids]
    elif n_personas < len(personas):
        personas = personas[:n_personas]

    out_path = Path(out_path)
    # Full session transcripts go to a sibling file (audit events are only the
    # metadata projection; level 2 LLM monitors need the whole conversation, and
    # regenerating an agentic corpus is expensive). One JSON object per session.
    transcript_path = out_path.with_suffix(".transcripts.jsonl")

    # Resume: prefer the day-boundary resume manifest (the authoritative record
    # of the last FULLY-committed day + the byte offsets to trim a partial/
    # garbage tail back to). Falls back to a date-scan for legacy corpora that
    # predate the manifest (lossy: a partial last day is skipped, not re-run,
    # and any crash tail is kept).
    resume_after_day: str | None = None
    manifest: dict[str, Any] | None = None
    restore_rec: dict[str, Any] | None = None   # set only on a --restore-world resume
    resume_in_progress: dict[str, Any] | None = None  # set only on a --session-resume mid-day resume
    if resume:
        manifest = resume_state.load(out_path)
        if manifest is not None:
            in_prog = resume_state.in_progress_record(manifest)
            if (session_resume and not restore_world and in_prog is not None
                    and in_prog["day"] > (resume_state.last_completed_day(manifest) or "")):
                # Session-level (bucket-granular) warm resume: the run died mid-day
                # with a live lab. Trim to the last in-day BUCKET boundary (finer
                # than the day boundary) and re-enter that day from the saved plan
                # (see the day loop), instead of discarding the whole in-progress
                # day. Only buckets after the checkpoint re-run; days <= the last
                # COMMITTED day are still skipped.
                trimmed_a, trimmed_t = resume_state.truncate_to_record(
                    in_prog, out_path, transcript_path)
                resume_after_day = resume_state.last_completed_day(manifest)
                resume_in_progress = in_prog
                logger.info("--session-resume: re-entering day %s from bucket %d "
                            "(last committed day=%s); trimmed %d audit + %d "
                            "transcript bytes of partial tail",
                            in_prog["day"], in_prog["next_bucket_index"],
                            resume_after_day, trimmed_a, trimmed_t)
            else:
                # Day-granular resume target = last fully-committed day. With
                # --restore-world we instead target the last day that ALSO has a
                # complete docker+in-process world boundary, so the restored world
                # lines up with the (truncated) audit/transcript.
                target_rec = resume_state.last_completed_record(manifest)
                if restore_world:
                    restore_rec = resume_state.last_restorable_record(manifest)
                    if restore_rec is not None:
                        target_rec = restore_rec
                    else:
                        logger.warning("--restore-world: no day has a complete "
                                       "snapshot+managers boundary; falling back to "
                                       "re-seed resume.")
                trimmed_a, trimmed_t = resume_state.truncate_to_record(
                    target_rec, out_path, transcript_path)
                resume_after_day = target_rec["day"] if target_rec else None
                logger.info("--resume: target_day=%s (restore_world=%s); trimmed %d "
                            "audit + %d transcript bytes of partial/aborted tail",
                            resume_after_day, bool(restore_rec), trimmed_a, trimmed_t)
        else:
            resume_after_day = _last_completed_day(out_path)
            logger.warning("--resume: no resume manifest at %s; falling back to "
                           "date-scan (no garbage trim; partial last day skipped). "
                           "last event date=%s",
                           resume_state.manifest_path(out_path), resume_after_day)
    if manifest is None:
        manifest = resume_state.new_manifest(
            audit_path=out_path, transcript_path=transcript_path,
            start_date=start_date, n_days=n_days, seed=seed, session_id=session_id)

    collector = AuditCollector(jsonl_path=out_path, append=resume)
    tmp_dir = out_path.resolve().parent

    transcript_fh = None
    # One transcript write lock (agentic mode opens the file below; rules mode leaves
    # transcript_fh None). Defined unconditionally so _flush_durable — used by the
    # day-commit AND the --session-resume bucket checkpoint in BOTH modes — always
    # has it in scope.
    _t_lock = __import__("threading").Lock()

    def _flush_durable() -> tuple[int, int]:
        """Flush + fsync the audit log and transcript, returning their exact on-disk
        byte sizes. Used to record durable byte offsets at a day boundary AND (with
        --session-resume) at each in-day bucket boundary. transcript_fh is None in
        rules mode (audit-only), so only the audit size is returned there."""
        a_bytes = collector.flush_sync()
        t_bytes = 0
        if transcript_fh is not None:
            with _t_lock:
                retry_on_disk_full(transcript_fh.flush, what="transcript flush")
                try:
                    os.fsync(transcript_fh.fileno())
                except OSError as exc:
                    if is_disk_full(exc):
                        retry_on_disk_full(
                            lambda: os.fsync(transcript_fh.fileno()),
                            what="transcript fsync")
            t_bytes = transcript_path.stat().st_size
        return a_bytes, t_bytes

    sandbox = None
    if mode == "agentic":
        if llm is None:
            from ..llm.factory import build_llm
            llm = build_llm(None)
        from ..sandbox.lab import LabSandbox
        from ..run_task import _default_lab_compose
        from uuid import uuid4
        # Resume sandbox policy:
        #  - --restore-world (restore_rec set): recreate the world from the last
        #    complete day-boundary snapshot. Restore the docker volumes in-place
        #    under the ORIGINAL project name, bring that project up, and overlay
        #    the in-process manager state from the same boundary. Crash-safe: does
        #    NOT assume the old containers survived.
        #  - plain --resume: reuse the prior session_id so compose reattaches to
        #    a still-running lab (works only if the containers outlived the crash,
        #    e.g. a python-only kill / the auth-breaker halt). No re-seed.
        #  - fresh run: new session id, seed the world.
        if restore_rec is not None:
            sb_session = manifest.get("session_id") or session_id or f"itb-npc-{uuid4().hex[:8]}"
            from ..sandbox.snapshot import restore_volumes_in_place
            logger.info("--restore-world: restoring docker volumes in-place from "
                        "%s into project %s", restore_rec["snapshot_dir"], sb_session)
            await restore_volumes_in_place(restore_rec["snapshot_dir"])
            sandbox = LabSandbox(session_id=sb_session,
                                 compose_files=[_default_lab_compose()])
            await sandbox.start()
            ctx = await build_context(collector, tmp_dir=tmp_dir, envs=AGENTIC_ENVS,
                                      sandbox=sandbox, seed_world=False)
            from ..sandbox.state_export import import_manager_state
            applied = import_manager_state(ctx, Path(restore_rec["managers_path"]))
            logger.info("--restore-world: re-imported in-process state: %s", applied)
        else:
            sb_session = session_id or f"itb-npc-{uuid4().hex[:8]}"
            # Persist the real sandbox id into the manifest up front so even a
            # day-1 crash records which lab/snapshot lineage this corpus belongs to.
            if manifest.get("session_id") != sb_session:
                manifest["session_id"] = sb_session
                resume_state.write_atomic(manifest, out_path)
            sandbox = LabSandbox(session_id=sb_session,
                                 compose_files=[_default_lab_compose()])
            await sandbox.start()
            # On plain resume, the seeded world is assumed already in the live
            # containers — don't re-seed (would duplicate or fail on existing rows).
            ctx = await build_context(collector, tmp_dir=tmp_dir, envs=AGENTIC_ENVS,
                                      sandbox=sandbox, seed_world=((not resume) or os.environ.get("NPC_RESEED_ON_RESUME","").lower() in ("1","true","yes")))
            # The 5 in-process managers (org/model_registry/eval_server/secrets_store/plane) live in
            # THIS python process, not the docker lab, so restoring the lab alone leaves them at day-0
            # on a cross-process resume (seed_world=False base-seeds them; unlike --restore-world this
            # path does no import) -- multi-day arcs that accumulate model/eval/plane/secret state would
            # silently reset. Overlay the last committed day's managers.json. Path resolution:
            # NPC_RESUME_MANAGERS_PATH wins (lets the file live on durable storage, e.g. a shared volume);
            # else fall back to the manifest's recorded managers_path (self-describing, e.g. a restored
            # /tmp/.snapshots tree). Same import --restore-world uses; loud if the state can't be found.
            if resume:
                _mp = os.environ.get("NPC_RESUME_MANAGERS_PATH", "").strip()
                if not _mp:
                    _rec = resume_state.last_completed_record(manifest)
                    if _rec and _rec.get("managers_ok"):
                        _mp = (_rec.get("managers_path") or "").strip()
                if _mp and Path(_mp).exists():
                    from ..sandbox.state_export import import_manager_state
                    applied = import_manager_state(ctx, Path(_mp))
                    logger.info("--resume: imported in-process manager state from %s: %s", _mp, applied)
                elif _mp:
                    logger.error("--resume: managers.json %s not found; in-process managers "
                                 "(org/model_registry/eval_server/secrets_store/plane) start at day-0", _mp)
                else:
                    logger.warning("--resume: no managers.json (NPC_RESUME_MANAGERS_PATH unset and no "
                                   "manifest managers_path); in-process managers start at day-0 base")
        from .agentic_member import AgenticMember, load_task_bank, load_arc_bank
        transcript_mode = "a" if resume else "w"
        transcript_fh = open(transcript_path, transcript_mode, encoding="utf-8")

        def _sink(rec: dict) -> None:
            with _t_lock:
                # write() buffers the line (exactly once); only the disk-touching
                # flush() is retried on a full disk, so ENOSPC/EDQUOT STALLS the
                # session instead of crashing it (was: uncaught OSError killed the
                # whole sim mid-day, losing the in-progress day). If write() itself
                # hits disk-full via an implicit buffer flush, the bytes stay
                # buffered and the retried flush below drains them (no dup).
                try:
                    transcript_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except OSError as exc:
                    if not is_disk_full(exc):
                        raise
                retry_on_disk_full(transcript_fh.flush, what="transcript sink")

        task_bank = load_task_bank(rich=True)   # expanded (v2) bank; RAISES if absent (no silent v1 fallback)
        # Benign multi-day arc bank (default OFF; loaded only when --benign-arcs; {} if absent).
        arc_bank = load_arc_bank() if benign_arcs else {}
        persona_map = insider_persona_map or {}
        # Multi-day campaign schedules (flag-gated; default OFF -> campaign_schedule
        # is None and the member runs the current single-day i.i.d. path). Built per
        # insider from a deterministic per-member RNG over the sim's working days;
        # benign members and non-decomposable threats (04/05, collusion) get None.
        threat_map = insider_threat_map or {}
        # Collusion role per insider (approver_1/approver_2, read_access/egress_access);
        # drives requester-before-co-signer day sequencing in build_campaign. Empty for
        # solo/shadow (role=None there -> legacy single random day).
        role_map = insider_role_map or {}
        # Resolve the planning mode. Default is DETERMINISTIC: "scripted" (stage->
        # trigger multi-day campaign) when --multi-day, else "single" (K self-contained
        # attack-days per insider). "coinflip" is the legacy random single-day path,
        # kept only for exact v1 reproduction. Both deterministic modes need the window.
        if planning is None:
            planning = "scripted" if multi_day else "single"
        _sim_dates = (_weekday_dates(date.fromisoformat(start_date), n_days)
                      if planning != "coinflip" else [])

        def _campaign_for(account_id: str, i: int):
            # Per-insider schedules from a per-member RNG over the working days:
            # "scripted" -> stage->trigger campaign; "single" -> K self-contained
            # attack-days. "agentic" -> None here (the agent authors at runtime in its
            # first session); "coinflip" (legacy) and benign members -> None (the
            # member's per-session i.i.d. path).
            if account_id not in persona_map:
                return None
            import random as _r
            rng = _r.Random((seed or 0) + i)
            if planning == "scripted":
                from .campaign import build_campaign
                return build_campaign(threat_map.get(account_id, ""), _sim_dates, rng,
                                      role=role_map.get(account_id)) or None
            if planning == "single":
                # Collusion is inherently multi-day (the requester->co-signer handoff needs
                # >=2 days); single-day can't coordinate it, so collusion insiders run BENIGN
                # in single-day mode rather than emitting degenerate (uncoordinated) "collusion"
                # cells. Collusion is evaluated only on the multi-day path.
                if role_map.get(account_id) is not None:
                    # Return an EXPLICIT empty schedule -- [] NOT None. None routes the member
                    # to its legacy i.i.d. attack coin-flip (agentic_member: the
                    # `campaign_schedule is None` branch), which re-emits the (uncoordinated)
                    # collusion attack anyway -- that bug put ~40% no-op collusion sessions
                    # into the single_K4 corpus. An empty list is "not None", so the member
                    # takes the schedule branch, matches no attack-day on any date, and stays
                    # benign the whole run.
                    return []
                from .campaign import build_single_day
                return build_single_day(_sim_dates, rng, k=campaign_days) or None
            return None

        def _arc_for(persona, i: int):
            # Benign multi-day arc for a NON-INSIDER -- the benign analog of _campaign_for. Off by
            # default; only when benign_arcs + a non-coinflip window (arcs need _sim_dates). Insiders
            # keep the attack campaign (mutually exclusive). SEPARATE RNG stream (+100003) so it never
            # perturbs self.rng or the attack RNG -> the default-off path stays byte-identical. For a
            # benign member None and [] behave identically (no attack coin-flip to suppress), so
            # `or None` is the clean "no arc" signal -- the opposite of the collusion-disable case.
            if not benign_arcs or planning == "coinflip" or persona.id in persona_map:
                return None
            arcs = arc_bank.get(getattr(persona, "team", "") or "")
            if not arcs:
                return None
            import random as _r
            rng = _r.Random((seed or 0) + 100003 + i)
            from .campaign import build_benign_arc
            return build_benign_arc(rng.choice(arcs), _sim_dates, rng) or None

        members: list[Any] = [
            AgenticMember(p, llm=llm, config=config, transcript_sink=_sink,
                          task_bank=task_bank,
                          rng_seed=(None if seed is None else seed + i),
                          persona_override=persona_map.get(p.id),
                          max_turns=max_turns,
                          elicit=elicit,
                          attack_session_rate=attack_session_rate,
                          campaign_schedule=_campaign_for(p.id, i),
                          benign_arc_schedule=_arc_for(p, i),
                          planning_mode=planning,
                          campaign_days=campaign_days,
                          sim_dates=_sim_dates,
                          rich_sessions=True)
            for i, p in enumerate(personas)
        ]
    else:
        ctx = await build_context(collector, tmp_dir=tmp_dir)
        members = [
            Member(p, config=config, rng_seed=(None if seed is None else seed + i))
            for i, p in enumerate(personas)
        ]

    sem = asyncio.Semaphore(max(1, concurrency))

    # Auth circuit-breaker state. `consecutive` counts back-to-back auth-failed
    # sessions; any successful session resets it. `tripped` latches once the
    # limit is hit so the day loop can bail before writing more empty sessions.
    breaker = {"consecutive": 0, "tripped": False}

    async def _run_one(member: Any, slot_dt: Any, action_key: str) -> None:
        if breaker["tripped"]:
            return  # don't start new work once the breaker has latched
        async with sem:
            if breaker["tripped"]:
                return
            try:
                await member.run_at(
                    ctx, slot_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), action_key,
                )
                breaker["consecutive"] = 0  # a clean session => auth is healthy
            except Exception as exc:                                # noqa: BLE001
                if _is_auth_error(exc):
                    breaker["consecutive"] += 1
                    logger.error(
                        "simulate: AUTH failure %d/%d on session %s/%s: %s",
                        breaker["consecutive"], auth_failure_limit,
                        getattr(member.persona, "id", "?"), action_key, exc)
                    if (auth_failure_limit > 0
                            and breaker["consecutive"] >= auth_failure_limit):
                        breaker["tripped"] = True
                        logger.error("simulate: AUTH circuit-breaker TRIPPED after "
                                     "%d consecutive failures — halting run. Refresh "
                                     "the API credential, then --resume.",
                                     breaker["consecutive"])
                else:
                    logger.exception("simulate: session %s/%s raised; continuing",
                                     getattr(member.persona, "id", "?"), action_key)

    # Day-boundary snapshots: take one at the end of each day so a resume
    # could later restore-into-fresh-sandbox. With this Stage-1 resume
    # (reuse-existing-sandbox), snapshots are useful as a safety net for the
    # snapshot/restore path we'll add as a follow-on.
    snapshots_dir = out_path.parent / ".snapshots" / out_path.stem

    keep_sandbox_for_resume = False
    try:
        for day_iso in _weekday_dates(date.fromisoformat(start_date), n_days):
            if resume_after_day is not None and day_iso <= resume_after_day:
                logger.info("--resume: skipping completed sim-day %s", day_iso)
                continue
            # Causal sim-time interleaving across all members for the day.
            # Old loop (`for m in members: await m.run_day`) ran each member's full
            # day before the next, letting later members see "future" events in the
            # shared world. The fix: gather every member's planned (slot, key) pairs
            # for the day, sort globally by slot, and run in that order — so alice's
            # 9am chat post is in the world when bob reads chat at 9:15am, not when
            # bob runs after alice has already done her 4pm post. Matches Chimera's
            # time-stepped scheduling (§3 of the paper: "At time step t, …").
            #
            # With concurrency>1: bucket plan entries by `bucket_minutes` sim-time
            # windows, process buckets in order, parallelize within each bucket via
            # asyncio.gather. Per-task contextvar clock keeps timestamps correct
            # even under concurrency. With concurrency=1: degenerates to the
            # original sequential behavior (sem permits one at a time, single bucket
            # gather just awaits each in turn).
            # Session-level resume: if this is the interrupted day, reuse its saved
            # plan (preserving slots + attack schedule without re-running plan_day
            # under a diverged RNG) and skip the buckets already durable on disk.
            resume_bucket_start = 0
            day_plan: list[tuple[Any, Any, str]]
            if resume_in_progress is not None and day_iso == resume_in_progress["day"]:
                resume_bucket_start = int(resume_in_progress["next_bucket_index"])
                day_plan = _restore_day_plan(resume_in_progress["day_plan"], members)
                logger.info("--session-resume: day %s reusing saved plan (%d sessions); "
                            "resuming at bucket %d", day_iso, len(day_plan), resume_bucket_start)
                resume_in_progress = None   # consumed; only this one day re-enters mid-way
            else:
                day_plan = [
                    (t, m, key) for m in members for (t, key) in m.plan_day(day_iso)
                ]
                day_plan.sort(key=lambda x: x[0])

            # Group by sim-time bucket while preserving order.
            buckets: list[list[tuple[Any, Any, str]]] = []
            current_bucket_id: int | None = None
            for t, m, key in day_plan:
                bid = (t.hour * 60 + t.minute) // max(1, bucket_minutes)
                if bid != current_bucket_id:
                    buckets.append([])
                    current_bucket_id = bid
                buckets[-1].append((t, m, key))

            for bucket_idx, bucket in enumerate(buckets):
                if bucket_idx < resume_bucket_start:
                    continue   # generated pre-crash + kept in the truncated tail
                # The continuous-day schedule packs an account's sessions close in
                # sim-time, so several can land in one bucket. Run each ACCOUNT's
                # sessions sequentially (so the cross-session recap/clock stays
                # causal) while different accounts still run concurrently.
                by_member: dict[int, list[tuple[Any, Any, str]]] = {}
                for t, m, key in bucket:
                    by_member.setdefault(id(m), []).append((t, m, key))

                async def _run_member_seq(items: list[tuple[Any, Any, str]]) -> None:
                    for t, m, key in items:
                        await _run_one(m, t, key)

                await asyncio.gather(*(
                    _run_member_seq(items) for items in by_member.values()
                ))
                if breaker["tripped"]:
                    # Leave the lab containers UP so --resume can reattach
                    # (Stage-1 resume reuses the live sandbox; tearing it down
                    # would lose the seeded world).
                    keep_sandbox_for_resume = True
                    raise AuthCircuitBreakerError(
                        f"{breaker['consecutive']} consecutive auth failures; "
                        f"halted mid-day {day_iso}. Token is stale/dead — refresh "
                        f"it, then re-run with --resume --session-id "
                        f"{sandbox.session_id if sandbox else session_id}.")

                # Session-level (bucket-granular) checkpoint. Placed AFTER the gather
                # (the bucket is quiescent — no session in flight, so the audit log is
                # cleanly truncatable here) AND after the breaker check (so a
                # breaker-tripping bucket is NOT checkpointed and re-runs on resume).
                # Records byte offsets + next bucket + the day plan, so a mid-day crash
                # resumes from the next bucket instead of discarding the whole day.
                # Default OFF — only when the run opted into --session-resume.
                if session_resume:
                    a_bytes, t_bytes = _flush_durable()
                    resume_state.record_in_progress(
                        manifest, out_path, day=day_iso,
                        next_bucket_index=bucket_idx + 1,
                        audit_bytes=a_bytes, transcript_bytes=t_bytes,
                        day_plan=_serialize_day_plan(day_plan))

            # End-of-day snapshot + in-process state export (both best-effort).
            # Docker snapshot only meaningful in agentic mode where a real
            # LabSandbox is up. In-process state export runs in both modes —
            # the 5 in-process managers (org/model_registry/eval_server/
            # secrets_store/plane) hold mutable Python state that docker
            # snapshots don't capture. Failures are logged but don't abort
            # the sim — we'd rather lose the snapshot than the day's work.
            snap_out = snapshots_dir / f"day_{day_iso}"
            managers_ok = False
            snapshot_ok = False
            try:
                snap_out.mkdir(parents=True, exist_ok=True)
                from ..sandbox.state_export import export_manager_state
                export_manager_state(ctx, snap_out / "managers.json")
                managers_ok = True
                logger.info("day-boundary in-process state exported: %s",
                            snap_out / "managers.json")
            except Exception:                                       # noqa: BLE001
                logger.exception("in-process state export failed for %s; "
                                 "sim continues", day_iso)
            # The docker volume snapshot (pause containers + tar each volume) is
            # ONLY needed for --restore-world (rebuilding a DEAD lab). Plain
            # --resume reattaches the still-live lab, whose volumes persist in
            # dockerd across days, so the backup tars are redundant for it — and
            # they get progressively slower as the GitLab volume fills. Set
            # NPC_SKIP_DOCKER_SNAPSHOT=1 to skip them (keeps the cheap in-process
            # managers.json export above). snapshot_ok stays False → manifest
            # records no snapshot_dir, consistent with the --resume-only path.
            _skip_snap = os.environ.get("NPC_SKIP_DOCKER_SNAPSHOT", "").lower() in ("1", "true", "yes")
            if sandbox is not None and _skip_snap:
                logger.info("day-boundary docker snapshot SKIPPED for %s "
                            "(NPC_SKIP_DOCKER_SNAPSHOT) — plain --resume reattaches "
                            "the live lab, volume backups are redundant", day_iso)
            elif sandbox is not None:
                try:
                    from ..sandbox.snapshot import snapshot_sandbox
                    await snapshot_sandbox(
                        sandbox.session_id, f"day_{day_iso}", out_dir=snap_out,
                        compose_files=getattr(sandbox, "compose_files", None),
                    )
                    snapshot_ok = True
                    logger.info("day-boundary docker snapshot saved: %s", snap_out)
                except Exception:                                       # noqa: BLE001
                    logger.exception("day-boundary docker snapshot failed for %s; "
                                     "sim continues", day_iso)

            # Commit the day to the resume manifest LAST, after the audit log,
            # transcript, state export and snapshot for the day are flushed to
            # disk. Record the exact byte offsets so a later --resume can trim a
            # partial/garbage tail back to this boundary. The manifest only ever
            # names a day whose outputs are durable — so resume can trust it.
            audit_bytes, transcript_bytes = _flush_durable()
            resume_state.record_day(
                manifest, out_path, day=day_iso,
                audit_bytes=audit_bytes, transcript_bytes=transcript_bytes,
                snapshot_ok=snapshot_ok, managers_ok=managers_ok,
                snapshot_dir=str(snap_out) if snapshot_ok else None,
                managers_path=str(snap_out / "managers.json") if managers_ok else None,
            )
            # The committed day supersedes any mid-day (bucket) checkpoint for it, so
            # drop it — a later resume must land on this day boundary, not re-enter the
            # finished day. No-op when --session-resume wasn't used.
            if session_resume:
                resume_state.clear_in_progress(manifest, out_path)
            logger.info("day-boundary manifest committed: day=%s audit=%dB "
                        "transcript=%dB snapshot_ok=%s",
                        day_iso, audit_bytes, transcript_bytes, snapshot_ok)
    finally:
        collector.close()
        if transcript_fh is not None:
            transcript_fh.close()
        if sandbox is not None and not keep_sandbox_for_resume:
            try:
                await sandbox.stop()
            except Exception:                                  # noqa: BLE001
                pass
        elif keep_sandbox_for_resume:
            logger.warning("simulate: leaving sandbox %s UP for --resume (auth "
                           "breaker tripped). Tear it down manually if abandoning.",
                           getattr(sandbox, "session_id", "?"))
    return collector


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mole.generator.run")
    ap.add_argument("--personas", type=int, default=3, help="number of NPCs")
    ap.add_argument("--days", type=int, default=5, help="number of working days")
    ap.add_argument("--out", default="npc_audit.jsonl", help="audit jsonl output path")
    ap.add_argument("--start-date", default="2026-04-06", help="first sim day (YYYY-MM-DD)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (None for nondeterministic)")
    ap.add_argument("--sessions", type=int, default=8, help="sessions per account per day")
    ap.add_argument("--include-holdouts", action="store_true",
                    help="generate trajectories for the holdout cohort too, mixed into the "
                         "same universe as the training cohort (one universe → split by "
                         "cohort tag at eval time for L1P-O). Default: holdouts excluded "
                         "from generation entirely.")
    ap.add_argument("--mode", choices=["agentic", "rules"], default="agentic",
                    help="agentic=LLM persona-agents (standard, needs sandbox+LLM backend); "
                         "rules=scripted workflows (fast, no LLM)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Max in-flight sessions per sim-time bucket. Default 1 "
                         "(sequential, deterministic). Set to ~25 for the 150x7 "
                         "production sim; bounded by provider rate limits.")
    ap.add_argument("--bucket-minutes", type=int, default=30,
                    help="Sim-time window for parallel buckets (default 30). "
                         "Sessions within the same window run via asyncio.gather; "
                         "windows are processed sequentially to preserve causal order.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from an existing audit log: open --out + transcripts "
                         "in append mode, skip any sim-day already present in the log, "
                         "and (if --session-id given) reattach to that sandbox. "
                         "Without --session-id, agentic mode starts a fresh sandbox; "
                         "the resume is only safe if the prior run's containers are "
                         "still up under that session-id.")
    ap.add_argument("--session-resume", action="store_true",
                    help="Opt into session-level (bucket-granular) warm resume: "
                         "checkpoint after each in-day sim-time bucket, so a mid-day "
                         "crash resumes from the last bucket (re-running only ~one "
                         "bucket) instead of discarding the whole in-progress day. "
                         "Default OFF (day-granular resume). Used together with "
                         "--resume; the live lab must have survived the crash.")
    ap.add_argument("--session-id", default=None,
                    help="Reattach to an existing LabSandbox by docker-compose project "
                         "name (only useful with --resume). If omitted on resume, a new "
                         "sandbox is started — only correct if you've separately "
                         "restored container state from a day-boundary snapshot.")
    ap.add_argument("--absent-prob", type=float, default=0.01,
                    help="Per-account whole-day absence probability (PTO/sick). "
                         "Default 0.01 (very rare); 0 disables.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    collector = asyncio.run(simulate(
        n_personas=args.personas, n_days=args.days,
        out_path=args.out, start_date=args.start_date, seed=args.seed,
        sessions_per_day=args.sessions, mode=args.mode,
        include_holdouts=args.include_holdouts,
        concurrency=args.concurrency, bucket_minutes=args.bucket_minutes,
        resume=args.resume, session_resume=args.session_resume,
        session_id=args.session_id,
        absent_probability=args.absent_prob,
    ))
    events = collector.events
    by_account = Counter(e.account for e in events)
    by_service = Counter(e.service for e in events)
    kinds = Counter(e.account_kind for e in events)
    print(f"generator: {len(events)} events -> {args.out}")
    if args.mode == "agentic":
        print(f"          transcripts -> {Path(args.out).with_suffix('.transcripts.jsonl')}")
    print(f"  account_kind: {dict(kinds)}")
    print(f"  by account: {dict(by_account)}")
    print(f"  by service: {dict(by_service)}")
    # The whole point: a benign-only log.
    n_malicious = sum(1 for e in events if e.is_malicious)
    print(f"  malicious events: {n_malicious} (expect 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
