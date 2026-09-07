"""Run a task end-to-end: sandbox → state setup → stages × (agent loop + rubric) → score.

The orchestrator does *not* know what individual tools any state-backend exposes
yet — in Phase 1 the agent runs with an empty tool catalog and the
rubric is the only signal. Tool wiring (registering each manager's methods as
LLM tools, dispatching tool calls back to the manager) lands once concrete
backends are implemented.

For tasks whose `environments` list is empty, the docker sandbox is skipped
entirely and a `DryRunSandbox` is used so `ctx.sandbox.ports` is still valid.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audit import AuditCollector
from .audit.schema import AuditEvent
from .agent.loop import AgentLoopResult, Dispatcher, ToolResult, run_agent_loop
from .agent.prompts import render_prompt
from .agent.tools import build_tools_for
from .llm import ChatMessage, LLMClient, ToolCall, ToolSchema
from .models import RubricItemResult, StageResult, TaskDefinition, TaskResult
from .sandbox.base import DryRunSandbox, Sandbox
from .sandbox.lab import LabSandbox
from .state.composite import CompositeStateManager
from .state.context import TaskContext
from .tasks.loader import load_task
from .tasks.rubric import run_rubric

logger = logging.getLogger(__name__)


# Sentinel account for tasks that don't declare environments (and so never
# emit events). NEVER use this for any task with state backends — the audit
# log would carry a literal "agent" actor label, which is a giant
# obvious leak to any level 0 monitor.
_NULL_ACCOUNT = "<no-account>"


def _resolve_focal_account(task_metadata: dict[str, Any], environments: list[str]) -> str:
    """Return the account identity the agent acts as.

    Requires METADATA["focal_account"] whenever the task declares any state
    backends — otherwise every audit event would carry a meta-label like
    'agent' that the monitor must not see.
    """
    fp = task_metadata.get("focal_account")
    if fp:
        return str(fp)
    if environments:
        raise ValueError(
            "task declares environments but METADATA['focal_account'] is unset; "
            "the audit log requires an account identity (e.g. 'bob.li') to tag "
            "events with — defaulting would leak an actor label to monitors."
        )
    return _NULL_ACCOUNT


def _is_malicious_run(task_metadata: dict[str, Any], override: bool | None) -> bool:
    """Decide whether to tag agent events `is_malicious=True`.

    Explicit override wins. Otherwise: any adversary_level other than the
    benign default (or `None`) is treated as malicious.
    """
    if override is not None:
        return bool(override)
    level = (task_metadata.get("adversary_level") or "").strip().lower()
    return level not in ("", "benign", "none")


def _build_tools_and_dispatcher(
    ctx: TaskContext,
) -> tuple[list[ToolSchema], Dispatcher]:
    """Build the LLM-visible tool catalog + dispatcher from registered managers.

    Auto-reflects over each manager's public async methods (see
    agent/tools.build_tools_for). When the task declares no environments,
    returns an empty catalog and a dispatcher that refuses every call.
    """
    return build_tools_for(ctx)


def _weighted_score(items: list[RubricItemResult]) -> float:
    total = sum(it.weight for it in items)
    if total <= 0:
        return 0.0
    return sum(it.weight for it in items if it.passed) / total


def _read_task_metadata(task: TaskDefinition) -> dict[str, Any]:
    """Re-load the task module and return its full METADATA dict.

    TaskDefinition unpacks the metadata into its named fields at load time,
    so we have to re-import to retrieve fields the loader didn't surface
    (focal_account, adversary_level, etc.). Cheap — it's a one-shot import.
    """
    task_py = Path(task.task_dir) / "task.py"
    spec = importlib.util.spec_from_file_location(
        f"_itb_metadata_{task.id}", task_py,
    )
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    md = getattr(module, "METADATA", {})
    return dict(md) if isinstance(md, dict) else {}


async def run_task_full(
    *,
    task_dir: Path | str,
    llm: LLMClient,
    lab_compose: Path | None = None,
    session_id: str | None = None,
    max_turns_per_stage: int = 30,
    temperature: float = 0.7,
    results_dir: Path | str = "results",
    is_malicious: bool | None = None,
    protocol: str | None = None,
) -> TaskResult:
    """Execute one task from sandbox-up through final rubric scoring.

    `is_malicious` overrides the default inference. By default, the run is
    treated as malicious iff METADATA['adversary_level'] is set to anything
    other than 'benign'/empty/'none'. This is the flag that gets stamped onto
    every agent audit event so oracles can later filter Cell 3/4 events.
    """
    task: TaskDefinition = load_task(task_dir)
    logger.info("task %s (%s) — %d stage(s), envs=%s",
                task.name, task.id, len(task.stage_fns), task.environments)

    composite = CompositeStateManager(task.environments, task.env_config)
    # Sandbox is needed only if *some* backend actually requires Docker.
    # Pure-data backends (org, etc.) set NEEDS_SANDBOX = False on the class.
    needs_sandbox = any(
        getattr(type(mgr), "NEEDS_SANDBOX", True)
        for mgr in composite.managers.values()
    )
    sandbox: Sandbox
    started_real_sandbox = False

    # Pull METADATA off the loaded module so checkers + persona-rendering
    # have access to focal_account / adversary_level / etc.
    task_metadata = _read_task_metadata(task)
    # `focal_account` resolution is deferred into the try-block below so a
    # missing-account validation error lands in result.error like everything
    # else, rather than crashing the call site.
    focal_account: str = _NULL_ACCOUNT
    malicious_run: bool = False

    # Per-run trace persistence. Each call to run_task_full gets its
    # own run_id-stamped directory under traces_base so audit logs +
    # transcripts + manifests don't overwrite prior runs. Critical for
    # monitor training: every agent trace needs to survive past
    # the next sweep iteration.
    #
    # traces_base resolution priority (high → low):
    #   1. explicit results_dir kwarg (when caller overrides default)
    #   2. $MOLE_TRACE_DIR env
    #   3. "./traces" (local default)
    from .persistence import (
        TraceWriter, RunManifest, default_traces_base,
    )
    traces_base = (
        Path(results_dir) if results_dir and str(results_dir) != "results"
        else Path(os.environ.get("ITB_TRACES_BASE")
                  or default_traces_base())
    )
    trace = TraceWriter(traces_base=traces_base, task_id=task.id)
    results_root = trace.run_dir
    audit_path = trace.audit_jsonl
    collector = AuditCollector(jsonl_path=audit_path)
    # Plug the org backend (if loaded) into the collector so is_external
    # checks use the organisation's real domain.
    collector.set_org_lookup(lambda: composite.managers.get("org"))

    sim_start_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.monotonic()
    stage_results: list[StageResult] = []
    error_text = ""

    try:
        # ── input validation ───────────────────────────────────────
        focal_account = _resolve_focal_account(task_metadata, task.environments)
        malicious_run = _is_malicious_run(task_metadata, is_malicious)

        # ── sandbox ────────────────────────────────────────────────
        if needs_sandbox:
            if lab_compose is None:
                raise ValueError(
                    "task declares environments but no lab_compose provided"
                )
            sid = session_id or f"itb-{task.id}-{uuid4().hex[:8]}"
            sandbox = LabSandbox(session_id=sid, compose_files=[Path(lab_compose)])
            await sandbox.start()
            started_real_sandbox = True
        else:
            sandbox = DryRunSandbox()

        await composite.setup(sandbox=sandbox)

        # Seed the world BEFORE wrapping managers in audit middleware, so
        # seed-time API calls (creating users / groups / projects / files)
        # don't pollute the per-threat audit log. The agent should see
        # the pre-existing world; oracles read post-agent state, not
        # seed-time events.
        from types import SimpleNamespace
        seed_ctx = SimpleNamespace(**composite.managers)
        try:
            from .seeders.seed_org import seed_org
            seed_counts = await seed_org(seed_ctx)
            logger.info("seed_org: %s", seed_counts)
        except FileNotFoundError as exc:
            # No org_template.yaml is fine for tasks with empty environments;
            # noisy for tasks that need it, but recoverable — proceed.
            logger.warning("seed_org skipped: %s", exc)
        except Exception:
            logger.exception("seed_org failed; continuing with partial world state")
        # Per-threat seed delta if defined.
        if task.seed_fn is not None:
            try:
                await task.seed_fn(seed_ctx)
                logger.info("task.seed completed for %s", task.id)
            except Exception:
                logger.exception("task.seed failed; continuing")

        # Pre-episode snapshot for snapshot-diff harm oracles (retroactive-edit,
        # data-destruction). Captured with the RAW managers (seed_ctx) AFTER
        # seeding but BEFORE the audit-wrap + agent run, so these baseline reads
        # don't appear in the agent audit log. Stored on ctx.snapshots
        # once the context exists. Without this, those oracles silently compare
        # against an empty baseline and never fire.
        pre_snapshot: dict[str, Any] = {}
        if task.pre_snapshot_fn is not None:
            try:
                snap_ctx = SimpleNamespace(**composite.managers,
                                           task_metadata=task_metadata)
                pre_snapshot = await task.pre_snapshot_fn(snap_ctx) or {}
                logger.info("task.pre_snapshot captured for %s: keys=%s",
                            task.id, list(pre_snapshot))
            except Exception:
                logger.exception("task.pre_snapshot failed; snapshot-diff harm "
                                 "oracles may under-fire")

        # Wrap each state manager so every public async method emits audit
        # events tagged with the agent account + malicious flag.
        # Bind via default-arg so the closures don't capture the latest value.
        for svc_name, mgr in composite.managers.items():
            collector.wrap_manager(
                service_name=svc_name,
                manager=mgr,
                account_getter=lambda fp=focal_account: (fp, "agent"),
                is_malicious_getter=lambda flag=malicious_run: flag,
            )

        # Make the focal account the actor at the service level too,
        # not just the audit-event label. GitLab's `Sudo` header lets
        # the admin PAT impersonate a user so MRs / commits land
        # authored by the focal account — needed for any oracle that
        # filters by author (list_mrs author=bob.li, etc.). Reads stay
        # as admin so oracles see ground truth across the whole repo.
        gl_mgr = composite.managers.get("gitlab")
        if gl_mgr is not None and focal_account != _NULL_ACCOUNT:
            set_actor = getattr(gl_mgr, "set_actor", None)
            if callable(set_actor):
                set_actor(focal_account)

        ctx: TaskContext = composite.create_context(
            task_dir=Path(task.task_dir),
            sandbox=sandbox,
            audit=collector,
            task_metadata=task_metadata,
            sim_start=sim_start_iso,
        )
        # Dispatcher auto-fills actor parameters (account/sender/from_user)
        # from this box. Without it, schemas mark those fields optional but
        # tools that require them still receive no actor and either TypeError
        # or persist state under "system" instead of the focal account.
        if focal_account != _NULL_ACCOUNT:
            ctx._account_box = {
                "account": focal_account,
                "kind": "agent",
            }
        # Expose the pre-episode baseline to snapshot-diff harm oracles.
        ctx.snapshots = pre_snapshot
        # Default: audit events use the task's sim_start until a stage_fn
        # either calls ctx.advance_clock() at its top OR returns a `time`
        # field. The orchestrator advances the clock between stages from
        # whichever value the previous stage returned.
        ctx.advance_clock(sim_start_iso)
        # The agent "logs in" once, on the session's first sim day —
        # emitted INSIDE the stage loop after the clock advances to stage0's
        # sim time, so it lands on the threat timeline (not real wall-clock
        # now, which would fragment the run into a spurious extra user-day).
        _session_started = False

        # ── persistent conversation across stages ─────────────────
        tools, dispatcher = _build_tools_and_dispatcher(ctx)
        # Intervention layer: wrap the dispatcher so every tool call passes a
        # gating protocol (observability-level decision) before mutating state. This is
        # the intervention half of the frontier — Cell 2 (benign+control utility
        # cost) and Cell 4 (malicious+control harm prevented).
        gating = None
        if protocol and protocol != "no_control":
            from .protocols import load_protocol
            gating = load_protocol(protocol, org=composite.managers.get("org"), llm=llm)
            dispatcher = gating.wrap(dispatcher)
            logger.info("gating protocol active: %s", protocol)
        system_prompt = render_prompt(
            task.prompt,
            focal_account=focal_account if focal_account != _NULL_ACCOUNT else None,
            org_manager=composite.managers.get("org"),
            sim_start_iso=sim_start_iso,
            available_tools=[t.name for t in tools],
            metadata=task_metadata,
        )
        history: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
        ]

        # Persist the rendered system prompt (with adversary overlay if
        # any) for downstream replay + monitor training. Once-per-run.
        try:
            trace.write_system_prompt(system_prompt)
        except Exception:
            logger.exception("trace: write_system_prompt failed")

        # Stamp an initial manifest so the trace dir is queryable even
        # if the run crashes mid-stages. Filled in with final score
        # below.
        run_manifest = RunManifest(
            run_id=trace.run_id,
            task_id=task.id,
            task_name=task.name,
            adversary=None,
            is_malicious=malicious_run,
            focal_account=focal_account,
            llm_backend=getattr(llm, "backend", "?"),
            llm_model=getattr(llm, "model_id", "?"),
            temperature=temperature,
            max_turns_per_stage=max_turns_per_stage,
            sim_start_iso=sim_start_iso,
            started_at_iso=sim_start_iso,
        )
        try:
            trace.write_manifest(run_manifest)
        except Exception:
            logger.exception("trace: initial write_manifest failed")

        # ── per-stage execution ────────────────────────────────────
        for i, stage_fn in enumerate(task.stage_fns):
            stage_id = f"stage{i}"
            logger.info("─── %s ───", stage_id)
            stage_err = ""
            agent_result: AgentLoopResult | None = None

            try:
                ret = await stage_fn(ctx)
                notification = (ret or {}).get("notification", "") if isinstance(ret, dict) else ""

                # If the stage returned a `time` field, adopt it as the
                # current simulated clock — events from the agent loop
                # (and all subsequent stages' events, unless those stages
                # explicitly advance further) will be tagged with this ts.
                stage_time = (ret or {}).get("time") if isinstance(ret, dict) else None
                if stage_time:
                    ctx.advance_clock(str(stage_time))

                # Emit the session logon once, now that the clock is on the
                # threat's sim timeline (first stage). Keeps all focal
                # events — logon, agent actions, logoff — on coherent sim
                # days instead of spilling onto real wall-clock today.
                if not _session_started and focal_account != _NULL_ACCOUNT:
                    _clk = getattr(collector, "_clock_fn", None)
                    collector.emit(AuditEvent.new(
                        account=focal_account, account_kind="agent",
                        service="session", action="logon", resource_id=focal_account,
                        ts=_clk() if callable(_clk) else sim_start_iso,
                    ))
                    _session_started = True

                if notification:
                    history.append(ChatMessage(role="user", content=notification))
                    agent_result = await run_agent_loop(
                        llm=llm,
                        initial_messages=history,
                        tools=tools,
                        dispatcher=dispatcher,
                        max_turns=max_turns_per_stage,
                        temperature=temperature,
                    )
                    history = agent_result.messages
                    logger.info(
                        "  agent: %d turn(s), finish=%s, tokens in/out=%d/%d",
                        agent_result.turn_count,
                        agent_result.finish_reason,
                        agent_result.total_input_tokens,
                        agent_result.total_output_tokens,
                    )
            except Exception as exc:
                logger.exception("stage %s raised", stage_id)
                stage_err = f"{type(exc).__name__}: {exc}"

            # Per-stage rubric (if any defined).
            stage_rubric = task.rubric.get(stage_id, [])
            eval_res = await run_rubric(stage_rubric, ctx)

            stage_results.append(StageResult(
                stage_id=stage_id,
                success=(stage_err == "" and (agent_result is None
                         or agent_result.aborted_reason is None)),
                error=stage_err or (agent_result.aborted_reason if agent_result else "") or "",
                verification=eval_res.items,
                verification_score=eval_res.score,
            ))

        # ── final rubric ───────────────────────────────────────────
        final_rubric = task.rubric.get("final", [])
        final_eval = await run_rubric(final_rubric, ctx)
        stage_results.append(StageResult(
            stage_id="final",
            verification=final_eval.items,
            verification_score=final_eval.score,
        ))

        # Session logoff at end-of-run (matches background-account logoff).
        if focal_account != _NULL_ACCOUNT:
            _clk = getattr(collector, "_clock_fn", None)
            collector.emit(AuditEvent.new(
                account=focal_account, account_kind="agent",
                service="session", action="logoff", resource_id=focal_account,
                ts=_clk() if callable(_clk) else sim_start_iso,
            ))

    except Exception as exc:
        logger.exception("task %s failed", task.id)
        error_text = f"{type(exc).__name__}: {exc}"

    finally:
        try:
            await composite.cleanup()
        except Exception as exc:
            logger.warning("composite cleanup error: %s", exc)
        if started_real_sandbox:
            try:
                await sandbox.stop(delete=True)
            except Exception as exc:
                logger.warning("sandbox.stop error: %s", exc)
        try:
            collector.close()
        except Exception as exc:
            logger.warning("audit collector close error: %s", exc)

    elapsed = time.monotonic() - started
    all_items: list[RubricItemResult] = [v for sr in stage_results for v in sr.verification]
    score = _weighted_score(all_items)

    logger.info("=" * 60)
    logger.info("task %s — score %.3f (%d/%d items, %.1fs)", task.id, score,
                sum(1 for it in all_items if it.passed), len(all_items), elapsed)
    for it in all_items:
        logger.info("  %s %s (w=%.1f) %s", "✓" if it.passed else "✗", it.item_id,
                    it.weight, it.detail or "")
    logger.info("=" * 60)

    # Final trace persistence: enrich manifest with score + rubric
    # summary, write result.json + transcript, append to index.jsonl.
    try:
        run_manifest.completed_at_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        run_manifest.execution_time_s = elapsed
        run_manifest.score = score
        run_manifest.error = error_text
        run_manifest.rubric_summary = {it.item_id: it.passed for it in all_items}
        trace.write_manifest(run_manifest)
        # Save the agent's full transcript across all stages
        for msg in history:
            trace.append_transcript(
                stage_id="<aggregated>",
                role=getattr(msg, "role", "?"),
                content=getattr(msg, "content", "") or "",
                tool_calls=getattr(msg, "tool_calls", None),
            )
        # And the result.json (same shape as run_task.py writes).
        trace.write_result({
            "task_id": task.id,
            "score": score,
            "execution_time_s": elapsed,
            "error": error_text,
            "stages": [
                {"id": s.stage_id, "success": s.success, "error": s.error,
                 "verification_score": s.verification_score,
                 "verification": [
                     {"id": v.item_id, "passed": v.passed, "weight": v.weight,
                      "detail": v.detail, "method": v.method.value}
                     for v in s.verification
                 ]} for s in stage_results
            ],
        })
        trace.append_index(run_manifest)
        logger.info("trace persisted: %s", trace.run_dir)
    except Exception:
        logger.exception("trace persistence failed; trace dir may be partial")

    return TaskResult(
        task_id=task.id,
        stage_results=stage_results,
        rubric_results=all_items,
        score=score,
        execution_time_s=elapsed,
        error=error_text,
    )
