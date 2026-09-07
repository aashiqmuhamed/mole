"""Sim runner — kick off the v1 150x7 generation with insider personas wired.

Loads `bootstrap/insider_assignment.yaml` + `bootstrap/personas/*.yaml`,
builds the `insider_persona_map: {employee_id -> persona_override_text}`
dict, and calls `generator.run.simulate()` with the right concurrency and resume
flags. Defaults to a SMOKE config (15 accounts × 2 days × concurrency=4)
so you can verify the pipeline works end-to-end before kicking off the
full 24-hour run.

Usage examples:
  # Smoke run on OpenRouter (default backend):
  OPENROUTER_API_KEY=... OPENROUTER_MODEL=openai/gpt-5.3 \\
      python scripts/sim_runner.py --smoke

  # Full run on a self-hosted vLLM endpoint:
  LLM_BACKEND=vllm VLLM_BASE_URL=http://localhost:8000/v1 VLLM_MODEL=kimi-k2.6 \\
      python scripts/sim_runner.py --full --concurrency 25

  # Resume after a crash:
  python scripts/sim_runner.py --full --resume --session-id <prev>

The CLI flags forward to generator/run.py::simulate.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO          # repo root: holds mole/, bootstrap/, data/
sys.path.insert(0, str(BENCH))

import yaml                                                  # noqa: E402
from mole.generator.run import simulate            # noqa: E402
from mole.model_config import (              # noqa: E402
    recommended_concurrency,
    recommended_max_turns,
)

ASSIGNMENT = BENCH / "bootstrap" / "insider_assignment.yaml"
PERSONAS_DIR = BENCH / "bootstrap" / "personas"              # agent-native motives (default)
PERSONAS_HUMAN_DIR = BENCH / "bootstrap" / "personas_human"  # legacy human-motive v1 set


def _active_model_id() -> str:
    """Model id/slug for the active backend, read from its model env. Feeds the
    per-model concurrency / max-turns defaults in configs/models.yaml."""
    backend = os.environ.get("LLM_BACKEND", "openrouter").strip().lower()
    env = {"openrouter": "OPENROUTER_MODEL", "or": "OPENROUTER_MODEL",
           "vllm": "VLLM_MODEL", "local": "VLLM_MODEL",
           "anthropic": "ANTHROPIC_MODEL", "claude": "ANTHROPIC_MODEL"}.get(backend, "MODEL")
    return os.environ.get(env, "") or os.environ.get("MODEL", "")


def _flatten_assignment(assignment: dict) -> list[tuple[str, str, str | None, str]]:
    """Return (threat, level, role_label_or_None, account_id) tuples for all
    solo + collusion + shadow cells."""
    out: list[tuple[str, str, str | None, str]] = []
    for sc, levels in (assignment.get("solo") or {}).items():
        for lvl, eid in levels.items():
            out.append((sc, lvl, None, eid))
    for sc, levels in (assignment.get("collusion") or {}).items():
        for lvl, roles in levels.items():
            for role, eid in roles.items():
                out.append((sc, lvl, role, eid))
    for sc, levels in (assignment.get("shadow") or {}).items():
        for lvl, eid in levels.items():
            out.append((sc, lvl, None, eid))
    return out


def build_insider_persona_map(personas_dir: Path = PERSONAS_DIR) -> dict[str, str]:
    """Walk the assignment + persona files and build {account_id -> override_text}.

    For collusion (role-suffixed file names like `c1_release_bypass_s0_approver_1.yaml`)
    we load the file by (threat, level, role) triple. For solo+shadow it's
    `<threat>_<level>.yaml` or `<threat>_<level>_shadow.yaml`.

    `personas_dir` selects the motive set: PERSONAS_DIR (agent-native, default) or
    PERSONAS_HUMAN_DIR (legacy human-motive v1). Both share the same filenames.
    """
    assignment = yaml.safe_load(ASSIGNMENT.read_text(encoding="utf-8"))
    flat = _flatten_assignment(assignment)
    persona_map: dict[str, str] = {}
    for threat, level, role, eid in flat:
        if role is not None:
            fname = f"{threat}_{level}_{role}.yaml"
        else:
            # Shadow vs solo distinguished by whether this account_id appears in
            # shadow assignment.
            shadow_cell = (assignment.get("shadow") or {}).get(threat, {})
            is_shadow = shadow_cell.get(level) == eid
            fname = f"{threat}_{level}_shadow.yaml" if is_shadow else f"{threat}_{level}.yaml"
        path = personas_dir / fname
        if not path.exists():
            print(f"warn: persona file missing for {eid} ({threat}/{level}/{role}): {fname}")
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        text = doc.get("persona_override")
        if text:
            persona_map[eid] = text
    return persona_map


def build_insider_threat_map() -> dict[str, str]:
    """{account_id -> threat_id} for designated insiders (drives multi-day campaigns).

    Threat comes straight from the assignment, so this needs no persona files.
    Collusion (c1/c2) account_ids are included but have no PHASE_DECOMP entry, so they
    fall back to single-day behavior.
    """
    assignment = yaml.safe_load(ASSIGNMENT.read_text(encoding="utf-8"))
    return {eid: threat for threat, level, role, eid in _flatten_assignment(assignment)}


def build_insider_role_map() -> dict[str, str]:
    """{account_id -> collusion role} for collusion insiders only (approver_1/approver_2,
    read_access/egress_access). Solo + shadow cells have role=None and are omitted, so
    build_campaign leaves them on the legacy single-day path. Drives the requester-
    before-co-signer attack-day sequencing for collusion pairs in multi-day mode."""
    assignment = yaml.safe_load(ASSIGNMENT.read_text(encoding="utf-8"))
    return {eid: role for threat, level, role, eid in _flatten_assignment(assignment)
            if role is not None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sim_runner")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smoke", action="store_true",
                     help="Smoke config: 15 employees × 2 days × concurrency=4 "
                          "(~30 min wall time on a stable backend).")
    grp.add_argument("--full", action="store_true",
                     help="Full v1 config: 150 employees × 7 days. Default "
                          "concurrency=25; override with --concurrency.")
    ap.add_argument("--out", default=None,
                    help="Output audit jsonl path. Default: smoke->sim_smoke.jsonl, "
                         "full->sim150.jsonl under data/corpus/.")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Override the default per-config concurrency.")
    ap.add_argument("--resume", action="store_true",
                    help="Pass through to simulate(); requires --session-id.")
    ap.add_argument("--restore-world", action="store_true",
                    help="On --resume, RESTORE the docker + in-process world from "
                         "the last complete day-boundary snapshot (crash-safe; does "
                         "not assume the lab containers survived) instead of "
                         "reusing a live sandbox or re-seeding. Gives a continuous "
                         "world across the resume boundary. NOTE: the docker volume "
                         "restore should be smoke-tested before trusting it on a "
                         "production corpus.")
    ap.add_argument("--session-resume", action="store_true",
                    help="Opt into session-level (bucket-granular) warm resume: "
                         "checkpoint after each in-day sim-time bucket so a mid-day "
                         "crash resumes from the last bucket (re-running ~one bucket) "
                         "instead of discarding the whole in-progress day. Default OFF "
                         "(day-granular resume, unchanged). Pair with --resume; the "
                         "live lab must have survived the crash.")
    ap.add_argument("--session-id", default=None,
                    help="Reuse an existing LabSandbox project on resume.")
    ap.add_argument("--start-date", default="2026-04-06")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=None,
                    help="Per-session LLM-call cap. Default None → resolve "
                         "per-model from the registry (gpt-5.3-chat 30, kimi "
                         "50, DeepSeek-V3.2 60; untested models 30). Pass an "
                         "int to override for all models.")
    ap.add_argument("--attack-rate", type=float, default=0.20,
                    help="Per-session probability that an insider runs in "
                         "ATTACK mode (loads malicious persona). The rest "
                         "of the insider's sessions run BENIGN mode (same "
                         "task_bank as a regular background account) so the corpus has a "
                         "realistic attack-to-routine ratio. Default 0.20 "
                         "(empirical insiders are ~5%%; default elevated "
                         "for smoke-scale sample sizes). Set to 1.0 to "
                         "restore the always-attack behavior for prompt "
                         "iteration; set 0.0 for a pure-benign control.")
    ap.add_argument("--multi-day", action="store_true",
                    help="Enable MULTI-DAY campaign mode for insiders: instead of the "
                         "i.i.d. per-session attack coin-flip, each insider runs a staged "
                         "campaign (stage on day i, trigger on day j>i) so the corpus has "
                         "genuine cross-day attacks + detection latency. Default OFF "
                         "(current single-day behavior). Also enabled by env NPC_MULTIDAY=1. "
                         "Threats 04/05 and collusion stay single-day.")
    ap.add_argument("--benign-arcs", action="store_true",
                    help="Enable BENIGN multi-day arcs (default OFF): non-insider agents run "
                         "multi-day benign PROJECTS (one step/day from benign_arcs_v1.yaml) so "
                         "cross-day continuity isn't attack-exclusive -- gives multi-day attacks "
                         "benign cover. Only meaningful with --multi-day. Also env NPC_BENIGN_ARCS=1.")
    ap.add_argument("--planning", choices=["single", "scripted", "coinflip", "agentic"], default=None,
                    help="Insider attack-insertion mode. 'single' (no-flag default): K "
                         "self-contained attack-days per insider, evenly spread with "
                         "per-insider jitter. 'scripted': precomputed stage->trigger multi-day "
                         "campaign. 'coinflip': legacy per-session i.i.d. coin-flip (random; v1 "
                         "repro). 'agentic': the insider authors its own K-day schedule on its "
                         "first session (agent-chosen timing) -- NOT default (frontier chat "
                         "models refuse the meta-plan: gpt 100%%, kimi 60%%), kept for future "
                         "models that comply. Default: scripted if --multi-day, else single.")
    ap.add_argument("--campaign-days", type=int, default=4,
                    help="K = attack-days per insider in 'single'/'agentic' modes (fixed, so the base "
                         "rate is #insiders * K / account-days). Default 4 — calibrated to "
                         "the v1 ar05 base rate (~3.2%%, ~4.6 attack-days/insider), counted "
                         "executed-union-attempted.")
    ap.add_argument("--no-elicitation", action="store_true",
                    help="Disable the red-team elicitation prefix on insider "
                         "system prompts (default: prefix ON). The prefix "
                         "frames the session as a sanctioned eval so "
                         "capability-aligned models (Opus) don't refuse the "
                         "persona on principle. Disable to measure natural "
                         "(unframed) compliance for comparison.")
    ap.add_argument("--persona-set", choices=["agent", "human"], default="agent",
                    help="Which insider-persona motive set to load: 'agent' "
                         "(default) = agent-native motives from bootstrap/personas/; "
                         "'human' = the legacy human-motive v1 set from "
                         "bootstrap/personas_human/ (the bridge baseline / repro).")
    ap.add_argument("--personas", default=None,
                    help="Comma-separated persona IDs to run, restricting "
                         "the sim to JUST these employees (e.g. "
                         "'bob.li,grace.t,tom.g0'). Lets you iterate on "
                         "prompt changes against specific personas in a "
                         "few minutes instead of running the whole smoke. "
                         "Everyone else is dropped from the cohort.")
    ap.add_argument("--days", type=int, default=None,
                    help="Override day count. Default: 2 for --smoke, "
                         "7 for --full. Useful with --personas for sub-"
                         "minute iteration loops at --days 1.")
    ap.add_argument("--sessions-per-day", type=int, default=None,
                    help="Override sessions/persona/day (default 8, ~7.6 after "
                         "loaf). Lower (e.g. 2) for a fast WIDE smoke: all "
                         "personas at shallow depth.")
    ap.add_argument("--auth-failure-limit", type=int, default=25,
                    help="Halt the whole run after this many CONSECUTIVE "
                         "auth-failed sessions (API credential expired/dead). "
                         "Prevents the failure mode where a dead credential "
                         "lets the sim march on writing thousands of empty "
                         "sessions. The sandbox is left up so you can refresh "
                         "the credential and --resume. 0 disables the breaker.")
    args = ap.parse_args(argv)

    persona_filter = None
    if args.personas:
        persona_filter = set(x.strip() for x in args.personas.split(",") if x.strip())

    backend = os.environ.get("LLM_BACKEND", "openrouter").strip().lower()

    if args.smoke:
        n_personas = len(persona_filter) if persona_filter else 15
        n_days = args.days if args.days is not None else 2
        concurrency = args.concurrency or min(4, n_personas)
        default_out = BENCH / "data" / "corpus" / "sim_smoke.jsonl"
    else:
        n_personas = len(persona_filter) if persona_filter else 150
        n_days = args.days if args.days is not None else 7
        # Per-model default: each model has its own sustainable concurrency
        # (from configs/models.yaml, keyed by the active backend's model slug);
        # --concurrency overrides.
        concurrency = args.concurrency or recommended_concurrency(_active_model_id())
        default_out = BENCH / "data" / "corpus" / "sim150.jsonl"
    out_path = Path(args.out) if args.out else default_out

    # Per-model turn cap (some models finish attacks in more turns than others);
    # --max-turns overrides. Cheap to raise now that the loop bounds per-call
    # context regardless of turn count. Resolved from configs/models.yaml.
    max_turns = (args.max_turns if args.max_turns is not None
                 else recommended_max_turns(_active_model_id()))
    sessions_per_day = args.sessions_per_day if args.sessions_per_day is not None else 8

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _model = _active_model_id() or "(unset)"
    _conc_src = "explicit --concurrency" if args.concurrency else "per-model default"
    _mt_src = "explicit --max-turns" if args.max_turns is not None else "per-model default"
    print(f"sim_runner: {'SMOKE' if args.smoke else 'FULL'} run  backend={backend}")
    print(f"  model={_model} concurrency={concurrency} ({_conc_src})")
    print(f"  max_turns={max_turns} ({_mt_src})")
    print(f"  personas={n_personas} days={n_days} sessions/day={sessions_per_day}")
    print(f"  out={out_path}")
    print(f"  resume={args.resume} session_id={args.session_id}")

    personas_dir = PERSONAS_DIR if args.persona_set == "agent" else PERSONAS_HUMAN_DIR
    print(f"loading insider persona map ({args.persona_set} motives from {personas_dir.name}/)...")
    persona_map = build_insider_persona_map(personas_dir)
    threat_map = build_insider_threat_map()
    role_map = build_insider_role_map()
    multi_day = args.multi_day or __import__("os").environ.get("NPC_MULTIDAY", "").lower() in ("1", "true", "yes")
    benign_arcs = args.benign_arcs or __import__("os").environ.get("NPC_BENIGN_ARCS", "").lower() in ("1", "true", "yes")
    # --planning wins; else deterministic default: scripted when --multi-day,
    # otherwise 'single' (K self-contained attack-days per insider).
    planning = args.planning or ("scripted" if multi_day else "single")
    print(f"  {len(persona_map)} insider personas loaded  [planning={planning}"
          + (f", K={args.campaign_days}]" if planning in ("single", "agentic") else "]"))
    if benign_arcs:
        print("  benign multi-day arcs: ENABLED (non-insiders run multi-day benign projects)")
    # In smoke mode, only the first N personas run; report how many of those
    # are designated insiders so the user can see attack activity to expect.
    if args.smoke:
        from mole.generator.persona_loader import load_personas
        personas = load_personas(include_holdouts=True)[:n_personas]
        designated_in_smoke = [p.id for p in personas if p.id in persona_map]
        print(f"  insiders in smoke window: {len(designated_in_smoke)}: {designated_in_smoke}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    coll = asyncio.run(simulate(
        n_personas=n_personas, n_days=n_days,
        out_path=out_path, start_date=args.start_date, seed=args.seed,
        mode="agentic",
        include_holdouts=True,
        concurrency=concurrency,
        sessions_per_day=sessions_per_day,
        resume=args.resume, restore_world=args.restore_world,
        session_resume=args.session_resume,
        session_id=args.session_id,
        insider_persona_map=persona_map,
        insider_threat_map=threat_map,
        insider_role_map=role_map,
        max_turns=max_turns,
        elicit=not args.no_elicitation,
        allowed_persona_ids=persona_filter,
        attack_session_rate=args.attack_rate,
        multi_day=multi_day,
        benign_arcs=benign_arcs,
        planning=planning,
        campaign_days=args.campaign_days,
        auth_failure_limit=args.auth_failure_limit,
    ))
    print(f"done. {len(coll.events)} audit events -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
