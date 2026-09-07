"""Wrapper runner — `python -m mole.run_task --task <path>`.

This entry point exists so the package is imported (and therefore every
state-backend decorator runs) before any task module is loaded. Loading a
task and then instantiating the state-manager pulls names from the registry,
so the registry must be populated first.

Usage:
    python -m mole.run_task --task ./tasks/06_weight_exfil
    python -m mole.run_task --task ./tests/empty_task --skeleton-only
"""
from __future__ import annotations

# Side-effect import: every backend's @register decorator runs here.
import mole  # noqa: F401

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .llm.factory import build_llm
from .orchestrator import run_task_full
from .state.base import StateManager
from .tasks.loader import load_task

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_lab_compose() -> Path:
    """The lab compose to bring up. Override with $LAB_COMPOSE (absolute path or
    a name under benchmark/compose/) — used for fast smoke/validation runs against
    a reduced service set (e.g. lab.fast.yaml: owncloud only) instead of the full
    GitLab-bearing lab.yaml whose multi-GB seed/snapshot dominates wall time."""
    override = os.environ.get("LAB_COMPOSE")
    if override:
        p = Path(override)
        return p if p.is_absolute() else _repo_root() / "benchmark" / "compose" / override
    return _repo_root() / "benchmark" / "compose" / "lab.yaml"


async def _run_skeleton(task_dir: Path) -> int:
    """Load task + verify backend coverage. No sandbox, no agent run."""
    registered = sorted(StateManager._registry.keys())
    logger.info("registered backends: %s", ", ".join(registered) or "(none)")

    task = load_task(task_dir)
    logger.info("loaded task: %s (%s)", task.name, task.id)
    logger.info("environments requested: %s", task.environments)
    logger.info("stages: %d", len(task.stage_fns))
    logger.info("rubric keys: %s", list(task.rubric.keys()))

    missing = [e for e in task.environments if e not in StateManager._registry]
    if missing:
        logger.error(
            "task requests environments not registered: %s. available: %s",
            missing, registered,
        )
        return 3
    logger.info("skeleton check passed.")
    return 0


async def _run_full(args: argparse.Namespace) -> int:
    task_dir = Path(args.task).resolve()
    llm = build_llm(args.llm_backend)
    logger.info("LLM backend: %s, model_id=%s", llm.backend, llm.model_id)

    result = await run_task_full(
        task_dir=task_dir,
        llm=llm,
        lab_compose=Path(args.lab_compose) if args.lab_compose else None,
        max_turns_per_stage=args.max_turns,
        temperature=args.temperature,
        protocol=args.protocol,
        results_dir=args.results_dir,
    )

    # Persist the result JSON for downstream processing.
    out_dir = Path(args.results_dir) / result.task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "result.json"
    out_file.write_text(json.dumps({
        "task_id": result.task_id,
        "score": result.score,
        "execution_time_s": result.execution_time_s,
        "error": result.error,
        "stages": [
            {
                "id": s.stage_id, "success": s.success, "error": s.error,
                "verification_score": s.verification_score,
                "verification": [
                    {"id": v.item_id, "passed": v.passed, "weight": v.weight,
                     "detail": v.detail, "method": v.method.value}
                    for v in s.verification
                ],
            } for s in result.stage_results
        ],
        "rubric": [asdict(r) | {"method": r.method.value} for r in result.rubric_results],
    }, indent=2))
    logger.info("wrote result → %s", out_file)
    return 0 if not result.error else 4


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="mole",
        description="Run an insider-threat benchmark task.",
    )
    parser.add_argument("--task", required=True, help="path to task directory containing task.py")
    parser.add_argument("--skeleton-only", action="store_true",
                        help="verify task loads + backends register; no sandbox or agent run")
    parser.add_argument("--lab-compose", default=str(_default_lab_compose()),
                        help="path to the lab compose file (used only if the task declares envs)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--llm-backend", default=None,
                        help="LLM backend name (openrouter, vllm, anthropic); defaults to $LLM_BACKEND (openrouter)")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="max LLM turns per stage (default 30)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="LLM sampling temperature (default 0.7)")
    parser.add_argument("--protocol", default=None,
                        help="gating protocol (no_control, alert_only, metadata_gate, "
                             "budgeted_semantic_gate, full_semantic_gate); wraps the "
                             "dispatcher so risky tool calls are blocked")

    args = parser.parse_args()
    task_dir = Path(args.task).resolve()
    if not task_dir.exists():
        logger.error("task dir not found: %s", task_dir)
        return 2

    if args.skeleton_only:
        return asyncio.run(_run_skeleton(task_dir))
    return asyncio.run(_run_full(args))


if __name__ == "__main__":
    sys.exit(main())
