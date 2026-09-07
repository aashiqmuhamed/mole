"""Core data models — task definitions, rubric entries, evaluation results."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class GradingMethod(str, Enum):
    RULE = "rule"
    LLM_JUDGE = "llm_judge"


@dataclass
class RubricEntry:
    """One checker in a task's rubric. The checker is an async function taking
    a TaskContext and returning a truthy/falsy value."""
    id: str
    checker: Callable          # async def check(ctx) -> bool
    weight: float = 1.0
    description: str = ""      # auto-filled from checker.__doc__ if blank


@dataclass
class TaskDefinition:
    """Validated contents of a task.py module."""
    id: str
    name: str
    category: str
    environments: list[str]
    prompt: str
    stage_fns: list[Callable]                          # [stage0, stage1, ...]
    rubric: dict[str, list[RubricEntry]]               # "stage0", "stage1", "final"
    env_config: dict[str, dict[str, Any]]
    task_dir: Path
    difficulty: str = "medium"
    role: str = ""
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 7200
    # Optional per-threat seed function. If defined in task.py as
    #   async def seed(ctx): ...
    # the orchestrator runs it after composite.setup + common seed_org,
    # BEFORE the audit-wrap, so seed-time API calls don't appear in
    # audit.jsonl. Default None = no threat-specific seed.
    seed_fn: Callable | None = None
    # Optional per-threat pre-episode snapshot hook. If defined in task.py as
    #   async def pre_snapshot(ctx): return {...}
    # the orchestrator calls it after seeding but BEFORE the agent runs
    # (using raw, un-audited managers so the reads don't pollute audit.jsonl),
    # and stores the returned dict on `ctx.snapshots`. Snapshot-diff harm
    # oracles (e.g. retroactive-edit, data-destruction) read it to compare
    # post-agent state against this pre-agent baseline. Default None.
    pre_snapshot_fn: Callable | None = None


@dataclass
class RubricItemResult:
    item_id: str
    passed: bool
    weight: float
    method: GradingMethod = GradingMethod.RULE
    detail: str = ""


@dataclass
class EvaluationResult:
    items: list[RubricItemResult] = field(default_factory=list)
    score: float = 0.0          # weighted pass-rate in [0, 1]


@dataclass
class StageResult:
    stage_id: str
    success: bool = True
    error: str = ""
    verification: list[RubricItemResult] = field(default_factory=list)
    verification_score: float = -1.0


@dataclass
class TaskResult:
    task_id: str
    stage_results: list[StageResult] = field(default_factory=list)
    rubric_results: list[RubricItemResult] = field(default_factory=list)
    score: float = 0.0
    execution_time_s: float = 0.0
    error: str = ""
