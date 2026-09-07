"""Load a task from its `task.py` module.

A valid task.py exposes:
  - METADATA: dict with id, name, category, environments (required) plus
              optional env_config / difficulty / role / tags.
  - PROMPT:   non-empty string handed to the agent at stage0.
  - stage0, stage1, ...: contiguous async functions; each returns
              `{"notification": str, "time": ISO8601}` (or similar).
  - RUBRIC:   dict mapping stage IDs (or "final") to a list of
              `{id, checker, weight, description?}` dicts.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path

from ..models import RubricEntry, TaskDefinition

logger = logging.getLogger(__name__)

_REQUIRED_METADATA = ("id", "name", "category", "environments")


def load_task(task_dir: Path | str) -> TaskDefinition:
    """Import a task.py and return a validated TaskDefinition."""
    task_dir = Path(task_dir)
    task_py = task_dir / "task.py"
    if not task_py.exists():
        raise FileNotFoundError(f"task.py not found in {task_dir}")

    # Unique module name avoids cross-task import collisions when multiple
    # tasks define classes/functions with the same names.
    module_name = f"_itb_task_{task_dir.name}_{id(task_dir)}"
    spec = importlib.util.spec_from_file_location(module_name, task_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build importlib spec for {task_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metadata = getattr(module, "METADATA", None)
    if not isinstance(metadata, dict):
        raise ValueError(f"METADATA dict required in {task_py}")
    for f in _REQUIRED_METADATA:
        if f not in metadata:
            raise ValueError(f"missing METADATA field {f!r} in {task_py}")

    prompt = getattr(module, "PROMPT", None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"non-empty PROMPT string required in {task_py}")

    # Stage functions must be contiguous (stage0, stage1, ...). Gaps are usually
    # a typo and we'd rather fail loudly than silently skip stages.
    stage_fns: list = []
    i = 0
    while True:
        fn = getattr(module, f"stage{i}", None)
        if fn is None:
            break
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(f"stage{i} must be `async def` in {task_py}")
        stage_fns.append(fn)
        i += 1
    if not stage_fns:
        raise ValueError(f"no stage functions found in {task_py}")
    # Look a few past the last contiguous one to catch gaps.
    for j in range(i, i + 10):
        if getattr(module, f"stage{j}", None) is not None:
            raise ValueError(
                f"stage{j} found but stage{i} missing — stage numbers must be "
                f"contiguous in {task_py}"
            )

    raw_rubric = getattr(module, "RUBRIC", None)
    if not isinstance(raw_rubric, dict):
        raise ValueError(f"RUBRIC dict required in {task_py}")
    stage_ids = {f"stage{k}" for k in range(len(stage_fns))}
    rubric: dict[str, list[RubricEntry]] = {}
    for key, entries in raw_rubric.items():
        if key != "final" and key not in stage_ids:
            raise ValueError(
                f"RUBRIC key {key!r} doesn't match any stage; expected one of "
                f"{sorted(stage_ids | {'final'})}"
            )
        parsed: list[RubricEntry] = []
        for e in entries:
            checker = e["checker"]
            if not asyncio.iscoroutinefunction(checker):
                raise TypeError(f"RUBRIC checker {e['id']!r} must be `async def`")
            parsed.append(RubricEntry(
                id=e["id"],
                checker=checker,
                weight=float(e.get("weight", 1.0)),
                description=e.get("description", "") or (checker.__doc__ or "").strip(),
            ))
        rubric[key] = parsed

    # Optional `async def seed(ctx)` for threat-specific world setup
    # (e.g., creating models/llama-finetune in GitLab for threat 06).
    seed_fn = getattr(module, "seed", None)
    if seed_fn is not None and not asyncio.iscoroutinefunction(seed_fn):
        raise TypeError(f"seed must be `async def` in {task_py}")

    # Optional `async def pre_snapshot(ctx) -> dict` for snapshot-diff harm
    # oracles (captured pre-agent; stored on ctx.snapshots).
    pre_snapshot_fn = getattr(module, "pre_snapshot", None)
    if pre_snapshot_fn is not None and not asyncio.iscoroutinefunction(pre_snapshot_fn):
        raise TypeError(f"pre_snapshot must be `async def` in {task_py}")

    return TaskDefinition(
        id=metadata["id"],
        name=metadata["name"],
        category=metadata["category"],
        environments=list(metadata["environments"]),
        prompt=prompt,
        stage_fns=stage_fns,
        rubric=rubric,
        env_config=metadata.get("env_config", {}),
        task_dir=task_dir,
        difficulty=metadata.get("difficulty", "medium"),
        role=metadata.get("role", ""),
        tags=list(metadata.get("tags", [])),
        timeout_seconds=int(metadata.get("timeout_seconds", 7200)),
        seed_fn=seed_fn,
        pre_snapshot_fn=pre_snapshot_fn,
    )


def discover_task_dirs(tasks_root: Path | str) -> list[Path]:
    """Return every immediate subdir of `tasks_root` that contains a task.py."""
    root = Path(tasks_root)
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (d / "task.py").exists()
    )
