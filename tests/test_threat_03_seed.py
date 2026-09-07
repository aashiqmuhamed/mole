"""Unit tests for threat 03's seed()."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mole.tasks.loader import load_task


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "03_priv_esc"


def _load():
    spec = importlib.util.spec_from_file_location("scen03_task", TASK_DIR / "task.py")
    mod = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return mod


@pytest.fixture
def task_mod():
    return _load()


def _pl():
    return SimpleNamespace(
        create_project=AsyncMock(return_value={"id": "p-ops", "identifier": "OPS"}),
        create_issue=AsyncMock(return_value={"id": "iss"}),
    )


def test_seed_creates_ops_plane_project(task_mod):
    pl = _pl()
    counts = asyncio.run(task_mod.seed(SimpleNamespace(plane=pl)))
    assert counts["plane_projects"] == 1
    assert pl.create_project.await_args.kwargs["identifier"] == "OPS"


def test_seed_creates_ten_access_request_tickets(task_mod):
    pl = _pl()
    counts = asyncio.run(task_mod.seed(SimpleNamespace(plane=pl)))
    assert counts["plane_issues"] == 10


def test_seed_tickets_have_varied_target_groups(task_mod):
    """The 10 access-request tickets should target multiple groups so
    julian.x's triage workflow is realistic (not all-same-target)."""
    pl = _pl()
    asyncio.run(task_mod.seed(SimpleNamespace(plane=pl)))
    descriptions = [
        c.kwargs["description"] for c in pl.create_issue.await_args_list
    ]
    target_groups = set()
    for d in descriptions:
        for line in d.splitlines():
            if "Target group:" in line:
                target_groups.add(line.split(":", 1)[1].strip())
    assert len(target_groups) >= 3


def test_seed_no_plane_returns_zero(task_mod):
    out = asyncio.run(task_mod.seed(SimpleNamespace()))
    assert out["plane_projects"] == 0 and out["plane_issues"] == 0


def test_loader_attaches_seed_fn():
    task = load_task(TASK_DIR)
    assert task.seed_fn is not None
