"""Unit tests for threat 02's seed()."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mole.tasks.loader import load_task


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "02_credential_exfil"


def _load():
    spec = importlib.util.spec_from_file_location("scen02_task", TASK_DIR / "task.py")
    mod = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return mod


@pytest.fixture
def task_mod():
    return _load()


def _pl():
    return SimpleNamespace(
        create_project=AsyncMock(return_value={"id": "p-infra", "identifier": "INFRA"}),
        create_issue=AsyncMock(return_value={"id": "iss-INFRA-204"}),
    )


def test_seed_creates_infra_plane_project(task_mod):
    pl = _pl()
    counts = asyncio.run(task_mod.seed(SimpleNamespace(plane=pl)))
    assert counts["plane_projects"] == 1
    args = pl.create_project.await_args.kwargs
    assert args["identifier"] == "INFRA"


def test_seed_creates_infra_204_issue(task_mod):
    pl = _pl()
    counts = asyncio.run(task_mod.seed(SimpleNamespace(plane=pl)))
    assert counts["plane_issues"] == 1
    issue_args = pl.create_issue.await_args.kwargs
    assert "INFRA-204" in issue_args["name"]
    assert issue_args["assignee"] == "frank.s"


def test_seed_no_plane_returns_zero(task_mod):
    out = asyncio.run(task_mod.seed(SimpleNamespace()))
    assert out["plane_projects"] == 0 and out["plane_issues"] == 0


def test_loader_attaches_seed_fn():
    task = load_task(TASK_DIR)
    assert task.seed_fn is not None
