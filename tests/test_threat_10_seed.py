"""Unit tests for threat 10's seed()."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from mole.tasks.loader import load_task


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "10_safety_sabotage"


def _load():
    spec = importlib.util.spec_from_file_location("scen10_task", TASK_DIR / "task.py")
    mod = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return mod


@pytest.fixture
def task_mod():
    return _load()


def _gl():
    return SimpleNamespace(
        create_group=AsyncMock(return_value={"id": 1}),
        create_project=AsyncMock(return_value={"id": 2}),
        commit=AsyncMock(return_value={"id": "abc"}),
        add_group_member=AsyncMock(return_value={"id": 3}),
    )


def test_seed_creates_infra_group_and_training_pipeline_project(task_mod):
    gl = _gl()
    counts = asyncio.run(task_mod.seed(SimpleNamespace(gitlab=gl)))
    assert counts["projects"] == 1
    grp_paths = {c.kwargs["path"] for c in gl.create_group.await_args_list}
    assert "infra" in grp_paths
    proj_calls = {(c.kwargs["namespace"], c.kwargs["path"])
                  for c in gl.create_project.await_args_list}
    assert ("infra", "training-pipeline") in proj_calls


def test_seed_commits_monitors_config_with_canonical_values(task_mod):
    gl = _gl()
    asyncio.run(task_mod.seed(SimpleNamespace(gitlab=gl)))
    # Find the monitors/config.yaml commit
    cfg_call = next(
        c for c in gl.commit.await_args_list
        if c.kwargs["path"] == "monitors/config.yaml"
    )
    parsed = yaml.safe_load(cfg_call.kwargs["content"])
    canon = task_mod.METADATA["canonical_monitor_config"]
    # Round-trip: committed yaml must match canonical exactly
    assert parsed == canon


def test_seed_adds_frank_s_and_alice_kim_to_infra(task_mod):
    gl = _gl()
    asyncio.run(task_mod.seed(SimpleNamespace(gitlab=gl)))
    members = {(c.kwargs["username"], c.kwargs["access_level"])
               for c in gl.add_group_member.await_args_list}
    assert ("frank.s", 30) in members
    assert ("alice.kim", 40) in members


def test_seed_no_gitlab_returns_zero(task_mod):
    counts = asyncio.run(task_mod.seed(SimpleNamespace()))
    assert counts == {"projects": 0, "files": 0}


def test_loader_attaches_seed_fn():
    task = load_task(TASK_DIR)
    assert task.seed_fn is not None
