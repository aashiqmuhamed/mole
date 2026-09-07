"""Unit tests for threat 07's seed() (no-op)."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from mole.tasks.loader import load_task


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "07_eval_manip"


def _load():
    spec = importlib.util.spec_from_file_location("scen07_task", TASK_DIR / "task.py")
    mod = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return mod


@pytest.fixture
def task_mod():
    return _load()


def test_seed_is_a_noop_returning_empty_dict(task_mod):
    """eval_server + model_registry + rocketchat are all set up by
    their respective manager.setup() / seed_org common-base, so
    threat 07's per-threat seed is intentionally empty."""
    out = asyncio.run(task_mod.seed(SimpleNamespace()))
    assert out == {}


def test_loader_attaches_seed_fn():
    task = load_task(TASK_DIR)
    assert task.seed_fn is not None
