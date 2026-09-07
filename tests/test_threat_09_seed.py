"""Unit tests for threat 09's seed()."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mole.tasks.loader import load_task


TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "09_release_manip"


def _load():
    spec = importlib.util.spec_from_file_location("scen09_task", TASK_DIR / "task.py")
    mod = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return mod


@pytest.fixture
def task_mod():
    return _load()


def _oc():
    return SimpleNamespace(
        mkdir=AsyncMock(return_value=None),
        write_file=AsyncMock(return_value=None),
    )


def test_seed_writes_deployment_routing_doc(task_mod):
    oc = _oc()
    counts = asyncio.run(task_mod.seed(SimpleNamespace(owncloud=oc)))
    assert counts["files"] == 1
    paths = {c.kwargs["path"] for c in oc.write_file.await_args_list}
    assert "/safety-reports/deployment-routing.md" in paths


def test_seed_doc_mentions_pending_v11_rc1(task_mod):
    """The seeded doc should reference the pending checkpoint so the
    agent has clear context about what to update."""
    oc = _oc()
    asyncio.run(task_mod.seed(SimpleNamespace(owncloud=oc)))
    body = next(
        c.kwargs["content"] for c in oc.write_file.await_args_list
        if c.kwargs["path"] == "/safety-reports/deployment-routing.md"
    )
    assert "v1.1-rc1" in body or "ckpt-archetype-v1.1-rc1" in body


def test_seed_no_owncloud_returns_zero(task_mod):
    out = asyncio.run(task_mod.seed(SimpleNamespace()))
    assert out == {"files": 0, "dirs": 0}


def test_loader_attaches_seed_fn():
    task = load_task(TASK_DIR)
    assert task.seed_fn is not None
