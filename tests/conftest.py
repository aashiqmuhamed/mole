"""Pytest configuration shared across the test suite."""
from __future__ import annotations

import itertools
import os
import re
import shutil
from pathlib import Path

import pytest


pytest_plugins = []   # placeholder; populate as needed


@pytest.fixture(autouse=True)
def _no_judge_cache(monkeypatch):
    """Keep the LLM-judge score cache OFF during tests so they never read/write the real
    repo cache (experiments/judge_cache/). Tests that exercise the cache set the env
    themselves to a tmp path, overriding this."""
    monkeypatch.setenv("JUDGE_SCORE_CACHE", "off")
    yield


@pytest.fixture(autouse=True)
def _reset_sim_clock():
    """Isolate the per-task simulated clock (a ContextVar in audit.collector) between
    tests. A test that sets it via set_task_clock() but runs code that doesn't reset the
    token can otherwise leak the clock into later tests that assert the no-clock default."""
    yield
    from mole.audit.collector import _sim_clock_var
    _sim_clock_var.set(None)

_TMP_COUNTER = itertools.count()


@pytest.fixture
def tmp_path(request) -> Path:
    """Workspace-local replacement for pytest's tmp_path fixture.

    In this Windows sandbox, pytest's built-in TempPathFactory creates
    `pytest-of-<user>` directories with an ACL that the same sandboxed
    process cannot read back, so tests error during fixture setup. Keep test
    temp dirs under the writable repository and create them with ordinary
    inherited ACLs instead.
    """
    root = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)
    path = root / f"{next(_TMP_COUNTER):04d}_{safe_name[:80]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def pytest_collection_modifyitems(config, items):
    """Skip the empty_task fixture directory — it's a task module, not a test module."""
    skip_marker = pytest.mark.skip(reason="fixture, not a test module")
    for item in items:
        if "empty_task" in item.nodeid:
            item.add_marker(skip_marker)
