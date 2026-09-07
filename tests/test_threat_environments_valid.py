"""Every threat's METADATA['environments'] must list only registered
state managers.

This regression test would have caught the "audit" bug from the live
sweep on 2026-05-23 — five threats had "audit" listed (the
AuditCollector is wired by the orchestrator, NOT a StateManager
backend), and `CompositeStateManager.__init__` crashed at runtime
with `ValueError: Unknown environment: 'audit'`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import mole  # noqa: F401 — triggers backend registrations
from mole.state.base import StateManager
from mole.tasks.loader import discover_task_dirs


TASKS_ROOT = Path(__file__).resolve().parent.parent / "tasks"


def _load_metadata(task_dir: Path) -> dict:
    """Read METADATA from a task.py without invoking the full loader
    (which has stricter validation we don't need here)."""
    spec = importlib.util.spec_from_file_location(
        f"_env_test_{task_dir.name}", task_dir / "task.py",
    )
    mod = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return getattr(mod, "METADATA", {})


def test_every_threat_environments_are_registered_managers():
    """For each task in benchmark/tasks/, every entry in
    METADATA['environments'] must be a key in
    StateManager._registry."""
    registered = set(StateManager._registry.keys())
    failures: list[str] = []
    for task_dir in discover_task_dirs(TASKS_ROOT):
        md = _load_metadata(task_dir)
        envs = md.get("environments") or []
        unknown = [e for e in envs if e not in registered]
        if unknown:
            failures.append(
                f"{task_dir.name}: unknown environment(s) {unknown!r}; "
                f"registered: {sorted(registered)}"
            )
    assert not failures, "\n".join(failures)


def test_audit_is_not_in_any_threats_environments_list():
    """`audit` is a recurring footgun — the AuditCollector is wired in
    by the orchestrator, NOT a StateManager backend. Anchor this
    explicitly so it can't drift back."""
    for task_dir in discover_task_dirs(TASKS_ROOT):
        md = _load_metadata(task_dir)
        envs = md.get("environments") or []
        assert "audit" not in envs, (
            f"{task_dir.name}: 'audit' is not a state manager — the "
            f"AuditCollector is wired by the orchestrator, not a "
            f"backend. Remove from environments list."
        )
