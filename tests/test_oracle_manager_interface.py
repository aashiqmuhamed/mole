"""Every manager method an oracle/seed calls must actually exist.

This is the static half of the per-threat integration smoke (#62). It
would have caught the 2026-05-24 interface-drift bugs where oracles called
methods the managers don't have — `rocketchat.read_channel` (real name:
channel_history), `eval_server.recent_logs` (didn't exist) — which raised
AttributeError at runtime, got swallowed by try/except, and silently
zeroed utility. The unit-test fakes mocked the same wrong names, so they
never caught it; a check against the REAL manager classes does.

Approach: AST-walk each task.py. Within each function, track local
variables bound from `getattr(ctx, "<env>", ...)` or `ctx.<env>`, then
verify every `var.method(...)` and `ctx.<env>.method(...)` call names a
method that exists on the registered manager class for that env.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mole  # noqa: F401 — triggers backend registration
from mole.state.base import StateManager

TASKS_ROOT = Path(__file__).resolve().parent.parent / "tasks"

# ctx attributes that are NOT state managers (wired by the orchestrator).
_NON_MANAGER_CTX_ATTRS = {"audit", "snapshots", "task_metadata", "sandbox"}


def _manager_attrs(env: str) -> set[str] | None:
    cls = StateManager._registry.get(env)
    if cls is None:
        return None
    return {n for n in dir(cls) if not n.startswith("_")}


def _threat_dirs() -> list[Path]:
    return [
        d for d in sorted(TASKS_ROOT.iterdir())
        if d.is_dir() and not d.name.startswith(".")
        and not d.name.startswith("_") and d.name != "00_persona_lookup"
        and (d / "task.py").exists()
    ]


def _env_of_value(node: ast.AST) -> str | None:
    """Return the env name if `node` is `getattr(ctx, "env", ...)` or
    `ctx.env`, else None."""
    # getattr(ctx, "env", default)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name) and node.args[0].id == "ctx"
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[1].value
    # ctx.env
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id == "ctx"
    ):
        return node.attr
    return None


def _check_function(fn: ast.FunctionDef, failures: list[str], threat: str) -> None:
    # Map local var -> env, from assignments in this function.
    var_env: dict[str, str] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            env = _env_of_value(node.value)
            if env and env not in _NON_MANAGER_CTX_ATTRS:
                var_env[node.targets[0].id] = env

    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        method = node.func.attr
        target = node.func.value
        env: str | None = None
        # var.method(...)
        if isinstance(target, ast.Name) and target.id in var_env:
            env = var_env[target.id]
        # ctx.env.method(...)
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                and target.value.id == "ctx":
            cand = target.attr
            if cand not in _NON_MANAGER_CTX_ATTRS and cand in StateManager._registry:
                env = cand
        if env is None:
            continue
        attrs = _manager_attrs(env)
        if attrs is None:
            continue
        if method not in attrs:
            failures.append(
                f"{threat}: {env}.{method}() called (line {node.lineno}) but "
                f"{StateManager._registry[env].__name__} has no such method"
            )


@pytest.mark.parametrize("task_dir", _threat_dirs(), ids=lambda d: d.name)
def test_oracle_manager_methods_exist(task_dir: Path):
    tree = ast.parse((task_dir / "task.py").read_text(encoding="utf-8"))
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function(node, failures, task_dir.name)
    assert not failures, "\n".join(failures)
