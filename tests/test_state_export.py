"""Tests for sandbox/state_export.py — day-boundary in-process state dump."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from mole.sandbox.state_export import (
    export_manager_state,
    import_manager_state,
)
from mole.state.composite import CompositeStateManager
from mole.sandbox.base import DryRunSandbox


def test_export_captures_known_manager_fields(tmp_path):
    """A composite context with the in-process managers should produce a
    JSON file containing at least the known state fields of each manager
    that is loaded."""
    async def _build():
        # In-process subset only — no containers needed for this test.
        comp = CompositeStateManager(environments=[
            "org", "model_registry", "eval_server", "secrets_store", "plane",
        ])
        await comp.setup(sandbox=DryRunSandbox())
        ctx = comp.create_context(task_dir=tmp_path, sandbox=DryRunSandbox())
        return ctx

    ctx = asyncio.run(_build())
    out_path = tmp_path / "managers.json"
    exported = export_manager_state(ctx, out_path)
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert exported == on_disk

    # Each in-process manager should have contributed at least one field.
    for name in ("org", "model_registry", "eval_server", "secrets_store", "plane"):
        assert name in on_disk, f"{name} missing from state dump"
        assert on_disk[name], f"{name} dumped empty"


def test_export_is_robust_to_missing_managers(tmp_path):
    """If the ctx only has some managers (e.g., minimal test env), the
    export should silently skip the missing ones rather than raise."""
    fake_ctx = SimpleNamespace(_managers={})
    out_path = tmp_path / "managers.json"
    exported = export_manager_state(fake_ctx, out_path)
    assert exported == {}
    assert out_path.exists()


def test_export_does_not_serialize_unknown_fields(tmp_path):
    """We whitelist `_STATE_FIELDS` so the export is stable even as
    managers grow internal caches we don't want to leak (e.g., http clients,
    LLM callbacks). Verify by passing a manager with an extra non-listed
    attribute and confirming it's NOT in the dump."""
    class FakeOrg:
        _doc = {"company": "agentlab"}
        _accounts_by_id = {"alice.kim": {"id": "alice.kim"}}
        _groups_by_id = {}
        _policies_by_action = {}
        _internal_http_client = "should-not-leak"   # NOT in _STATE_FIELDS

    ctx = SimpleNamespace(_managers={"org": FakeOrg()})
    out_path = tmp_path / "managers.json"
    exported = export_manager_state(ctx, out_path)
    assert "org" in exported
    assert "_internal_http_client" not in exported["org"]
    assert exported["org"]["_doc"] == {"company": "agentlab"}


def test_import_round_trips_mutated_state(tmp_path):
    """export then import onto fresh day-0 managers restores the mutated state
    (the --restore-world resume path). Symmetric inverse of export."""
    class Mgr:
        pass

    src = Mgr()
    src._checkpoints = {"ckpt-9": {"approved": False}}
    src._deployments = [{"model": "x"}]
    src._required_approvers = 2
    src_secrets = Mgr()
    src_secrets._secrets = {"api-key-prod": {"rotated": True}}
    ctx = SimpleNamespace(_managers={"model_registry": src, "secrets_store": src_secrets})
    mj = tmp_path / "managers.json"
    export_manager_state(ctx, mj)

    # fresh, empty day-0 managers
    dst = Mgr()
    dst._checkpoints, dst._deployments, dst._required_approvers = {}, [], 1
    dst_secrets = Mgr()
    dst_secrets._secrets = {}
    ctx2 = SimpleNamespace(
        _managers={"model_registry": dst, "secrets_store": dst_secrets})
    applied = import_manager_state(ctx2, mj)

    assert dst._checkpoints == {"ckpt-9": {"approved": False}}
    assert dst._required_approvers == 2
    assert dst_secrets._secrets == {"api-key-prod": {"rotated": True}}
    assert set(applied) == {"model_registry", "secrets_store"}


def test_import_missing_file_is_noop(tmp_path):
    """A resume pointed at a missing managers.json must not raise."""
    ctx = SimpleNamespace(_managers={})
    assert import_manager_state(ctx, tmp_path / "nope.json") == {}
