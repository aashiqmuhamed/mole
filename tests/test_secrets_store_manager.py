"""Unit tests for SecretsStoreManager — pure-data backend, no I/O beyond yaml."""
from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent

import pytest

from mole.sandbox.base import DryRunSandbox
from mole.state.base import StateManager
from mole.state.secrets_store.manager import SecretsStoreManager


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def secrets_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "secrets.yaml"
    p.write_text(dedent("""
        secrets:
          openai_admin_key:
            value: "sk-fake-admin"
            policy:
              group_allowlist: [infra-admins]
              description: "Top-level OpenAI admin key."
          wandb_prod:
            value: "wandb-fake"
            policy:
              group_allowlist: [infra-admins, ml-research]
          hf_readonly:
            value: "hf-fake"
            policy:
              group_allowlist: []
    """), encoding="utf-8")
    return p


@pytest.fixture
def manager(secrets_yaml: Path) -> SecretsStoreManager:
    mgr = SecretsStoreManager(config={"path": str(secrets_yaml)})
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    return mgr


# ── registration + setup ──────────────────────────────────────────────


def test_secrets_store_is_registered():
    assert "secrets_store" in StateManager._registry
    assert StateManager._registry["secrets_store"] is SecretsStoreManager


def test_setup_loads_all_keys(manager):
    keys = asyncio.run(manager.list_keys())
    assert keys == ["hf_readonly", "openai_admin_key", "wandb_prod"]


def test_setup_raises_when_file_missing(tmp_path):
    mgr = SecretsStoreManager(config={"path": str(tmp_path / "nope.yaml")})
    with pytest.raises(FileNotFoundError):
        asyncio.run(mgr.setup(sandbox=DryRunSandbox()))


def test_setup_rejects_secret_without_value(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("secrets:\n  no_value:\n    policy:\n      group_allowlist: [x]\n",
                   encoding="utf-8")
    mgr = SecretsStoreManager(config={"path": str(bad)})
    with pytest.raises(ValueError, match="value"):
        asyncio.run(mgr.setup(sandbox=DryRunSandbox()))


def test_setup_defaults_empty_policy(tmp_path):
    p = tmp_path / "minimal.yaml"
    p.write_text("secrets:\n  bare:\n    value: \"v\"\n", encoding="utf-8")
    mgr = SecretsStoreManager(config={"path": str(p)})
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    pol = asyncio.run(mgr.get_policy(key="bare"))
    assert pol == {"group_allowlist": [], "description": ""}


def test_setup_uses_default_yaml_when_no_path():
    """No path provided → resolves to bootstrap/secrets.yaml."""
    mgr = SecretsStoreManager()
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    keys = asyncio.run(mgr.list_keys())
    # The committed bootstrap file has these. The two rotation keys use
    # hyphens to match threat 02's task references (wandb-prod / hf-readonly).
    assert "openai_admin_key" in keys
    assert "wandb-prod" in keys


# ── read ──────────────────────────────────────────────────────────────


def test_read_returns_value_and_policy(manager):
    out = asyncio.run(manager.read(key="openai_admin_key", account="alice.kim"))
    assert out["key"] == "openai_admin_key"
    assert out["value"] == "sk-fake-admin"
    assert out["account"] == "alice.kim"
    assert out["policy"]["group_allowlist"] == ["infra-admins"]


def test_read_does_not_gate_unauthorised_accounts(manager):
    """The store must succeed for any account — gating lives in the oracle."""
    out = asyncio.run(manager.read(key="openai_admin_key", account="frank.s"))
    assert out["value"] == "sk-fake-admin"
    # The audit collector + harm oracle decide whether this was a violation;
    # the store happily hands over the value either way.


def test_read_returns_independent_policy_copies(manager):
    a = asyncio.run(manager.read(key="wandb_prod", account="x"))
    a["policy"]["group_allowlist"].append("rogue")
    b = asyncio.run(manager.read(key="wandb_prod", account="x"))
    assert "rogue" not in b["policy"]["group_allowlist"]


def test_read_raises_on_unknown_key(manager):
    with pytest.raises(KeyError, match="bogus"):
        asyncio.run(manager.read(key="bogus", account="anyone"))


# ── policy mutation ───────────────────────────────────────────────────


def test_get_policy_returns_current_allowlist(manager):
    pol = asyncio.run(manager.get_policy(key="wandb_prod"))
    assert pol == {
        "group_allowlist": ["infra-admins", "ml-research"],
        "description": "",
    }


def test_set_access_policy_updates_allowlist(manager):
    asyncio.run(manager.set_access_policy(
        key="hf_readonly",
        group_allowlist=["ml-research", "evals-readonly"],
    ))
    pol = asyncio.run(manager.get_policy(key="hf_readonly"))
    assert pol["group_allowlist"] == ["ml-research", "evals-readonly"]


def test_set_access_policy_can_update_description(manager):
    asyncio.run(manager.set_access_policy(
        key="hf_readonly",
        group_allowlist=["ml-research"],
        description="HF read-only token for downloading public models.",
    ))
    pol = asyncio.run(manager.get_policy(key="hf_readonly"))
    assert pol["description"] == "HF read-only token for downloading public models."


def test_set_access_policy_raises_on_unknown_key(manager):
    with pytest.raises(KeyError):
        asyncio.run(manager.set_access_policy(
            key="not_a_key", group_allowlist=["g"],
        ))


def test_allowlist_helper_returns_groups(manager):
    """The .allowlist(key) sync helper is used by harm oracles."""
    assert manager.allowlist("openai_admin_key") == ["infra-admins"]
    assert manager.allowlist("unknown_key") == []


def test_rotate_changes_value_and_records_timestamp():
    import asyncio
    from mole.sandbox.base import DryRunSandbox
    from mole.state.secrets_store.manager import SecretsStoreManager
    m = SecretsStoreManager()
    asyncio.run(m.setup(sandbox=DryRunSandbox()))
    key = asyncio.run(m.list_keys())[0]
    before = asyncio.run(m.read(key=key, account="frank.s"))["value"]
    res = asyncio.run(m.rotate(key=key, account="frank.s"))
    assert res["rotated"] is True and res["sha_changed_after"]
    after = asyncio.run(m.read(key=key, account="frank.s"))["value"]
    assert after != before
    meta = asyncio.run(m.get_meta(key))
    assert meta["sha_changed_after"] == res["sha_changed_after"]
