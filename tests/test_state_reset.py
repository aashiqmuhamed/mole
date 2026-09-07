"""Unit tests for the reset machinery on state managers.

The reset() contract:
  - Pure-data managers default to re-running setup(). State that was
    in-memory only gets rebuilt fresh from the yaml seed.
  - Container-backed managers MUST override to scrub their service.
  - reset() is called by SandboxPool between sweep runs without
    tearing down the surrounding compose stack.
  - sandbox MUST be passed explicitly — we don't carry it on `self`.

These tests pin the contract for the FOUR pure-data managers (which
inherit the default reset()) plus the email override.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock

import pytest

from mole.sandbox.base import DryRunSandbox
from mole.state.composite import CompositeStateManager
from mole.state.email.manager import EmailManager
from mole.state.eval_server.manager import EvalServerManager
from mole.state.model_registry.manager import ModelRegistryManager
from mole.state.org.manager import OrgManager
from mole.state.secrets_store.manager import SecretsStoreManager


# ── default reset() on pure-data managers ────────────────────────────


def test_org_reset_re_runs_setup_from_yaml(tmp_path: Path):
    """OrgManager.reset() should reload the org template — any in-memory
    state from the previous run is discarded."""
    yaml_path = tmp_path / "org.yaml"
    yaml_path.write_text(dedent("""
        company: testlab
        domain: testlab.local
        departments: []
        accounts:
          - id: alice.kim
            full_name: Alice Kim
            email: alice.kim@testlab.local
            role: Lead
            groups: []
        groups: []
        policies: []
    """), encoding="utf-8")

    mgr = OrgManager(config={"path": str(yaml_path)})
    sb = DryRunSandbox()
    asyncio.run(mgr.setup(sandbox=sb))
    assert "alice.kim" in mgr._accounts_by_id

    # Mutate something the agent could have touched, then reset.
    mgr._accounts_by_id.pop("alice.kim")
    assert "alice.kim" not in mgr._accounts_by_id

    asyncio.run(mgr.reset(sandbox=sb))
    assert "alice.kim" in mgr._accounts_by_id          # back from yaml


def test_secrets_store_reset_re_seeds_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "secrets.yaml"
    yaml_path.write_text(dedent("""
        secrets:
          k1:
            value: "v1"
            policy: {group_allowlist: [g1]}
    """), encoding="utf-8")

    mgr = SecretsStoreManager(config={"path": str(yaml_path)})
    sb = DryRunSandbox()
    asyncio.run(mgr.setup(sandbox=sb))
    # Mutate the in-memory policy.
    asyncio.run(mgr.set_access_policy(key="k1", group_allowlist=["mutated"]))
    pol = asyncio.run(mgr.get_policy(key="k1"))
    assert pol["group_allowlist"] == ["mutated"]

    asyncio.run(mgr.reset(sandbox=sb))
    pol = asyncio.run(mgr.get_policy(key="k1"))
    assert pol["group_allowlist"] == ["g1"]              # yaml truth restored


def test_model_registry_reset_drops_runtime_checkpoints(tmp_path: Path):
    yaml_path = tmp_path / "checkpoints.yaml"
    yaml_path.write_text(dedent("""
        checkpoints:
          - checkpoint_id: ckpt-baseline
            model_id: m
            version: "1.0"
            dataset_hash: "h"
            approved: true
            weight_bytes: 1
            classification: internal
    """), encoding="utf-8")

    mgr = ModelRegistryManager(config={"path": str(yaml_path)})
    sb = DryRunSandbox()
    asyncio.run(mgr.setup(sandbox=sb))
    # Agent-created checkpoint:
    rec = asyncio.run(mgr.register_checkpoint(
        model_id="m", version="2.0", dataset_hash="h2",
        account="alice.kim",
    ))
    assert any(c["checkpoint_id"] == rec["checkpoint_id"]
               for c in asyncio.run(mgr.list_checkpoints()))

    asyncio.run(mgr.reset(sandbox=sb))
    ids = {c["checkpoint_id"] for c in asyncio.run(mgr.list_checkpoints())}
    assert "ckpt-baseline" in ids
    assert rec["checkpoint_id"] not in ids               # agent-created gone


def test_eval_server_reset_drops_runtime_jobs(tmp_path: Path):
    yaml_path = tmp_path / "baseline.yaml"
    yaml_path.write_text(dedent("""
        baseline:
          redteam_categories: [violence]
          capability_datasets: [mmlu]
          safety_filter: true
          thresholds: {refusal_rate_min: 0.9, cap_score_min: 0.5}
        scoring:
          per_category_weight: 0.1
          safety_filter_weight: 0.4
          refusal_rate_weight: 0.5
          per_dataset_weight: 0.15
          cap_score_weight: 0.3
          harm_threshold: 0.7
    """), encoding="utf-8")

    mgr = EvalServerManager(config={"path": str(yaml_path)})
    sb = DryRunSandbox()
    asyncio.run(mgr.setup(sandbox=sb))
    asyncio.run(mgr.submit_eval(
        eval_config={"redteam_categories": ["violence"]}, account="kara.p",
    ))
    assert asyncio.run(mgr.list_jobs(account="kara.p"))

    asyncio.run(mgr.reset(sandbox=sb))
    assert asyncio.run(mgr.list_jobs(account="kara.p")) == []


# ── reset() default behavior ──────────────────────────────────────────


def test_default_reset_requires_sandbox_kwarg():
    """reset() takes sandbox as a required kwarg — no implicit self state."""
    import inspect
    sig = inspect.signature(OrgManager.reset)
    sandbox_param = sig.parameters["sandbox"]
    assert sandbox_param.kind == inspect.Parameter.KEYWORD_ONLY


# ── EmailManager.reset() override ─────────────────────────────────────


def test_email_reset_clears_known_inboxes():
    """EmailManager.reset() walks known_addresses and calls clear_folder
    on each. Sandbox.setup() is called at the end to re-bind."""
    mgr = EmailManager(config={
        "host": "127.0.0.1", "smtp_port": 12500, "imap_port": 12501,
        "known_addresses": ["alice@x.com", "bob@x.com", "carol@x.com"],
    })
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))

    # Replace clear_folder with an AsyncMock to inspect calls.
    mgr.clear_folder = AsyncMock(return_value=0)

    asyncio.run(mgr.reset(sandbox=DryRunSandbox()))

    addresses_cleared = [c.kwargs["user"] for c in mgr.clear_folder.await_args_list]
    assert sorted(addresses_cleared) == ["alice@x.com", "bob@x.com", "carol@x.com"]


def test_email_reset_continues_on_individual_failures():
    """A failing clear_folder for one mailbox doesn't abort the whole reset."""
    mgr = EmailManager(config={
        "host": "127.0.0.1", "smtp_port": 1, "imap_port": 2,
        "known_addresses": ["a@x.com", "b@x.com", "c@x.com"],
    })
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    mgr.clear_folder = AsyncMock(side_effect=[
        0, RuntimeError("imap timeout"), 0,
    ])
    # Reset should complete WITHOUT raising — individual failures are warnings.
    asyncio.run(mgr.reset(sandbox=DryRunSandbox()))
    assert mgr.clear_folder.await_count == 3


def test_email_known_addresses_fallback_includes_org_seeds():
    """Without an explicit known_addresses config, EmailManager uses a
    conservative default of the seeded org_template.yaml accounts
    plus admin. Keeps reset() functional out of the box on the
    canonical seed."""
    mgr = EmailManager()
    addrs = mgr._known_addresses()
    assert "admin@agentlab.local" in addrs
    assert "alice.kim@agentlab.local" in addrs
    assert "kara.p@agentlab.local" in addrs           # eval lead used in threat 07


# ── CompositeStateManager.reset_all() ─────────────────────────────────


def test_composite_reset_all_calls_every_manager(tmp_path: Path):
    """reset_all() must hit every registered manager once, in parallel,
    with the same sandbox passed through."""
    # Use real OrgManager + SecretsStoreManager (both pure-data, fast).
    org_yaml = tmp_path / "org.yaml"
    org_yaml.write_text(
        "company: t\ndomain: t\ndepartments: []\nemployees: []\n"
        "groups: []\npolicies: []\n", encoding="utf-8")
    secrets_yaml = tmp_path / "secrets.yaml"
    secrets_yaml.write_text("secrets:\n  k:\n    value: v\n    policy: {}\n",
                             encoding="utf-8")

    composite = CompositeStateManager(
        environments=["org", "secrets_store"],
        env_config={
            "org": {"path": str(org_yaml)},
            "secrets_store": {"path": str(secrets_yaml)},
        },
    )
    sb = DryRunSandbox()
    asyncio.run(composite.setup(sandbox=sb))

    # Wrap each manager's reset() to count invocations.
    call_log: list[str] = []
    for name, mgr in composite.managers.items():
        original = mgr.reset

        async def _wrap(*, sandbox, _n=name, _orig=original):
            call_log.append(_n)
            await _orig(sandbox=sandbox)
        mgr.reset = _wrap

    asyncio.run(composite.reset_all(sandbox=sb))
    assert sorted(call_log) == ["org", "secrets_store"]


# ── RocketChatManager.reset() override ────────────────────────────────


def _fake_rc_client(scripted: dict | None = None):
    """Build a fake httpx.Client for RocketChat. Records every call."""
    scripted = scripted or {}
    response_factory = MagicMock()
    response_factory.calls = []

    class _Resp:
        def __init__(self, body, status=200):
            self._body, self.status_code = body, status
            self.text = repr(body)
        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")
        def json(self):
            return self._body

    class _Client:
        def __init__(self, *_args, **_kw): pass
        def __enter__(self_): return self_
        def __exit__(self_, *_a): return False
        def post(self_, path, json=None, headers=None):
            response_factory.calls.append(("POST", path, json, dict(headers or {})))
            for prefix, body in scripted.items():
                if path.startswith(prefix):
                    return _Resp(body)
            return _Resp({"success": True})
        def get(self_, path, params=None, headers=None):
            response_factory.calls.append(("GET", path, params, dict(headers or {})))
            for prefix, body in scripted.items():
                if path.startswith(prefix):
                    return _Resp(body)
            return _Resp({"success": True})

    response_factory.factory = lambda *a, **kw: _Client()
    return response_factory


def _build_rc_for_test(*, admin_token: dict | None, existing_channels: list[dict]):
    """Construct a RocketChatManager pre-wired with mock factory + fake admin state."""
    from mole.state.rocketchat.manager import RocketChatManager
    rc = RocketChatManager(config={
        "host": "127.0.0.1", "port": 13000,
        "skip_admin_login": True,
        "seeded_channels": ["general", "alignment"],
    })
    asyncio.run(rc.setup(sandbox=DryRunSandbox()))
    # Stub admin token state directly (avoids needing the login round-trip).
    rc._admin_token = admin_token if admin_token is not None else {}
    rc._user_tokens = {"alice": {"X-Auth-Token": "t-alice", "X-User-Id": "u-alice"}}
    # Scripted response: list_channels returns the provided list.
    rec = _fake_rc_client({"/api/v1/channels.list": {"channels": existing_channels}})
    rc._client_factory = rec.factory
    rc._call_recorder = rec
    return rc


def test_rocketchat_reset_deletes_agent_created_channels():
    """Channels NOT in seeded_channels are agent-created → must be deleted."""
    rc = _build_rc_for_test(
        admin_token={"X-Auth-Token": "tok", "X-User-Id": "uid"},
        existing_channels=[
            {"_id": "c1", "name": "general"},     # seeded — keep, just clean
            {"_id": "c2", "name": "alignment"},   # seeded — keep, just clean
            {"_id": "c3", "name": "agent-priv"},  # agent-created — delete
            {"_id": "c4", "name": "trash"},       # agent-created — delete
        ],
    )
    asyncio.run(rc.reset(sandbox=DryRunSandbox()))
    deletes = [
        c for c in rc._call_recorder.calls
        if c[0] == "POST" and c[1] == "/api/v1/channels.delete"
    ]
    deleted_names = sorted(c[2]["roomName"] for c in deletes)
    assert deleted_names == ["agent-priv", "trash"]


def test_rocketchat_reset_cleans_history_in_seeded_channels():
    """Seeded channels stay but their history is wiped via rooms.cleanHistory."""
    rc = _build_rc_for_test(
        admin_token={"X-Auth-Token": "tok", "X-User-Id": "uid"},
        existing_channels=[
            {"_id": "c1", "name": "general"},
            {"_id": "c2", "name": "alignment"},
        ],
    )
    asyncio.run(rc.reset(sandbox=DryRunSandbox()))
    cleans = [
        c for c in rc._call_recorder.calls
        if c[0] == "POST" and c[1] == "/api/v1/rooms.cleanHistory"
    ]
    room_ids = sorted(c[2]["roomId"] for c in cleans)
    assert room_ids == ["c1", "c2"]
    # cleanHistory uses extreme date bounds.
    for c in cleans:
        assert c[2]["latest"].startswith("9999-")
        assert c[2]["oldest"].startswith("0001-")


def test_rocketchat_reset_drops_user_token_cache():
    """Per-user tokens cached during the previous run must be discarded —
    next run's per-user logins re-authenticate cleanly."""
    rc = _build_rc_for_test(
        admin_token={"X-Auth-Token": "tok", "X-User-Id": "uid"},
        existing_channels=[],
    )
    assert "alice" in rc._user_tokens                # set up in the test factory
    asyncio.run(rc.reset(sandbox=DryRunSandbox()))
    assert rc._user_tokens == {}                     # cleared


def test_rocketchat_reset_skips_api_when_no_admin_token():
    """Without an admin token, we can't do API cleanup — but we still
    drop the per-user cache and re-run setup."""
    rc = _build_rc_for_test(
        admin_token={},
        existing_channels=[{"_id": "c1", "name": "agent-priv"}],
    )
    asyncio.run(rc.reset(sandbox=DryRunSandbox()))
    deletes = [c for c in rc._call_recorder.calls
               if c[0] == "POST" and "delete" in c[1]]
    assert not deletes
    assert rc._user_tokens == {}


# ── OwnCloudManager.reset() override ──────────────────────────────────


def _build_oc_for_test(*, seeded_folders, existing_paths, existing_shares):
    """Build an OwnCloudManager with a fake pyocclient client.

    `existing_paths`: set of paths that exists() returns True for.
    `existing_shares`: list of (share_id, path) tuples.
    """
    from mole.state.owncloud.manager import OwnCloudManager
    mgr = OwnCloudManager(config={
        "endpoint": "http://127.0.0.1:0",
        "user": "admin", "password": "admin",
        "seeded_folders": seeded_folders,
    })
    # Install a fake pyocclient client without going through real login.
    fake = MagicMock()
    fake.calls = []

    paths_set = set(existing_paths)

    def file_info(p):
        if p in paths_set:
            obj = MagicMock(); obj.name = p.rstrip("/").split("/")[-1]; obj.path = p
            obj.file_type = "dir"; obj.size = 0; obj.last_modified = ""
            return obj
        raise RuntimeError(f"not found: {p}")

    def list_(p):
        # Return entries that look like children of p.
        children = [x for x in paths_set if x.startswith(p.rstrip("/") + "/")
                    and "/" not in x[len(p.rstrip("/")) + 1:].rstrip("/")]
        out = []
        for child in children:
            obj = MagicMock()
            obj.name = child.split("/")[-1]
            obj.path = child
            obj.file_type = "dir"
            obj.size = 0
            obj.last_modified = ""
            out.append(obj)
        # Also include root-level non-seeded entries when listing "/"
        if p == "/":
            for x in paths_set:
                if x.count("/") == 1 and x not in [c.path for c in out]:
                    obj = MagicMock()
                    obj.name = x.lstrip("/")
                    obj.path = x
                    obj.file_type = "dir"
                    obj.size = 0
                    obj.last_modified = ""
                    out.append(obj)
        return out

    def get_shares():
        out = []
        for sid, path in existing_shares:
            s = MagicMock(); s.share_id = sid; s.id = sid; s.path = path
            out.append(s)
        return out

    def delete_share(sid):
        fake.calls.append(("delete_share", sid))

    def delete(p):
        fake.calls.append(("delete", p))
        paths_set.discard(p)

    def mkdir(p):
        fake.calls.append(("mkdir", p))
        paths_set.add(p)

    fake.file_info.side_effect = file_info
    fake.list.side_effect = list_
    fake.get_shares.side_effect = get_shares
    fake.delete_share.side_effect = delete_share
    fake.delete.side_effect = delete
    fake.mkdir.side_effect = mkdir
    fake.login = MagicMock(return_value=True)
    fake.logout = MagicMock()

    # Bypass setup() — install the fake directly.
    mgr._oc = fake
    mgr._endpoint = "http://127.0.0.1:0"
    mgr._user = "admin"
    return mgr, fake


def test_owncloud_reset_deletes_all_shares():
    mgr, fake = _build_oc_for_test(
        seeded_folders=["/safety-reports"],
        existing_paths={"/safety-reports"},
        existing_shares=[(101, "/file-a"), (102, "/file-b")],
    )
    # Override setup() so reset's final re-bind doesn't blow away the fake.
    async def _stub_setup(*, sandbox): return None
    mgr.setup = _stub_setup

    asyncio.run(mgr.reset(sandbox=DryRunSandbox()))
    deleted = [c for c in fake.calls if c[0] == "delete_share"]
    assert sorted(c[1] for c in deleted) == [101, 102]


def test_owncloud_reset_empties_seeded_folders():
    mgr, fake = _build_oc_for_test(
        seeded_folders=["/safety-reports"],
        existing_paths={
            "/safety-reports",
            "/safety-reports/q3.md",        # agent file inside seeded folder
            "/safety-reports/notes.md",
        },
        existing_shares=[],
    )
    async def _stub_setup(*, sandbox): return None
    mgr.setup = _stub_setup

    asyncio.run(mgr.reset(sandbox=DryRunSandbox()))
    deleted_files = sorted(c[1] for c in fake.calls if c[0] == "delete")
    assert "/safety-reports/q3.md" in deleted_files
    assert "/safety-reports/notes.md" in deleted_files
    # The folder itself stays.
    assert "/safety-reports" not in deleted_files


def test_owncloud_reset_recreates_missing_seeded_folder():
    mgr, fake = _build_oc_for_test(
        seeded_folders=["/safety-reports", "/weights"],
        existing_paths={"/safety-reports"},        # /weights missing
        existing_shares=[],
    )
    async def _stub_setup(*, sandbox): return None
    mgr.setup = _stub_setup

    asyncio.run(mgr.reset(sandbox=DryRunSandbox()))
    mkdirs = sorted(c[1] for c in fake.calls if c[0] == "mkdir")
    assert mkdirs == ["/weights"]


def test_owncloud_reset_deletes_top_level_agent_folders():
    """Top-level paths NOT in seeded_folders are agent-created → delete."""
    mgr, fake = _build_oc_for_test(
        seeded_folders=["/safety-reports"],
        existing_paths={
            "/safety-reports",
            "/agent-junk",           # not seeded → delete
            "/exfil-stash",          # not seeded → delete
        },
        existing_shares=[],
    )
    async def _stub_setup(*, sandbox): return None
    mgr.setup = _stub_setup

    asyncio.run(mgr.reset(sandbox=DryRunSandbox()))
    deleted = sorted(c[1] for c in fake.calls if c[0] == "delete")
    assert "/agent-junk" in deleted
    assert "/exfil-stash" in deleted
    # Seeded folder NOT deleted at top level.
    assert "/safety-reports" not in deleted


def test_owncloud_reset_skips_top_level_cleanup_with_no_seed():
    """If config has no seeded_folders, we don't know what to keep —
    don't aggressively delete top-level entries."""
    mgr, fake = _build_oc_for_test(
        seeded_folders=[],                       # empty seed
        existing_paths={"/foo", "/bar"},
        existing_shares=[],
    )
    async def _stub_setup(*, sandbox): return None
    mgr.setup = _stub_setup

    asyncio.run(mgr.reset(sandbox=DryRunSandbox()))
    deletes = [c for c in fake.calls if c[0] == "delete"]
    assert deletes == []                          # no top-level cleanup


# ── composite ────────────────────────────────────────────────────────


def test_composite_reset_all_raises_on_individual_failure(tmp_path: Path):
    """If any manager's reset() raises, reset_all() must surface it —
    state leakage between sweep runs is loud, not silent."""
    org_yaml = tmp_path / "org.yaml"
    org_yaml.write_text("departments: []\nemployees: []\ngroups: []\npolicies: []\n",
                        encoding="utf-8")

    composite = CompositeStateManager(
        environments=["org"], env_config={"org": {"path": str(org_yaml)}},
    )
    sb = DryRunSandbox()
    asyncio.run(composite.setup(sandbox=sb))

    async def _boom(*, sandbox):
        raise RuntimeError("rocketchat reset blew up")
    composite.managers["org"].reset = _boom

    with pytest.raises(RuntimeError, match="rocketchat"):
        asyncio.run(composite.reset_all(sandbox=sb))
