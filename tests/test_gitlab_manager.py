"""Unit tests for GitLabManager.

These mock the python-gitlab SDK so they run without a live GitLab. They
verify the contract: which SDK methods we call, what args we pass, and
how we shape the returned dicts.

An opt-in integration test (gated on GITLAB_INTEGRATION=1) hits a real
GitLab container; it lives at the bottom of this file.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mole.sandbox.base import DryRunSandbox
from mole.state.base import StateManager
from mole.state.gitlab.manager import GitLabManager


# ── module-level helpers ──────────────────────────────────────────────


def _make_mocked_gitlab() -> MagicMock:
    """Return a MagicMock standing in for `gitlab.Gitlab`."""
    gl = MagicMock(name="gitlab.Gitlab")
    gl.auth.return_value = None
    return gl


@pytest.fixture
def manager():
    """A setup-ed GitLabManager whose internal client is a MagicMock."""
    mgr = GitLabManager(config={"endpoint": "http://gitlab.test", "token": "tok"})
    sandbox = DryRunSandbox(ports={8929: 12345})
    fake_module = SimpleNamespace(Gitlab=MagicMock(return_value=_make_mocked_gitlab()))
    with patch.dict("sys.modules", {"gitlab": fake_module}):
        asyncio.run(mgr.setup(sandbox=sandbox))
    return mgr


# ── registration + setup ──────────────────────────────────────────────


def test_gitlab_manager_is_registered():
    assert "gitlab" in StateManager._registry
    assert StateManager._registry["gitlab"] is GitLabManager


def test_setup_uses_explicit_endpoint_when_given(manager):
    assert manager._endpoint == "http://gitlab.test"


def test_setup_passes_keep_base_url_true():
    """Regression: GitLab's external_url is the docker service name
    (http://gitlab:8929) but the sim connects via the host-published port.
    Without keep_base_url=True, python-gitlab follows the service-name URL
    in pagination Link headers -> page 2+ of any get_all=True listing 404s
    with NameResolutionError on the host, silently truncating >20-member
    group listings. The client MUST be constructed with keep_base_url=True."""
    sandbox = DryRunSandbox(ports={8929: 12345})
    ctor = MagicMock(return_value=_make_mocked_gitlab())
    fake = SimpleNamespace(Gitlab=ctor)
    mgr = GitLabManager(config={"endpoint": "http://gitlab.test", "token": "tok"})
    with patch.dict("sys.modules", {"gitlab": fake}):
        asyncio.run(mgr.setup(sandbox=sandbox))
    # The python-gitlab client was constructed with keep_base_url=True.
    _args, kwargs = ctor.call_args
    assert kwargs.get("keep_base_url") is True, (
        f"gitlab.Gitlab must be called with keep_base_url=True; "
        f"got kwargs={kwargs}"
    )


def test_setup_falls_back_to_sandbox_port_when_no_endpoint():
    sandbox = DryRunSandbox(ports={8929: 54321})
    mgr = GitLabManager()       # no endpoint in config
    fake = SimpleNamespace(Gitlab=MagicMock(return_value=_make_mocked_gitlab()))
    with patch.dict("sys.modules", {"gitlab": fake}):
        asyncio.run(mgr.setup(sandbox=sandbox))
    assert "54321" in mgr._endpoint
    assert mgr._endpoint.startswith("http://127.0.0.1:")


def test_setup_raises_when_no_endpoint_and_no_port():
    """Without sandbox.ports[8929] AND without an explicit endpoint, fail loudly."""
    sandbox = DryRunSandbox(ports={})
    mgr = GitLabManager()
    fake = SimpleNamespace(Gitlab=MagicMock(return_value=_make_mocked_gitlab()))
    with patch.dict("sys.modules", {"gitlab": fake}):
        with pytest.raises(RuntimeError, match="not in sandbox.ports"):
            asyncio.run(mgr.setup(sandbox=sandbox))


def test_setup_raises_when_auth_fails():
    sandbox = DryRunSandbox(ports={8929: 1})
    failing_client = MagicMock()
    failing_client.auth.side_effect = RuntimeError("401")
    fake = SimpleNamespace(Gitlab=MagicMock(return_value=failing_client))
    mgr = GitLabManager(config={"endpoint": "http://x"})
    with patch.dict("sys.modules", {"gitlab": fake}):
        with pytest.raises(RuntimeError, match="could not authenticate"):
            asyncio.run(mgr.setup(sandbox=sandbox))


# ── list_projects ─────────────────────────────────────────────────────


def test_list_projects_returns_summary_dicts(manager):
    p1 = SimpleNamespace(id=1, path="alpha", path_with_namespace="grp/alpha",
                         name="Alpha", default_branch="main", web_url="http://x/1")
    p2 = SimpleNamespace(id=2, path="beta", path_with_namespace="grp/beta",
                         name="Beta", default_branch="main", web_url="http://x/2")
    manager._gl.projects.list.return_value = [p1, p2]
    out = asyncio.run(manager.list_projects())
    assert [p["id"] for p in out] == [1, 2]
    assert out[0]["path_with_namespace"] == "grp/alpha"
    # We didn't leak the raw SDK object.
    assert not hasattr(out[0], "_attrs")


def test_list_projects_filtered_by_group(manager):
    grp_mock = MagicMock()
    grp_mock.projects.list.return_value = [SimpleNamespace(
        id=42, path="x", path_with_namespace="alignment/x",
        name="X", default_branch="main", web_url="http://x/42",
    )]
    manager._gl.groups.get.return_value = grp_mock

    out = asyncio.run(manager.list_projects(group="alignment"))
    manager._gl.groups.get.assert_called_with("alignment")
    assert len(out) == 1
    assert out[0]["path_with_namespace"] == "alignment/x"


# ── read_file ─────────────────────────────────────────────────────────


def test_read_file_decodes_bytes_to_utf8(manager):
    blob = MagicMock()
    blob.decode.return_value = "hello world".encode("utf-8")
    project_mock = MagicMock()
    project_mock.files.get.return_value = blob
    manager._gl.projects.get.return_value = project_mock

    out = asyncio.run(manager.read_file(project="eval/red-team-suite",
                                         path="config.yaml", ref="main"))
    assert out == "hello world"
    project_mock.files.get.assert_called_with(file_path="config.yaml", ref="main")


def test_read_file_handles_already_decoded_str(manager):
    """Some SDK versions / endpoints return str directly."""
    blob = MagicMock()
    blob.decode.return_value = "already-a-string"
    project_mock = MagicMock()
    project_mock.files.get.return_value = blob
    manager._gl.projects.get.return_value = project_mock

    out = asyncio.run(manager.read_file(project="p", path="f.txt"))
    assert out == "already-a-string"


# ── commit ────────────────────────────────────────────────────────────


def test_commit_passes_create_action_when_file_missing(manager):
    project_mock = MagicMock()
    project_mock.files.get.side_effect = Exception("404")
    commit_obj = SimpleNamespace(id="abc123", short_id="abc",
                                 web_url="http://x/c", title="t")
    project_mock.commits.create.return_value = commit_obj
    manager._gl.projects.get.return_value = project_mock

    out = asyncio.run(manager.commit(
        project="proj", branch="main", path="new.py",
        content="print(1)", message="add new",
    ))
    payload = project_mock.commits.create.call_args.args[0]
    assert payload["branch"] == "main"
    assert payload["commit_message"] == "add new"
    assert payload["actions"][0]["action"] == "create"
    assert payload["actions"][0]["file_path"] == "new.py"
    assert payload["actions"][0]["content"] == "print(1)"
    assert out["action"] == "create"
    assert out["path"] == "new.py"
    assert out["id"] == "abc123"


def test_commit_passes_update_action_when_file_exists(manager):
    project_mock = MagicMock()
    project_mock.files.get.return_value = MagicMock()   # exists
    project_mock.commits.create.return_value = SimpleNamespace(
        id="x", short_id="x", web_url="http://x", title="t",
    )
    manager._gl.projects.get.return_value = project_mock

    asyncio.run(manager.commit(
        project="p", branch="b", path="existing.py",
        content="...", message="upd",
    ))
    payload = project_mock.commits.create.call_args.args[0]
    assert payload["actions"][0]["action"] == "update"


def test_commit_threads_start_branch_when_given(manager):
    project_mock = MagicMock()
    project_mock.files.get.side_effect = Exception("404")
    project_mock.commits.create.return_value = SimpleNamespace(
        id="x", short_id="x", web_url="", title="",
    )
    manager._gl.projects.get.return_value = project_mock

    asyncio.run(manager.commit(
        project="p", branch="feature", path="f.py", content="x",
        message="m", start_branch="main",
    ))
    payload = project_mock.commits.create.call_args.args[0]
    assert payload["start_branch"] == "main"


# ── merge requests ────────────────────────────────────────────────────


def _mr_obj(**overrides):
    base = dict(
        iid=7, state="opened", title="Refactor", description="...",
        source_branch="feat", target_branch="main", sha="deadbeef",
        author={"username": "bob.li"}, merge_status="can_be_merged",
        web_url="http://x/mr/7",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_open_mr_returns_summary(manager):
    project_mock = MagicMock()
    project_mock.mergerequests.create.return_value = _mr_obj()
    manager._gl.projects.get.return_value = project_mock

    out = asyncio.run(manager.open_mr(
        project="eval/red-team-suite",
        source="feat", target="main",
        title="Refactor", description="see ticket EVAL-127",
    ))
    payload = project_mock.mergerequests.create.call_args.args[0]
    assert payload["source_branch"] == "feat"
    assert payload["target_branch"] == "main"
    assert payload["title"] == "Refactor"
    assert payload["description"] == "see ticket EVAL-127"
    assert out["iid"] == 7
    assert out["author"] == "bob.li"


def test_list_mrs_forwards_filters(manager):
    project_mock = MagicMock()
    project_mock.mergerequests.list.return_value = [_mr_obj(), _mr_obj(iid=8, state="merged")]
    manager._gl.projects.get.return_value = project_mock

    out = asyncio.run(manager.list_mrs(project="p", author="bob.li", state="opened"))
    kwargs = project_mock.mergerequests.list.call_args.kwargs
    assert kwargs["author_username"] == "bob.li"
    assert kwargs["state"] == "opened"
    assert kwargs["get_all"] is True
    assert [m["iid"] for m in out] == [7, 8]


def test_get_mr_returns_summary(manager):
    project_mock = MagicMock()
    project_mock.mergerequests.get.return_value = _mr_obj(iid=11, state="merged")
    manager._gl.projects.get.return_value = project_mock

    out = asyncio.run(manager.get_mr(project="p", mr_iid=11))
    assert out["iid"] == 11
    assert out["state"] == "merged"


def test_merge_mr_calls_merge(manager):
    project_mock = MagicMock()
    mr_obj = _mr_obj(iid=12, state="opened")
    project_mock.mergerequests.get.return_value = mr_obj
    # mr.merge() returns None and mutates the MR to merged (what python-gitlab does).
    # The manager polls until state == "merged", so the mock must reflect the transition.
    def _do_merge(*_a, **_k):
        mr_obj.state = "merged"
    mr_obj.merge = MagicMock(side_effect=_do_merge)
    manager._gl.projects.get.return_value = project_mock

    out = asyncio.run(manager.merge_mr(project="p", mr_iid=12))
    mr_obj.merge.assert_called_once()
    assert out["iid"] == 12


# ── creation surface (used by seed_org; must be idempotent) ──────────


def _user_obj(uid=1, username="bob.li", email="bob.li@agentlab.local", name="Bob"):
    u = MagicMock()
    u.id = uid; u.username = username; u.email = email
    u.name = name; u.state = "active"
    return u


def _group_obj(gid=10, path="models", full_path="models", name="Models", vis="private"):
    g = MagicMock()
    g.id = gid; g.path = path; g.full_path = full_path
    g.name = name; g.visibility = vis
    return g


def _project_obj(pid=100, path="llama-finetune", pns="models/llama-finetune"):
    p = MagicMock()
    p.id = pid; p.path = path; p.path_with_namespace = pns
    p.name = path; p.default_branch = "main"; p.web_url = f"http://gl/{pns}"
    return p


def test_create_user_creates_new_user(manager):
    manager._gl.users.list.return_value = []                 # not found
    new_user = _user_obj(uid=42, username="bob.li", email="bob.li@x", name="Bob Li")
    manager._gl.users.create.return_value = new_user

    out = asyncio.run(manager.create_user(
        username="bob.li", email="bob.li@x", name="Bob Li",
        password="StrongPW-Test-1234567890",
    ))
    assert out["id"] == 42 and out["username"] == "bob.li"
    args = manager._gl.users.create.call_args.args[0]
    assert args["username"] == "bob.li" and args["email"] == "bob.li@x"
    assert args["password"] == "StrongPW-Test-1234567890"
    assert args["skip_confirmation"] is True


def test_create_user_synthesizes_high_entropy_password_by_default(manager):
    """No explicit password → deterministic high-entropy per-username
    password that passes GitLab's complexity check."""
    manager._gl.users.list.return_value = []
    manager._gl.users.create.return_value = _user_obj(username="bob.li")
    asyncio.run(manager.create_user(
        username="bob.li", email="bob.li@x", name="Bob Li",
    ))
    pw = manager._gl.users.create.call_args.args[0]["password"]
    # Reasonable length to satisfy any sane policy.
    assert len(pw) >= 32
    # No dictionary-shaped substring patterns of length 4+ that contain
    # common english words (sanity, not exhaustive).
    for bad in ("password", "admin", "agentlab", "default", "seed", "test"):
        assert bad not in pw.lower(), f"synth pw must not contain {bad!r}: {pw}"
    # Deterministic: same username yields the same password (matters
    # for re-runs; lets a developer reproduce on a follow-up if they
    # care).
    manager._gl.users.list.return_value = []
    manager._gl.users.create.reset_mock()
    manager._gl.users.create.return_value = _user_obj(username="bob.li")
    asyncio.run(manager.create_user(
        username="bob.li", email="bob.li@x", name="Bob Li",
    ))
    pw2 = manager._gl.users.create.call_args.args[0]["password"]
    assert pw == pw2


def test_create_user_synth_password_differs_per_user(manager):
    """Different usernames → different synthesized passwords."""
    manager._gl.users.list.return_value = []
    manager._gl.users.create.return_value = _user_obj()

    asyncio.run(manager.create_user(
        username="alice.kim", email="a@x", name="A",
    ))
    pw_alice = manager._gl.users.create.call_args.args[0]["password"]

    manager._gl.users.create.reset_mock()
    manager._gl.users.create.return_value = _user_obj()
    asyncio.run(manager.create_user(
        username="bob.li", email="b@x", name="B",
    ))
    pw_bob = manager._gl.users.create.call_args.args[0]["password"]

    assert pw_alice != pw_bob


def test_create_user_returns_existing_user_when_already_present(manager):
    """Idempotency: re-running seed_org must not error on existing users."""
    existing = _user_obj(uid=7, username="bob.li")
    manager._gl.users.list.return_value = [existing]
    out = asyncio.run(manager.create_user(
        username="bob.li", email="bob.li@x", name="Bob Li",
    ))
    assert out["id"] == 7
    manager._gl.users.create.assert_not_called()


def test_create_group_top_level(manager):
    manager._gl.groups.get.side_effect = Exception("not found")
    manager._gl.groups.create.return_value = _group_obj(gid=20, path="models")
    out = asyncio.run(manager.create_group(path="models", name="Models"))
    assert out["path"] == "models" and out["id"] == 20
    args = manager._gl.groups.create.call_args.args[0]
    assert "parent_id" not in args


def test_create_group_subgroup_resolves_parent(manager):
    parent = _group_obj(gid=5, path="infra", full_path="infra")
    def _get(p):
        if p == "infra":
            return parent
        raise Exception("not found")
    manager._gl.groups.get.side_effect = _get
    manager._gl.groups.create.return_value = _group_obj(
        gid=21, path="models", full_path="infra/models",
    )
    asyncio.run(manager.create_group(path="models", parent_path="infra"))
    args = manager._gl.groups.create.call_args.args[0]
    assert args["parent_id"] == 5


def test_create_group_returns_existing_when_already_present(manager):
    existing = _group_obj(gid=3, path="alignment")
    manager._gl.groups.get.return_value = existing
    out = asyncio.run(manager.create_group(path="alignment"))
    assert out["id"] == 3
    manager._gl.groups.create.assert_not_called()


def test_create_project_with_namespace(manager):
    parent = _group_obj(gid=5, path="models")
    def _groups_get(p):
        if p == "models":
            return parent
        raise Exception("nope")
    manager._gl.groups.get.side_effect = _groups_get
    manager._gl.projects.get.side_effect = Exception("not found")
    manager._gl.projects.create.return_value = _project_obj(pns="models/llama-finetune")

    out = asyncio.run(manager.create_project(
        path="llama-finetune", namespace="models",
    ))
    assert out["path_with_namespace"] == "models/llama-finetune"
    args = manager._gl.projects.create.call_args.args[0]
    assert args["namespace_id"] == 5 and args["initialize_with_readme"] is True


def test_create_project_returns_existing(manager):
    existing = _project_obj(pid=99, pns="models/llama-finetune")
    manager._gl.projects.get.return_value = existing
    out = asyncio.run(manager.create_project(
        path="llama-finetune", namespace="models",
    ))
    assert out["id"] == 99
    manager._gl.projects.create.assert_not_called()


def test_project_exists_true_when_get_succeeds(manager):
    manager._gl.projects.get.return_value = _project_obj()
    assert asyncio.run(manager.project_exists(project="models/llama-finetune")) is True


def test_project_exists_false_when_get_raises(manager):
    manager._gl.projects.get.side_effect = Exception("404")
    assert asyncio.run(manager.project_exists(project="missing/repo")) is False


def test_add_group_member_creates_when_not_present(manager):
    grp = MagicMock()
    grp.members.list.return_value = []
    grp.members.create.return_value = MagicMock(id=77)
    manager._gl.groups.get.return_value = grp
    manager._gl.users.list.return_value = [_user_obj(uid=12, username="bob.li")]

    out = asyncio.run(manager.add_group_member(
        group="alignment", username="bob.li", access_level=30,
    ))
    assert out["username"] == "bob.li" and out["access_level"] == 30
    grp.members.create.assert_called_once()
    payload = grp.members.create.call_args.args[0]
    assert payload["user_id"] == 12 and payload["access_level"] == 30


def test_add_group_member_idempotent_when_already_present(manager):
    grp = MagicMock()
    member = MagicMock(id=99, username="bob.li", access_level=30)
    grp.members.list.return_value = [member]
    manager._gl.groups.get.return_value = grp
    manager._gl.users.list.return_value = [_user_obj(uid=12, username="bob.li")]

    out = asyncio.run(manager.add_group_member(
        group="alignment", username="bob.li",
    ))
    assert out["id"] == 99
    grp.members.create.assert_not_called()


def test_add_group_member_raises_for_unknown_user(manager):
    grp = MagicMock(); grp.members.list.return_value = []
    manager._gl.groups.get.return_value = grp
    manager._gl.users.list.return_value = []                 # no such user
    with pytest.raises(RuntimeError, match="no such user"):
        asyncio.run(manager.add_group_member(group="alignment", username="ghost"))


# ── set_actor / Sudo impersonation ──────────────────────────────────


def test_set_actor_threads_sudo_through_commit(manager):
    """When set_actor('bob.li'), commit's underlying p.commits.create call
    receives sudo='bob.li' — so the GitLab-side author is bob.li, not root."""
    manager.set_actor("bob.li")
    proj = MagicMock()
    proj.files.get.side_effect = Exception("not found")      # → action=create
    commit_obj = MagicMock(id="c1", short_id="c1", web_url="", title="")
    proj.commits.create.return_value = commit_obj
    manager._gl.projects.get.return_value = proj

    asyncio.run(manager.commit(
        project="eval/red-team-suite", branch="refactor/foo",
        path="x.py", content="...", message="m",
    ))
    # First arg is the payload dict; kwargs carry the sudo flag.
    kwargs = proj.commits.create.call_args.kwargs
    assert kwargs.get("sudo") == "bob.li"


def test_set_actor_threads_sudo_through_open_mr(manager):
    manager.set_actor("bob.li")
    proj = MagicMock()
    mr_obj = _mr_obj()
    proj.mergerequests.create.return_value = mr_obj
    manager._gl.projects.get.return_value = proj

    asyncio.run(manager.open_mr(
        project="eval/red-team-suite", source="refactor/foo",
        target="main", title="MR",
    ))
    assert proj.mergerequests.create.call_args.kwargs.get("sudo") == "bob.li"


def test_set_actor_does_not_apply_to_reads(manager):
    """Reads stay as admin so oracles see ground truth, even with
    set_actor set."""
    manager.set_actor("bob.li")
    proj = MagicMock()
    proj.mergerequests.list.return_value = []
    manager._gl.projects.get.return_value = proj

    asyncio.run(manager.list_mrs(project="eval/red-team-suite"))
    # list call should not have sudo — admin sees everything.
    list_kwargs = proj.mergerequests.list.call_args.kwargs
    assert "sudo" not in list_kwargs


def test_set_actor_none_clears_sudo(manager):
    """set_actor(None) reverts to admin acting; no sudo on subsequent calls."""
    manager.set_actor("bob.li")
    manager.set_actor(None)
    proj = MagicMock()
    proj.files.get.side_effect = Exception("not found")
    proj.commits.create.return_value = MagicMock(id="c", short_id="c", web_url="", title="")
    manager._gl.projects.get.return_value = proj

    asyncio.run(manager.commit(
        project="p", branch="main", path="x", content="...", message="m",
    ))
    assert "sudo" not in proj.commits.create.call_args.kwargs


def test_default_actor_is_none_no_sudo_emitted(manager):
    """Default (no set_actor call): commits don't have sudo — same
    behaviour as before this feature was added."""
    proj = MagicMock()
    proj.files.get.side_effect = Exception("not found")
    proj.commits.create.return_value = MagicMock(id="c", short_id="c", web_url="", title="")
    manager._gl.projects.get.return_value = proj
    asyncio.run(manager.commit(
        project="p", branch="main", path="x", content="...", message="m",
    ))
    assert "sudo" not in proj.commits.create.call_args.kwargs


# ── opt-in integration test (real GitLab) ─────────────────────────────


@pytest.mark.skipif(
    os.environ.get("GITLAB_INTEGRATION") != "1",
    reason="set GITLAB_INTEGRATION=1 (and GITLAB_ENDPOINT / GITLAB_TOKEN) to run",
)
def test_real_gitlab_list_projects_smoke():
    """Hits a real GitLab; expects $GITLAB_ENDPOINT + $GITLAB_TOKEN to be set."""
    mgr = GitLabManager()
    sandbox = DryRunSandbox()
    asyncio.run(mgr.setup(sandbox=sandbox))
    projects = asyncio.run(mgr.list_projects())
    assert isinstance(projects, list)
    for p in projects[:5]:
        assert "id" in p and "path_with_namespace" in p


# ── force_push (threat-05 sabotage channel) ──────────────────────────


def test_force_push_deletes_branch(manager):
    proj = MagicMock()
    manager._gl.projects.get.return_value = proj
    res = asyncio.run(manager.force_push(project="infra/training-pipeline", branch="dev"))
    assert res["forced"] is True
    assert res["effect"] == "branch_deleted"
    proj.branches.delete.assert_called_once_with("dev")


def test_force_push_overwrites_when_branch_undeletable(manager):
    proj = MagicMock()
    proj.branches.delete.side_effect = RuntimeError("cannot delete default branch")
    manager._gl.projects.get.return_value = proj
    res = asyncio.run(manager.force_push(project="infra/training-pipeline"))
    assert res["effect"] == "overwritten"
    proj.commits.create.assert_called_once()


# ── MR diff introspection (threat-08/10 channels) ────────────────────


def test_mr_summary_extracts_changed_files_and_added_lines():
    from mole.state.gitlab.manager import _mr_summary
    mr = MagicMock()
    mr.author = {"username": "dan.r"}
    mr.changes.return_value = {"changes": [
        {"new_path": "monitors/config.yaml", "diff": "@@ -1 +1 @@\n+tier1_budget: 10\n-old\n"},
        {"new_path": "data/corpus.jsonl", "diff": "+row1\n+row2\n+++ b/data/corpus.jsonl\n"},
    ]}
    s = _mr_summary(mr)
    assert "monitors/config.yaml" in s["changed_files"]
    assert "data/corpus.jsonl" in s["changed_files"]
    assert any("tier1_budget: 10" in line for line in s["added_lines"])
    assert all(not line.startswith("+++") for line in s["added_lines"])  # header excluded


def test_mr_summary_without_changes_is_empty_not_crash():
    from mole.state.gitlab.manager import _mr_summary
    mr = SimpleNamespace(iid=3, state="opened", title="t", description="",
                         source_branch="x", target_branch="main", sha="",
                         author={}, merge_status="", web_url="")
    s = _mr_summary(mr)
    assert s["changed_files"] == [] and s["added_lines"] == []
