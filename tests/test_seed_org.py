"""Tests for bootstrap/seed_org.py — populates live backends from yaml.

All backends are mocked so the tests run without Docker; the contract
under test is "did seed_org issue the right async calls for the yaml's
accounts?"
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mole.seeders import seed_org
from mole.seeders.seed_org import _STANDARD_CHANNELS


# ── helpers ───────────────────────────────────────────────────────────


@pytest.fixture
def org_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "org.yaml"
    p.write_text(dedent("""
        accounts:
          - id: alice.kim
            full_name: Alice Kim
            email: alice.kim@agentlab.local
            services:
              gitlab: {username: alice.kim, groups: [alignment, eval-reviewers], permissions: maintainer}
              owncloud: {folders: {/safety-reports: rw, /weights: r}}
              rocketchat: {channels: [alignment, general]}
          - id: bob.li
            full_name: Bob Li
            email: bob.li@agentlab.local
            services:
              gitlab: {username: bob.li, groups: [alignment], permissions: developer}
              owncloud: {folders: {/safety-reports: r}}
              rocketchat: {channels: [alignment]}
          - id: noaccess.user
            full_name: No Access User
            email: noaccess@agentlab.local
            services: {}
    """), encoding="utf-8")
    return p


def _ctx(*, email=None, owncloud=None, rocketchat=None, gitlab=None):
    return SimpleNamespace(
        email=email, owncloud=owncloud, rocketchat=rocketchat, gitlab=gitlab,
    )


# ── email ─────────────────────────────────────────────────────────────


def test_seed_email_sends_one_welcome_per_employee(org_yaml: Path):
    email = SimpleNamespace(send_email=AsyncMock())
    counts = asyncio.run(seed_org(_ctx(email=email), org_yaml=org_yaml, only={"email"}))
    # 3 accounts + the standard shared role mailboxes (procurement@, finance@, ...) now seeded.
    assert counts["email"] >= 3
    addresses = {c.kwargs["to"] for c in email.send_email.await_args_list}
    assert {"alice.kim@agentlab.local", "bob.li@agentlab.local",
            "noaccess@agentlab.local"} <= addresses
    assert "procurement@agentlab.local" in addresses


def test_seed_email_skipped_when_backend_missing(org_yaml: Path):
    counts = asyncio.run(seed_org(_ctx(), org_yaml=org_yaml, only={"email"}))
    assert counts["email"] == 0


def test_seed_email_continues_on_individual_failure(org_yaml: Path):
    email = SimpleNamespace(send_email=AsyncMock(
        side_effect=[None, RuntimeError("blip"), None],
    ))
    counts = asyncio.run(seed_org(_ctx(email=email), org_yaml=org_yaml, only={"email"}))
    assert counts["email"] == 2                     # 1 of 3 failed, 2 succeeded


# ── owncloud ──────────────────────────────────────────────────────────


def test_seed_owncloud_creates_each_unique_folder_once(org_yaml: Path):
    """/safety-reports appears for two accounts → only one mkdir."""
    owncloud = SimpleNamespace(
        exists=AsyncMock(return_value=False),
        mkdir=AsyncMock(),
    )
    counts = asyncio.run(seed_org(_ctx(owncloud=owncloud),
                                  org_yaml=org_yaml, only={"owncloud"}))
    paths = sorted(c.kwargs["path"] for c in owncloud.mkdir.await_args_list)
    assert paths == ["/safety-reports", "/weights"]
    assert counts["owncloud"] == 2


def test_seed_owncloud_skips_existing_folders(org_yaml: Path):
    owncloud = SimpleNamespace(
        exists=AsyncMock(return_value=True),         # everything already there
        mkdir=AsyncMock(),
    )
    counts = asyncio.run(seed_org(_ctx(owncloud=owncloud),
                                  org_yaml=org_yaml, only={"owncloud"}))
    assert counts["owncloud"] == 0
    owncloud.mkdir.assert_not_called()


def test_seed_owncloud_handles_per_folder_mkdir_failures(org_yaml: Path):
    owncloud = SimpleNamespace(
        exists=AsyncMock(return_value=False),
        mkdir=AsyncMock(side_effect=[None, RuntimeError("permission denied")]),
    )
    counts = asyncio.run(seed_org(_ctx(owncloud=owncloud),
                                  org_yaml=org_yaml, only={"owncloud"}))
    assert counts["owncloud"] == 1                  # second mkdir failed


# ── rocketchat ───────────────────────────────────────────────────────


def _rc_mgr(*, existing_channels: list[str], admin_token: dict | None = None):
    """A fake RocketChatManager with the bits seed_org pokes."""
    mgr = SimpleNamespace()
    mgr._admin_token = admin_token if admin_token is not None else {
        "X-Auth-Token": "tok", "X-User-Id": "u",
    }
    mgr._base_url = "http://rocketchat:3000"
    mgr.list_channels = AsyncMock(return_value=[
        {"_id": f"c{i}", "name": n} for i, n in enumerate(existing_channels)
    ])
    # _client_factory is a callable returning a context manager.
    cm = MagicMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"success": True})
    cm.post = MagicMock(return_value=response)
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    mgr._client_factory = MagicMock(return_value=cm)
    mgr._fake_post = cm.post              # for assertion convenience
    return mgr


def _posts_to(rc, endpoint: str):
    return [c for c in rc._fake_post.call_args_list if endpoint in c.args[0]]


def test_seed_rocketchat_creates_missing_channels(org_yaml: Path):
    rc = _rc_mgr(existing_channels=["general"])      # alignment missing
    counts = asyncio.run(seed_org(_ctx(rocketchat=rc),
                                  org_yaml=org_yaml, only={"rocketchat"}))
    # Channels: the missing persona channel 'alignment' plus the standard company-wide
    # channels are created ('general' already existed).
    channel_names = {c.kwargs["json"]["name"] for c in _posts_to(rc, "channels.create")}
    assert "alignment" in channel_names
    assert set(_STANDARD_CHANNELS) <= channel_names
    assert "general" not in channel_names
    # Accounts are also provisioned as RC users (password == username) so
    # the agent / background accounts can authenticate and post.
    user_posts = _posts_to(rc, "users.create")
    assert user_posts, "expected RC users to be created"
    assert all(c.kwargs["json"]["password"] == c.kwargs["json"]["username"]
               for c in user_posts)
    # count = channels created + users created
    assert counts["rocketchat"] == len(channel_names) + len(user_posts)


def test_seed_rocketchat_skips_existing_channels(org_yaml: Path):
    rc = _rc_mgr(existing_channels=["alignment", "general", *_STANDARD_CHANNELS])
    counts = asyncio.run(seed_org(_ctx(rocketchat=rc),
                                  org_yaml=org_yaml, only={"rocketchat"}))
    # No channel creation (both already exist) — but users are still provisioned.
    assert not _posts_to(rc, "channels.create")
    user_posts = _posts_to(rc, "users.create")
    assert counts["rocketchat"] == len(user_posts)


def test_seed_rocketchat_skips_when_admin_token_missing(org_yaml: Path):
    rc = _rc_mgr(existing_channels=[], admin_token={})
    counts = asyncio.run(seed_org(_ctx(rocketchat=rc),
                                  org_yaml=org_yaml, only={"rocketchat"}))
    assert counts["rocketchat"] == 0
    rc._fake_post.assert_not_called()


def test_seed_rocketchat_disables_rate_limiter_before_user_creates(org_yaml: Path):
    """Regression test for the seed_org rate-limit hang.

    RocketChat's default REST rate limiter (~10 users.create/min) caused the
    seed loop to stall around account #48-58. Fix: POST API_Enable_Rate_Limiter
    -> false in the admin-settings block that runs BEFORE user creation. This
    test asserts the disable lands before any users.create call so a flipped
    ordering (e.g. someone moves the settings block below the loop) can't
    regress the fix silently."""
    rc = _rc_mgr(existing_channels=[])
    asyncio.run(seed_org(_ctx(rocketchat=rc),
                         org_yaml=org_yaml, only={"rocketchat"}))
    # Find indices of the rate-limit disable + first users.create in the
    # ordered call_args_list — the disable must come first.
    posts = list(rc._fake_post.call_args_list)
    rate_limit_idx = next(
        (i for i, c in enumerate(posts)
         if "settings/API_Enable_Rate_Limiter" in c.args[0]),
        None,
    )
    first_user_create_idx = next(
        (i for i, c in enumerate(posts)
         if "users.create" in c.args[0]),
        None,
    )
    assert rate_limit_idx is not None, \
        "expected POST to /api/v1/settings/API_Enable_Rate_Limiter"
    assert first_user_create_idx is not None, \
        "expected at least one POST to /api/v1/users.create"
    assert rate_limit_idx < first_user_create_idx, \
        (f"rate-limit disable must precede user creation "
         f"(disable at {rate_limit_idx}, first users.create at "
         f"{first_user_create_idx})")
    # And the POSTed value must actually be False (not e.g. truthy by mistake).
    disable_call = posts[rate_limit_idx]
    assert disable_call.kwargs["json"]["value"] is False


# ── gitlab (users, groups, memberships) ─────────────────────────────


def _gitlab_mock():
    return SimpleNamespace(
        create_group=AsyncMock(return_value={"id": 1, "path": "x"}),
        create_user=AsyncMock(return_value={"id": 2, "username": "x"}),
        add_group_member=AsyncMock(return_value={"id": 3}),
    )


def test_seed_gitlab_creates_groups_from_employee_memberships(org_yaml: Path):
    gl = _gitlab_mock()
    asyncio.run(seed_org(_ctx(gitlab=gl), org_yaml=org_yaml, only={"gitlab"}))
    paths = {c.kwargs["path"] for c in gl.create_group.await_args_list}
    assert paths == {"alignment", "eval-reviewers"}


def test_seed_gitlab_creates_one_user_per_gitlab_block(org_yaml: Path):
    """noaccess.user has services:{} (no gitlab block) → no user created."""
    gl = _gitlab_mock()
    asyncio.run(seed_org(_ctx(gitlab=gl), org_yaml=org_yaml, only={"gitlab"}))
    usernames = {c.kwargs["username"] for c in gl.create_user.await_args_list}
    assert usernames == {"alice.kim", "bob.li"}
    alice = next(
        c for c in gl.create_user.await_args_list
        if c.kwargs["username"] == "alice.kim"
    )
    assert alice.kwargs["email"] == "alice.kim@agentlab.local"
    assert alice.kwargs["name"] == "Alice Kim"


def test_seed_gitlab_adds_members_with_role_specific_access_level(org_yaml: Path):
    gl = _gitlab_mock()
    asyncio.run(seed_org(_ctx(gitlab=gl), org_yaml=org_yaml, only={"gitlab"}))
    by_user: dict[str, list[tuple[str, int]]] = {}
    for c in gl.add_group_member.await_args_list:
        by_user.setdefault(c.kwargs["username"], []).append(
            (c.kwargs["group"], c.kwargs["access_level"]),
        )
    # alice = maintainer (40); bob = developer (30); noaccess has no gitlab block.
    assert set(by_user["alice.kim"]) == {("alignment", 40), ("eval-reviewers", 40)}
    assert by_user["bob.li"] == [("alignment", 30)]
    assert "noaccess.user" not in by_user


def test_seed_gitlab_skipped_when_backend_missing(org_yaml: Path):
    counts = asyncio.run(seed_org(_ctx(), org_yaml=org_yaml, only={"gitlab"}))
    assert counts["gitlab"] == 0


def test_seed_gitlab_swallows_individual_failures(org_yaml: Path):
    """One failing create_user call must NOT abort the rest of the seed."""
    gl = SimpleNamespace(
        create_group=AsyncMock(return_value={"id": 1}),
        create_user=AsyncMock(side_effect=[{"id": 1}, RuntimeError("boom")]),
        add_group_member=AsyncMock(return_value={"id": 9}),
    )
    counts = asyncio.run(seed_org(_ctx(gitlab=gl), org_yaml=org_yaml, only={"gitlab"}))
    # 2 groups created + 1 user (2nd raised) + 3 memberships (alice ×2 + bob ×1).
    # noaccess.user has no gitlab block, contributes nothing.
    assert counts["gitlab"] == 2 + 1 + 3


def test_seed_gitlab_idempotent_at_call_level(org_yaml: Path):
    """Re-running yields the same call shape (manager-level idempotency is
    verified separately in test_gitlab_manager — here we just check the
    seed doesn't blow up on a second invocation)."""
    gl = _gitlab_mock()
    counts_1 = asyncio.run(seed_org(_ctx(gitlab=gl), org_yaml=org_yaml, only={"gitlab"}))
    counts_2 = asyncio.run(seed_org(_ctx(gitlab=gl), org_yaml=org_yaml, only={"gitlab"}))
    assert counts_1 == counts_2


# ── selection / defaults ─────────────────────────────────────────────


def test_only_filter_runs_only_named_backends(org_yaml: Path):
    email = SimpleNamespace(send_email=AsyncMock())
    owncloud = SimpleNamespace(exists=AsyncMock(return_value=False),
                               mkdir=AsyncMock())
    counts = asyncio.run(seed_org(
        _ctx(email=email, owncloud=owncloud),
        org_yaml=org_yaml, only={"email"},
    ))
    assert "email" in counts
    assert "owncloud" not in counts
    owncloud.mkdir.assert_not_called()


def test_default_targets_include_gitlab(org_yaml: Path):
    """Without `only`, gitlab is part of the default target set."""
    gl = _gitlab_mock()
    counts = asyncio.run(seed_org(_ctx(gitlab=gl), org_yaml=org_yaml))
    assert "gitlab" in counts


def test_missing_yaml_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        asyncio.run(seed_org(_ctx(), org_yaml=tmp_path / "nope.yaml"))
