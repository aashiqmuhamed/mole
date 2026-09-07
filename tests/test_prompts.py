"""Unit tests for agent/prompts.py — persona + today rendering."""
from __future__ import annotations

from types import SimpleNamespace

from mole.agent.prompts import render_prompt


class _StubOrg:
    """Minimal org-manager surface needed by the prompt renderer."""

    def __init__(self, whoami_record: dict | None = None,
                 raises: Exception | None = None) -> None:
        self._record = whoami_record
        self._raises = raises

    def whoami(self, account: str) -> dict:
        if self._raises:
            raise self._raises
        return self._record or {}


def test_render_prompt_with_org_lookup_inserts_role_team_manager():
    org = _StubOrg({
        "full_name": "Alice Kim",
        "role": "Alignment Team Lead",
        "team": "alignment",
        "manager": None,
        "groups": ["model-approval", "eval-reviewers"],
        "email": "alice.kim@agentlab.local",
    })
    out = render_prompt(
        "{persona}\n\nToday is {today}. Begin.",
        focal_account="alice.kim",
        org_manager=org,
        sim_start_iso="2026-04-06T09:00:00Z",
    )
    assert "Alice Kim" in out
    assert "Alignment Team Lead" in out
    assert "alignment team" in out
    assert "model-approval" in out
    assert "alice.kim@agentlab.local" in out
    assert "2026-04-06" in out


def test_render_prompt_omits_team_when_org_returns_none():
    org = _StubOrg({"full_name": "Bob", "role": "Researcher",
                    "team": None, "manager": None, "groups": []})
    out = render_prompt("{persona}", focal_account="bob.li", org_manager=org)
    assert "Bob" in out
    assert "team" not in out.lower()
    assert "manager" not in out.lower()


def test_render_prompt_includes_manager_when_present():
    org = _StubOrg({"full_name": "Bob", "role": "Researcher",
                    "team": "alignment", "manager": "alice.kim",
                    "groups": ["ml-research"]})
    out = render_prompt("{persona}", focal_account="bob.li", org_manager=org)
    assert "manager is alice.kim" in out


def test_render_prompt_falls_back_to_account_when_no_org():
    out = render_prompt(
        "{persona}\nDo work.",
        focal_account="bob.li",
        org_manager=None,
    )
    assert "bob.li" in out


def test_render_prompt_handles_org_exception_gracefully():
    org = _StubOrg(raises=KeyError("unknown account"))
    out = render_prompt(
        "{persona}", focal_account="ghost.user", org_manager=org,
    )
    assert "ghost.user" in out
    # Should NOT propagate the exception.


def test_render_prompt_today_strips_time_component():
    out = render_prompt("today={today}", sim_start_iso="2026-04-06T09:00:00Z",
                        focal_account=None)
    assert "today=2026-04-06" in out


def test_render_prompt_unknown_placeholders_stay_literal():
    """A typo like {prsona} shouldn't crash; just leave it as-is."""
    out = render_prompt(
        "{persona}\nUnused: {something_else}",
        focal_account=None,
    )
    assert "{something_else}" in out


def test_render_prompt_includes_tool_catalog():
    out = render_prompt(
        "{persona}", focal_account="bob.li",
        available_tools=["gitlab.commit", "owncloud.read_file"],
    )
    assert "gitlab.commit" in out
    assert "owncloud.read_file" in out


def test_render_prompt_no_placeholder_passes_through_unchanged():
    out = render_prompt(
        "Plain prompt without placeholders.",
        focal_account="alice.kim",
    )
    assert out == "Plain prompt without placeholders."


def test_render_prompt_with_no_focal_account_generic_persona():
    out = render_prompt("{persona}", focal_account=None)
    assert "autonomous" in out.lower() or "employee" in out.lower()
