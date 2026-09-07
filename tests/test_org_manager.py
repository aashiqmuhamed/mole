"""Unit tests for OrgManager — uses the shipped org_template.yaml."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mole.sandbox.base import DryRunSandbox
from mole.state.base import StateManager
from mole.state.org.manager import OrgManager, _default_org_yaml


def _mgr() -> OrgManager:
    """Build + set up an OrgManager backed by the canonical YAML."""
    m = OrgManager()
    asyncio.run(m.setup(sandbox=DryRunSandbox()))
    return m


def test_org_manager_is_registered():
    assert "org" in StateManager._registry
    assert StateManager._registry["org"] is OrgManager


def test_default_org_yaml_resolves_to_bootstrap_template():
    path = _default_org_yaml()
    assert path.name == "org_template.yaml"
    assert path.exists(), f"expected canonical org yaml at {path}"


def test_setup_loads_employees_groups_policies():
    m = _mgr()
    employees = m.list_accounts()
    assert len(employees) >= 9  # canonical fixture ships with ~10
    ids = {e["id"] for e in employees}
    assert {"alice.kim", "bob.li", "frank.s", "kara.p"} <= ids


def test_whoami_returns_role_and_groups_for_alice():
    m = _mgr()
    info = m.whoami("alice.kim")
    assert info["id"] == "alice.kim"
    assert info["role"] == "Alignment Team Lead"
    assert "model-approval" in info["groups"]
    assert info["team"] == "alignment"
    # Alice is the team lead, so her manager is None (she's a team manager herself).
    assert info["manager"] is None


def test_whoami_returns_team_and_manager_for_member():
    m = _mgr()
    info = m.whoami("bob.li")
    assert info["team"] == "alignment"
    assert info["manager"] == "alice.kim"


def test_whoami_raises_for_unknown_account():
    m = _mgr()
    with pytest.raises(KeyError):
        m.whoami("not.a.real.person")


def test_list_group_members():
    m = _mgr()
    members = m.list_group_members("model-approval")
    # Per the policy, model-approval gates checkpoint-tag-approved with N=2.
    assert "alice.kim" in members
    assert "ivan.o" in members


def test_list_group_members_unknown_group_returns_empty():
    m = _mgr()
    assert m.list_group_members("does-not-exist") == []


def test_who_can_approve_tag_approved():
    m = _mgr()
    approvers = m.who_can_approve("model.registry.tag_approved")
    assert {"alice.kim", "ivan.o"} <= set(approvers)
    # The 2-of-N policy is captured separately via get_policy().
    pol = m.get_policy("model.registry.tag_approved")
    assert pol is not None
    assert pol["requires_n"] == 2


def test_list_my_groups():
    m = _mgr()
    assert "model-approval" in m.list_my_groups("alice.kim")
    assert "infra-admins" in m.list_my_groups("frank.s")
    # Interns aren't in any group.
    assert m.list_my_groups("julian.x") == []


def test_get_policy_returns_copy_not_reference():
    m = _mgr()
    pol = m.get_policy("model.registry.tag_approved")
    assert pol is not None
    pol["requires_n"] = 999
    # Mutating the returned dict shouldn't affect internal state.
    fresh = m.get_policy("model.registry.tag_approved")
    assert fresh["requires_n"] == 2


def test_get_policy_unknown_action_returns_none():
    m = _mgr()
    assert m.get_policy("not.a.real.action") is None


def test_is_external_uses_company_domain():
    m = _mgr()
    assert m.is_external("attacker@gmail.com") is True
    assert m.is_external("alice.kim@agentlab.local") is False


def test_list_accounts_omits_service_credentials():
    m = _mgr()
    for emp in m.list_accounts():
        # Public surface must NOT leak service-account info.
        assert "services" not in emp
        assert "groups" not in emp
        assert "mbti" not in emp
        assert "persona_text" not in emp
        # But it must include the basics every consumer needs.
        assert {"id", "full_name", "role", "email"} <= set(emp.keys())


def test_get_employee_returns_full_record_with_services():
    """For stages/checkers that need to look up service credentials."""
    m = _mgr()
    emp = m.get_employee("alice.kim")
    assert "services" in emp
    assert "gitlab" in emp["services"]
    assert emp["services"]["gitlab"]["permissions"] == "maintainer"


def test_setup_with_custom_yaml_path(tmp_path: Path):
    """Manager honors config['path'] when given."""
    yml = tmp_path / "tiny.yaml"
    yml.write_text(
        "company: tiny\n"
        "domain: tiny.local\n"
        "accounts:\n"
        "  - {id: solo, full_name: Solo, email: solo@tiny.local, role: It, groups: []}\n"
        "groups: []\n"
        "policies: []\n",
        encoding="utf-8",
    )
    m = OrgManager(config={"path": str(yml)})
    asyncio.run(m.setup(sandbox=DryRunSandbox()))
    assert {e["id"] for e in m.list_accounts()} == {"solo"}
    assert m.domain() == "tiny.local"


def test_add_group_member_mutates_membership():
    """Regression: OrgManager was read-only, so threat-03 privilege-escalation
    harm was unreachable (no way to change group membership). add_group_member
    must add to the group AND mirror onto the account record, idempotently."""
    m = _mgr()
    grp = "model-approval"
    before = set(m.list_group_members(grp))
    person = "julian.x"
    res = asyncio.run(m.add_group_member(group=grp, account=person, actor=person))
    assert res["added"] is True
    assert person in m.list_group_members(grp)
    assert grp in m.list_my_groups(person)
    # Idempotent: a second add reports added=False, no duplicate.
    res2 = asyncio.run(m.add_group_member(group=grp, account=person))
    assert res2["added"] is False
    assert m.list_group_members(grp).count(person) == 1
    # Membership of others preserved.
    assert before <= set(m.list_group_members(grp))


def test_add_group_member_unknown_group_raises():
    m = _mgr()
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(m.add_group_member(group="no-such-group", account="julian.x"))


def test_cohorts_partition_focal_background_holdout():
    """The org has 3 cohorts; the 9 threat focals are all cohort=focal."""
    m = _mgr()
    focal = set(m.list_by_cohort("focal"))
    background = m.list_by_cohort("background")
    holdout = m.list_by_cohort("holdout")
    expected_focal = {"alice.kim", "bob.li", "dan.r", "frank.s", "grace.t",
                      "hannah.m", "ivan.o", "julian.x", "kara.p"}
    assert focal == expected_focal
    assert len(background) >= 10        # realistic background account population
    assert len(holdout) >= 3            # L1P-O eval reserve
    assert m.cohort_of("alice.kim") == "focal"
    # cohorts are disjoint
    assert not (set(background) & focal) and not (set(holdout) & focal)


def test_holdouts_excluded_from_default_persona_load():
    from mole.generator.persona_loader import load_personas
    m = _mgr()
    holdout = set(m.list_by_cohort("holdout"))
    default_ids = {p.id for p in load_personas()}
    all_ids = {p.id for p in load_personas(include_holdouts=True)}
    assert holdout and not (holdout & default_ids)     # holdouts dropped by default
    assert holdout <= all_ids                          # but present when requested
    # focals + background present in the training population
    assert {"alice.kim", "bob.li"} <= default_ids
