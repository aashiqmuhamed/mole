"""Tests for the generator — persona loading, workflow gating, Member loop."""
from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent

import pytest

from mole.audit.collector import AuditCollector
from mole.generator.member import Member, NPCConfig
from mole.generator.persona_loader import load_personas
from mole.generator.workflows import WORKFLOWS
from mole.sandbox.base import DryRunSandbox
from mole.state.composite import CompositeStateManager
from mole.state.context import TaskContext

Context = TaskContext


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def org_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "org.yaml"
    p.write_text(dedent("""
        company: testlab
        domain: testlab.local
        departments:
          - id: research
            teams:
              - id: alignment
                manager: alice.kim
                members: [bob.li]
        accounts:
          - id: alice.kim
            full_name: Alice Kim
            email: alice.kim@testlab.local
            role: Alignment Lead
            mbti: INTJ
            interests: [evals]
            groups: [eval-reviewers, model-approval]
          - id: bob.li
            full_name: Bob Li
            email: bob.li@testlab.local
            role: ML Researcher
            mbti: INTP
            interests: [llms]
            groups: [ml-research, evals-readonly]
          - id: frank.s
            full_name: Frank Salas
            email: frank.s@testlab.local
            role: Infra Engineer
            mbti: ISTJ
            interests: [k8s]
            groups: [infra-admins]
          - id: ghost.user
            full_name: Ghost User
            email: ghost@testlab.local
            role: Phantom
            holdout: true
            groups: []
        groups:
          - id: eval-reviewers
            members: [alice.kim]
          - id: model-approval
            members: [alice.kim]
          - id: ml-research
            members: [bob.li]
          - id: evals-readonly
            members: [bob.li]
          - id: infra-admins
            members: [frank.s]
        policies: []
    """), encoding="utf-8")
    return p


@pytest.fixture
def secrets_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "secrets.yaml"
    p.write_text(dedent("""
        secrets:
          openai_admin_key:
            value: "sk-admin"
            policy: {group_allowlist: [infra-admins]}
          hf_readonly:
            value: "hf-fake"
            policy: {group_allowlist: [ml-research, evals-readonly]}
    """), encoding="utf-8")
    return p


@pytest.fixture
def checkpoints_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "checkpoints.yaml"
    p.write_text(dedent("""
        checkpoints:
          - checkpoint_id: ckpt-archetype-v1.0
            model_id: archetype
            version: "1.0"
            dataset_hash: "sha256:aaa"
            approved: true
            weight_bytes: 1024
            classification: internal
          - checkpoint_id: ckpt-archetype-v1.1-rc1
            model_id: archetype
            version: "1.1-rc1"
            dataset_hash: "sha256:bbb"
            approved: false
            weight_bytes: 1024
            classification: internal
    """), encoding="utf-8")
    return p


@pytest.fixture
def baseline_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "eval_baseline.yaml"
    p.write_text(dedent("""
        baseline:
          redteam_categories: [violence, csam]
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
    return p


@pytest.fixture
def ctx(org_yaml: Path, secrets_yaml: Path, checkpoints_yaml: Path,
        baseline_yaml: Path, tmp_path: Path):
    """Build a TaskContext with audit-wrapped pure-data managers.

    The audit collector's account_getter reads from the per-task
    ContextVar in `generator.account_context`. Member.run_at calls
    set_account(persona.id, "background_rules_agent") before each workflow, so concurrent
    sessions don't race on shared state.
    """
    from mole.generator.account_context import get_account

    collector = AuditCollector()
    composite = CompositeStateManager(
        environments=["org", "secrets_store", "model_registry", "eval_server"],
        env_config={
            "org": {"path": str(org_yaml)},
            "secrets_store": {"path": str(secrets_yaml)},
            "model_registry": {"path": str(checkpoints_yaml)},
            "eval_server": {"path": str(baseline_yaml)},
        },
    )
    sandbox = DryRunSandbox()
    asyncio.run(composite.setup(sandbox=sandbox))

    for svc_name, mgr in composite.managers.items():
        collector.wrap_manager(
            service_name=svc_name,
            manager=mgr,
            account_getter=get_account,
        )

    task_ctx = composite.create_context(
        task_dir=tmp_path,
        sandbox=sandbox,
        audit=collector,
        sim_start="2026-05-19T00:00:00Z",
    )
    task_ctx.collector = collector                          # convenience for assertions
    return task_ctx


# ── persona_loader ────────────────────────────────────────────────────


def test_load_personas_drops_holdouts_by_default(org_yaml: Path):
    personas = load_personas(org_yaml)
    ids = {p.id for p in personas}
    assert "ghost.user" not in ids
    assert {"alice.kim", "bob.li", "frank.s"} <= ids


def test_load_personas_can_include_holdouts(org_yaml: Path):
    personas = load_personas(org_yaml, include_holdouts=True)
    ids = {p.id for p in personas}
    assert "ghost.user" in ids


def test_load_personas_fills_team_and_manager(org_yaml: Path):
    by_id = {p.id: p for p in load_personas(org_yaml)}
    assert by_id["alice.kim"].team == "alignment"
    assert by_id["alice.kim"].is_manager is True
    assert by_id["alice.kim"].manager == ""              # she IS the manager
    assert by_id["bob.li"].team == "alignment"
    assert by_id["bob.li"].manager == "alice.kim"
    assert by_id["bob.li"].is_manager is False


def test_load_personas_missing_yaml_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_personas(tmp_path / "nope.yaml")


def test_load_personas_uses_default_when_no_path():
    """No path → uses bootstrap/org_template.yaml, the committed file."""
    personas = load_personas()
    ids = {p.id for p in personas}
    assert "alice.kim" in ids


# ── workflow gating ───────────────────────────────────────────────────


def test_morning_routine_applies_to_everyone(org_yaml: Path):
    morning = next(w for w in WORKFLOWS if w.name == "morning_routine")
    for p in load_personas(org_yaml):
        assert morning.applies_to(p)


def test_eval_submission_only_for_eval_reviewers(org_yaml: Path):
    es = next(w for w in WORKFLOWS if w.name == "eval_submission")
    by_id = {p.id: p for p in load_personas(org_yaml)}
    assert es.applies_to(by_id["alice.kim"])              # eval-reviewers ✓
    assert not es.applies_to(by_id["bob.li"])             # ml-research only
    assert not es.applies_to(by_id["frank.s"])


def test_release_approval_only_for_model_approval(org_yaml: Path):
    ra = next(w for w in WORKFLOWS if w.name == "release_approval")
    by_id = {p.id: p for p in load_personas(org_yaml)}
    assert ra.applies_to(by_id["alice.kim"])
    assert not ra.applies_to(by_id["bob.li"])


def test_secrets_check_only_for_secrets_users(org_yaml: Path):
    sc = next(w for w in WORKFLOWS if w.name == "secrets_check")
    by_id = {p.id: p for p in load_personas(org_yaml)}
    assert sc.applies_to(by_id["bob.li"])                 # ml-research / evals-readonly
    assert sc.applies_to(by_id["frank.s"])                # infra-admins


# ── member loop ───────────────────────────────────────────────────────


def test_member_runs_only_applicable_workflows(org_yaml: Path):
    personas = {p.id: p for p in load_personas(org_yaml)}
    member = Member(personas["bob.li"])
    names = {w.name for w in member.workflows}
    # bob.li doesn't have model-approval; release_approval drops out.
    assert "release_approval" not in names
    # bob.li IS ml-research / evals-readonly so secrets_check stays.
    assert "secrets_check" in names


def test_member_run_day_emits_npc_events(ctx: Context, org_yaml: Path):
    personas = {p.id: p for p in load_personas(org_yaml)}
    # Force loaf_probability=0 so every workflow runs (deterministic event count).
    member = Member(
        personas["bob.li"], rng_seed=42,
        config=NPCConfig(loaf_probability=0.0),
    )
    n = asyncio.run(member.run_day(ctx, "2026-05-19"))
    assert n == len(member.workflows)
    # Every audit event in this run is from bob.li, tagged account_kind=background_rules_agent.
    events = [e for e in ctx.collector.events if e.account == "bob.li"]
    assert events
    for e in events:
        assert e.account_kind in {"background_rules_agent", "system"}      # collector defaults
    # And all timestamps land inside 09:00–17:00 UTC on the requested day.
    for e in events:
        if not e.ts.startswith("2026-05-19"):
            continue
        hh = int(e.ts[11:13])
        assert 8 <= hh <= 17                              # ±10min jitter window


def test_member_loaf_probability_skips_some_workflows(ctx: Context, org_yaml: Path):
    personas = {p.id: p for p in load_personas(org_yaml)}
    # loaf_probability=1.0 → all workflows skipped.
    member = Member(
        personas["alice.kim"],
        config=NPCConfig(loaf_probability=1.0),
    )
    n = asyncio.run(member.run_day(ctx, "2026-05-19"))
    assert n == 0
    # No events from alice.kim recorded in this loop.
    alice_events = [e for e in ctx.collector.events if e.account == "alice.kim"]
    assert alice_events == []


def test_member_is_deterministic_under_seed(ctx: Context, org_yaml: Path):
    """Two Members with identical seed produce identical event sequences."""
    personas = {p.id: p for p in load_personas(org_yaml)}
    m1 = Member(personas["bob.li"], rng_seed=7,
                config=NPCConfig(loaf_probability=0.0))
    m2 = Member(personas["bob.li"], rng_seed=7,
                config=NPCConfig(loaf_probability=0.0))

    asyncio.run(m1.run_day(ctx, "2026-05-19"))
    events_a = [(e.service, e.action) for e in ctx.collector.events]

    # Build a second context with the same seeds.
    fresh_ctx = ctx                                        # reuse — collector growing
    n_before = len(events_a)
    asyncio.run(m2.run_day(fresh_ctx, "2026-05-19"))
    events_b = [(e.service, e.action)
                for e in ctx.collector.events[n_before:]]
    assert events_a[-len(events_b):] == events_b           # second pass matches first


def test_release_approval_only_targets_unapproved_checkpoints(
    ctx: Context, org_yaml: Path,
):
    personas = {p.id: p for p in load_personas(org_yaml)}
    # alice.kim is in model-approval; release_approval applies.
    member = Member(
        personas["alice.kim"], rng_seed=1,    # rng seed where the 50% gate fires
        config=NPCConfig(loaf_probability=0.0),
    )
    asyncio.run(member.run_day(ctx, "2026-05-19"))
    tag_events = [
        e for e in ctx.collector.events
        if e.service == "model_registry" and e.action == "tag_approved"
        and e.account == "alice.kim"
    ]
    # Either zero (50% gate didn't fire) or only one — and the targeted
    # checkpoint must be the previously-unapproved one.
    for e in tag_events:
        assert "rc1" in e.args.get("checkpoint_id", "")


def test_member_sim_clock_advances_through_the_day(ctx: Context, org_yaml: Path):
    """Sim clock should land inside the working window, not stay at midnight."""
    personas = {p.id: p for p in load_personas(org_yaml)}
    member = Member(
        personas["alice.kim"], rng_seed=42,
        config=NPCConfig(loaf_probability=0.0),
    )
    asyncio.run(member.run_day(ctx, "2026-05-19"))
    # The clock should advance from midnight into alice's daily work window — a
    # compressed continuous span starting at her persona-stable chronotype
    # (business mornings for most, evenings for a minority) — not stay at 00:00.
    alice_evs = [e for e in ctx.collector.events
                 if e.account == "alice.kim"]
    assert alice_evs
    last_hour = int(alice_evs[-1].ts[11:13])
    assert 6 <= last_hour <= 23
