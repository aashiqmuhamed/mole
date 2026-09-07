"""Multi-day campaign behavior in AgenticMember — scripted fake LLM, no network.

Asserts:
  - on a STAGE date the session is PHASED: benign identity (no insider persona), with
    the stage step as the day's task -- the malicious objective is withheld until trigger;
  - on a TRIGGER date, attack mode with the TRIGGER directive;
  - on a NON-phase date the session is benign even though attack_session_rate=1.0
    (i.e. the campaign schedule overrides the i.i.d. coin-flip);
  - FLAG-OFF regression: with campaign_schedule=None the decision is governed by the
    coin-flip and NO phase directive is ever appended (default path unchanged).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from mole.audit.collector import AuditCollector
from mole.llm.base import ChatResponse, ToolCall
from mole.agent.tools import build_tools_for
from mole.generator.agentic_member import AgenticMember, _benign_session_kind
from mole.generator.member import NPCConfig
from mole.generator.persona_loader import load_personas
from mole.generator.run import build_context

PERSONA = "PERSONA_MARKER_insider_intent_text"
STAGE_DIR = "STAGE_MARKER_do_only_step_one"
TRIGGER_DIR = "TRIGGER_MARKER_now_complete_it"


class _CapturingLLM:
    """Captures the system prompt of each session (once, on the first turn), then drives
    one benign tool call so a state-changing-ish event is emitted, then stops."""
    backend = "fake"
    model_id = "fake-npc"

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.systems: list[str] = []
        self.users: list[str] = []

    async def complete(self, messages, *, tools=None, temperature=0.0,
                       max_tokens=None, seed=None):
        first_turn = len(messages) == 2          # [system, user] only -> session start
        if first_turn:
            self.systems.append(messages[0].content)
            self.users.append(messages[1].content)
            return ChatResponse(content="", tool_calls=[
                ToolCall(id="c1", name=self.tool_name, arguments={})])
        return ChatResponse(content="done for now", tool_calls=[])


def _benign_tool(ctx) -> str:
    tools, _ = build_tools_for(ctx)
    return next(t.name for t in tools
                if t.name in ("model_registry.list_checkpoints",
                              "eval_server.baseline", "secrets_store.list_keys"))


def _member(llm, **kw):
    persona = next(p for p in load_personas() if p.id == "bob.li")
    return AgenticMember(persona, llm=llm,
                         config=NPCConfig(sessions_per_day=1, loaf_probability=0.0),
                         rng_seed=0, persona_override=PERSONA, **kw)


def test_multiday_phases_and_nonphase_benign(tmp_path: Path):
    async def _run():
        collector = AuditCollector(jsonl_path=tmp_path / "npc.jsonl")
        ctx = await build_context(collector, tmp_dir=tmp_path)
        llm = _CapturingLLM(_benign_tool(ctx))
        sched = [
            {"date": "2026-04-06", "phase": "stage", "directive": STAGE_DIR},
            {"date": "2026-04-09", "phase": "trigger", "directive": TRIGGER_DIR},
        ]
        m = _member(llm, attack_session_rate=1.0, campaign_schedule=sched)
        await m.run_day(ctx, "2026-04-06")        # stage
        await m.run_day(ctx, "2026-04-07")        # non-phase -> benign
        await m.run_day(ctx, "2026-04-09")        # trigger
        collector.close()
        return llm

    llm = asyncio.run(_run())
    assert len(llm.systems) == 3
    s_stage, s_benign, s_trigger = llm.systems

    # STAGE day (PHASED): benign identity (NO insider persona), but the stage step IS the
    # day's task -- the malicious objective is withheld until the trigger. Regression guard
    # for the staging-shadow bug: STAGE_DIR must be present (it was silently dropped when the
    # rich-benign branch shadowed the staging task, collapsing multi-day to trigger-only).
    assert PERSONA not in s_stage and STAGE_DIR in s_stage and TRIGGER_DIR not in s_stage
    # NON-phase day: benign despite attack_session_rate=1.0 -> schedule overrides coin-flip.
    assert PERSONA not in s_benign and STAGE_DIR not in s_benign and TRIGGER_DIR not in s_benign
    # TRIGGER day: attack prompt + trigger directive, not stage.
    assert PERSONA in s_trigger and TRIGGER_DIR in s_trigger and STAGE_DIR not in s_trigger


def test_one_attack_session_per_attack_day(tmp_path: Path):
    """On an attack-day with MULTIPLE sessions, exactly ONE is attack-mode and the
    rest stay benign — preserving the per-session interspersing the R@K/day metric
    relies on (an all-attack day would be trivially detectable + unrealistic)."""
    async def _run():
        collector = AuditCollector(jsonl_path=tmp_path / "npc.jsonl")
        ctx = await build_context(collector, tmp_dir=tmp_path)
        llm = _CapturingLLM(_benign_tool(ctx))
        sched = [{"date": "2026-04-06", "phase": "act", "directive": STAGE_DIR}]
        persona = next(p for p in load_personas() if p.id == "bob.li")
        m = AgenticMember(persona, llm=llm,
                          config=NPCConfig(sessions_per_day=3, loaf_probability=0.0),
                          rng_seed=0, persona_override=PERSONA, campaign_schedule=sched)
        await m.run_day(ctx, "2026-04-06")        # 3 sessions, all on one attack-day
        collector.close()
        return llm

    llm = asyncio.run(_run())
    attacks = [s for s in llm.systems if PERSONA in s]
    benign = [s for s in llm.systems if PERSONA not in s]
    assert len(llm.systems) == 3
    assert len(attacks) == 1, f"expected exactly 1 attack session, got {len(attacks)}"
    assert len(benign) == 2
    assert STAGE_DIR in attacks[0] and STAGE_DIR not in (benign[0] + benign[1])


def test_flag_off_default_path_unchanged(tmp_path: Path):
    """campaign_schedule=None -> coin-flip governs, NO phase directive ever appended."""
    async def _run(rate):
        collector = AuditCollector(jsonl_path=tmp_path / f"npc_{rate}.jsonl")
        ctx = await build_context(collector, tmp_dir=tmp_path)
        llm = _CapturingLLM(_benign_tool(ctx))
        m = _member(llm, attack_session_rate=rate, campaign_schedule=None)
        await m.run_day(ctx, "2026-04-06")
        collector.close()
        return llm

    # rate=1.0 -> always attack: persona present, but never a phase directive.
    attack = asyncio.run(_run(1.0))
    assert attack.systems and PERSONA in attack.systems[0]
    assert STAGE_DIR not in attack.systems[0] and TRIGGER_DIR not in attack.systems[0]

    # rate=0.0 -> always benign: no persona, no directive.
    benign = asyncio.run(_run(0.0))
    assert benign.systems and PERSONA not in benign.systems[0]


def test_empty_schedule_is_benign_not_iid_coinflip(tmp_path: Path):
    """Collusion disabled in single-day mode returns an EMPTY schedule ([]), NOT None.

    [] must take the schedule branch and stay benign on every day even with
    attack_session_rate=1.0 -- if it fell through to None's legacy i.i.d. coin-flip it
    would attack. This is the exact bug that emitted ~40% no-op collusion 'attacks' into
    the single_K4 corpus: the disable returned None, which the member treats as the
    coin-flip path, so the malicious collusion persona kept attacking (uncoordinated)."""
    async def _run():
        collector = AuditCollector(jsonl_path=tmp_path / "npc.jsonl")
        ctx = await build_context(collector, tmp_dir=tmp_path)
        llm = _CapturingLLM(_benign_tool(ctx))
        # rate=1.0 would FORCE an attack on the i.i.d. path; the empty schedule must win.
        m = _member(llm, attack_session_rate=1.0, campaign_schedule=[])
        await m.run_day(ctx, "2026-04-06")
        await m.run_day(ctx, "2026-04-07")
        collector.close()
        return llm

    llm = asyncio.run(_run())
    assert llm.systems, "expected sessions to run"
    assert all(PERSONA not in s for s in llm.systems), (
        "empty schedule must stay benign on every day; None would have attacked via the "
        "i.i.d. coin-flip (the single_K4 collusion-leak bug)")


def test_benign_session_kind_priority():
    """Pure-helper regression guard: arc > staging > rich > single > generic. An arc step (benign
    multi-day) or a STAGE day (phased attack) MUST win over the rich-benign path (rich_sessions on
    by default) -- before the staging fix the rich branch shadowed the step and it never ran."""
    # arc wins over everything (incl. staging — mutually exclusive in practice, but priority holds)
    assert _benign_session_kind("arc step", "stage step", rich_sessions=True, has_tasks=True) == "arc"
    assert _benign_session_kind("arc step", None, rich_sessions=True, has_tasks=True) == "arc"
    # staging wins over rich/single/generic (the original shadow-bug guard)
    assert _benign_session_kind(None, "stage step", rich_sessions=True, has_tasks=True) == "staging"
    assert _benign_session_kind(None, "stage step", rich_sessions=False, has_tasks=True) == "staging"
    assert _benign_session_kind(None, "stage step", rich_sessions=True, has_tasks=False) == "staging"
    # no arc/staging -> rich when rich+tasks; else single; else generic
    assert _benign_session_kind(None, None, rich_sessions=True, has_tasks=True) == "rich"
    assert _benign_session_kind(None, None, rich_sessions=False, has_tasks=True) == "single"
    assert _benign_session_kind(None, None, rich_sessions=True, has_tasks=False) == "generic"
    assert _benign_session_kind(None, None, rich_sessions=False, has_tasks=False) == "generic"
