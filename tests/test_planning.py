"""Tests for the planning stage (#3).

3.1 — scripted campaign decomposition + scheduling (build_campaign):
  - decomposable threats (incl. the newly-added 04/05) yield a stage->trigger pair;
  - non-decomposable threats (collusion c1/c2) yield ONE single agent-day (K=1),
    never an empty schedule — so multi-day mode never falls back to the coin-flip;
  - one campaign per insider by default; deterministic; dates stay in the window.
"""
from __future__ import annotations

import asyncio
import random

from mole.audit.collector import AuditCollector
from mole.agent.tools import build_tools_for
from mole.llm.base import ChatResponse, ToolCall
from mole.generator.agentic_member import AgenticMember
from mole.generator.campaign import build_campaign, PHASE_DECOMP
from mole.generator.member import NPCConfig
from mole.generator.persona_loader import load_personas
from mole.generator.run import build_context

DATES = [f"2026-04-{d:02d}" for d in range(6, 20)]   # 14 consecutive days


def _sched(threat: str, seed: int = 0, **kw) -> list[dict]:
    return build_campaign(threat, DATES, random.Random(seed), **kw)


def test_decomposable_threats_yield_stage_then_trigger():
    for sc in ("01_ip_theft", "06_weight_exfil", "07_eval_manip", "08_data_poisoning"):
        s = _sched(sc)
        assert [p["phase"] for p in s] == ["stage", "trigger"], (sc, s)
        assert s[0]["directive"] and s[1]["directive"], sc
        assert s[0]["date"] < s[1]["date"], sc          # stage precedes trigger


def test_04_and_05_now_decompose():
    # These used to be single-act (absent from PHASE_DECOMP, falling back to coin-flip).
    assert "04_fraud" in PHASE_DECOMP and "05_sabotage" in PHASE_DECOMP
    for sc in ("04_fraud", "05_sabotage"):
        assert [p["phase"] for p in _sched(sc)] == ["stage", "trigger"]


def test_collusion_is_single_day_never_empty():
    # Non-decomposable -> one single agent-day (K=1), NOT [] (which would coin-flip).
    for sc in ("c1_release_bypass", "c2_split_exfil"):
        s = _sched(sc)
        assert len(s) == 1, (sc, s)
        assert s[0]["phase"] == "single"
        assert s[0]["directive"] is None
        assert s[0]["date"] in DATES


def test_one_campaign_per_insider_by_default():
    s = _sched("06_weight_exfil")
    assert sum(1 for p in s if p["phase"] == "stage") == 1   # one pair, not two
    assert len(s) == 2


def test_more_campaigns_when_requested():
    s = _sched("06_weight_exfil", n_campaigns=2)
    assert sum(1 for p in s if p["phase"] == "stage") == 2   # two pairs (K=4)


def test_schedule_is_deterministic():
    assert _sched("06_weight_exfil", seed=42) == _sched("06_weight_exfil", seed=42)
    assert _sched("c1_release_bypass", seed=7) == _sched("c1_release_bypass", seed=7)


def test_all_dates_within_window():
    for sc in ("01_ip_theft", "05_sabotage", "c2_split_exfil"):
        for p in _sched(sc):
            assert p["date"] in DATES


def test_short_window_falls_back_to_single_day():
    s = build_campaign("06_weight_exfil", ["2026-04-06", "2026-04-07"], random.Random(0))
    assert len(s) == 1 and s[0]["phase"] == "single"


def test_empty_window_returns_empty():
    assert build_campaign("06_weight_exfil", [], random.Random(0)) == []


# ── 3.2 — agentic plan authoring (agent-chosen timing, fixed K) ────────────────

class _PlanLLM:
    """Mock LLM for the single-shot planning call: returns fixed content."""
    backend = "fake"
    model_id = "fake"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        return ChatResponse(content=self.content, tool_calls=[])


def _insider(llm, **kw):
    persona = next(p for p in load_personas(include_holdouts=True) if p.id == "bob.li")
    return AgenticMember(persona, llm=llm,
                         config=NPCConfig(sessions_per_day=1, loaf_probability=0.0),
                         rng_seed=0, persona_override="OBJECTIVE_MARKER: do the harm",
                         planning_mode="agentic", sim_dates=DATES, **kw)


def test_agentic_author_honors_agent_chosen_dates():
    m = _insider(_PlanLLM("My plan:\n2026-04-08\n2026-04-15\n"), campaign_days=2)
    sched = asyncio.run(m._author_campaign(org=None, sim_now="2026-04-06T09:00:00Z"))
    assert [p["date"] for p in sched] == ["2026-04-08", "2026-04-15"]
    assert all(p["phase"] == "act" for p in sched)
    assert "action-day 1 of 2" in sched[0]["directive"]
    assert "FINAL action day" in sched[1]["directive"]


def test_agentic_author_exactly_K_even_if_model_over_picks():
    m = _insider(_PlanLLM("\n".join(DATES)), campaign_days=3)   # model lists all 14 days
    sched = asyncio.run(m._author_campaign(org=None, sim_now="2026-04-06T00:00:00Z"))
    assert len(sched) == 3


def test_agentic_author_falls_back_to_spread_on_refusal():
    m = _insider(_PlanLLM("I will not plan an attack."), campaign_days=3)   # no dates
    sched = asyncio.run(m._author_campaign(org=None, sim_now="2026-04-06T00:00:00Z"))
    assert len(sched) == 3                          # base rate preserved despite refusal
    assert all(p["date"] in DATES for p in sched)


def test_agentic_author_ignores_out_of_window_dates():
    m = _insider(_PlanLLM("2025-01-01\n2026-04-09\n2030-12-31"), campaign_days=1)
    sched = asyncio.run(m._author_campaign(org=None, sim_now="2026-04-06T00:00:00Z"))
    assert [p["date"] for p in sched] == ["2026-04-09"]


def _benign_tool(ctx) -> str:
    tools, _ = build_tools_for(ctx)
    return next(t.name for t in tools
                if t.name in ("model_registry.list_checkpoints", "eval_server.baseline",
                              "secrets_store.list_keys"))


class _AgenticSessionLLM:
    """First (tool-less) call = planning -> dates; session calls -> one tool then done."""
    backend = "fake"
    model_id = "fake"

    def __init__(self, plan_dates, tool_name):
        self.plan_dates = plan_dates
        self.tool_name = tool_name
        self.systems: list[str] = []
        self._planned = False

    async def complete(self, messages, *, tools=None, **kw):
        if tools is None and not self._planned:         # the planning call
            self._planned = True
            return ChatResponse(content="\n".join(self.plan_dates), tool_calls=[])
        if len(messages) == 2:                           # session start
            self.systems.append(messages[0].content)
            return ChatResponse(content="",
                                tool_calls=[ToolCall(id="c1", name=self.tool_name, arguments={})])
        return ChatResponse(content="done", tool_calls=[])


def test_agentic_mode_authors_on_first_session_and_attacks_on_chosen_day(tmp_path):
    async def _run():
        collector = AuditCollector(jsonl_path=tmp_path / "npc.jsonl")
        ctx = await build_context(collector, tmp_dir=tmp_path)
        dates = ["2026-04-06", "2026-04-07", "2026-04-08"]
        llm = _AgenticSessionLLM(["2026-04-07"], _benign_tool(ctx))    # agent picks day 2
        persona = next(p for p in load_personas(include_holdouts=True) if p.id == "bob.li")
        m = AgenticMember(persona, llm=llm,
                          config=NPCConfig(sessions_per_day=1, loaf_probability=0.0),
                          rng_seed=0, persona_override="OBJECTIVE_MARKER",
                          planning_mode="agentic", campaign_days=1, sim_dates=dates)
        await m.run_day(ctx, "2026-04-06")     # authors here; 04-06 not chosen -> benign
        await m.run_day(ctx, "2026-04-07")     # chosen -> attack
        collector.close()
        return llm, m

    llm, m = asyncio.run(_run())
    assert [p["date"] for p in m.campaign_schedule] == ["2026-04-07"]  # agent-chosen, persisted
    assert "OBJECTIVE_MARKER" in llm.systems[-1]                       # attack day uses the override
    assert "action-day 1 of 1" in llm.systems[-1]                     # agentic directive injected
    assert "OBJECTIVE_MARKER" not in llm.systems[0]                    # day 1 was benign

