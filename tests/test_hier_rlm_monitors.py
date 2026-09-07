"""Mechanics tests for the hierarchical (map-reduce) and recursive (RLM-lite) level 2
judges, driven by a scripted fake LLM — no network. Validates: cross-day context,
full-body retention (no 600-char shear), the RLM command loop, and the dispatch handlers.
"""
from __future__ import annotations

import asyncio
import types

from mole.audit.schema import AuditEvent
from mole.monitors.llm_judge.hier import HierJudgeMonitor
from mole.monitors.llm_judge.rlm import RLMJudgeMonitor


def ev(account, day, action, args=None, service="owncloud", external=False):
    e = AuditEvent.new(account=account, account_kind="human", service=service,
                       action=action, resource_id="r", args=args or {}, ts=f"{day}T10:00:00Z")
    e.is_external = external
    return e


class FakeLLM:
    """Routes by system prompt: summary (compress) / RLM loop (investigating) / judge."""

    def __init__(self, script=None, judge='{"score": 0.8, "rationale": "t"}', summary="routine day."):
        self.script = list(script or [])
        self.judge = judge
        self.summary = summary
        self.calls = []

    async def complete(self, messages, *, temperature=0.0, max_tokens=2000, **kw):
        self.calls.append(messages)
        sys = messages[0].content if messages else ""
        if "compress" in sys:
            return types.SimpleNamespace(content=self.summary)
        if "investigating ONE account" in sys:
            return types.SimpleNamespace(
                content=self.script.pop(0) if self.script else 'FINAL {"score":0.5,"rationale":"d"}')
        return types.SimpleNamespace(content=self.judge)


# ── hierarchical ────────────────────────────────────────────────────
def test_hier_uses_prior_summaries_and_keeps_full_body():
    p = "alice.k"
    world = [ev(p, "2025-01-01", "read"),
             ev(p, "2025-01-02", "upload", {"body": "x" * 5000}, external=True)]
    fake = FakeLLM(judge='{"score":0.82,"rationale":"egress"}', summary="uploaded a large file")
    m = HierJudgeMonitor(fake, world_events=world, window_days=7, today_arg_chars=4000)
    today = [e for e in world if e.ts[:10] == "2025-01-02"]
    s = asyncio.run(m.score_episode(today))

    assert 0.81 <= s.score <= 0.83                              # judge verdict parsed
    assert ("alice.k", "2025-01-01") in m._summary_cache       # prior day was summarized + cached
    judge_calls = [c for c in fake.calls if "reviewing one account" in c[0].content]
    assert judge_calls, "judge call should use the flat-judge system prompt"
    user = judge_calls[-1][1].content
    assert "uploaded a large file" in user                     # prior-day summary injected (cross-day)
    assert "x" * 601 in user                                   # body kept past the 600-char flat cap


def test_hier_falls_back_without_world():
    m = HierJudgeMonitor(FakeLLM(), window_days=7)
    s = asyncio.run(m.score_episode([ev("z", "2025-01-03", "read")]))
    assert 0.0 <= s.score <= 1.0


# ── RLM-lite ────────────────────────────────────────────────────────
def test_rlm_command_handlers():
    p = "bob.j"
    evs = [ev(p, "2025-01-01", "read"),
           ev(p, "2025-01-02", "upload", {"body": "SECRET" * 2000}, external=True)]
    m = RLMJudgeMonitor(types.SimpleNamespace())               # llm unused for handler unit tests
    assert "2025-01-02" in m._cmd_peek(evs) and "external" in m._cmd_peek(evs)
    assert "[1]" in m._cmd_grep(evs, "upload")
    exp = m._cmd_expand(evs, 1)
    assert "SECRET" in exp and "FULL ARGS" in exp              # full untruncated body on demand
    assert "read" in m._cmd_day(evs, "2025-01-01")
    assert "out of range" in m._cmd_expand(evs, 99)
    assert "no matches" in m._cmd_grep(evs, "zzz-nope")


def test_rlm_loop_navigates_then_finals():
    p = "carol.x"
    world = [ev(p, "2025-01-01", "read"),
             ev(p, "2025-01-02", "upload", {"body": "x" * 3000}, external=True)]
    fake = FakeLLM(script=["PEEK", "GREP upload", "EXPAND 1", 'FINAL {"score":0.9,"rationale":"exfil"}'])
    m = RLMJudgeMonitor(fake, world_events=world, max_steps=6)
    today = [e for e in world if e.ts[:10] == "2025-01-02"]
    s = asyncio.run(m.score_episode(today))

    assert 0.89 <= s.score <= 0.91                             # FINAL verdict parsed
    assert len(fake.calls) == 4                                # navigated 3 commands then finalized


def test_rlm_step_budget_exhaustion_is_safe():
    p = "dave.y"
    world = [ev(p, "2025-01-02", "read")]
    fake = FakeLLM(script=["PEEK", "PEEK", "PEEK"])            # never FINALs
    m = RLMJudgeMonitor(fake, world_events=world, max_steps=3)
    s = asyncio.run(m.score_episode(world))
    assert 0.0 <= s.score <= 1.0                               # no crash, defined score
