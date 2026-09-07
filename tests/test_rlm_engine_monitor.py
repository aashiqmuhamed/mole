"""Mechanics for the RLM-engine level 2 judge, driven by a scripted fake. No network.

The load-bearing test is `test_day_chunks_rejoin_to_flat_render`: it pins the apples-to-apples
invariant that makes the raw-full comparison meaningful. If chunking ever stops reproducing
`FullHistoryJudgeMonitor`'s prompt byte-for-byte, the cell measures rendering differences instead
of the access path, and the whole experiment is confounded.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from mole.audit.schema import AuditEvent
from mole.monitors.llm_judge.judge import FAILED_SCORE, _SYS, _parse, _render

rlm_engine = pytest.importorskip("mole.monitors.llm_judge.rlm_engine",
                                 reason="needs the optional `rlms` engine")
pytest.importorskip("rlm", reason="needs the optional `rlms` engine")

RLMEngineJudgeMonitor = rlm_engine.RLMEngineJudgeMonitor
_has_score = rlm_engine._has_score


def ev(account="alice.k", day="2026-04-06", hour=10, action="upload", args=None):
    return AuditEvent.new(account=account, account_kind="human", service="owncloud",
                          action=action, resource_id="r", args=args or {"body": "x" * 900},
                          ts=f"{day}T{hour:02d}:00:00Z")


def world_3d(p="alice.k"):
    return [ev(p, "2026-04-06", 9), ev(p, "2026-04-06", 11),
            ev(p, "2026-04-07", 10),
            ev(p, "2026-04-08", 8), ev(p, "2026-04-08", 15)]


class ScriptedFake:
    """Plays an orchestrator: probes `context`, then submits via the answer dict."""
    model_id = "fake"

    def __init__(self, verdict='{"score": 0.83, "rationale": "external egress"}'):
        self.verdict = verdict
        self.turn = 0
        self.systems: list[str] = []

    async def complete(self, messages, *, temperature=0.0, max_tokens=None, **kw):
        self.turn += 1
        self.systems.append(messages[0].content if messages else "")
        if self.turn == 1:
            body = "Probing.\n```repl\nprint(len(context))\n```"
        else:
            body = (f'Done.\n```repl\nanswer["content"] = {self.verdict!r}\n'
                    'answer["ready"] = True\n```')
        return types.SimpleNamespace(content=body,
                                     usage=types.SimpleNamespace(input_tokens=9, output_tokens=3))


# ── the invariant ────────────────────────────────────────────────────
def test_day_chunks_rejoin_to_flat_render():
    """Rejoined chunks == the flat judge's prompt, byte-for-byte. Only the ACCESS PATH may differ."""
    world = world_3d()
    m = RLMEngineJudgeMonitor(ScriptedFake(), world_events=world)
    chunks = m._day_chunks(world)
    flat, _ = _render(world, max_arg_chars=600)

    assert len(chunks) == 3, "one chunk per day"
    assert "\n".join(chunks) == flat


def test_chunks_respect_the_arg_cap(monkeypatch):
    monkeypatch.setenv("JUDGE_ARG_CHARS", "50")
    world = world_3d()
    m = RLMEngineJudgeMonitor(ScriptedFake(), world_events=world)
    flat, _ = _render(world, max_arg_chars=50)
    assert "\n".join(m._day_chunks(world)) == flat


def test_history_is_causal_no_future_leak():
    world = world_3d()
    m = RLMEngineJudgeMonitor(ScriptedFake(), world_events=world)
    today = [e for e in world if e.ts[:10] == "2026-04-07"]
    causal = [e for e in m._by_account["alice.k"] if e.ts[:10] <= "2026-04-07"]
    assert len(m._day_chunks(causal)) == 2, "the 04-08 events must not be visible on 04-07"


# ── the prompt ───────────────────────────────────────────────────────
def test_root_prompt_carries_sys_verbatim():
    m = RLMEngineJudgeMonitor(ScriptedFake(), world_events=world_3d())
    rp = m._root_prompt("alice.k", "2026-04-08", 3)
    assert _SYS in rp, "the rubric must be identical to the flat judge's"
    assert "context[-1] is 2026-04-08" in rp


def test_sys_cannot_be_a_custom_system_prompt():
    """Pins the trap: _SYS holds a literal {"score": ...} and the engine .format()s
    custom_system_prompt, so wiring it there raises. This is WHY _SYS lives in root_prompt."""
    with pytest.raises(KeyError):
        _SYS.format(custom_tools_section="")


# ── failure handling ─────────────────────────────────────────────────
@pytest.mark.parametrize("content,expected", [
    ('{"score": 0.7, "rationale": "x"}', True),
    ('prose then {"score":0.0,"rationale":"benign"} trailing', True),
    ('score: 0.7', True),
    ('{"rationale": "no score key"}', False),      # _parse yields 0.0 here -> must NOT count
    ('completely unparseable prose', False),
    ('', False),
])
def test_has_score_agrees_with_parse(content, expected):
    assert _has_score(content) is expected
    if expected:
        _parse(content)                              # a real score must parse


def test_unparseable_engine_output_is_failed_not_benign(monkeypatch):
    monkeypatch.setenv("RLM_EMPTY_RETRIES", "1")
    monkeypatch.setenv("RLM_ENV", "local")           # subprocess kernel is slow; wiring is the SUT
    m = RLMEngineJudgeMonitor(ScriptedFake(verdict="I could not decide"), world_events=world_3d())
    s = asyncio.run(m.score_episode([ev("alice.k", "2026-04-08", 8)]))

    # The whole point: junk must not land on 0.0, which _parse also returns for "confidently benign".
    assert s.score == FAILED_SCORE and s.score < 0.0
    assert "JUDGE_FAILED" in s.rationale
    with pytest.raises(RuntimeError, match="DEGRADED CELL"):
        m.raise_if_degraded()


def test_subcall_counter_reads_the_attribute_that_actually_exists():
    """rlm 0.1.3's REPLResult declares a dataclass field `llm_calls` but its hand-written
    __init__ assigns `self.rlm_calls`, so the annotated name never exists at runtime. Reading
    it yields 0 and makes a real orchestration look like the degenerate depth-0 case — which
    is exactly what the first pilot reported. Pin the real attribute."""
    from rlm.core.types import REPLResult

    r = REPLResult(stdout="", stderr="", locals={}, execution_time=0.1,
                   rlm_calls=["call-a", "call-b"])
    assert not hasattr(r, "llm_calls"), "if this fails, upstream fixed the name; keep both paths"
    assert rlm_engine._n_subcalls(r) == 2
    assert rlm_engine._n_subcalls(REPLResult(stdout="", stderr="", locals={},
                                             execution_time=0.1)) == 0
    assert rlm_engine._n_subcalls(None) == 0


# ── end-to-end through the real engine ───────────────────────────────
def test_engine_end_to_end_scores_and_parses(monkeypatch):
    monkeypatch.setenv("RLM_ENV", "local")
    fake = ScriptedFake()
    m = RLMEngineJudgeMonitor(fake, world_events=world_3d())
    s = asyncio.run(m.score_episode([ev("alice.k", "2026-04-08", 8)]))

    assert 0.82 <= s.score <= 0.84
    assert s.rationale == "external egress"
    assert m._fail.n_failed == 0
    m.raise_if_degraded()
    # The engine's own scaffold must own the system slot (this is the stated confound).
    assert any("Recursive Language Model" in c for c in fake.systems)
    assert any("orchestrator, not a solver" in c for c in fake.systems)
