"""Rich-session realism patch (multi-day flag) — fake-LLM, no network.

Asserts:
  - rich mode turns a benign session into a multi-item agenda (the loop feeds the
    remaining items via the `next_task` continuation); default does ONE task and stops;
  - rich mode advances the sim clock within a session so events get spread timestamps;
    default leaves every event on the one per-session timestamp;
  - flag-off (`rich_sessions=False`) is the unchanged single-task / single-timestamp path.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from mole.audit.collector import AuditCollector
from mole.llm.base import ChatResponse, ToolCall
from mole.agent.tools import build_tools_for
from mole.generator.agentic_member import AgenticMember
from mole.generator.member import NPCConfig
from mole.generator.persona_loader import load_personas

SIM_NOW = "2026-04-06T09:00:00"
TASKS = [f"Routine item {c}" for c in "ABCDEFGH"]


def _persona(pid="bob.li"):
    return next(p for p in load_personas() if p.id == pid)


class _AlwaysStop:
    """Finishes every item immediately (no tool call) — so rich mode keeps feeding the
    next agenda item until the queue is exhausted."""
    backend = "fake"
    model_id = "fake"

    def __init__(self):
        self.calls = 0
        self.system = None
        self.next_count = 0

    async def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=None, seed=None):
        self.calls += 1
        if messages and messages[0].role == "system":
            self.system = messages[0].content
        c = sum(1 for m in messages
                if m.role == "user" and "Next on your agenda:" in (m.content or ""))
        self.next_count = max(self.next_count, c)
        return ChatResponse(content="ok", tool_calls=[])


class _NToolThenStop:
    """Calls one in-process tool on the first `n` turns (≥1 event per turn), then stops."""
    backend = "fake"
    model_id = "fake"

    def __init__(self, tool: str, n: int = 2):
        self.tool = tool
        self.n = n
        self.calls = 0

    async def complete(self, messages, *, tools=None, temperature=0.0, max_tokens=None, seed=None):
        self.calls += 1
        if self.calls <= self.n:
            return ChatResponse(content="", tool_calls=[
                ToolCall(id=f"c{self.calls}", name=self.tool, arguments={})])
        return ChatResponse(content="done", tool_calls=[])


def _import_build_context():
    from mole.generator.run import build_context
    return build_context


def test_agenda_continuation_rich_vs_default(tmp_path: Path):
    build_context = _import_build_context()

    def run(rich):
        async def go():
            coll = AuditCollector(jsonl_path=tmp_path / f"npc_{rich}.jsonl")
            ctx = await build_context(coll, tmp_dir=tmp_path)
            p = _persona()
            llm = _AlwaysStop()
            m = AgenticMember(p, llm=llm,
                              config=NPCConfig(sessions_per_day=1, loaf_probability=0.0),
                              rng_seed=0, task_bank={p.team: list(TASKS)}, rich_sessions=rich)
            await m.run_at(ctx, SIM_NOW, "session_0")
            coll.close()
            return llm
        return asyncio.run(go())

    rich = run(True)
    assert rich.calls >= 2, "rich session should work multiple agenda items"
    assert rich.next_count >= 1, "rich session should feed a 'Next on your agenda' continuation"
    assert "several things to handle" in (rich.system or ""), "rich uses the work-block prompt"

    deflt = run(False)
    assert deflt.calls == 1, "default session does ONE task then stops"
    assert deflt.next_count == 0, "default never feeds a continuation"
    assert "focus this session" in (deflt.system or ""), "default uses the single-task prompt"


def test_intra_session_clock_spread(tmp_path: Path):
    build_context = _import_build_context()

    def run(rich):
        async def go():
            coll = AuditCollector(jsonl_path=tmp_path / f"clk_{rich}.jsonl")
            ctx = await build_context(coll, tmp_dir=tmp_path)
            tools, _ = build_tools_for(ctx)
            tname = next(t.name for t in tools
                         if t.name in ("model_registry.list_checkpoints",
                                       "eval_server.baseline", "secrets_store.list_keys"))
            p = _persona()
            # empty task bank → no agenda; isolates the clock-advance behavior
            m = AgenticMember(p, llm=_NToolThenStop(tname, 2),
                              config=NPCConfig(sessions_per_day=1, loaf_probability=0.0),
                              rng_seed=0, task_bank={}, rich_sessions=rich)
            await m.run_at(ctx, SIM_NOW, "session_0")
            coll.close()
            return [e.ts for e in coll.events if e.account == p.id]
        return asyncio.run(go())

    rich_ts = run(True)
    assert len(rich_ts) >= 2, "expected ≥2 events"
    assert len(set(rich_ts)) >= 2, "rich mode should spread events across timestamps"

    def_ts = run(False)
    assert len(def_ts) >= 2
    assert len(set(def_ts)) == 1, "default mode keeps all events on one per-session timestamp"
