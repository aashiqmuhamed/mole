"""AgenticMember unit test — scripted fake LLM, no live calls / no sandbox.

Drives one benign agent session over the in-process generator managers and asserts the
emitted events are attributed to the persona with kind 'background_llm_agent'.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from mole.audit.collector import AuditCollector
from mole.llm.base import ChatMessage, ChatResponse, ToolCall, ToolSchema
from mole.generator.agentic_member import AgenticMember
from mole.generator.member import NPCConfig
from mole.generator.persona_loader import load_personas
from mole.generator.run import build_context


class _ScriptedLLM:
    """Turn 1: call one in-process tool. Turn 2: stop (no tool calls)."""
    backend = "fake"
    model_id = "fake-npc"

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self._calls = 0

    async def complete(self, messages, *, tools=None, temperature=0.0,
                       max_tokens=None, seed=None):
        self._calls += 1
        if self._calls == 1:
            return ChatResponse(content="", tool_calls=[
                ToolCall(id="c1", name=self.tool_name, arguments={})])
        return ChatResponse(content="done for now", tool_calls=[])


def test_agentic_member_attributes_events_to_persona(tmp_path: Path):
    async def _run():
        out = tmp_path / "npc.jsonl"
        collector = AuditCollector(jsonl_path=out)
        ctx = await build_context(collector, tmp_dir=tmp_path)   # rules envs, in-process
        # Pick a real async tool from the in-process managers.
        from mole.agent.tools import build_tools_for
        built = build_tools_for(ctx)
        tools, _ = built
        tool_name = next(t.name for t in tools
                         if t.name in ("model_registry.list_checkpoints",
                                       "eval_server.baseline", "secrets_store.list_keys"))
        persona = next(p for p in load_personas() if p.id == "bob.li")
        m = AgenticMember(persona, llm=_ScriptedLLM(tool_name),
                          config=NPCConfig(sessions_per_day=1), rng_seed=0)
        executed = await m.run_day(ctx, "2026-04-06")
        collector.close()
        return executed, collector

    executed, collector = asyncio.run(_run())
    assert executed == 1
    npc_events = [e for e in collector.events if e.account_kind == "background_llm_agent"]
    assert npc_events, "no events attributed to the agentic NPC"
    assert all(e.account == "bob.li" for e in npc_events)
    assert not any(e.is_malicious for e in collector.events)    # benign baseline
