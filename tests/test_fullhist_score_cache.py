"""fullhist_judge must resume from the score cache; it sends the biggest prompts and runs longest.

Until now `judge.py` was the ONLY monitor with a durable cache, so a long-running judge cell
that died mid-run resumed free from that cache on the next attempt. `fullhist_judge` had none,
so an uncapped raw-full cell (many hours of calls) would lose everything to a single crash.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from mole.audit.schema import AuditEvent
from mole.monitors.llm_judge.full_history import FullHistoryJudgeMonitor


def ev(account="alice.k", day="2026-04-06"):
    return AuditEvent.new(account=account, account_kind="human", service="owncloud",
                          action="upload", resource_id="r", args={"b": "x" * 80},
                          ts=f"{day}T10:00:00Z")


class CountingLLM:
    model_id = "fake-judge"

    def __init__(self, body='{"score": 0.6, "rationale": "egress"}'):
        self.body = body
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        return types.SimpleNamespace(content=self.body)


def _cache_file(tmp_path):
    return str(tmp_path / "cache.jsonl")


def test_second_run_resumes_from_cache_without_recalling(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_SCORE_CACHE", _cache_file(tmp_path))
    world = [ev()]
    llm = CountingLLM()
    m1 = FullHistoryJudgeMonitor(llm, world_events=world)
    s1 = asyncio.run(m1.score_episode(world))
    assert llm.calls == 1

    # A fresh monitor = the next retry attempt after a CLI wedge. It must NOT re-call.
    llm2 = CountingLLM()
    m2 = FullHistoryJudgeMonitor(llm2, world_events=world)
    s2 = asyncio.run(m2.score_episode(world))
    assert llm2.calls == 0, "no resume: an 18h cell would restart from zero after a wedge"
    assert s2.score == s1.score


def test_failures_are_never_cached(tmp_path, monkeypatch):
    """A transient dip must retry on the next run, not be pinned to a bogus score forever."""
    monkeypatch.setenv("JUDGE_SCORE_CACHE", _cache_file(tmp_path))
    monkeypatch.setenv("JUDGE_EMPTY_RETRIES", "1")

    class Boom:
        model_id = "fake-judge"
        async def complete(self, messages, **kw):
            raise RuntimeError("token limit exceeded")

    world = [ev()]
    m = FullHistoryJudgeMonitor(Boom(), world_events=world)
    asyncio.run(m.score_episode(world))
    assert m._cache.stored == 0


def test_cache_tag_separates_rawfull_from_rawrecent(tmp_path, monkeypatch):
    """A short account renders the SAME prompt at any event cap, so raw-full and raw-recent
    would collide on a content-only key and silently share a score."""
    monkeypatch.setenv("JUDGE_SCORE_CACHE", _cache_file(tmp_path))
    world = [ev()]

    monkeypatch.setenv("JUDGE_MAX_EVENTS", "1500")           # H4 raw-recent
    llm_a = CountingLLM()
    m_a = FullHistoryJudgeMonitor(llm_a, world_events=world)
    asyncio.run(m_a.score_episode(world))
    assert llm_a.calls == 1

    monkeypatch.setenv("JUDGE_MAX_EVENTS", "40000")          # H5 raw-full
    llm_b = CountingLLM()
    m_b = FullHistoryJudgeMonitor(llm_b, world_events=world)
    asyncio.run(m_b.score_episode(world))
    assert llm_b.calls == 1, "raw-full reused raw-recent's cached score; the rungs collided"


def test_cache_tag_separates_backends(tmp_path, monkeypatch):
    """Two different (backend, model) judges must never share a cached score."""
    monkeypatch.setenv("JUDGE_SCORE_CACHE", _cache_file(tmp_path))
    world = [ev()]

    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    a = CountingLLM()
    a.model_id = "openai/gpt-4.1"
    asyncio.run(FullHistoryJudgeMonitor(a, world_events=world).score_episode(world))

    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    b = CountingLLM()
    b.model_id = "anthropic/claude-opus-4"
    asyncio.run(FullHistoryJudgeMonitor(b, world_events=world).score_episode(world))
    assert b.calls == 1, "opus reused gpt-4.1's cached score"
