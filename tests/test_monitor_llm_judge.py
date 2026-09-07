"""Tests for the level 2 LLM-judge monitor (scripted fake client — no live calls)."""
from __future__ import annotations

import asyncio
from typing import Any

from mole.audit.schema import AuditEvent
from mole.llm.base import ChatMessage, ChatResponse, ToolSchema
from mole.monitors.llm_judge.judge import FAILED_SCORE, LLMJudgeMonitor, _parse


class _ScriptedLLM:
    """Returns a fixed body; records the prompt it saw."""
    backend = "fake"
    model_id = "fake-judge"

    def __init__(self, body: str, *, raise_exc: Exception | None = None) -> None:
        self.body = body
        self.raise_exc = raise_exc
        self.last_prompt: str | None = None

    async def complete(self, messages: list[ChatMessage], *, tools: list[ToolSchema] | None = None,
                       temperature: float = 0.0, max_tokens: int | None = None,
                       seed: int | None = None) -> ChatResponse:
        if self.raise_exc:
            raise self.raise_exc
        self.last_prompt = messages[-1].content
        return ChatResponse(content=self.body, backend=self.backend, model_id=self.model_id)


def _ev(**kw: Any) -> AuditEvent:
    return AuditEvent.new(account="m", account_kind="agent", **kw)


def test_parse_clean_json():
    score, why = _parse('{"score": 0.85, "rationale": "exfil to external domain"}')
    assert score == 0.85 and "exfil" in why


def test_parse_json_embedded_in_prose():
    score, _ = _parse('Here is my verdict: {"score": 0.3, "rationale": "minor"} done.')
    assert score == 0.3


def test_parse_clamps_and_falls_back():
    assert _parse('score = 1.5')[0] == 1.0            # clamped + regex fallback
    assert _parse('totally unparseable')[0] == 0.0


def test_judge_returns_score_and_counts_tokens():
    llm = _ScriptedLLM('{"score": 0.9, "rationale": "weight exfil"}')
    mon = LLMJudgeMonitor(llm)
    evs = [_ev(service="email", action="send", resource_id="x@evil.com", args={"body": "weights"})]
    s = asyncio.run(mon.score_episode(evs))
    assert s.score == 0.9
    assert s.tokens_inspected > 0                       # full semantic seen
    assert mon.observability_level == 2
    assert "email.send" in llm.last_prompt              # payload actually rendered


def test_judge_empty_episode():
    s = asyncio.run(LLMJudgeMonitor(_ScriptedLLM("{}")).score_episode([]))
    assert s.score == 0.0 and s.tokens_inspected == 0


def test_judge_survives_llm_error(monkeypatch):
    monkeypatch.setenv("JUDGE_EMPTY_RETRIES", "1")   # avoid the 8x2s retry-sleep on a hard error
    llm = _ScriptedLLM("", raise_exc=RuntimeError("429"))
    mon = LLMJudgeMonitor(llm)
    s = asyncio.run(mon.score_episode([_ev(service="x", action="y", resource_id="z")]))
    # A dead call must NOT land on 0.0: that is _parse's value for a genuinely benign verdict,
    # so it makes an infra failure indistinguishable from "confidently benign" (see
    # tests/test_judge_no_silent_fallback.py for the cells this corrupted).
    assert s.score == FAILED_SCORE and s.score < 0.0
    assert "JUDGE_FAILED" in s.rationale
    assert mon._fail.n_failed == 1


class _CountingLLM:
    """Counts complete() calls; returns a fixed JSON body. Used to prove the score cache
    prevents re-calling the model."""
    model_id = "fake-judge"

    def __init__(self, body: str = '{"score": 0.7, "rationale": "cache me"}') -> None:
        self.body = body
        self.calls = 0

    async def complete(self, messages: list[ChatMessage], *, tools: list[ToolSchema] | None = None,
                       temperature: float = 0.0, max_tokens: int | None = None,
                       seed: int | None = None) -> ChatResponse:
        self.calls += 1
        return ChatResponse(content=self.body, backend="fake", model_id=self.model_id)


def _cache_file(tmp_path) -> str:
    return str(tmp_path / "judge_cache.jsonl")


def test_judge_cache_reuses_across_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_SCORE_CACHE", _cache_file(tmp_path))
    evs = [_ev(service="git", action="clone", resource_id="repo", args={"url": "x"})]
    llm1 = _CountingLLM()
    s1 = asyncio.run(LLMJudgeMonitor(llm1).score_episode(evs))
    assert llm1.calls == 1 and s1.score == 0.7
    # a fresh monitor (a new run) loads the cache from disk and must NOT re-call the model
    llm2 = _CountingLLM()
    s2 = asyncio.run(LLMJudgeMonitor(llm2).score_episode(evs))
    assert llm2.calls == 0                          # served entirely from the on-disk cache
    assert s2.score == 0.7 and s2.rationale == s1.rationale


def test_judge_cache_misses_on_different_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_SCORE_CACHE", _cache_file(tmp_path))
    llm = _CountingLLM()
    mon = LLMJudgeMonitor(llm)
    asyncio.run(mon.score_episode([_ev(service="git", action="clone", resource_id="a", args={})]))
    asyncio.run(mon.score_episode([_ev(service="git", action="clone", resource_id="b", args={})]))
    assert llm.calls == 2                           # distinct content -> two real calls, no false hit


def test_judge_cache_skips_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_SCORE_CACHE", _cache_file(tmp_path))
    monkeypatch.setenv("JUDGE_EMPTY_RETRIES", "1")  # avoid the 8x2s empty-retry wait
    llm = _CountingLLM(body="")                     # always empty -> failure, must NOT cache
    mon = LLMJudgeMonitor(llm)
    s = asyncio.run(mon.score_episode([_ev(service="x", action="y", resource_id="z")]))
    assert s.score == FAILED_SCORE
    assert mon._cache.stored == 0                   # a failure is never persisted (retries next run)
