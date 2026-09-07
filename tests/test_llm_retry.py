"""Tests for the LLM retry layer.

Covers the contract: which exceptions are retried, which propagate
immediately, that the backoff schedule is exponential with jitter, and
that the agent loop's existing error path still receives the *final*
exception when retries are exhausted.
"""
from __future__ import annotations

import asyncio
import os
import random
from typing import Any
from unittest.mock import MagicMock

import pytest

from mole.llm.base import ChatMessage, ChatResponse, TokenUsage
from mole.llm.retry import (
    RetryingLLMClient,
    RetryPolicy,
    compute_backoff,
    is_retryable,
    policy_from_env,
    wrap_with_retry,
)


# ── helpers ───────────────────────────────────────────────────────────


class _FailingThenOk:
    """Fake LLMClient that raises the given exceptions, then returns a response."""

    backend = "fake"
    model_id = "fake-model"

    def __init__(self, fails_with: list[BaseException], ok: ChatResponse | None = None) -> None:
        self._fails = list(fails_with)
        self._ok = ok or ChatResponse(
            content="done", usage=TokenUsage(input_tokens=1, output_tokens=1),
            backend="fake", model_id="fake-model",
        )
        self.calls = 0

    async def complete(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        if self._fails:
            raise self._fails.pop(0)
        return self._ok


def _msgs() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="hi")]


def _mk_openai_status_error(status: int, code: str | None = None) -> Exception:
    """Construct a fake exception that looks like openai's APIStatusError."""
    exc = Exception(f"openai {status}")
    exc.status_code = status                     # what is_retryable inspects
    if code is not None:
        exc.code = code
    return exc


# ── is_retryable ──────────────────────────────────────────────────────


def test_retryable_status_codes_default_policy():
    p = RetryPolicy()
    # 403 is retryable on purpose: some hosted providers return a transient
    # empty-body 403 for a stretch, then recover, so we wait it out like a 503
    # rather than silently gut a long generation run. See retry.py docstring.
    for code in (403, 408, 429, 500, 502, 503, 504):
        assert is_retryable(_mk_openai_status_error(code), p), code


def test_non_retryable_4xx_status_codes():
    p = RetryPolicy()
    # 401 stays non-retryable: a dead/expired token needs a re-mint, not a re-ask.
    # 403 is deliberately NOT here anymore (see the retryable test above).
    for code in (400, 401, 404, 422):
        assert not is_retryable(_mk_openai_status_error(code), p), code


def test_500_with_model_error_code_is_retryable():
    p = RetryPolicy()
    assert is_retryable(_mk_openai_status_error(500, code="model_error"), p)


def test_500_with_unknown_code_is_NOT_retryable():
    """A 500 carrying a code we don't recognise stays a hard failure.
    This is the deliberate guard against retrying genuine bugs."""
    p = RetryPolicy()
    assert not is_retryable(
        _mk_openai_status_error(500, code="invalid_request_error"), p,
    )


def test_known_exception_class_names_are_retryable():
    p = RetryPolicy()
    # Each of these matches by class-name lookup in retry_on_exception_names.
    class APIConnectionError(Exception): pass
    class APITimeoutError(Exception): pass
    class RemoteProtocolError(Exception): pass
    assert is_retryable(APIConnectionError("blip"), p)
    assert is_retryable(APITimeoutError("slow"), p)
    assert is_retryable(RemoteProtocolError("disconnect"), p)


def test_unknown_exception_class_is_not_retryable():
    p = RetryPolicy()
    assert not is_retryable(ValueError("bad input"), p)
    assert not is_retryable(TypeError("nope"), p)


def test_message_substring_fallback_catches_socket_resets():
    """Even unknown exception types are retried if the message smells transient."""
    p = RetryPolicy()
    assert is_retryable(Exception("Connection reset by peer"), p)
    assert is_retryable(Exception("Server disconnected without sending a response"), p)


def test_status_on_nested_response_attribute():
    """httpx-style HTTPStatusError wraps the response under .response.status_code."""
    p = RetryPolicy()
    exc = Exception("http error")
    exc.response = MagicMock()
    exc.response.status_code = 503
    assert is_retryable(exc, p)


# ── compute_backoff ───────────────────────────────────────────────────


def test_compute_backoff_is_exponential_with_jitter():
    p = RetryPolicy(initial_backoff_s=1.0, max_backoff_s=60.0, jitter=0.0)
    # No jitter → exact powers of two.
    assert compute_backoff(0, p) == 1.0
    assert compute_backoff(1, p) == 2.0
    assert compute_backoff(2, p) == 4.0
    assert compute_backoff(3, p) == 8.0


def test_compute_backoff_caps_at_max():
    p = RetryPolicy(initial_backoff_s=1.0, max_backoff_s=5.0, jitter=0.0)
    assert compute_backoff(10, p) == 5.0          # 2**10 capped


def test_compute_backoff_jitter_bounds():
    p = RetryPolicy(initial_backoff_s=10.0, max_backoff_s=60.0, jitter=0.3)
    # With jitter=0.3, returned value is in [0.7×base, 1.3×base].
    rng = random.Random(0)
    samples = [compute_backoff(0, p, rng=rng) for _ in range(200)]
    assert all(7.0 <= s <= 13.0 for s in samples)
    # Spread is non-trivial.
    assert max(samples) - min(samples) > 1.0


# ── RetryingLLMClient ────────────────────────────────────────────────


def test_retrying_client_passes_through_on_success():
    inner = _FailingThenOk(fails_with=[])
    client = RetryingLLMClient(inner, policy=RetryPolicy(max_attempts=3), sleep=_noop_sleep)
    resp = asyncio.run(client.complete(_msgs()))
    assert resp.content == "done"
    assert inner.calls == 1


def test_retrying_client_retries_until_success():
    fails = [
        _mk_openai_status_error(429),
        _mk_openai_status_error(500, code="model_error"),
    ]
    inner = _FailingThenOk(fails_with=fails)
    client = RetryingLLMClient(
        inner,
        policy=RetryPolicy(max_attempts=5),
        sleep=_noop_sleep,
    )
    resp = asyncio.run(client.complete(_msgs()))
    assert resp.content == "done"
    assert inner.calls == 3                       # 2 failures then success


def test_retrying_client_raises_non_retryable_immediately():
    inner = _FailingThenOk(fails_with=[_mk_openai_status_error(401)])
    client = RetryingLLMClient(inner, policy=RetryPolicy(max_attempts=5),
                               sleep=_noop_sleep)
    with pytest.raises(Exception) as info:
        asyncio.run(client.complete(_msgs()))
    assert info.value.status_code == 401
    assert inner.calls == 1                       # never retried


def test_retrying_client_raises_after_max_attempts():
    """All attempts fail with retryable errors → raise the last one."""
    fails = [_mk_openai_status_error(429) for _ in range(10)]
    inner = _FailingThenOk(fails_with=fails)
    client = RetryingLLMClient(inner, policy=RetryPolicy(max_attempts=3),
                               sleep=_noop_sleep)
    with pytest.raises(Exception) as info:
        asyncio.run(client.complete(_msgs()))
    assert info.value.status_code == 429
    assert inner.calls == 3                       # exactly max_attempts tries


def test_default_policy_is_unlimited():
    """The default is UNLIMITED (max_attempts=0): a transient provider outage
    must wait itself out, never abort a session with a truncated transcript."""
    assert RetryPolicy().max_attempts == 0


def test_unlimited_retries_until_success():
    """Default (unlimited) policy keeps retrying a transient failure no matter
    how many times it fails, until it finally succeeds."""
    fails = [_mk_openai_status_error(503) for _ in range(25)]
    inner = _FailingThenOk(fails_with=fails)
    client = RetryingLLMClient(inner, policy=RetryPolicy(), sleep=_noop_sleep)
    resp = asyncio.run(client.complete(_msgs()))
    assert resp.content == "done"
    assert inner.calls == 26                          # 25 failures then success


def test_max_attempts_zero_means_unlimited():
    """Explicit max_attempts=0 is the same 'unlimited' sentinel as the default."""
    fails = [_mk_openai_status_error(429) for _ in range(40)]
    inner = _FailingThenOk(fails_with=fails)
    client = RetryingLLMClient(inner, policy=RetryPolicy(max_attempts=0),
                               sleep=_noop_sleep)
    resp = asyncio.run(client.complete(_msgs()))
    assert resp.content == "done"
    assert inner.calls == 41


def test_unlimited_still_raises_non_retryable_immediately():
    """Unlimited applies only to *transient* failures — a 401 still fails fast,
    so a dead token or bad request never spins forever."""
    inner = _FailingThenOk(fails_with=[_mk_openai_status_error(401)])
    client = RetryingLLMClient(inner, policy=RetryPolicy(), sleep=_noop_sleep)
    with pytest.raises(Exception) as info:
        asyncio.run(client.complete(_msgs()))
    assert info.value.status_code == 401
    assert inner.calls == 1


def test_compute_backoff_handles_unbounded_attempt():
    """With unlimited retries `attempt` grows without bound; backoff must stay
    capped and cheap (no multi-thousand-digit 2**attempt)."""
    p = RetryPolicy(initial_backoff_s=1.0, max_backoff_s=30.0, jitter=0.0)
    assert compute_backoff(100_000, p) == 30.0


def test_retrying_client_forwards_backend_and_model_id():
    inner = _FailingThenOk(fails_with=[])
    inner.backend = "openrouter"
    inner.model_id = "gpt-5.1_2025-11-13"
    client = RetryingLLMClient(inner, policy=RetryPolicy(), sleep=_noop_sleep)
    assert client.backend == "openrouter"
    assert client.model_id == "gpt-5.1_2025-11-13"


def test_retrying_client_sleeps_between_attempts():
    """Verifies we actually call sleep with the computed backoff each time."""
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    fails = [_mk_openai_status_error(429), _mk_openai_status_error(503)]
    inner = _FailingThenOk(fails_with=fails)
    client = RetryingLLMClient(
        inner,
        policy=RetryPolicy(max_attempts=5, initial_backoff_s=0.1,
                           max_backoff_s=10.0, jitter=0.0),
        sleep=fake_sleep,
    )
    asyncio.run(client.complete(_msgs()))
    # 2 failures → 2 sleeps with no jitter → 0.1, 0.2.
    assert len(sleeps) == 2
    assert sleeps[0] == pytest.approx(0.1)
    assert sleeps[1] == pytest.approx(0.2)


def test_retrying_client_does_not_sleep_after_final_failure():
    """If the last attempt fails, raise immediately — no point sleeping."""
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    fails = [_mk_openai_status_error(429) for _ in range(5)]
    inner = _FailingThenOk(fails_with=fails)
    client = RetryingLLMClient(
        inner, policy=RetryPolicy(max_attempts=3, initial_backoff_s=0.1, jitter=0.0),
        sleep=fake_sleep,
    )
    with pytest.raises(Exception):
        asyncio.run(client.complete(_msgs()))
    # 3 attempts → 2 sleeps (between 1↔2 and 2↔3), not 3.
    assert len(sleeps) == 2


# ── empty-body 200 retry (transient success-shaped flap) ─────────────


class _EmptyThenOk:
    """Fake LLMClient that returns N empty-body responses, then a real one."""

    backend = "fake"
    model_id = "fake-model"

    def __init__(self, empties: int, *, empty: ChatResponse | None = None,
                 ok: ChatResponse | None = None) -> None:
        self._empties_left = empties
        self._empty = empty if empty is not None else ChatResponse(
            content="", backend="fake", model_id="fake-model")
        self._ok = ok or ChatResponse(
            content="done", backend="fake", model_id="fake-model")
        self.calls = 0

    async def complete(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        if self._empties_left > 0:
            self._empties_left -= 1
            return self._empty
        return self._ok


def test_empty_completion_is_retried_until_content():
    """An empty-body 200 (no content, no tool call) is re-rolled like a 503."""
    inner = _EmptyThenOk(empties=3)
    client = RetryingLLMClient(inner, policy=RetryPolicy(max_attempts=10),
                               sleep=_noop_sleep)
    resp = asyncio.run(client.complete(_msgs()))
    assert resp.content == "done"
    assert inner.calls == 4                         # 3 empties then content


def test_unlimited_policy_retries_empty_until_content():
    """The default (unlimited) policy waits out a long empty-200 window."""
    inner = _EmptyThenOk(empties=20)
    client = RetryingLLMClient(inner, policy=RetryPolicy(), sleep=_noop_sleep)
    resp = asyncio.run(client.complete(_msgs()))
    assert resp.content == "done"
    assert inner.calls == 21


def test_empty_with_tool_call_is_not_retried():
    """Empty *content* with a tool call is a legitimate turn — must NOT retry,
    else every tool-calling turn would loop forever."""
    from mole.llm.base import ToolCall
    tool_turn = ChatResponse(
        content="", tool_calls=[ToolCall(id="c1", name="f", arguments={})],
        backend="fake", model_id="fake-model")
    inner = _EmptyThenOk(empties=5, empty=tool_turn)
    client = RetryingLLMClient(inner, policy=RetryPolicy(max_attempts=10),
                               sleep=_noop_sleep)
    resp = asyncio.run(client.complete(_msgs()))
    assert resp.tool_calls and resp.tool_calls[0].name == "f"
    assert inner.calls == 1                         # returned immediately


def test_empty_returns_after_max_attempts_without_raising():
    """A bounded policy that stays empty returns the empty response — it does
    not loop forever, and does not raise (there's no exception to raise)."""
    inner = _EmptyThenOk(empties=100)               # never yields content
    client = RetryingLLMClient(inner, policy=RetryPolicy(max_attempts=3),
                               sleep=_noop_sleep)
    resp = asyncio.run(client.complete(_msgs()))
    assert resp.content == ""                        # gave up, returned the empty
    assert inner.calls == 3                          # exactly max_attempts tries


def test_empty_retry_disabled_returns_empty_immediately():
    """retry_on_empty=False (LLM_RETRY_ON_EMPTY=0) accepts the empty as-is."""
    inner = _EmptyThenOk(empties=5)
    client = RetryingLLMClient(
        inner, policy=RetryPolicy(max_attempts=10, retry_on_empty=False),
        sleep=_noop_sleep)
    resp = asyncio.run(client.complete(_msgs()))
    assert resp.content == ""
    assert inner.calls == 1                          # not retried


def test_empty_retry_sleeps_between_attempts():
    """Each empty re-roll backs off, same schedule as a transient-error retry."""
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    inner = _EmptyThenOk(empties=2)
    client = RetryingLLMClient(
        inner, policy=RetryPolicy(max_attempts=10, initial_backoff_s=0.1,
                                  max_backoff_s=10.0, jitter=0.0),
        sleep=fake_sleep)
    asyncio.run(client.complete(_msgs()))
    assert len(sleeps) == 2
    assert sleeps[0] == pytest.approx(0.1)
    assert sleeps[1] == pytest.approx(0.2)


def test_is_empty_completion_helper():
    from mole.llm.base import ToolCall
    from mole.llm.retry import _is_empty_completion
    def mk(**kw):
        return ChatResponse(backend="fake", model_id="fake-model", **kw)
    assert _is_empty_completion(mk(content=""))               # empty
    assert _is_empty_completion(mk(content="   \n "))         # whitespace only
    assert not _is_empty_completion(mk(content="hi"))         # has content
    # empty content but a tool call → legitimate turn, not "empty"
    assert not _is_empty_completion(
        mk(content="", tool_calls=[ToolCall(id="c1", name="f", arguments={})]))


# ── env + factory wiring ─────────────────────────────────────────────


def test_policy_from_env_picks_up_overrides(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("LLM_RETRY_INITIAL_BACKOFF_S", "0.25")
    monkeypatch.setenv("LLM_RETRY_MAX_BACKOFF_S", "12.5")
    monkeypatch.setenv("LLM_RETRY_JITTER", "0.5")
    p = policy_from_env()
    assert p.max_attempts == 7
    assert p.initial_backoff_s == 0.25
    assert p.max_backoff_s == 12.5
    assert p.jitter == 0.5


def test_policy_from_env_retry_on_empty(monkeypatch):
    monkeypatch.delenv("LLM_RETRY_ON_EMPTY", raising=False)
    assert policy_from_env().retry_on_empty is True       # default on
    monkeypatch.setenv("LLM_RETRY_ON_EMPTY", "0")
    assert policy_from_env().retry_on_empty is False      # explicit off
    monkeypatch.setenv("LLM_RETRY_ON_EMPTY", "1")
    assert policy_from_env().retry_on_empty is True


def test_wrap_with_retry_disabled_when_env_says_so(monkeypatch):
    monkeypatch.setenv("LLM_RETRY", "0")
    inner = _FailingThenOk(fails_with=[])
    wrapped = wrap_with_retry(inner)
    # With LLM_RETRY=0, we get the inner client back untouched.
    assert wrapped is inner


def test_wrap_with_retry_enabled_by_default(monkeypatch):
    monkeypatch.delenv("LLM_RETRY", raising=False)
    inner = _FailingThenOk(fails_with=[])
    wrapped = wrap_with_retry(inner)
    assert isinstance(wrapped, RetryingLLMClient)


# ── async helper ──────────────────────────────────────────────────────


async def _noop_sleep(_s: float) -> None:
    return None
