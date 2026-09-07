"""Retry layer for LLMClient.

Wraps any client implementing the LLMClient Protocol and re-issues
`complete()` on transient failures with jittered exponential backoff.

What this catches that the OpenAI SDK's built-in retry does NOT:

  - `openai.InternalServerError` with `code='model_error'`. The
    "model produced invalid content" 500 fires when the model emits
    output the wrapper can't parse (malformed tool-call JSON, content
    that trips a post-filter, etc.). OpenAI SDK treats this as a
    request-level failure and does not retry. Empirically it's
    transient ~80% of the time — a single re-roll of the same prompt
    usually succeeds.
  - 408/502/503/504 in some SDK versions.
  - `httpx.RemoteProtocolError` mid-handshake (RocketChat-style
    "Server disconnected without a response").
  - 403 PermissionDenied from **some hosted providers**. A 403 is normally a
    permanent auth failure, but a few providers' per-instance completion
    *grant* flaps: it returns an empty-body 403 for a stretch, then recovers,
    while the token stays valid. Treating it as permanent silently guts long
    generation runs (every session in the dead window fails closed → empty
    days), so we wait the grant back like a 503. A genuinely revoked grant
    just retries until an operator notices — the same trade we already accept
    for a long 503 outage.
  - An **empty-body 200**. The same grant flap can also surface as a 200
    whose body has no content and no tool call — a "successful" call that
    returned nothing. Status-based retry can't see it (200 == success), so
    accepting it writes an empty turn into the transcript: exactly the
    silently-degraded corpus the unlimited retry exists to prevent. We treat
    it as transient and re-roll like a 503. A tool-call turn legitimately has
    empty content, so the check also requires no tool call. Gated by
    `retry_on_empty` (LLM_RETRY_ON_EMPTY=0).

What it deliberately does NOT retry:

  - 401 Unauthorized. A bad/expired token won't fix itself by re-asking on
    the same credential — that needs a re-mint — so fail loud. (403 is the
    exception, handled above: for some providers it's a transient grant flap.)
  - 4xx validation errors (400 with no `model_error` code). Same: a
    bad request stays bad.
  - Synchronous exceptions that don't smell transient
    (ValueError, KeyError, TypeError in our own code).

Tuning surface (env vars, all optional):
  LLM_RETRY                      "0" to disable; anything else is on
  LLM_RETRY_MAX_ATTEMPTS         default 0 = UNLIMITED (retry transient
                                 failures forever until success). Set a
                                 positive integer to cap. Unlimited is the
                                 default so a transient provider outage waits
                                 itself out instead of aborting a session with
                                 a truncated transcript — a silently degraded
                                 corpus the auth-failure breaker can't catch.
                                 Non-retryable errors (4xx, bugs) still raise
                                 immediately regardless.
  LLM_RETRY_INITIAL_BACKOFF_S    default 1.0
  LLM_RETRY_MAX_BACKOFF_S        default 30.0
  LLM_RETRY_JITTER               default 0.3 (±30 %)
  LLM_RETRY_ON_EMPTY             "0" to disable the empty-body-200 retry;
                                 anything else (default) leaves it on.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

from .base import ChatMessage, ChatResponse, LLMClient, ToolSchema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Tunable retry parameters. All times in seconds."""
    # 0 (or negative) = UNLIMITED: retry transient failures forever until they
    # succeed. This is the default because aborting a session mid-generation on
    # a transient provider outage writes a truncated transcript — a silently
    # degraded corpus the auth-failure breaker won't catch. A positive value
    # caps the attempts (used by tests and any caller that wants fail-fast).
    max_attempts: int = 0
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    jitter: float = 0.3
    # Retry an empty-body 200 — a response with no content AND no tool call —
    # as if it were a transient failure. This is a provider grant-flap tell: a 200
    # that returns nothing. Status-based retry can't catch it (200 == success),
    # so it's a separate gate. A tool-call turn legitimately has empty content,
    # so the emptiness check (see `_is_empty_completion`) also requires no tool
    # call, else we'd retry every tool-calling turn forever.
    retry_on_empty: bool = True
    # HTTP status codes that signal "try again later". 403 is included
    # because some providers' per-instance completion grant
    # *flaps* — empty-body 403 for a stretch, then it recovers, token still
    # valid (see the module docstring). Not retrying it silently guts long
    # generation runs. 401 is deliberately NOT here: a dead token needs a
    # re-mint, not a re-ask.
    retry_on_status: tuple[int, ...] = (403, 408, 429, 500, 502, 503, 504)
    # OpenAI-style error codes inside a 500 body that we still want to retry.
    retry_on_codes: tuple[str, ...] = (
        "model_error",
        "server_error",
        "service_unavailable",
    )
    # Exception class names (kept as strings so we don't import optional deps).
    retry_on_exception_names: tuple[str, ...] = (
        "APIConnectionError",
        "APITimeoutError",
        "ConnectionError",
        "ReadTimeout",
        "ConnectTimeout",
        "RemoteProtocolError",
        "TimeoutError",
    )


def policy_from_env(default: RetryPolicy | None = None) -> RetryPolicy:
    """Construct a RetryPolicy from `LLM_RETRY_*` env vars (with defaults)."""
    d = default or RetryPolicy()
    return RetryPolicy(
        max_attempts=int(os.environ.get("LLM_RETRY_MAX_ATTEMPTS", d.max_attempts)),
        initial_backoff_s=float(
            os.environ.get("LLM_RETRY_INITIAL_BACKOFF_S", d.initial_backoff_s),
        ),
        max_backoff_s=float(
            os.environ.get("LLM_RETRY_MAX_BACKOFF_S", d.max_backoff_s),
        ),
        jitter=float(os.environ.get("LLM_RETRY_JITTER", d.jitter)),
        retry_on_status=d.retry_on_status,
        retry_on_codes=d.retry_on_codes,
        retry_on_exception_names=d.retry_on_exception_names,
        retry_on_empty=(
            d.retry_on_empty
            if os.environ.get("LLM_RETRY_ON_EMPTY") is None
            else os.environ.get("LLM_RETRY_ON_EMPTY") != "0"
        ),
    )


def is_retryable(exc: BaseException, policy: RetryPolicy) -> bool:
    """Decide whether `exc` should be retried under `policy`."""
    cls = type(exc).__name__
    if cls in policy.retry_on_exception_names:
        return True
    # OpenAI SDK exceptions carry .status_code and sometimes .code.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in policy.retry_on_status:
        # If the body carries a code, prefer that for the 5xx-with-code case.
        code = _extract_openai_code(exc)
        if status == 500 and code is not None:
            return code in policy.retry_on_codes
        return True
    # Some exceptions wrap a response object on `.response`; check that path too.
    resp = getattr(exc, "response", None)
    if resp is not None:
        rs = getattr(resp, "status_code", None)
        if isinstance(rs, int) and rs in policy.retry_on_status:
            return True
    # Plain socket / connection errors that aren't class-matched above.
    msg = str(exc).lower()
    if any(token in msg for token in (
        "connection reset", "connection aborted", "server disconnected",
        "remote protocol error", "incomplete read",
    )):
        return True
    return False


def compute_backoff(attempt: int, policy: RetryPolicy, *, rng: random.Random | None = None) -> float:
    """Exponential backoff with multiplicative jitter, capped at max_backoff_s.

    `attempt` is 0-indexed (first retry uses attempt=0).
    """
    rng = rng or random
    # Cap the exponent before computing 2**attempt: with unlimited retries
    # `attempt` grows without bound, and 2**attempt would build a multi-thousand
    # -digit int every call. Once base exceeds max_backoff_s the cap dominates
    # anyway, so an exponent of 30 (2**30 ≈ 1e9 s) is already saturating.
    base = policy.initial_backoff_s * (2 ** min(attempt, 30))
    base = min(base, policy.max_backoff_s)
    jitter_lo = max(0.0, 1.0 - policy.jitter)
    jitter_hi = 1.0 + policy.jitter
    return base * rng.uniform(jitter_lo, jitter_hi)


class RetryingLLMClient:
    """Wraps an LLMClient and retries transient `complete()` failures.

    The wrapped client must implement the LLMClient Protocol. We forward
    `backend` and `model_id` through so callers can't tell they're talking
    to a wrapper.
    """

    def __init__(
        self,
        inner: LLMClient,
        policy: RetryPolicy | None = None,
        *,
        rng: random.Random | None = None,
        sleep: Any = None,
    ) -> None:
        self._inner = inner
        self._policy = policy or RetryPolicy()
        self._rng = rng or random.Random()
        # Allow tests to inject a fake sleep.
        self._sleep = sleep if sleep is not None else asyncio.sleep

    # Forward Protocol attributes.
    @property
    def backend(self) -> str:
        return getattr(self._inner, "backend", "")

    @property
    def model_id(self) -> str:
        return getattr(self._inner, "model_id", "")

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        # max_attempts <= 0 means UNLIMITED: keep retrying transient failures
        # forever so a provider outage waits itself out. A positive value caps.
        unlimited = self._policy.max_attempts <= 0
        cap = "unlimited" if unlimited else str(self._policy.max_attempts)
        attempts = itertools.count() if unlimited else range(self._policy.max_attempts)
        last_exc: BaseException | None = None
        for attempt in attempts:
            try:
                resp = await self._inner.complete(
                    messages, tools=tools, temperature=temperature,
                    max_tokens=max_tokens, seed=seed,
                )
            except BaseException as exc:                  # noqa: BLE001
                last_exc = exc
                if not is_retryable(exc, self._policy):
                    raise
                # Bounded policy on its last attempt? re-raise the original.
                # Unlimited never gives up on a *retryable* failure.
                if not unlimited and attempt + 1 >= self._policy.max_attempts:
                    logger.warning(
                        "LLM complete() exhausted %d attempts; last error: %r",
                        self._policy.max_attempts, exc,
                    )
                    raise
                delay = compute_backoff(attempt, self._policy, rng=self._rng)
                logger.info(
                    "LLM complete() retryable failure on attempt %d/%s (%r); "
                    "sleeping %.2fs",
                    attempt + 1, cap, exc, delay,
                )
                await self._sleep(delay)
                continue
            # The call succeeded at the HTTP layer — but an empty-body 200 (no
            # content AND no tool call) is a provider grant-flap tell: a "success"
            # that returned nothing. Accepting it writes an empty turn into the
            # transcript (the silently-degraded corpus status-based retry can't
            # catch, since 200 == success). Re-roll it like a transient failure.
            if not self._policy.retry_on_empty or not _is_empty_completion(resp):
                return resp
            if not unlimited and attempt + 1 >= self._policy.max_attempts:
                logger.warning(
                    "LLM complete() still empty-body after %d attempts; "
                    "returning the empty response",
                    self._policy.max_attempts,
                )
                return resp
            delay = compute_backoff(attempt, self._policy, rng=self._rng)
            logger.info(
                "LLM complete() empty-body 200 on attempt %d/%s (provider "
                "grant-flap tell); sleeping %.2fs",
                attempt + 1, cap, delay,
            )
            await self._sleep(delay)
        # Defensive: only reachable for a bounded policy (range exhausted) — the
        # last attempt above already returned or raised, so this is a safety net.
        assert last_exc is not None
        raise last_exc


# ── helpers ───────────────────────────────────────────────────────────


def _is_empty_completion(resp: ChatResponse) -> bool:
    """True for an empty-body 200: no text content AND no tool call.

    This is the shape some providers return during a completion-grant flap — the HTTP
    call succeeds but the model produced nothing. A tool-call turn legitimately
    carries empty content, so we must require *no tool call* as well, else we'd
    retry every tool-calling turn forever.
    """
    content = getattr(resp, "content", None) or ""
    tool_calls = getattr(resp, "tool_calls", None) or []
    return not content.strip() and not tool_calls


def _extract_openai_code(exc: BaseException) -> str | None:
    """Pull the OpenAI error `code` out of an exception body, if present."""
    # Path 1: SDK >= 1.x sets .code directly on the exception.
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    # Path 2: body is a dict {"error": {"code": "..."}} on .body
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            c = err.get("code")
            if isinstance(c, str):
                return c
    return None


def wrap_with_retry(client: LLMClient, policy: RetryPolicy | None = None) -> LLMClient:
    """Wrap `client` in retry logic unless `LLM_RETRY=0` is set."""
    if os.environ.get("LLM_RETRY") == "0":
        return client
    return RetryingLLMClient(client, policy=policy or policy_from_env())
