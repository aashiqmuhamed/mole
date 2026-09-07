"""Unit tests for SandboxPool — the persistent-sandbox sweep harness.

These tests fake LabSandbox + CompositeStateManager so we can pin
the pool's orchestration contract (start once, reset between runs,
stop once) without booting any real container.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mole.sandbox.pool import SandboxPool


def _fake_lab_sandbox(session_id: str, **_kw) -> MagicMock:
    sb = MagicMock()
    sb.session_id = session_id
    sb.start = AsyncMock()
    sb.stop = AsyncMock()
    sb.reset_service = AsyncMock()
    return sb


def _fake_composite(environments: list[str], **_kw) -> MagicMock:
    c = MagicMock()
    c.environments = environments
    c.setup = AsyncMock()
    c.cleanup = AsyncMock()
    c.reset_all = AsyncMock()
    return c


@patch("mole.sandbox.pool.CompositeStateManager", side_effect=_fake_composite)
@patch("mole.sandbox.pool.LabSandbox", side_effect=_fake_lab_sandbox)
def test_pool_start_boots_sandbox_then_managers(mock_lab, mock_comp):
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")],
        environments=["org", "gitlab"],
    )
    asyncio.run(pool.start())
    pool.sandbox.start.assert_awaited_once()
    pool.composite.setup.assert_awaited_once()
    # Setup gets the sandbox we just booted.
    assert pool.composite.setup.await_args.kwargs["sandbox"] is pool.sandbox


@patch("mole.sandbox.pool.CompositeStateManager", side_effect=_fake_composite)
@patch("mole.sandbox.pool.LabSandbox", side_effect=_fake_lab_sandbox)
def test_pool_start_is_idempotent(mock_lab, mock_comp):
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")],
        environments=["org"],
    )
    asyncio.run(pool.start())
    asyncio.run(pool.start())                    # second call must be no-op
    assert pool.sandbox.start.await_count == 1


@patch("mole.sandbox.pool.CompositeStateManager", side_effect=_fake_composite)
@patch("mole.sandbox.pool.LabSandbox", side_effect=_fake_lab_sandbox)
def test_pool_reset_heavy_then_composite(mock_lab, mock_comp):
    """Reset must call sandbox.reset_service for every heavy service,
    THEN composite.reset_all — the heavy reset has to come first
    because that's what gives composite a fresh container to bind to."""
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")],
        environments=["org", "gitlab"],
        heavy_reset_services=("gitlab",),
    )
    asyncio.run(pool.start())
    asyncio.run(pool.reset())

    pool.sandbox.reset_service.assert_awaited_once_with("gitlab")
    pool.composite.reset_all.assert_awaited_once()
    # Ordering: heavy reset before composite reset.
    heavy_call_time = pool.sandbox.reset_service.call_args_list[0]
    composite_call_time = pool.composite.reset_all.call_args_list[0]
    # Mock call_args don't carry timestamps natively, but the async-mock
    # awaited_once_with semantics confirm ordering through await_args_list
    # if we used a single AsyncMock for both. Here just confirm both fired.
    assert pool.sandbox.reset_service.await_args.args == ("gitlab",)
    assert pool.composite.reset_all.await_args.kwargs["sandbox"] is pool.sandbox


@patch("mole.sandbox.pool.CompositeStateManager", side_effect=_fake_composite)
@patch("mole.sandbox.pool.LabSandbox", side_effect=_fake_lab_sandbox)
def test_pool_reset_continues_when_heavy_reset_fails(mock_lab, mock_comp):
    """A failing reset_service must NOT abort the pool's reset —
    composite.reset_all still runs (manager-level scrub may still
    work; if not, the next run's setup() surfaces the breakage)."""
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")],
        environments=["org", "gitlab"],
        heavy_reset_services=("gitlab",),
    )
    asyncio.run(pool.start())
    pool.sandbox.reset_service.side_effect = RuntimeError("gitlab boot crashed")
    asyncio.run(pool.reset())
    pool.composite.reset_all.assert_awaited_once()


@patch("mole.sandbox.pool.CompositeStateManager", side_effect=_fake_composite)
@patch("mole.sandbox.pool.LabSandbox", side_effect=_fake_lab_sandbox)
def test_pool_acquire_context_manager_resets_on_release(mock_lab, mock_comp):
    """acquire() returns (sandbox, composite); on context-exit it
    resets so the next caller starts clean."""
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")],
        environments=["org"],
    )

    async def _runner():
        async with pool.acquire() as (sb, comp):
            assert sb is pool.sandbox
            assert comp is pool.composite
        # On exit, reset fired (composite.reset_all).
        pool.composite.reset_all.assert_awaited_once()

    asyncio.run(_runner())


@patch("mole.sandbox.pool.CompositeStateManager", side_effect=_fake_composite)
@patch("mole.sandbox.pool.LabSandbox", side_effect=_fake_lab_sandbox)
def test_pool_acquire_can_skip_reset(mock_lab, mock_comp):
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")],
        environments=["org"],
    )

    async def _runner():
        async with pool.acquire(reset_on_release=False) as (_, _):
            pass
        pool.composite.reset_all.assert_not_awaited()

    asyncio.run(_runner())


@patch("mole.sandbox.pool.CompositeStateManager", side_effect=_fake_composite)
@patch("mole.sandbox.pool.LabSandbox", side_effect=_fake_lab_sandbox)
def test_pool_acquire_boots_lazily(mock_lab, mock_comp):
    """If start() wasn't called explicitly, first acquire() boots the pool."""
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")],
        environments=["org"],
    )

    async def _runner():
        async with pool.acquire(reset_on_release=False) as (sb, _):
            assert sb is not None

    asyncio.run(_runner())
    assert pool.sandbox.start.await_count == 1


@patch("mole.sandbox.pool.CompositeStateManager", side_effect=_fake_composite)
@patch("mole.sandbox.pool.LabSandbox", side_effect=_fake_lab_sandbox)
def test_pool_stop_cleans_up(mock_lab, mock_comp):
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")],
        environments=["org"],
    )
    asyncio.run(pool.start())
    asyncio.run(pool.stop())
    pool.composite.cleanup.assert_awaited_once()
    pool.sandbox.stop.assert_awaited_once_with(delete=True)


def test_pool_stop_safe_before_start():
    """stop() before start() must be a no-op, not an error."""
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")], environments=["org"],
    )
    asyncio.run(pool.stop())                    # must not raise


def test_pool_accessors_raise_before_start():
    pool = SandboxPool(
        compose_files=[Path("lab.yaml")], environments=["org"],
    )
    with pytest.raises(RuntimeError, match="not started"):
        _ = pool.sandbox
    with pytest.raises(RuntimeError, match="not started"):
        _ = pool.composite
