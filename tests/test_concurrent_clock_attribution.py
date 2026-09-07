"""Regression test for the per-task simulated-clock attribution under
concurrency.

Companion to test_account_context.py. The audit collector reads the
sim-time for each event from a per-task ContextVar (`_sim_clock_var` in
audit/collector.py). Without that, two concurrent sessions calling
ctx.advance_clock() would race on a shared fallback and stamp each
other's events with the wrong sim_now.

This test asserts the existing fix stays in place — every event emitted
by a given session bears that session's clock value, regardless of how
many sibling sessions are in flight.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from mole.audit.collector import (
    AuditCollector,
    set_task_clock,
)


@dataclass
class _FakeMgr:
    """Minimal in-memory manager — one method we wrap so wrap_manager
    actually emits an event."""
    async def write_file(self, *, path: str = "") -> int:
        return len(path)


@pytest.mark.asyncio
async def test_concurrent_sessions_each_get_own_clock():
    """N tasks each set their own sim-time, then issue an audited tool
    call. After all tasks finish, every recorded event's `ts` matches
    the account-task's clock — not any sibling's."""
    collector = AuditCollector()
    mgr = _FakeMgr()
    # Constant account — we only care about the clock here.
    collector.wrap_manager(
        service_name="owncloud",
        manager=mgr,
        account_getter=lambda: ("test", "system"),
    )

    N = 6

    async def task(i: int) -> None:
        my_ts = f"2026-04-{i+1:02d}T09:00:00Z"
        set_task_clock(my_ts)
        # Yield several times to give sibling tasks chances to call
        # set_task_clock with THEIR values. With a contextvar-scoped
        # clock, our value survives the interleaving.
        for _ in range(3):
            await asyncio.sleep(0)
        await mgr.write_file(path=f"/p{i}")

    await asyncio.gather(*(task(i) for i in range(N)))

    # One event per task; sorted by path-index so we can pair them.
    by_path = {e.args.get("path"): e for e in collector.events}
    assert len(by_path) == N
    for i in range(N):
        ev = by_path[f"/p{i}"]
        expected = f"2026-04-{i+1:02d}T09:00:00Z"
        assert ev.ts == expected, (
            f"task {i} wrote /p{i} expecting ts={expected!r}, "
            f"but audit recorded ts={ev.ts!r} — clock raced under concurrency"
        )


@pytest.mark.asyncio
async def test_clock_default_falls_back_to_collector_clock_fn():
    """When a task hasn't called set_task_clock, the audit collector falls
    back to its set_clock() callable — covers sequential code paths."""
    collector = AuditCollector()
    collector.set_clock(lambda: "2026-04-06T00:00:00Z")
    mgr = _FakeMgr()
    collector.wrap_manager(
        service_name="owncloud", manager=mgr,
        account_getter=lambda: ("test", "system"),
    )
    # No set_task_clock call in this task.
    await mgr.write_file(path="/no-clock-set")
    ev = collector.events[-1]
    assert ev.ts == "2026-04-06T00:00:00Z"
