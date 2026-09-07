"""Regression test for the account-attribution race condition.

Pre-fix behavior: ctx._account_box was a mutable dict shared across
concurrent AgenticMember sessions. When 4 sessions ran under asyncio.gather()
they all wrote to box["account"] between awaits — so by the time the
audit-collector middleware called account_getter() inside an in-flight
tool call, it read whichever sibling task had written most recently. The
2026-05-28 Opus v3 corpus had julian.x's org.add_group_member events
attributed to deepa.a1 because the writes interleaved.

This test simulates the race: N async tasks each set their own account,
yield (giving sibling tasks a chance to overwrite the old dict), then
record what the account-getter returns. With the ContextVar fix every
task sees its own value regardless of interleaving.
"""
from __future__ import annotations

import asyncio

import pytest

from mole.generator.account_context import (
    get_account,
    set_account,
)


@pytest.mark.asyncio
async def test_concurrent_tasks_each_see_their_own_account():
    """N tasks set their own account, then check the contextvar reads it back
    correctly — even after siblings have raced through their own set+yield."""
    N = 8

    async def task(name: str) -> tuple[str, str]:
        set_account(name, "background_llm_agent")
        # Yield control multiple times to give sibling tasks a chance to
        # race the contextvar set. With the old dict-box pattern, by the
        # time we re-acquire control, box["account"] would be some other
        # sibling's value. ContextVar isolates per task — get_account()
        # still returns OUR name.
        for _ in range(5):
            await asyncio.sleep(0)
        return get_account()

    results = await asyncio.gather(*(task(f"persona_{i}") for i in range(N)))

    for i, (account, kind) in enumerate(results):
        assert account == f"persona_{i}", (
            f"task {i} saw account={account!r} after interleaving — "
            f"this is the race the ContextVar fix is supposed to prevent"
        )
        assert kind == "background_llm_agent"


@pytest.mark.asyncio
async def test_default_account_is_system_before_set():
    """Before any set_account() in this task, get_account() returns the
    default ('system','system') — covers setup paths that do tool calls
    before a session formally starts."""
    # Run in a fresh task so its Context is clean.
    async def task():
        return get_account()

    account, kind = await asyncio.create_task(task())
    assert account == "system"
    assert kind == "system"


@pytest.mark.asyncio
async def test_set_account_mints_unique_task_id_per_session():
    """Every set_account() call generates a fresh task_id — needed so the
    attribution audit can pair audit events with the issuing session 1:1."""
    from mole.generator.account_context import get_task_id

    async def task(name):
        tid = set_account(name, "background_llm_agent")
        await asyncio.sleep(0)
        return tid, get_task_id()

    results = await asyncio.gather(*(task(f"p_{i}") for i in range(6)))
    returned_tids = [r[0] for r in results]
    observed_tids = [r[1] for r in results]
    # Every task got a unique id.
    assert len(set(returned_tids)) == 6
    # The id returned by set_account matches the id read back via get_task_id
    # in the same task (no race even across yields).
    assert returned_tids == observed_tids


@pytest.mark.asyncio
async def test_nested_set_in_subtask_does_not_leak_to_parent():
    """ContextVar.set() in a child task doesn't propagate to the parent.
    This guarantees no cross-task contamination — even if a tool implementation
    awkwardly spawns its own task, the parent's account stays intact."""
    set_account("parent", "background_llm_agent")

    async def child():
        set_account("child", "background_rules_agent")
        return get_account()

    child_p, _ = await asyncio.create_task(child())
    parent_p, _ = get_account()

    assert child_p == "child"
    assert parent_p == "parent"
