"""Async-safe per-task account attribution via contextvars.

Replaces the older `ctx._account_box` mutable-dict pattern, which silently
race-corrupted audit-event attribution under concurrency. With concurrency>1
multiple AgenticMember sessions all mutated the same dict between awaits, so
when a tool call passed through the audit middleware the middleware read
whichever sibling-task happened to have written `box["account"]` most
recently — NOT the account of the task that actually issued the call.

Symptom seen in the 2026-05-28 v3 corpora: julian.x's `org.add_group_member`
priv-esc attack (clearly visible in the julian.x transcript) was attributed
to deepa.a1 in the labeled audit log, so the harm matcher correctly skipped
it (deepa.a1 doesn't own threat 03). After this fix, each session sees
its own contextvar value because asyncio.gather() per-task isolation
preserves Context, and the middleware reads the right account regardless
of which other tasks are mid-flight.

Use:
    from .account_context import set_account, get_account

    async def run_at(self, ctx, sim_now, action_key):
        set_account(self.persona.id, "background_llm_agent")
        ...

The wired `account_getter` in `audit/collector.py::wrap_manager` calls
`get_account()`, which returns the contextvar-scoped value.
"""
from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

# Module-level ContextVars. Defaults cover code paths that haven't yet
# called set_account() — typically setup/seeding, not in-session tool
# calls, so "system" is the right attribution.
_account_cv: ContextVar[str] = ContextVar("account", default="system")
_kind_cv: ContextVar[str] = ContextVar("kind", default="system")
# Per-session task UUID — set fresh at each set_account() so every audit
# event from one session shares a stable id. This lets an audit-attribution pass
# pair events with sessions deterministically,
# without args-fuzzy heuristics. Default "" so pre-session events (setup,
# seeding) are uniformly unmarked rather than carrying a random id.
_task_id_cv: ContextVar[str] = ContextVar("task_id", default="")


def set_account(account: str, kind: str, task_id: str | None = None) -> str:
    """Set the per-task account for audit attribution.

    Call at the start of each session's run_at()/run_day() before any tool
    calls. asyncio.gather() preserves Context per task, so concurrent
    sessions don't see each other's values.

    Mints a fresh task_id if none is supplied; callers should pass the
    returned id to the transcript writer so transcripts and audit events
    can be cross-referenced 1:1.
    """
    _account_cv.set(account)
    _kind_cv.set(kind)
    if task_id is None:
        task_id = uuid4().hex
    _task_id_cv.set(task_id)
    return task_id


def get_account() -> tuple[str, str]:
    """Return (account_id, account_kind) for the current task.

    Used as the `account_getter` callback wired into the audit middleware.
    """
    return _account_cv.get(), _kind_cv.get()


def get_task_id() -> str:
    """Return the current session's task id, or "" if unset."""
    return _task_id_cv.get()
