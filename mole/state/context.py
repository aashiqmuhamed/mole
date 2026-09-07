"""TaskContext — handed to every stage function and every rubric checker.

Each registered StateManager is exposed on the context under its registry
name (e.g. `ctx.email`, `ctx.gitlab`, `ctx.owncloud`). Tasks read fixtures
from `ctx.task_dir`. The sandbox is reachable as `ctx.sandbox` if a stage
needs the underlying port map. The audit collector is reachable as
`ctx.audit` so oracle checkers can query the event log without touching
disk. `ctx.task_metadata` exposes the task's METADATA dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class TaskContext:
    def __init__(
        self,
        managers: dict[str, Any],
        sandbox: Any,
        task_dir: Path,
        *,
        audit: Any = None,
        task_metadata: dict[str, Any] | None = None,
        sim_start: str | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.task_dir = task_dir
        self.audit = audit                              # AuditCollector (may be None for dry-run/tests)
        self.task_metadata: dict[str, Any] = task_metadata or {}
        self.sim_start: str | None = sim_start          # ISO8601 marker for "task started"
        self._sim_now: str | None = sim_start            # advances per stage / explicit calls
        self._managers = managers
        for name, mgr in managers.items():
            setattr(self, name, mgr)
        # Convenience alias.
        if "filesystem" in managers:
            self.fs = managers["filesystem"]

    @property
    def sim_now(self) -> str | None:
        """The current simulated wall-clock time the audit log is using."""
        return self._sim_now

    def advance_clock(self, iso_ts: str) -> None:
        """Set the simulated clock to `iso_ts` (ISO-8601).

        All subsequent audit events emitted by wrapped state-manager methods
        will be timestamped with this value until the next advance_clock or
        the next stage. Call this at the top of a stage_fn if you want
        events from the stage body to use a specific in-universe time;
        otherwise the orchestrator advances the clock between stages from
        each stage's return-value `time` field.

        Sets BOTH the per-task contextvar (so `asyncio.gather`-parallelized
        sessions in the generator each carry their own clock) AND the
        global fallback (for sequential code paths and backwards compat).
        """
        self._sim_now = iso_ts
        if self.audit is not None:
            from ..audit.collector import set_task_clock
            set_task_clock(iso_ts)
            self.audit.set_clock(lambda v=iso_ts: v)
