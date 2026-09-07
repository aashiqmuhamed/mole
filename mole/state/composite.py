"""CompositeStateManager — owns the per-task set of state managers.

Given a task's declared environments list, instantiates each via the
StateManager registry, sets them up in parallel against the active sandbox,
hands out a TaskContext, and cleans up at the end.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..sandbox.base import Sandbox
from .base import StateManager
from .context import TaskContext

logger = logging.getLogger(__name__)


class CompositeStateManager:
    def __init__(
        self,
        environments: list[str],
        env_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        env_config = env_config or {}
        self.managers: dict[str, StateManager] = {
            env: StateManager.create(env, config=env_config.get(env))
            for env in environments
        }

    async def setup(self, *, sandbox: Sandbox) -> None:
        """Set up all managers in parallel. Surfaces the first exception."""
        async def _one(name: str) -> None:
            mgr = self.managers[name]
            logger.debug("setup → %s", name)
            await mgr.setup(sandbox=sandbox)

        results = await asyncio.gather(
            *[_one(n) for n in self.managers],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, BaseException):
                raise r

    async def cleanup(self) -> None:
        """Clean up all managers in parallel. Exceptions are logged, not raised."""
        async def _one(name: str) -> None:
            try:
                await self.managers[name].cleanup()
            except Exception as exc:
                logger.warning("cleanup error in %s: %s", name, exc)

        await asyncio.gather(*[_one(n) for n in self.managers])

    async def reset_all(self, *, sandbox: Sandbox) -> None:
        """Call reset() on every managed backend.

        Used by SandboxPool between sweep runs to scrub per-service
        state without tearing down the surrounding compose stack.
        Exceptions are raised — a backend that can't reset cleanly
        signals state leakage between runs, which we want loud.
        """
        async def _one(name: str) -> None:
            logger.debug("reset → %s", name)
            await self.managers[name].reset(sandbox=sandbox)

        results = await asyncio.gather(
            *[_one(n) for n in self.managers],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, BaseException):
                raise r

    def get(self, env_name: str) -> StateManager | None:
        return self.managers.get(env_name)

    def create_context(
        self,
        *,
        task_dir: Path,
        sandbox: Sandbox,
        audit: Any = None,
        task_metadata: dict[str, Any] | None = None,
        sim_start: str | None = None,
    ) -> TaskContext:
        return TaskContext(
            managers=self.managers,
            sandbox=sandbox,
            task_dir=task_dir,
            audit=audit,
            task_metadata=task_metadata,
            sim_start=sim_start,
        )
