"""SandboxPool — long-lived sandbox + composite, reset between runs.

Bridges Phase 1 (one-off `python -m mole.run_task ...`)
and Phase A (~500 unattended rollouts).

Single-instance: one LabSandbox boots once, every run reuses it, the
pool's `release(reset=True)` calls `Sandbox.reset_service()` for the
heavy services + `CompositeStateManager.reset_all()` for the
managers. Container image cache + warmup cost are amortized across
every sweep run instead of paid per-run.

Multi-instance pool (4-way docker parallelism for the paper grid)
sits on top of this — one Pool per worker, parallel via asyncio
or multiprocessing.

Usage:

    pool = SandboxPool(
        compose_files=[Path("compose/lab.yaml")],
        environments=["org", "gitlab", "owncloud", "email", "rocketchat",
                      "model_registry", "eval_server", "secrets_store"],
        heavy_reset_services=("gitlab",),    # ask Sandbox.reset_service for these
    )
    await pool.start()
    try:
        for cfg in sweep_configs:
            async with pool.acquire() as (sandbox, composite):
                result = await run_task_full(
                    task_dir=cfg.task_dir,
                    llm=cfg.llm,
                    sandbox=sandbox,         # reuse, skip teardown
                    composite=composite,
                )
            # pool.acquire's __aexit__ calls release(reset=True)
            sweep_outputs.append(result)
    finally:
        await pool.stop()
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable

from ..state.composite import CompositeStateManager
from .lab import LabSandbox

logger = logging.getLogger(__name__)


class SandboxPool:
    """Owns a single long-lived LabSandbox + CompositeStateManager.

    Not thread-safe; one pool per asyncio task. Compose state is
    process-wide via docker.
    """

    def __init__(
        self,
        *,
        compose_files: Iterable[Path] | Path,
        environments: list[str],
        env_config: dict[str, dict[str, Any]] | None = None,
        session_id: str | None = None,
        heavy_reset_services: tuple[str, ...] = ("gitlab",),
    ) -> None:
        self._compose_files = compose_files
        self._environments = list(environments)
        self._env_config = env_config or {}
        self._session_id = session_id or f"itb-pool-{uuid.uuid4().hex[:8]}"
        self._heavy_reset_services = tuple(heavy_reset_services)
        self._sandbox: LabSandbox | None = None
        self._composite: CompositeStateManager | None = None
        self._started = False

    @property
    def sandbox(self) -> LabSandbox:
        if self._sandbox is None:
            raise RuntimeError("SandboxPool not started; call start() first")
        return self._sandbox

    @property
    def composite(self) -> CompositeStateManager:
        if self._composite is None:
            raise RuntimeError("SandboxPool not started; call start() first")
        return self._composite

    async def start(self) -> None:
        """Boot the sandbox + initialise every manager once.

        Idempotent — second call is a no-op.
        """
        if self._started:
            logger.debug("pool %s: start() called on already-started pool", self._session_id)
            return
        logger.info("pool %s: starting sandbox + managers", self._session_id)
        self._sandbox = LabSandbox(
            session_id=self._session_id,
            compose_files=self._compose_files,
        )
        await self._sandbox.start()
        self._composite = CompositeStateManager(
            environments=self._environments,
            env_config=self._env_config,
        )
        await self._composite.setup(sandbox=self._sandbox)
        self._started = True

    async def stop(self) -> None:
        """Tear down the sandbox + composite. Final teardown for the pool."""
        if not self._started:
            return
        logger.info("pool %s: stopping", self._session_id)
        if self._composite is not None:
            try:
                await self._composite.cleanup()
            except Exception:
                logger.exception("pool %s: composite cleanup error", self._session_id)
        if self._sandbox is not None:
            try:
                await self._sandbox.stop(delete=True)
            except Exception:
                logger.exception("pool %s: sandbox stop error", self._session_id)
        self._started = False

    async def reset(self) -> None:
        """Scrub state between sweep runs without tearing down.

        Two-step:
          1. For each name in heavy_reset_services: sandbox.reset_service(name).
             Recycles the container (rm -fsv + up -d + reseed). Used
             when service state can't be cleanly scrubbed from outside
             (gitlab is the canonical case).
          2. composite.reset_all(sandbox=self._sandbox). Every manager
             does its own state scrub — email IMAP-clear, rocketchat
             channel delete, etc.
        """
        if not self._started:
            raise RuntimeError("SandboxPool not started; reset() requires start()")
        assert self._sandbox is not None and self._composite is not None
        logger.info(
            "pool %s: reset — heavy=%s + composite.reset_all",
            self._session_id, ",".join(self._heavy_reset_services) or "(none)",
        )
        for name in self._heavy_reset_services:
            try:
                await self._sandbox.reset_service(name)
            except Exception:
                logger.exception(
                    "pool %s: heavy reset of %s failed (continuing)",
                    self._session_id, name,
                )
        await self._composite.reset_all(sandbox=self._sandbox)

    @asynccontextmanager
    async def acquire(self, *, reset_on_release: bool = True):
        """Yield (sandbox, composite) for one sweep run.

        On __aexit__: if reset_on_release=True, run reset() so the
        next caller sees a clean baseline.

        Boots the pool on first acquire if start() hasn't been called.
        """
        if not self._started:
            await self.start()
        try:
            yield (self._sandbox, self._composite)
        finally:
            if reset_on_release:
                await self.reset()


__all__ = ["SandboxPool"]
