"""State-manager base class + registry for ctx-backends.

Each backend (gitlab, owncloud, email, etc.) is a `StateManager` subclass
decorated with `@StateManager.register("<name>")`. The decorator adds the
class to a module-global dict; tasks list their needed backends via
`METADATA["environments"]`, and `CompositeStateManager` instantiates them
from the registry at run time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..sandbox.base import Sandbox


class StateManager(ABC):
    """Abstract base for one per-service backend (gitlab, owncloud, email, ...)."""

    # Override to False on pure-data backends (org directory, etc.) that need
    # no docker container. The orchestrator skips sandbox setup if every
    # requested backend has NEEDS_SANDBOX == False.
    NEEDS_SANDBOX: bool = True

    _registry: dict[str, type["StateManager"]] = {}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self._sandbox: Sandbox | None = None

    # ── registry ────────────────────────────────────────────────────

    @classmethod
    def register(cls, name: str):
        """Decorator: `@StateManager.register("gitlab")` adds the subclass."""
        def _decorator(subclass: type[StateManager]) -> type[StateManager]:
            cls._registry[name] = subclass
            return subclass
        return _decorator

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> "StateManager":
        if name not in cls._registry:
            raise ValueError(
                f"Unknown environment: {name!r}. Registered: {sorted(cls._registry)}"
            )
        return cls._registry[name](config=config)

    # ── lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    async def setup(self, *, sandbox: Sandbox) -> None:
        """Initialize client connections. Called once before any stage runs."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Tear down any resources created during the task."""

    async def reset(self, *, sandbox: Sandbox) -> None:
        """Return the backend's state to a clean baseline between sweep runs.

        Called by SandboxPool when one run finishes and the next is about
        to start, without tearing down the surrounding sandbox. The
        caller (CompositeStateManager.reset_all) passes the sandbox
        explicitly — we don't carry it on `self` so the implementation
        stays stateless.

        Default implementation: just re-run setup() against the same
        sandbox. Right for pure-data backends (org / secrets_store /
        model_registry / eval_server) whose state lives entirely in
        process memory and is rebuilt from yaml each time.

        Container-backed backends (email / gitlab / owncloud /
        rocketchat / plane) MUST override this. The general pattern:
          1. Scrub state the agent could have created (mailboxes,
             commits, files, channel messages, tickets, jobs).
          2. If the service can't be cleaned from outside (e.g.,
             GitLab repos), have the caller invoke
             Sandbox.reset_service(name) before us, then we
             re-run any post-warmup seed (e.g., refresh root-token).
          3. Re-call setup() to re-establish client connections.
        """
        await self.setup(sandbox=sandbox)
