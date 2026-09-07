"""Sandbox interface for per-task service isolation.

A sandbox is a per-session group of docker-compose'd services that the
agent (running on the host) connects to over network. There is no
"main" container for the agent; it runs in the host's Python process and
reaches services on `127.0.0.1:<dynamic_port>`. That means a sandbox
only needs three things: start, stop, and a port map.

DryRunSandbox provides a no-op implementation for unit tests that want a
`Sandbox` shaped object without spinning Docker up.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int


class Sandbox(ABC):
    """Per-task service-host abstraction."""

    # Discovered host ports, keyed by container port.
    # e.g. ports[8929] = 51237  ←  GitLab inside, port 51237 on the host
    ports: dict[int, int]

    @abstractmethod
    async def start(self) -> None:
        """Spin up the sandbox and populate `self.ports`."""

    @abstractmethod
    async def stop(self, delete: bool = True) -> None:
        """Tear down. With `delete=True`, also remove volumes/networks."""


class DryRunSandbox(Sandbox):
    """No-op sandbox for unit-test fixtures that don't need Docker.

    `ports` is the legacy container-port → host-port dict; `ports_by_service`
    is the unambiguous (service, container_port) → host_port dict that
    managers should prefer to avoid port-collision bugs when two
    services share the same container port (owncloud + plane-proxy
    both on port 80).
    """

    def __init__(
        self,
        ports: dict[int, int] | None = None,
        ports_by_service: dict[tuple[str, int], int] | None = None,
    ) -> None:
        self.ports = ports or {}
        self.ports_by_service = ports_by_service or {}

    async def start(self) -> None:
        pass

    async def stop(self, delete: bool = True) -> None:
        pass
