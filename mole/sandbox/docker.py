"""Per-session docker-compose driver.

Drives `docker compose -p <session_id> -f <compose_file>` for the lifecycle
of one task. Unlike traditional agent-harness sandboxes, there is no "main"
agent container: the agent runs in the host Python process and reaches
service ports on the loopback interface. This keeps the driver tiny —
~80 LOC including port discovery, no `exec`, no file copy.

Multi-file compose chains (base + overlay) are supported by `LabSandbox`,
which subclasses this and emits a `-f a -f b ...` flag chain.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

from .base import ExecResult, Sandbox

logger = logging.getLogger(__name__)


async def _async_run(cmd: str, timeout_s: int = 300, *, check: bool = False) -> ExecResult:
    """Spawn a shell command, capture stdout/stderr/exit-code, time-bound.

    `check=False` (default) never raises — the caller inspects `return_code`. This
    matches the snapshot/restore best-effort calls (`docker pause`/`unpause`/`volume
    rm`) that tolerate failure. `check=True` raises RuntimeError on a non-zero exit.
    """
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        if check:
            raise RuntimeError(f"command timed out after {timeout_s}s: {cmd}")
        return ExecResult(stdout="", stderr=f"timed out after {timeout_s}s", return_code=-1)
    result = ExecResult(
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        return_code=proc.returncode or 0,
    )
    if check and result.return_code != 0:
        raise RuntimeError(
            f"command failed ({result.return_code}): {cmd}\n{result.stderr[:500]}")
    return result


class DockerSandbox(Sandbox):
    """Single-compose-file docker driver. Subclass for multi-file overlays."""

    # (service-name, container-port) pairs to look up after `up`.
    # Override on subclasses for the lab/overlay services.
    SERVICE_PORTS: tuple[tuple[str, int], ...] = ()

    def __init__(
        self,
        session_id: str,
        compose_files: Iterable[Path] | Path,
        *,
        service_ports: Iterable[tuple[str, int]] | None = None,
    ) -> None:
        files = [Path(f) for f in ([compose_files] if isinstance(compose_files, Path) else compose_files)]
        if not files:
            raise ValueError("at least one compose file is required")
        self.session_id = session_id
        self._compose_files: list[Path] = files
        self._service_ports: tuple[tuple[str, int], ...] = tuple(
            service_ports if service_ports is not None else self.SERVICE_PORTS
        )
        # Keyed by container_port — kept for callers that look up unambiguous
        # services (gitlab 8929, rocketchat 3000, etc.). When two services
        # share a container port (owncloud 80 + plane-proxy 80), the
        # last-discovered wins here; use ports_by_service for disambiguation.
        self.ports: dict[int, int] = {}
        # Keyed by (service_name, container_port) — always unambiguous.
        self.ports_by_service: dict[tuple[str, int], int] = {}

    # ── command building ────────────────────────────────────────────

    @property
    def compose_files(self) -> list[Path]:
        """The compose files this sandbox drives — passed to snapshot_sandbox so
        it enumerates containers via the same project, not a fragile label filter."""
        return list(self._compose_files)

    def _compose_cmd(self, subcmd: str) -> str:
        flags = " ".join(f"-f {f}" for f in self._compose_files)
        return f"docker compose -p {self.session_id} {flags} {subcmd}"

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("sandbox %s: starting", self.session_id)
        result = await _async_run(self._compose_cmd("up -d --wait"), timeout_s=900)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose up failed for sandbox {self.session_id}:\n{result.stderr}"
            )

        # Discover dynamically-allocated host ports.
        for service, container_port in self._service_ports:
            r = await _async_run(self._compose_cmd(f"port {service} {container_port}"))
            if r.return_code != 0:
                logger.debug(
                    "port lookup failed for %s:%d (service may not be in this compose): %s",
                    service, container_port, r.stderr.strip()[:200],
                )
                continue
            line = r.stdout.strip()
            if not line:
                logger.debug("no bound port for %s:%d", service, container_port)
                continue
            try:
                host_port = int(line.rsplit(":", 1)[-1])
                self.ports[container_port] = host_port
                self.ports_by_service[(service, container_port)] = host_port
                logger.info("  %s:%d → host port %d", service, container_port, host_port)
            except (ValueError, IndexError):
                logger.warning("could not parse port for %s:%d (output: %r)",
                               service, container_port, line)

        logger.info("sandbox %s: up; %d port(s) discovered", self.session_id, len(self.ports))

    async def stop(self, delete: bool = True) -> None:
        logger.info("sandbox %s: stopping", self.session_id)
        subcmd = "down" + (" -v --remove-orphans" if delete else "")
        await _async_run(self._compose_cmd(subcmd), timeout_s=120)
