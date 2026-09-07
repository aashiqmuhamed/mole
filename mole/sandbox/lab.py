"""LabSandbox — the docker sandbox configured with the AI-lab services we ship.

This is a thin wrapper over `DockerSandbox` that pre-populates the
`(service-name, container-port)` list with everything in the lab compose
(GitLab, OwnCloud, RocketChat, Plane, GreenMail, eval-server, model-registry,
secrets-store) so the port map is automatically populated after start().

Per-task code calls `LabSandbox.ports[<container_port>]` to find each
service's host port and connects from the agent (running on the host).

Two-phase boot. When GitLab Omnibus comes up alongside rocketchat / mongo /
owncloud / collabora, its chef-style `gitlab-ctl reconfigure` races runit:
`sv restart gitlab-workhorse` fires before the runsv supervisor has
registered the workhorse service, and GitLab exits(1) ~20s in. In
isolation the same image boots cleanly in ~80s. We work around this by
bringing GitLab up first with `up -d --wait gitlab` before the rest of
the stack. Opt out by setting LAB_TWO_PHASE_BOOT=0 (e.g., on a compose
that omits gitlab entirely so the warmup is wasted work).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from .docker import DockerSandbox, _async_run

logger = logging.getLogger(__name__)


class LabSandbox(DockerSandbox):
    SERVICE_PORTS: tuple[tuple[str, int], ...] = (
        # Workspace services. Plane is in-process (no container) — see
        # state/plane/manager.py.
        ("gitlab", 8929),
        ("owncloud", 80),
        ("rocketchat", 3000),
        # Mail (used as both an inbox surface and an external-boundary signal)
        ("greenmail", 3025),
        ("greenmail", 3143),
        ("greenmail", 8080),
        # AI-lab services
        ("eval-server", 8000),
        ("model-registry", 8001),
        ("secrets-store", 8002),
    )

    # Service name(s) we bring up first if present in the active compose.
    # Empty values are skipped silently (e.g., on lab.minimal.yaml which has
    # no gitlab).
    WARMUP_SERVICES: tuple[str, ...] = ("gitlab",)

    def __init__(
        self,
        session_id: str,
        compose_files: Iterable[Path] | Path,
        *,
        service_ports: Iterable[tuple[str, int]] | None = None,
    ) -> None:
        super().__init__(
            session_id=session_id,
            compose_files=compose_files,
            service_ports=service_ports,
        )

    async def start(self) -> None:
        """Bring up warmup services first, then the full stack.

        Falls through to `DockerSandbox.start()` if either:
          - LAB_TWO_PHASE_BOOT=0 explicitly disables the warmup
          - none of WARMUP_SERVICES are defined in the active compose
        """
        if os.environ.get("LAB_TWO_PHASE_BOOT") == "0":
            logger.info("sandbox %s: two-phase boot disabled by env", self.session_id)
            await super().start()
            return

        present = await self._warmup_services_present()
        if not present:
            logger.debug(
                "sandbox %s: no warmup services in active compose; skipping warmup",
                self.session_id,
            )
            await super().start()
            return

        logger.info(
            "sandbox %s: warming up %s before the rest of the stack",
            self.session_id, ",".join(present),
        )
        # GitLab Omnibus's chef reconfigure has a stochastic race against
        # the runit supervisor on cold boot (workhorse `sv restart` fires
        # before runsv has registered the service). shm_size raises the
        # success rate but doesn't pin it. We work around it the way TAC
        # does: bring the service up detached, poll for healthy, and if
        # the container exits during the wait window, rm -f and start
        # again. Default max 8 attempts (override via LAB_WARMUP_ATTEMPTS).
        _max_attempts = int(os.environ.get("LAB_WARMUP_ATTEMPTS", "8") or "8")
        for attempt in range(1, _max_attempts + 1):
            await self._bring_warmup_up(present)
            ok = await self._wait_warmup_healthy(present, timeout_s=600)
            if ok:
                break
            logger.warning(
                "sandbox %s: warmup attempt %d/%d — one of %s exited(1); "
                "removing and retrying (likely the GitLab runit/workhorse race)",
                self.session_id, attempt, _max_attempts, ",".join(present),
            )
            await _async_run(
                self._compose_cmd(f"rm -f -s {' '.join(present)}"),
                timeout_s=120,
            )
        else:
            raise RuntimeError(
                f"docker compose warmup failed after {_max_attempts} attempts for "
                f"sandbox {self.session_id}"
            )

        # Per-service post-warmup seeding. Each service's seed step runs
        # inside the running container via `docker compose exec` and is
        # idempotent. This is where we patch up state that the published
        # TAC images bake at build time but doesn't survive the
        # image-to-anon-volume copy cleanly (notably: GitLab's root-token
        # PAT expires 365 days after the image was built — TAC's image
        # was published 2024-11-18, so any user who runs it after
        # 2025-11-18 hits 401 unless we refresh).
        if "gitlab" in present:
            await self._seed_gitlab_root_token()

        logger.info("sandbox %s: warmup complete; bringing up the rest", self.session_id)
        await super().start()

    async def reset_service(self, name: str) -> None:
        """Heavy single-service reset between sweep runs.

        `compose rm -fsv <name>` (stop + force + with-volumes) then
        `compose up -d <name>` and poll healthy. Re-runs the
        per-service post-warmup seed (currently just gitlab root-token
        refresh).

        Used by SandboxPool when a manager.reset() can't cleanly scrub
        the service state from the outside — most relevant for GitLab,
        whose state lives across thousands of files + a postgres DB,
        and where compose-level reset is cheaper than enumerating
        state via the API.

        Inherits the same 3-attempt retry loop as warmup so the
        workhorse race doesn't poison reset, either.
        """
        logger.info(
            "sandbox %s: reset_service('%s') — rm -fsv + up -d",
            self.session_id, name,
        )
        r = await _async_run(
            self._compose_cmd(f"rm -fsv {name}"), timeout_s=120,
        )
        if r.return_code != 0:
            logger.warning(
                "sandbox %s: rm -fsv %s returned %d (%s)",
                self.session_id, name, r.return_code, r.stderr.strip()[:200],
            )

        ok = False
        for attempt in range(1, 4):
            await _async_run(
                self._compose_cmd(f"up -d {name}"), timeout_s=120,
            )
            ok = await self._wait_warmup_healthy((name,), timeout_s=600)
            if ok:
                break
            logger.warning(
                "sandbox %s: reset_service('%s') attempt %d/3 — exited(1); retrying",
                self.session_id, name, attempt,
            )
            await _async_run(
                self._compose_cmd(f"rm -fs {name}"), timeout_s=120,
            )
        if not ok:
            raise RuntimeError(
                f"sandbox {self.session_id}: reset_service('{name}') "
                "failed after 3 attempts"
            )
        # Re-run any service-specific post-warmup seed.
        if name == "gitlab":
            await self._seed_gitlab_root_token()
        # Re-discover the port in case it rotated (`rm -fsv` releases it).
        for svc, container_port in self._service_ports:
            if svc != name:
                continue
            pr = await _async_run(
                self._compose_cmd(f"port {svc} {container_port}"),
            )
            if pr.return_code == 0 and pr.stdout.strip():
                try:
                    host_port = int(pr.stdout.strip().rsplit(":", 1)[-1])
                    self.ports[container_port] = host_port
                    self.ports_by_service[(svc, container_port)] = host_port
                except (ValueError, IndexError):
                    pass

    async def _seed_gitlab_root_token(self) -> None:
        """Refresh the seeded `root-token` PAT to 1 year from today.

        The TAC GitLab image ships with a baked PAT created at image-build
        time with `expires_at: 365.days.from_now`. Once that day passes,
        every PAT-authed call returns 401 with body
        `{"error":"invalid_token","error_description":"Token is expired."}`.
        TAC's init.sh does the same Ruby snippet at image-build time;
        we replay it at runtime against the live database via
        `gitlab-rails runner`.

        Idempotent: if no root-token exists, creates one; if one exists,
        bumps its expires_at + resets its value to the literal
        "root-token" string (which is what `gl.set_token('root-token')`
        does — the displayed value matches the secret).
        """
        refresh_script = (
            "existing = User.find_by_username('root')"
            ".personal_access_tokens.where(name: 'root-token').first; "
            "if existing; "
            "  existing.expires_at = 365.days.from_now; "
            "  existing.set_token('root-token'); "
            "  existing.save!; "
            "  puts 'refreshed existing root-token'; "
            "else; "
            "  t = User.find_by_username('root').personal_access_tokens.create("
            "    scopes: ['api','read_user','read_api','read_repository',"
            "             'write_repository','sudo','admin_mode'],"
            "    name: 'root-token', expires_at: 365.days.from_now); "
            "  t.set_token('root-token'); t.save!; "
            "  puts 'created new root-token'; "
            "end"
        )
        # Wrap the script in shell-safe double quotes; rails-runner takes
        # the script as its first positional arg.
        cmd = self._compose_cmd(
            f'exec -T gitlab gitlab-rails runner "{refresh_script}"',
        )
        logger.info("sandbox %s: refreshing gitlab root-token PAT", self.session_id)
        r = await _async_run(cmd, timeout_s=180)
        if r.return_code != 0:
            # Surface but don't abort — if the token happens to be valid
            # already, manager.setup() will succeed regardless.
            logger.warning(
                "sandbox %s: gitlab root-token refresh returned %d (stderr: %s); "
                "GitLabManager will surface a real error if auth fails",
                self.session_id, r.return_code, r.stderr.strip()[:200],
            )
        else:
            logger.info(
                "sandbox %s: %s",
                self.session_id, (r.stdout or "").strip()[:200],
            )

    async def _warmup_services_present(self) -> tuple[str, ...]:
        """Return WARMUP_SERVICES that actually appear in the active compose."""
        r = await _async_run(self._compose_cmd("config --services"))
        if r.return_code != 0:
            logger.warning(
                "sandbox %s: `compose config --services` failed; assuming "
                "no warmup services present (%s)",
                self.session_id, r.stderr.strip()[:200],
            )
            return ()
        defined = {line.strip() for line in r.stdout.splitlines() if line.strip()}
        return tuple(s for s in self.WARMUP_SERVICES if s in defined)

    async def _bring_warmup_up(self, services: tuple[str, ...]) -> None:
        """`docker compose up -d <services>` — detached, no --wait."""
        await _async_run(
            self._compose_cmd(f"up -d {' '.join(services)}"),
            timeout_s=120,
        )

    async def _wait_warmup_healthy(
        self, services: tuple[str, ...], *, timeout_s: int,
    ) -> bool:
        """Poll `compose ps` until every warmup service is healthy.

        Returns False if any of the services exits (status starts with
        "Exited") during the wait window — that's the signal that the
        cold-boot race fired and we should re-create the container.
        """
        import asyncio as _asyncio
        deadline = _asyncio.get_event_loop().time() + timeout_s
        target = set(services)
        while _asyncio.get_event_loop().time() < deadline:
            r = await _async_run(
                self._compose_cmd("ps --format json"), timeout_s=30,
            )
            healthy: set[str] = set()
            exited: set[str] = set()
            if r.return_code == 0 and r.stdout.strip():
                import json as _json
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    svc = rec.get("Service") or ""
                    state = (rec.get("State") or "").lower()
                    health = (rec.get("Health") or "").lower()
                    if svc not in target:
                        continue
                    if state == "exited":
                        exited.add(svc)
                    elif health == "healthy" or (state == "running" and health == ""):
                        # Services without a healthcheck count as healthy once running.
                        # Services WITH a healthcheck need the explicit "healthy" verdict.
                        if health == "healthy":
                            healthy.add(svc)
            if exited:
                logger.warning(
                    "sandbox %s: warmup service(s) exited during wait: %s",
                    self.session_id, ",".join(sorted(exited)),
                )
                return False
            if healthy >= target:
                logger.info(
                    "sandbox %s: all warmup services healthy: %s",
                    self.session_id, ",".join(sorted(target)),
                )
                return True
            await _asyncio.sleep(5)
        logger.warning(
            "sandbox %s: warmup poll timed out after %ds", self.session_id, timeout_s,
        )
        return False
