"""Unit tests for the docker-compose sandbox + port discovery."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from mole.sandbox.base import ExecResult
from mole.sandbox.docker import DockerSandbox
from mole.sandbox.lab import LabSandbox


def test_sandbox_requires_at_least_one_compose_file():
    with pytest.raises(ValueError):
        DockerSandbox(session_id="s", compose_files=[])


def test_compose_cmd_chains_multiple_f_flags():
    sb = LabSandbox("sess-1", [Path("base.yaml"), Path("lab.yaml")])
    cmd = sb._compose_cmd("up -d")
    assert "-p sess-1" in cmd
    assert cmd.count("-f ") == 2
    assert cmd.index("base.yaml") < cmd.index("lab.yaml")
    assert cmd.endswith("up -d")


def test_compose_cmd_single_file_works():
    sb = LabSandbox("sess-1", [Path("only.yaml")])
    cmd = sb._compose_cmd("ps")
    assert cmd.count("-f ") == 1
    assert "only.yaml" in cmd


def test_compose_cmd_accepts_single_path_argument():
    sb = DockerSandbox("sess-1", Path("only.yaml"))
    cmd = sb._compose_cmd("ps")
    assert "only.yaml" in cmd
    assert cmd.count("-f ") == 1


def test_default_service_ports_cover_workspace_and_ai_services():
    sb = LabSandbox("sess-1", [Path("a.yaml")])
    service_names = {svc for svc, _port in sb._service_ports}
    # Workspace services. Plane is in-process (no container) — see
    # state/plane/manager.py for the rationale.
    assert {"gitlab", "owncloud", "rocketchat"} <= service_names
    assert "plane-proxy" not in service_names
    # Mail
    assert "greenmail" in service_names
    # AI-lab services
    assert {"eval-server", "model-registry", "secrets-store"} <= service_names


def test_default_sandbox_exposes_ports_by_service_dict():
    """ports_by_service is the canonical lookup when two services share a
    container port (owncloud 80 + plane-proxy 80)."""
    sb = LabSandbox("sess-1", [Path("a.yaml")])
    assert hasattr(sb, "ports_by_service")
    assert sb.ports_by_service == {}                     # empty pre-start


def test_service_ports_can_be_overridden_per_sandbox():
    custom = [("custom-svc", 1234), ("another", 4321)]
    sb = LabSandbox("sess-1", [Path("a.yaml")], service_ports=custom)
    assert sb._service_ports == tuple(custom)


def test_initial_ports_dict_is_empty():
    """Until start() runs, no host ports are known."""
    sb = LabSandbox("sess-1", [Path("a.yaml")])
    assert sb.ports == {}


def test_port_discovery_populates_ports_by_service_when_two_services_share_a_port():
    """Owncloud and plane-proxy both listen on container port 80. The
    legacy `ports[80]` dict can only hold one — whichever discovered
    last wins. But `ports_by_service[(svc, 80)]` is unambiguous and
    must carry BOTH host ports."""
    rec = _RunRecorder({
        "up -d --wait":      ExecResult(stdout="ok", stderr="", return_code=0),
        "port owncloud":     ExecResult(stdout="127.0.0.1:55001",
                                         stderr="", return_code=0),
        "port plane-proxy":  ExecResult(stdout="127.0.0.1:55002",
                                         stderr="", return_code=0),
    })
    # Custom SERVICE_PORTS containing exactly the two port-80 services so
    # we don't depend on the rest of the LabSandbox port list.
    sb = DockerSandbox(
        session_id="port-clash", compose_files=[Path("lab.yaml")],
        service_ports=[("owncloud", 80), ("plane-proxy", 80)],
    )
    with patch("mole.sandbox.docker._async_run", rec):
        asyncio.run(sb.start())

    # Legacy dict: last-discovered (plane-proxy) wins; owncloud overwritten.
    assert sb.ports[80] == 55002
    # Per-service dict: both entries preserved unambiguously.
    assert sb.ports_by_service[("owncloud", 80)] == 55001
    assert sb.ports_by_service[("plane-proxy", 80)] == 55002


# ── two-phase boot ────────────────────────────────────────────────────


class _RunRecorder:
    """Stand-in for `_async_run`. Records each invocation and returns a
    scripted result keyed by a substring of the command."""

    def __init__(self, responses: dict[str, ExecResult]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def __call__(self, cmd: str, timeout_s: int = 300) -> ExecResult:
        self.calls.append(cmd)
        for key, resp in self.responses.items():
            if key in cmd:
                return resp
        return ExecResult(stdout="", stderr="no scripted response", return_code=1)


def _healthy_ps_json(services: tuple[str, ...]) -> str:
    """Scripted `docker compose ps --format json` body — one record per
    service, each one healthy. Matches the polling logic in
    LabSandbox._wait_warmup_healthy."""
    import json
    lines = []
    for s in services:
        lines.append(json.dumps({
            "Service": s, "State": "running", "Health": "healthy",
        }))
    return "\n".join(lines)


def test_two_phase_boot_warmups_gitlab_first_when_present():
    """Active compose contains gitlab → `up -d gitlab` fires AND the
    poll loop confirms gitlab healthy BEFORE the full `up -d --wait`
    of the rest of the stack."""
    rec = _RunRecorder({
        "config --services": ExecResult(
            stdout="gitlab\nowncloud\nrocketchat\ngreenmail\n",
            stderr="", return_code=0,
        ),
        # Warmup phase: detached up (no --wait), then poll.
        "up -d gitlab":       ExecResult(stdout="ok", stderr="", return_code=0),
        "ps --format json":   ExecResult(
            stdout=_healthy_ps_json(("gitlab",)), stderr="", return_code=0,
        ),
        # Post-warmup seed: rails-runner for root-token refresh.
        "exec -T gitlab gitlab-rails runner": ExecResult(
            stdout="refreshed existing root-token", stderr="", return_code=0,
        ),
        # Full bringup of the stack.
        "up -d --wait":        ExecResult(stdout="ok", stderr="", return_code=0),
        "port gitlab":         ExecResult(stdout="127.0.0.1:12345",
                                          stderr="", return_code=0),
        "port owncloud":       ExecResult(stdout="127.0.0.1:12346",
                                          stderr="", return_code=0),
        "port rocketchat":     ExecResult(stdout="127.0.0.1:12347",
                                          stderr="", return_code=0),
        "port greenmail":      ExecResult(stdout="127.0.0.1:12348",
                                          stderr="", return_code=0),
    })

    sb = LabSandbox("test-tp", [Path("lab.yaml")])
    with patch("mole.sandbox.lab._async_run", rec):
        with patch("mole.sandbox.docker._async_run", rec):
            asyncio.run(sb.start())

    cmd_seq = [c for c in rec.calls if "compose" in c]
    config_idx = next(i for i, c in enumerate(cmd_seq) if "config --services" in c)
    warmup_idx = next(i for i, c in enumerate(cmd_seq) if "up -d gitlab" in c)
    seed_idx   = next(
        i for i, c in enumerate(cmd_seq)
        if "exec -T gitlab gitlab-rails runner" in c
    )
    fullup_idx = next(
        i for i, c in enumerate(cmd_seq)
        if "up -d --wait" in c and "gitlab" not in c
    )
    # Order: config → warmup-detached → seed → fullup
    assert config_idx < warmup_idx < seed_idx < fullup_idx


def test_two_phase_boot_skipped_when_compose_lacks_gitlab():
    """compose/lab.minimal.yaml has no gitlab → no warmup phase fires."""
    rec = _RunRecorder({
        "config --services": ExecResult(
            stdout="greenmail\nrocketchat\nrocketchat-mongodb\n",
            stderr="", return_code=0,
        ),
        "up -d --wait":        ExecResult(stdout="ok", stderr="", return_code=0),
        "port greenmail":      ExecResult(stdout="127.0.0.1:1", stderr="", return_code=0),
        "port rocketchat":     ExecResult(stdout="127.0.0.1:2", stderr="", return_code=0),
    })

    sb = LabSandbox("test-tp-minimal", [Path("lab.minimal.yaml")])
    with patch("mole.sandbox.lab._async_run", rec):
        with patch("mole.sandbox.docker._async_run", rec):
            asyncio.run(sb.start())

    # No detached `up -d gitlab` and no warmup-style `up -d --wait gitlab`.
    assert not any("up -d gitlab" in c for c in rec.calls)
    assert not any("up -d --wait gitlab" in c for c in rec.calls)


def test_two_phase_boot_disabled_by_env(monkeypatch):
    monkeypatch.setenv("LAB_TWO_PHASE_BOOT", "0")
    rec = _RunRecorder({
        "up -d --wait":   ExecResult(stdout="ok", stderr="", return_code=0),
        "port gitlab":    ExecResult(stdout="127.0.0.1:1", stderr="", return_code=0),
        "port owncloud":  ExecResult(stdout="127.0.0.1:2", stderr="", return_code=0),
        "port rocketchat":ExecResult(stdout="127.0.0.1:3", stderr="", return_code=0),
        "port greenmail": ExecResult(stdout="127.0.0.1:4", stderr="", return_code=0),
    })
    sb = LabSandbox("test-tp-off", [Path("lab.yaml")])
    with patch("mole.sandbox.lab._async_run", rec):
        with patch("mole.sandbox.docker._async_run", rec):
            asyncio.run(sb.start())
    # No `config --services` probe, no warmup-gitlab command.
    assert not any("config --services" in c for c in rec.calls)
    assert not any("up -d --wait gitlab" in c for c in rec.calls)


def test_two_phase_boot_raises_when_warmup_fails(monkeypatch):
    """If every warmup retry surfaces an Exited gitlab, the sandbox
    aborts. Scripted ps-json returns an exited record on every poll
    so all retry attempts fail. Pin the retry count to 3 for a fast,
    deterministic run (the default is 8, via LAB_WARMUP_ATTEMPTS).

    Note: this test patches asyncio.sleep to skip the 5s poll interval
    so the test completes in milliseconds instead of ~30s."""
    monkeypatch.setenv("LAB_WARMUP_ATTEMPTS", "3")
    import json
    exited_ps = json.dumps({
        "Service": "gitlab", "State": "exited", "Health": "",
    })
    rec = _RunRecorder({
        "config --services": ExecResult(
            stdout="gitlab\nowncloud\n", stderr="", return_code=0,
        ),
        "up -d gitlab": ExecResult(stdout="ok", stderr="", return_code=0),
        "ps --format json": ExecResult(stdout=exited_ps, stderr="", return_code=0),
        "rm -f -s gitlab": ExecResult(stdout="ok", stderr="", return_code=0),
    })
    sb = LabSandbox("test-tp-fail", [Path("lab.yaml")])
    async def _noop_sleep(_s): return None
    with patch("mole.sandbox.lab._async_run", rec):
        with patch("mole.sandbox.docker._async_run", rec):
            with patch("asyncio.sleep", _noop_sleep):
                with pytest.raises(RuntimeError, match="3 attempts"):
                    asyncio.run(sb.start())


def test_two_phase_boot_refreshes_gitlab_root_token_after_warmup():
    """After gitlab warmup, the seed step must exec gitlab-rails runner
    with our PAT-refresh script."""
    rec = _RunRecorder({
        "config --services": ExecResult(
            stdout="gitlab\nrocketchat\nowncloud\n", stderr="", return_code=0,
        ),
        "up -d gitlab":     ExecResult(stdout="ok", stderr="", return_code=0),
        "ps --format json": ExecResult(
            stdout=_healthy_ps_json(("gitlab",)), stderr="", return_code=0,
        ),
        "exec -T gitlab gitlab-rails runner": ExecResult(
            stdout="refreshed existing root-token", stderr="", return_code=0,
        ),
        "up -d --wait":   ExecResult(stdout="ok", stderr="", return_code=0),
        "port gitlab":    ExecResult(stdout="127.0.0.1:1", stderr="", return_code=0),
        "port owncloud":  ExecResult(stdout="127.0.0.1:2", stderr="", return_code=0),
        "port rocketchat":ExecResult(stdout="127.0.0.1:3", stderr="", return_code=0),
        "port greenmail": ExecResult(stdout="127.0.0.1:4", stderr="", return_code=0),
    })

    sb = LabSandbox("test-tp-seed", [Path("lab.yaml")])
    with patch("mole.sandbox.lab._async_run", rec):
        with patch("mole.sandbox.docker._async_run", rec):
            asyncio.run(sb.start())

    cmds = [c for c in rec.calls if "compose" in c]
    warmup_idx = next(i for i, c in enumerate(cmds) if "up -d gitlab" in c)
    seed_idx = next(
        i for i, c in enumerate(cmds)
        if "exec -T gitlab gitlab-rails runner" in c
    )
    fullup_idx = next(
        i for i, c in enumerate(cmds)
        if "up -d --wait" in c and "gitlab" not in c
    )
    # Order: warmup-detached → poll → seed → fullup
    assert warmup_idx < seed_idx < fullup_idx
    seed_cmd = cmds[seed_idx]
    assert "root-token" in seed_cmd
    assert "personal_access_tokens" in seed_cmd
    assert "expires_at" in seed_cmd


def test_reset_service_recycles_gitlab_and_reseeds_token():
    """SandboxPool will call this between sweep runs to scrub gitlab
    state without touching the rest of the stack. Behaviour:

      rm -fsv gitlab  →  up -d gitlab  →  poll healthy  →  rails-runner
      token refresh  →  re-discover host port.

    Order + presence of all four steps is the contract."""
    rec = _RunRecorder({
        "rm -fsv gitlab":   ExecResult(stdout="ok", stderr="", return_code=0),
        "up -d gitlab":     ExecResult(stdout="ok", stderr="", return_code=0),
        "ps --format json": ExecResult(
            stdout=_healthy_ps_json(("gitlab",)), stderr="", return_code=0,
        ),
        "exec -T gitlab gitlab-rails runner": ExecResult(
            stdout="refreshed existing root-token", stderr="", return_code=0,
        ),
        "port gitlab":      ExecResult(stdout="127.0.0.1:33333",
                                       stderr="", return_code=0),
    })
    sb = LabSandbox("test-reset-gl", [Path("lab.yaml")])
    with patch("mole.sandbox.lab._async_run", rec):
        with patch("mole.sandbox.docker._async_run", rec):
            asyncio.run(sb.reset_service("gitlab"))

    cmds = [c for c in rec.calls if "compose" in c]
    rm_idx = next(i for i, c in enumerate(cmds) if "rm -fsv gitlab" in c)
    up_idx = next(i for i, c in enumerate(cmds) if "up -d gitlab" in c)
    poll_idx = next(i for i, c in enumerate(cmds) if "ps --format json" in c)
    seed_idx = next(i for i, c in enumerate(cmds)
                    if "exec -T gitlab gitlab-rails runner" in c)
    port_idx = next(i for i, c in enumerate(cmds) if "port gitlab" in c)
    assert rm_idx < up_idx < poll_idx < seed_idx < port_idx
    # The new host port lands in sandbox.ports.
    assert sb.ports.get(8929) == 33333


def test_reset_service_retries_on_workhorse_race():
    """First boot in reset crashes (workhorse race); second succeeds.
    The reset_service loop should survive one failed attempt
    transparently."""
    import json
    exited_ps = json.dumps({
        "Service": "gitlab", "State": "exited", "Health": "",
    })
    healthy_ps = _healthy_ps_json(("gitlab",))

    # First poll: exited. Second: healthy.
    calls_seen = {"ps": 0}

    class _R:
        def __init__(self):
            self.calls = []
        async def __call__(self, cmd, timeout_s=300):
            self.calls.append(cmd)
            if "ps --format json" in cmd:
                calls_seen["ps"] += 1
                return ExecResult(
                    stdout=exited_ps if calls_seen["ps"] == 1 else healthy_ps,
                    stderr="", return_code=0,
                )
            return ExecResult(stdout="ok", stderr="", return_code=0)

    rec = _R()
    sb = LabSandbox("test-reset-retry", [Path("lab.yaml")])
    async def _noop_sleep(_s): return None
    with patch("mole.sandbox.lab._async_run", rec):
        with patch("mole.sandbox.docker._async_run", rec):
            with patch("asyncio.sleep", _noop_sleep):
                asyncio.run(sb.reset_service("gitlab"))

    up_count = sum("up -d gitlab" in c for c in rec.calls)
    assert up_count == 2                # one failed, one succeeded


def test_reset_service_raises_after_max_attempts():
    """All 3 attempts fail → reset_service raises."""
    import json
    exited_ps = json.dumps({
        "Service": "gitlab", "State": "exited", "Health": "",
    })
    rec = _RunRecorder({
        "rm -fsv gitlab":   ExecResult(stdout="ok", stderr="", return_code=0),
        "rm -fs gitlab":    ExecResult(stdout="ok", stderr="", return_code=0),
        "up -d gitlab":     ExecResult(stdout="ok", stderr="", return_code=0),
        "ps --format json": ExecResult(stdout=exited_ps, stderr="", return_code=0),
    })
    sb = LabSandbox("test-reset-fail", [Path("lab.yaml")])
    async def _noop_sleep(_s): return None
    with patch("mole.sandbox.lab._async_run", rec):
        with patch("mole.sandbox.docker._async_run", rec):
            with patch("asyncio.sleep", _noop_sleep):
                with pytest.raises(RuntimeError, match="3 attempts"):
                    asyncio.run(sb.reset_service("gitlab"))


def test_two_phase_boot_continues_when_token_seed_fails():
    """A failing seed step must NOT abort the boot."""
    rec = _RunRecorder({
        "config --services": ExecResult(
            stdout="gitlab\n", stderr="", return_code=0,
        ),
        "up -d gitlab":     ExecResult(stdout="ok", stderr="", return_code=0),
        "ps --format json": ExecResult(
            stdout=_healthy_ps_json(("gitlab",)), stderr="", return_code=0,
        ),
        "exec -T gitlab gitlab-rails runner": ExecResult(
            stdout="", stderr="rails-runner: connection refused", return_code=1,
        ),
        "up -d --wait":   ExecResult(stdout="ok", stderr="", return_code=0),
        "port gitlab":    ExecResult(stdout="127.0.0.1:1", stderr="", return_code=0),
    })

    sb = LabSandbox("test-tp-seedfail", [Path("lab.yaml")])
    with patch("mole.sandbox.lab._async_run", rec):
        with patch("mole.sandbox.docker._async_run", rec):
            asyncio.run(sb.start())
    assert any(
        "up -d --wait" in c and "gitlab" not in c
        for c in rec.calls if "compose" in c
    )
