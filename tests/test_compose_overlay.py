"""Structural tests for the per-session docker-compose overlay.

These run without Docker — they parse the YAML and assert invariants
required for safe per-session isolation:
  - No fixed `container_name` (would clash across sessions).
  - Every service joins the shared bridge network.
  - Host ports use the `127.0.0.1:0:<container>` dynamic-allocation pattern.
  - Pull policy doesn't force re-fetch on every run (slows tests; we want `missing`).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


COMPOSE_PATH = Path(__file__).parent.parent / "compose" / "lab.yaml"


@pytest.fixture(scope="module")
def compose_doc() -> dict:
    with COMPOSE_PATH.open() as f:
        return yaml.safe_load(f)


def test_compose_file_parses(compose_doc):
    assert "services" in compose_doc
    assert "networks" in compose_doc


def test_no_fixed_container_names(compose_doc):
    offenders = [
        svc for svc, cfg in compose_doc["services"].items()
        if "container_name" in cfg
    ]
    assert not offenders, f"Services with container_name would clash across sessions: {offenders}"


def test_all_services_join_bench_net(compose_doc):
    for svc, cfg in compose_doc["services"].items():
        nets = cfg.get("networks", [])
        assert "bench-net" in nets, f"Service {svc!r} not on bench-net (networks={nets})"


def test_host_ports_use_dynamic_allocation(compose_doc):
    """Every published host port must bind to 127.0.0.1:0:<container_port>."""
    for svc, cfg in compose_doc["services"].items():
        for port_spec in cfg.get("ports", []):
            # Spec is a string like "127.0.0.1:0:<port>" or shorter forms we reject.
            assert isinstance(port_spec, str), f"{svc} has non-string port spec: {port_spec!r}"
            parts = port_spec.split(":")
            assert len(parts) == 3, (
                f"{svc} port spec {port_spec!r} should be 'host_ip:host_port:container_port'"
            )
            assert parts[0] == "127.0.0.1", f"{svc} not bound to loopback: {port_spec}"
            assert parts[1] == "0", f"{svc} not using dynamic host port: {port_spec}"


def test_pull_policy_is_not_always(compose_doc):
    """`pull_policy: always` slows every up command. Prefer `missing`."""
    for svc, cfg in compose_doc["services"].items():
        if "pull_policy" in cfg:
            assert cfg["pull_policy"] != "always", (
                f"{svc} has pull_policy: always — slows per-session startup"
            )


def test_gitlab_uses_internal_service_url(compose_doc):
    """GitLab's external_url must resolve inside bench-net (not the host)."""
    gl = compose_doc["services"]["gitlab"]
    config_blob = gl["environment"]["GITLAB_OMNIBUS_CONFIG"]
    external_url_lines = [
        line.strip() for line in config_blob.splitlines()
        if line.strip().startswith("external_url")
    ]
    assert external_url_lines, "GitLab omnibus config has no external_url"
    # Should point at the service name, not 'the-agent-company.com' or 'localhost'.
    assert any("gitlab" in url for url in external_url_lines), (
        f"GitLab external_url should be the service-name URL; got: {external_url_lines}"
    )
