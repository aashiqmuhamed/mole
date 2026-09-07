"""Opt-in live integration test — exercises the full env stack.

Brings up the lab compose, runs `seed_org` over real services, verifies
mailboxes were created and audit events were recorded with the right
accounts. Gated on `ITB_LIVE=1` so it doesn't run in CI by default.

Prereqs:
  - Docker daemon running
  - `compose/lab.yaml` images already pulled (otherwise first run will
    download multi-GB images and time out)

Run:
  ITB_LIVE=1 pytest tests/test_live_integration.py -v
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from mole.audit.collector import AuditCollector
from mole.sandbox.lab import LabSandbox
from mole.seeders import seed_org
from mole.state.composite import CompositeStateManager


_LIVE = os.environ.get("ITB_LIVE") == "1"


@pytest.mark.skipif(not _LIVE, reason="set ITB_LIVE=1 to run live integration test")
def test_seed_org_against_live_greenmail():
    """End-to-end: lab compose up → seed_org → IMAP verify → tear down.

    Brings up only the GreenMail container (via service_ports filter) so
    the test stays fast — about 15-30s start-to-stop on a warm box.
    The rest of the compose stack still gets started by `docker compose
    up --wait`; this is upstream behaviour, not a test bug. To run a
    truly minimal lab, point at a stripped compose file via
    LAB_COMPOSE_FILE env var.
    """
    compose_file = Path(
        os.environ.get("LAB_COMPOSE_FILE")
        or Path(__file__).resolve().parents[1] / "compose" / "lab.yaml"
    )
    if not compose_file.exists():
        pytest.skip(f"compose file not found at {compose_file}")

    sandbox = LabSandbox(
        session_id="itb-live-it",
        compose_files=[compose_file],
        # Discover only the GreenMail ports; the lab brings everything up
        # but we don't need port maps for services we won't touch.
        service_ports=[
            ("greenmail", 3025),
            ("greenmail", 3143),
            ("greenmail", 8080),
        ],
    )

    async def _run() -> dict:
        await sandbox.start()
        try:
            composite = CompositeStateManager(environments=["email", "org"])
            await composite.setup(sandbox=sandbox)

            from mole.generator.account_context import get_account
            collector = AuditCollector()
            for svc_name, mgr in composite.managers.items():
                collector.wrap_manager(
                    service_name=svc_name, manager=mgr,
                    account_getter=get_account,
                )
            ctx = composite.create_context(
                task_dir=Path("."), sandbox=sandbox, audit=collector,
            )

            counts = await seed_org(ctx, only={"email"})
            inboxes = {}
            for emp in ["alice.kim", "bob.li", "frank.s", "kara.p"]:
                msgs = await ctx.email.find_emails(
                    user=emp, subject_contains="Welcome",
                )
                inboxes[emp] = len(msgs)
            return {
                "counts": counts,
                "ports": dict(sandbox.ports),
                "inboxes": inboxes,
                "audit_events": len(collector.events),
                "audit_services": sorted({e.service for e in collector.events}),
            }
        finally:
            await sandbox.stop(delete=True)

    result = asyncio.run(_run())

    assert result["counts"]["email"] >= 4
    # GreenMail returned the welcome mail for every spot-checked account.
    for emp, n in result["inboxes"].items():
        assert n >= 1, f"{emp} inbox empty after seed"
    # And the audit collector captured every send_email + find_emails call.
    assert result["audit_events"] >= 8
    assert "email" in result["audit_services"]
    # Sandbox port discovery populated all three GreenMail ports.
    assert set(result["ports"]) == {3025, 3143, 8080}
