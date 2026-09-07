"""Seed the shared agentlab company world once and dump its real resource inventory.

Validates seeders/seed_company.py composes (org + all 10 threat seeds, union), and
emits bootstrap/company_inventory.yaml — the actual repos/files/channels/checkpoints/
secrets that exist — so the benign task bank can be GROUNDED in real resources instead of
LLM-invented names. Run under the lab sandbox (needs services; no LLM required here).

  python scripts/dump_company_inventory.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

BENCH = Path(__file__).resolve().parent.parent   # repo root: holds mole/, bootstrap/, data/
sys.path.insert(0, str(BENCH))

import yaml
from mole.audit.collector import AuditCollector
from mole.generator.run import AGENTIC_ENVS, build_context


async def _safe(coro, default=None):
    try:
        return await coro
    except Exception as e:                                      # noqa: BLE001
        return default if default is not None else f"<err: {type(e).__name__}>"


async def main() -> int:
    from mole.sandbox.lab import LabSandbox
    from mole.run_task import _default_lab_compose

    sandbox = LabSandbox(session_id=f"itb-inv-{uuid4().hex[:8]}",
                         compose_files=[_default_lab_compose()])
    print("starting sandbox + seeding shared company world…")
    await sandbox.start()
    collector = AuditCollector(jsonl_path=str(BENCH / "data" / "corpus" / "_inv.jsonl"))
    tmp = BENCH / "data" / "corpus"
    ctx = await build_context(collector, tmp_dir=tmp, envs=AGENTIC_ENVS,
                              sandbox=sandbox, seed_world=True)
    m = ctx._managers
    inv: dict = {}
    try:
        gl = m.get("gitlab")
        if gl:
            inv["gitlab_projects"] = sorted(
                p.get("path_with_namespace", p.get("name", str(p)))
                for p in (await _safe(gl.list_projects(), []) or []))
        rc = m.get("rocketchat")
        if rc:
            inv["rocketchat_channels"] = sorted(
                c.get("name", str(c)) for c in (await _safe(rc.list_channels(), []) or []))
        pl = m.get("plane")
        if pl:
            inv["plane_projects"] = sorted(
                p.get("name", str(p)) for p in (await _safe(pl.list_projects(), []) or []))
        mr = m.get("model_registry")
        if mr:
            inv["checkpoints"] = await _safe(mr.list_checkpoints(), [])
        ss = m.get("secrets_store")
        if ss:
            inv["secret_keys"] = await _safe(ss.list_keys(), [])
        oc = m.get("owncloud")
        if oc:
            inv["owncloud_root"] = await _safe(oc.list_dir(path="/"), [])
    finally:
        out = BENCH / "bootstrap" / "company_inventory.yaml"
        out.write_text(yaml.safe_dump(inv, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")
        print(f"\nwrote {out}")
        for k, v in inv.items():
            n = len(v) if isinstance(v, list) else "?"
            print(f"  {k}: {n}")
        try:
            await sandbox.stop()
        except Exception:                                      # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
