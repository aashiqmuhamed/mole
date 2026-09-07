"""Seed the one shared `agentlab` company world.

One company, one world. `seed_org` provisions the common org (users, channels,
mailboxes) from `org_template.yaml`; then every threat's `task.seed()` provisions the
resources that threat needs (repos, files, checkpoints, …). The union is the full AI
lab — the model-release pipeline, training corpus, infra, eval server, safety reports,
finance, customer data all coexist — so benign background accounts and focal attackers operate in the
SAME environment and their distributions differ only by the malicious act.

The threat seeds are disjoint (distinct groups/projects — verified), so they compose by
union with no collisions. Failures in any one threat seed are logged and skipped so a
single bad seed doesn't sink the whole world.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..tasks.loader import discover_task_dirs, load_task
from .seed_org import seed_org
from .seed_s1_variants import seed_s1_variants

logger = logging.getLogger(__name__)


def _tasks_root() -> Path:
    # seeders/seed_company.py -> seeders -> mole -> benchmark; tasks/ is
    # a sibling of the package at benchmark/tasks.
    return Path(__file__).resolve().parents[2] / "tasks"


def _is_threat_threat(name: str) -> bool:
    """The 10 numbered threats (01..10), not 00_* demos or _template."""
    return len(name) >= 2 and name[:2].isdigit() and name[:2] != "00"


async def seed_company(
    ctx: Any,
    *,
    tasks_root: Path | str | None = None,
    threats: list[str] | None = None,
) -> dict[str, Any]:
    """Seed the full shared company world: common org seed + every threat's resources.

    Returns {"org": <counts>, "threats": {threat_id: <counts>}}.
    """
    counts: dict[str, Any] = {"org": await seed_org(ctx)}
    root = Path(tasks_root) if tasks_root else _tasks_root()
    seeded: dict[str, Any] = {}
    for d in discover_task_dirs(root):
        if not _is_threat_threat(d.name):
            continue
        if threats is not None and d.name not in threats:
            continue
        try:
            task = load_task(d)
            if task.seed_fn is not None:
                seeded[task.id] = await task.seed_fn(ctx)
        except Exception:                                       # noqa: BLE001
            logger.exception("seed_company: threat %s seed failed; continuing", d.name)
    counts["threats"] = seeded
    # Layer S1 disjoint-target mirrors on top of the S0 seeds so both
    # stealth-level insiders in the v1 eval have distinct targets and
    # don't contaminate each other's cell. Best-effort: failures are logged
    # and skipped, the run continues with whatever S1 fixtures landed.
    try:
        counts["s1_variants"] = await seed_s1_variants(ctx)
    except Exception:                                               # noqa: BLE001
        logger.exception("seed_company: S1-variant seeding failed; continuing")
        counts["s1_variants"] = {}
    logger.info("seed_company: org + %d/%d threat worlds seeded + %d S1 variants",
                len(seeded),
                sum(1 for d in discover_task_dirs(root) if _is_threat_threat(d.name)),
                sum(counts.get("s1_variants", {}).values()))
    return counts
