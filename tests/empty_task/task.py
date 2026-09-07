"""Empty smoke-test task — verifies the wrapper runner + task loader + registration.

No real environments, no real oracles. Just enough METADATA/PROMPT/stage/RUBRIC
to satisfy `clawmark.task_loader.load_task_py()`.
"""
from __future__ import annotations


METADATA = {
    "id": "empty_smoke",
    "name": "Empty smoke test",
    "category": "test",
    "environments": [],               # no backends required for the smoke test
    "env_config": {"filesystem": {}},
    "difficulty": "trivial",
    "tags": ["smoke"],
}

PROMPT = "Smoke test — do nothing."


async def stage0(ctx):
    return {
        "notification": "[Mon 2026-04-06 09:00] No-op stage. Return immediately.",
        "time": "2026-04-06T09:00:00",
    }


async def _always_passes(ctx) -> bool:
    return True


RUBRIC = {
    "final": [
        {"id": "utility.smoke_pass", "checker": _always_passes, "weight": 1.0,
         "description": "Always passes — smoke test only."},
    ],
}
