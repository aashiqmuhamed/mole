"""Run a rubric stage end-to-end and return weighted results.

A rubric stage is a list of `RubricEntry`. Each entry's `checker` is an async
function that takes a `TaskContext` and returns truthy/falsy. The overall
stage score is `sum(weight for passed) / sum(weight)` in [0, 1]. Checker
exceptions are caught and surfaced as failed items with the exception
message in `detail` — we don't want one buggy checker to abort a whole
evaluation pass.
"""
from __future__ import annotations

import logging
from typing import Any

from ..models import EvaluationResult, RubricEntry, RubricItemResult

logger = logging.getLogger(__name__)


async def run_rubric(entries: list[RubricEntry], ctx: Any) -> EvaluationResult:
    items: list[RubricItemResult] = []
    for e in entries:
        detail = ""
        try:
            passed = bool(await e.checker(ctx))
        except Exception as exc:
            logger.error("rubric checker %s raised: %s", e.id, exc)
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
        items.append(RubricItemResult(
            item_id=e.id, passed=passed, weight=e.weight, detail=detail,
        ))

    total = sum(it.weight for it in items)
    passed_w = sum(it.weight for it in items if it.passed)
    score = passed_w / total if total > 0 else 0.0
    return EvaluationResult(items=items, score=score)
