"""RandomBaselineMonitor — a deliberately useless monitor for pipeline validation.

Use case: confirm the end-to-end eval pipeline produces sensible numbers.
A random monitor should give:
  - recall@top-X% ≈ X%   (catches a uniform-random fraction of attacks)
  - AUROC          ≈ 0.5 (no signal)
  - PR-AUC         ≈ base rate

If a real monitor doesn't beat these floors, it's not actually detecting
anything. The Random baseline is the lower bound every real monitor must
clear; if eval_matrix's output puts a real monitor at or below random, the
result is "no detection" rather than a meaningful recall claim.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from ..base import Monitor, Suspicion
from ...audit.schema import AuditEvent


class RandomBaselineMonitor(Monitor):
    """Assigns a uniformly random score to each episode.

    Deterministic per (account, day) — we hash the account+date so two
    runs with the same input produce the same ranking. This lets us reason
    about random-baseline numbers reproducibly.
    """

    observability_level = 0
    name = "random-baseline"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit_events(self, _events: Any) -> "RandomBaselineMonitor":
        """No-op fit — the random baseline doesn't learn from anything."""
        return self

    async def score_episode(
        self,
        events: Sequence[AuditEvent],
        *,
        budget: Any | None = None,
    ) -> Suspicion:
        if not events:
            return Suspicion(score=0.0, rationale="empty episode")
        account = events[0].account
        day = (events[0].ts or "")[:10]
        # Hash gives a uniform score in [0, 1); seed mixes in so callers can
        # reproduce a specific random ordering.
        key = f"{account}:{day}:{self.seed}".encode("utf-8")
        h = hashlib.sha256(key).digest()
        # First 8 bytes -> uint64 -> normalize to [0, 1)
        n = int.from_bytes(h[:8], "big")
        score = n / (2**64)
        return Suspicion(score=score, rationale="random", tokens_inspected=0)
