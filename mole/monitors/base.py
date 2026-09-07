"""Monitor ABC + Suspicion — the common interface every monitor implements.

A Monitor observes an audit-event stream (already projected to its
observability level — see audit/projections.py) and returns a Suspicion:
a scalar score (higher = more suspicious), a short rationale, and how many
inspection tokens it spent (0 for level 0 metadata-only monitors; >0 for
level 1/2 monitors that pay to look at semantic payloads under a budget).

The interface is async so level 2 LLM-as-judge monitors fit the same shape;
level 0/1 monitors that do no I/O simply don't await anything.

Scoring granularity:
  - `score_episode(events)` — one Suspicion for a whole trace (e.g. a
    per-(account, day) audit log). This is the primary entry point the
    offline replay harness uses.
  - `score_event(event)` — optional per-event score; the default wraps the
    single event as a one-element episode.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Sequence

from ..audit.schema import AuditEvent


class Suspicion(NamedTuple):
    score: float                 # higher = more suspicious; range is monitor-defined
    rationale: str = ""
    tokens_inspected: int = 0    # semantic-inspection budget spent (level 1/2)


class Monitor(ABC):
    """Base class for all monitors.

    Subclasses set `tier` (0/1/2) and `name`, and implement
    `score_episode`. level 1/2 monitors accept an inspection `budget` they
    may decrement; level 0 monitors ignore it.
    """

    observability_level: int = 0
    name: str = "monitor"

    @abstractmethod
    async def score_episode(
        self,
        events: Sequence[AuditEvent],
        *,
        budget: Any | None = None,
    ) -> Suspicion:
        """Return a Suspicion for the whole event sequence."""
        ...

    async def score_event(
        self,
        event: AuditEvent,
        *,
        budget: Any | None = None,
    ) -> Suspicion:
        """Per-event score. Default: treat as a one-event episode."""
        return await self.score_episode([event], budget=budget)
