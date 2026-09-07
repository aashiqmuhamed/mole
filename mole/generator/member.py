"""Member — single background-account simulation loop.

A Member owns a Persona + RNG seed + a list of workflows that apply to
that persona. Calling `await member.run_day(ctx, sim_date_iso)` runs
each applicable workflow once during the persona's working-hours window
on the given date, advancing `ctx.sim_now` so each call lands at a
realistic timestamp (the audit collector tags events with `ctx.sim_now`,
so the per-user-day rollup's `after_hours_rate` feature works as
intended).

Working hours, jitter, and `loaf_probability` are configurable via
NPCConfig; defaults match the working-hours band in the rollup
(09:00–17:00 UTC) and a 10% chance a workflow is skipped on any given
day.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any

from .persona_loader import Persona
from .workflows import WORKFLOWS, Workflow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NPCConfig:
    workhours_start: time = time(9, 0)
    workhours_end: time = time(17, 0)
    jitter_minutes: int = 10
    # ~5%: an account occasionally skips a planned touch-point (in a meeting,
    # interrupted, etc.). Low enough to keep the day dense/continuous.
    loaf_probability: float = 0.05
    # Per-account whole-day absence (PTO/sick). Kept very rare by default so the
    # benign substrate stays dense; it just breaks the unrealistic "everyone present
    # every weekday" pattern. Tunable via --absent-prob; set 0 to disable.
    absent_probability: float = 0.01
    # How many times each applicable workflow is attempted across the day.
    # Real accounts touch the same surfaces repeatedly (morning standup,
    # midday eval, afternoon review, EOD check). Default 1 keeps unit-test
    # expectations exact; the runner bumps it to model realistic volume.
    sessions_per_day: int = 1
    # Continuous-day model. Sessions run back-to-back from a per-account daily
    # START time, spaced ~session_budget_s apart with a small inter-session gap
    # drawn from the SAME range as intra-session pauses — so a monitor (which sees
    # only timestamps, no session_id/logon/parent links) cannot segment the day
    # into sessions and key on session-starts (where an insider's attack opens).
    # One occasional longer break models lunch/a meeting. Net: a day is one
    # near-continuous stream, not a few isolated bursts with hours-long gaps.
    session_budget_s: int = 120          # sim-seconds budgeted per session
    break_probability: float = 0.5       # chance of one longer mid-day break


class Member:
    """One simulated background account; drives workflows against a shared Context."""

    def __init__(
        self,
        persona: Persona,
        *,
        config: NPCConfig | None = None,
        rng_seed: int | None = None,
        workflows: tuple[Workflow, ...] = WORKFLOWS,
    ) -> None:
        self.persona = persona
        self.config = config or NPCConfig()
        # Deterministic per-persona RNG so two runs with the same seed
        # produce the same trajectory (essential for FACADE pretraining
        # reproducibility).
        seed_basis = rng_seed if rng_seed is not None else hash(persona.id) & 0xFFFFFFFF
        self.rng = random.Random(seed_basis)
        self.workflows = tuple(w for w in workflows if w.applies_to(persona))

    def plan_day(self, sim_date_iso: str) -> list[tuple[datetime, str]]:
        """Compute this background account's scheduled actions for the day as (slot_time, key) pairs.

        Consumes RNG for jitter + loaf checks (so each call advances the RNG state).
        The caller (`simulate`) merges per-member plans across all members and sorts
        by `slot_time`, so events run in causally-correct sim-time order across the
        org — alice's 9 am chat post happens before bob's 10 am inbox read, regardless
        of which member appears first in the iteration. (The old `for member: run_day`
        pattern ran each member's full day before the next member, which let later
        members "see the future" via shared world state.)

        Returns: list of (slot_datetime, action_key) for this day, loafed slots dropped.
        `action_key` is the workflow name for the rules-based Member.
        """
        if self.config.absent_probability > 0 and self.rng.random() < self.config.absent_probability:
            logger.debug("account %s absent (PTO) on %s", self.persona.id, sim_date_iso)
            return []
        base = self._iso_date_to_dt(sim_date_iso)
        sessions = max(1, self.config.sessions_per_day)
        schedule = [wf for _ in range(sessions) for wf in self.workflows]
        slots = self._slot_times(base, len(schedule))
        plan: list[tuple[datetime, str]] = []
        for wf, t in zip(schedule, slots):
            if self.rng.random() < self.config.loaf_probability:
                logger.debug("account %s loafing on %s", self.persona.id, wf.name)
                continue
            plan.append((t, wf.name))
        return plan

    async def run_at(self, ctx: Any, sim_now: str, action_key: str) -> None:
        """Run a single planned action at the given sim_now timestamp.

        Sets `ctx._account_box` so audit events attribute to this persona (kind="background_rules_agent"
        for the scripted Member; AgenticMember overrides with "background_llm_agent").
        """
        wf = next((w for w in self.workflows if w.name == action_key), None)
        if wf is None:
            return
        ctx.advance_clock(sim_now)
        # Per-task account via ContextVar — see account_context.py for
        # why the old mutable-box pattern raced under concurrency.
        from .account_context import set_account
        set_account(self.persona.id, "background_rules_agent")
        try:
            await wf.run(ctx, self.persona, self.rng)
        except Exception:                           # noqa: BLE001
            logger.exception(
                "account %s workflow %s raised; continuing",
                self.persona.id, action_key,
            )

    async def run_day(self, ctx: Any, sim_date_iso: str) -> int:
        """Run this background account's full day sequentially. Kept for tests / single-member callers;
        the corpus `simulate()` interleaves via plan_day + run_at across members instead.
        """
        plan = self.plan_day(sim_date_iso)
        for t, key in plan:
            await self.run_at(ctx, t.strftime("%Y-%m-%dT%H:%M:%SZ"), key)
        return len(plan)

    # ── internals ────────────────────────────────────────────────────

    def _iso_date_to_dt(self, sim_date_iso: str) -> datetime:
        """Parse YYYY-MM-DD (or full ISO ts) → UTC midnight on that date."""
        day = sim_date_iso[:10]
        try:
            return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"bad sim_date_iso: {sim_date_iso!r}") from exc

    def _personal_start(self, base: datetime) -> datetime:
        """This account's daily start time: a persona-stable chronotype + small
        daily jitter. Most start in the business morning (≈7:00–10:30); a minority
        are evening workers (≈17:00–21:00). Varied across the org so each person's
        day is compact while the org-wide log stays active all day, and so the
        after-hours signal survives (an evening/2 a.m. window reads after-hours).
        The chronotype is a STABLE hash of the persona id (not Python's salted
        hash()), so it reproduces across runs; jitter comes from the seeded RNG."""
        import hashlib
        h = int(hashlib.md5(self.persona.id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < 0.92:                                   # business chronotype
            start_min = 7 * 60 + int(h / 0.92 * 210)        # 07:00–10:30
        else:                                          # evening/late chronotype
            start_min = 17 * 60 + int((h - 0.92) / 0.08 * 240)  # 17:00–21:00
        start_min += int(self.rng.uniform(
            -self.config.jitter_minutes, self.config.jitter_minutes))
        start_min = max(0, min(24 * 60 - 1, start_min))
        return base + timedelta(minutes=start_min)

    def _slot_times(self, base: datetime, n: int) -> list[datetime]:
        """Continuous-day schedule: n sessions back-to-back from the account's
        START time, spaced one session-budget apart plus a small inter-session gap.
        These slots only drive execution ORDER + bucketing; for agentic background accounts the
        emitted timestamps are CHAINED off each session's real end (see
        agentic_member._run_session), and the occasional lunch break is injected
        there. Replaces the old even-spread-across-9–5 (hours-long gaps that
        exposed each session's start)."""
        if n <= 0:
            return []
        t = self._personal_start(base)
        budget = max(30, self.config.session_budget_s)
        out: list[datetime] = []
        for i in range(n):
            out.append(t)
            t = t + timedelta(seconds=budget + self.rng.uniform(2, 30))
        return out
