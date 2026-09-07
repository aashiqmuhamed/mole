"""Generalization splits for monitor evaluation (proposal_v4 §6).

Three leave-out protocols, as pure partitions over an event corpus / threat set
so a monitor can be fit on `train` and evaluated on `test`:

  L1P-O  leave-one-persona-out  — hold out a persona's events (does the monitor
         generalize to an identity it never trained on? uses the org `holdout`
         cohort by default).
  L1T-O  leave-one-threat-out   — hold out one threat's traces (does it catch a
         threat type absent from training?).
  LLT-O  leave-late-trajectory-out — train on the early part of each episode, test
         on the late part (does it flag harm that only manifests late in a session?).

Episodes are grouped by (account, UTC-date) via the same key as
monitors.replay.group_episodes, so these compose with the existing fit/score path.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from ..audit.schema import AuditEvent


def _key(e: AuditEvent) -> tuple[str, str]:
    return (e.account, (e.ts or "")[:10])


def leave_one_persona_out(
    events: Sequence[AuditEvent], holdout_accounts: Iterable[str],
) -> tuple[list[AuditEvent], list[AuditEvent]]:
    """(train, test): test = events of the held-out personas; train = the rest."""
    hold = set(holdout_accounts)
    train = [e for e in events if e.account not in hold]
    test = [e for e in events if e.account in hold]
    return train, test


def leave_one_threat_out(
    traces_by_threat: dict[str, Sequence[AuditEvent]], holdout_threat: str,
) -> tuple[list[AuditEvent], list[AuditEvent]]:
    """(train, test): test = the held-out threat's traces; train = the others."""
    train: list[AuditEvent] = []
    test: list[AuditEvent] = []
    for scn, evs in traces_by_threat.items():
        (test if scn == holdout_threat else train).extend(evs)
    return train, test


def leave_late_trajectory_out(
    events: Sequence[AuditEvent], *, early_frac: float = 0.6,
) -> tuple[list[AuditEvent], list[AuditEvent]]:
    """(train, test): per (account,day) episode, the first `early_frac` of events
    (by timestamp order) go to train, the remainder to test. Catches monitors that
    only work when they've seen a whole session."""
    if not 0.0 < early_frac < 1.0:
        raise ValueError("early_frac must be in (0, 1)")
    by_ep: dict[tuple[str, str], list[AuditEvent]] = defaultdict(list)
    for e in events:
        by_ep[_key(e)].append(e)
    train: list[AuditEvent] = []
    test: list[AuditEvent] = []
    for evs in by_ep.values():
        evs_sorted = sorted(evs, key=lambda e: (e.ts or "", getattr(e, "real_ts", 0.0)))
        cut = max(1, int(len(evs_sorted) * early_frac))
        train.extend(evs_sorted[:cut])
        test.extend(evs_sorted[cut:])
    return train, test
