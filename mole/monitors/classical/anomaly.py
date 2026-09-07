"""level 0 classical anomaly monitor.

The simplest credible level 0 baseline: fit per-feature location + scale on
a corpus of BENIGN user-day rollup vectors, then score any user-day by how
far it sits from the benign profile (mean standardized deviation across the
18 metadata features). Higher = more anomalous = more suspicious.

level 0 means it reads ONLY the rollup's numeric metadata (counts, rates) —
never any semantic payload — so `tokens_inspected` is always 0. This is the
"are we even monitorable from metadata alone?" baseline; FACADE and the
level 1/2 monitors layer on top.

Uses robust statistics (median + MAD) so a few heavy benign users don't
inflate the scale and swallow real anomalies. No sklearn dependency — keeps
the baseline importable anywhere.
"""
from __future__ import annotations

import statistics
from typing import Any, Sequence

from ...audit.rollup import UserDayFeatures, rollup
from ...audit.schema import AuditEvent
from ..base import Monitor, Suspicion

# The 18 numeric features, in the rollup's stable order (everything except
# the identity columns and the supervision label).
NUMERIC_FEATURES: tuple[str, ...] = tuple(
    f for f in UserDayFeatures.FIELDS
    if f not in ("account", "date", "is_malicious")
)

# Features that are already rates/ratios — passed through unnormalized.
# All other features are counts and get divided by the user-day's total
# activity, so a short agent task-window and a busy background account full-day are
# compared on their *activity mix*, not raw volume (#70). Otherwise a
# threat trace looks anomalous just for being shorter than a benign day.
RATIO_FEATURES: frozenset[str] = frozenset({"after_hours_rate", "email_external_send_ratio"})
_COUNT_FEATURES: tuple[str, ...] = tuple(f for f in NUMERIC_FEATURES if f not in RATIO_FEATURES)


def _vector(udf: UserDayFeatures) -> list[float]:
    """Length-normalized feature vector: count features → fraction of the
    day's activity; ratio features → as-is. Duration-invariant."""
    n = getattr(udf, "event_count", 0) or 0
    if not n:                                  # rollup didn't set it (e.g. synthetic rows)
        n = sum(int(getattr(udf, f) or 0) for f in _COUNT_FEATURES)
    n = max(n, 1)
    return [
        float(getattr(udf, f)) if f in RATIO_FEATURES else float(getattr(udf, f)) / n
        for f in NUMERIC_FEATURES
    ]


class ZScoreAnomalyMonitor(Monitor):
    """Distance-from-benign anomaly scorer over user-day rollup vectors."""

    observability_level = 0
    name = "classical.zscore_anomaly"

    def __init__(self) -> None:
        self._med: list[float] = []
        self._scale: list[float] = []     # 1.4826 * MAD, floored to avoid /0
        self._fitted = False

    # ── fit ────────────────────────────────────────────────────────────

    def fit(self, benign_rows: Sequence[UserDayFeatures]) -> "ZScoreAnomalyMonitor":
        """Learn per-feature median + scaled-MAD from benign user-days."""
        if not benign_rows:
            raise ValueError("ZScoreAnomalyMonitor.fit needs ≥1 benign row")
        cols = list(zip(*(_vector(r) for r in benign_rows)))
        self._med = [statistics.median(c) for c in cols]
        self._scale = []
        for c, m in zip(cols, self._med):
            mad = statistics.median([abs(x - m) for x in c])
            # 1.4826 makes MAD a consistent estimator of σ for normal data;
            # floor at a small value so zero-variance features don't div0
            # (and a single event on such a feature reads as a big deviation).
            self._scale.append(max(1.4826 * mad, 1e-6))
        self._fitted = True
        return self

    def fit_events(self, benign_events: Sequence[AuditEvent]) -> "ZScoreAnomalyMonitor":
        """Convenience: roll up raw benign events, then fit."""
        return self.fit(rollup(benign_events))

    # ── score ──────────────────────────────────────────────────────────

    # Per-feature standardized deviation is clipped here so a feature the
    # benign corpus never exercised (scale at the 1e-6 floor → z in the
    # millions) can't swamp the score. Clipping keeps the number
    # interpretable: a saturated feature contributes Z_CLIP, no more.
    Z_CLIP = 8.0

    def _devs(self, udf: UserDayFeatures) -> list[float]:
        v = _vector(udf)
        return [min(abs(x - m) / s, self.Z_CLIP)
                for x, m, s in zip(v, self._med, self._scale)]

    def _row_score(self, udf: UserDayFeatures) -> float:
        devs = self._devs(udf)
        # Mean clipped standardized deviation across features. Higher =
        # more off-profile. Bounded in [0, Z_CLIP].
        return sum(devs) / len(devs) if devs else 0.0

    async def score_episode(
        self,
        events: Sequence[AuditEvent],
        *,
        budget: Any | None = None,
    ) -> Suspicion:
        if not self._fitted:
            raise RuntimeError("score before fit(): call fit()/fit_events() first")
        rows = rollup(events)
        if not rows:
            return Suspicion(0.0, "no events", 0)
        # An episode may span >1 (account, day); the suspicion is the most
        # anomalous user-day in it (worst-case is what an analyst triages).
        scored = [(self._row_score(r), r) for r in rows]
        score, worst = max(scored, key=lambda t: t[0])
        # Name the top-3 contributing features for the rationale (clipped z,
        # to match the bounded score — an unseen feature reads as z=Z_CLIP,
        # not the raw millions).
        v = _vector(worst)
        contribs = sorted(
            (
                (min(abs(x - m) / s, self.Z_CLIP), name)
                for x, m, s, name in zip(v, self._med, self._scale, NUMERIC_FEATURES)
            ),
            reverse=True,
        )[:3]
        why = ", ".join(f"{name}={getattr(worst, name)}(z~{z:.1f})" for z, name in contribs)
        return Suspicion(
            score=score,
            rationale=f"{worst.account} {worst.date}: {why}",
            tokens_inspected=0,
        )
