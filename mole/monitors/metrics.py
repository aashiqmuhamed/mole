"""Ranking metrics for monitor evaluation — works for any monitor.

Every monitor produces a ranked list of `(key, score)` pairs (typically via
`monitors/replay.py:replay()`, which scores `(account, day)` episodes). This module
takes that ranked list + the set of true-malicious keys and reports the operational
metric framework — **percentile-based recall@top-X%**, recall vs K, PR-AUC, AUROC.

Why percentile rather than absolute K: at our 28-account scale K=1 means 3.6% of
account-days/day; at Chimera's 20-agent-org scale it's 5%; at real enterprise it's
0.001%. Reporting top-X% makes the same number meaningful at any scale, while
recall_at_k(K=1,3,5,10) gives the concrete analyst-budget intuition for our current
scale.

Standard classification metrics (P/R/F1) are also computed for Chimera-comparability
but they should NOT be the headline at realistic insider-threat base rates.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..audit.schema import AuditEvent


@dataclass
class RankingMetrics:
    """All ranking metrics for one monitor's evaluation on one universe."""

    monitor_name: str
    n_total: int
    n_malicious: int
    recall_at_pct: dict[float, float] = field(default_factory=dict)   # e.g. {0.01: 0.85}
    precision_at_pct: dict[float, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)        # absolute K too
    auroc: float = 0.0
    pr_auc: float = 0.0

    def summary(self) -> str:
        """One-line summary suitable for table rows."""
        pct = " ".join(
            f"R@{int(p * 100)}%={self.recall_at_pct.get(p, 0.0):.2f}"
            for p in sorted(self.recall_at_pct)
        )
        ks = " ".join(
            f"R@{k}={self.recall_at_k.get(k, 0.0):.2f}"
            for k in sorted(self.recall_at_k)
        )
        return (f"{self.monitor_name:<22} {pct}  {ks}  "
                f"AUROC={self.auroc:.2f} PR-AUC={self.pr_auc:.2f} "
                f"(N={self.n_total}, mal={self.n_malicious})")


def _extract_score(item: Any) -> float:
    """Tolerate Suspicion objects or bare floats."""
    s = getattr(item, "score", None)
    return float(item if s is None else s)


def compute_ranking_metrics(
    monitor_name: str,
    ranked: Sequence[tuple[Any, Any]],
    malicious_keys: set,
    *,
    pcts: Sequence[float] = (0.01, 0.05, 0.10),
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> RankingMetrics:
    """Compute ranking metrics from a list of `(key, score-or-Suspicion)` pairs sorted
    by score descending.

    `malicious_keys` is the ground-truth set of keys that should be flagged. For
    account-day ranking that's typically `(account, sim_date)` tuples; the same
    function works for any keying scheme as long as `malicious_keys` matches.
    """
    n = len(ranked)
    if n == 0:
        return RankingMetrics(monitor_name=monitor_name, n_total=0, n_malicious=0)

    labels = np.array([1 if k in malicious_keys else 0 for k, _ in ranked], dtype=int)
    scores = np.array([_extract_score(s) for _, s in ranked], dtype=float)
    n_mal = int(labels.sum())

    def _recall_at(k: int) -> float:
        k = max(0, min(k, n))
        if n_mal == 0 or k == 0:
            return 0.0
        return float(labels[:k].sum()) / n_mal

    def _precision_at(k: int) -> float:
        k = max(1, min(k, n))
        return float(labels[:k].sum()) / k

    def _k_from_pct(p: float) -> int:
        return max(1, int(np.ceil(p * n)))

    recall_at_pct = {p: _recall_at(_k_from_pct(p)) for p in pcts}
    precision_at_pct = {p: _precision_at(_k_from_pct(p)) for p in pcts}
    recall_at_k = {k: _recall_at(k) for k in ks}

    # AUROC via Mann-Whitney U: probability that a malicious item is ranked above
    # a benign one. Robust to tied scores via the 0.5 term.
    if 0 < n_mal < n:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        # Vectorized via pairwise comparison
        u = float(np.sum(pos[:, None] > neg[None, :])) + 0.5 * float(
            np.sum(pos[:, None] == neg[None, :])
        )
        auroc = u / (n_mal * (n - n_mal))
    else:
        auroc = 0.0

    # PR-AUC = average precision (sum of precision at each TP / n_mal)
    if n_mal > 0:
        cum_tp = 0
        ap_sum = 0.0
        for i, lbl in enumerate(labels):
            if lbl == 1:
                cum_tp += 1
                ap_sum += cum_tp / (i + 1)
        pr_auc = ap_sum / n_mal
    else:
        pr_auc = 0.0

    return RankingMetrics(
        monitor_name=monitor_name,
        n_total=n,
        n_malicious=n_mal,
        recall_at_pct=recall_at_pct,
        precision_at_pct=precision_at_pct,
        recall_at_k=recall_at_k,
        auroc=auroc,
        pr_auc=pr_auc,
    )


def malicious_account_days(events: Sequence[AuditEvent]) -> set[tuple[str, str]]:
    """Derive the ground-truth malicious-key set from `is_malicious` tags.

    A `(account, sim_date)` key is positive iff at least one event for that
    account-day has `is_malicious=True`. Use the result as `malicious_keys` for
    account-day ranking (the default unit for level 0/level 2).
    """
    out: set[tuple[str, str]] = set()
    for e in events:
        if getattr(e, "is_malicious", False):
            day = (e.ts or "")[:10]
            out.add((e.account, day))
    return out


# K-budgets and tabu window for the daily-protocol metric (FACADE §8). K=10 is
# the headline analyst budget.
DAILY_K_VALUES: tuple[int, ...] = (1, 3, 10, 30, 100)
TABU_DAYS: int = 7
WINDOW_DAYS: int = 7


def daily_recall_at_k(
    ranked: "Sequence[tuple[Any, Any]]",
    malicious_keys: set[tuple[str, str]],
    *,
    k_values: "Sequence[int]" = DAILY_K_VALUES,
    tabu_days: int = TABU_DAYS,
    window_days: int = WINDOW_DAYS,
) -> dict[int, float]:
    """FACADE §8 daily-protocol Recall@K/day (per-day rank + audit tabu + campaign-level).

    This is the OPERATIONAL metric (the FACADE operational aggregation metric),
    not a global percentile: it models an analyst with a fixed daily
    budget K, auditing each day's top-ranked NEW accounts, and asks what fraction
    of attack *campaigns* are caught. Unlike recall@top-X% it is causal (per-day),
    streaming (extends to any future day — no dependence on total N), and respects
    a `tabu_days` window (don't re-audit someone just investigated).

        for each day (ascending):
            rank accounts by their (account, day) score, desc
            audit the top-K not audited in the preceding `tabu_days`
        a campaign = a malicious account; it is DETECTED if its target is
        audited on at least one of its attack-active days (the days it has a
        malicious account-day).

    ranked: [((account, day_str), score-or-Suspicion)] over the eval slice.
    malicious_keys: ground-truth malicious (account, day_str) — from
        malicious_account_days(); also defines each campaign's active days.
    """
    days_sorted = sorted({k[1] for k, _ in ranked})
    day_idx = {d: i for i, d in enumerate(days_sorted)}
    by_day: dict[str, list[tuple[str, float]]] = {}
    for (p, d), s in ranked:
        by_day.setdefault(d, []).append((p, _extract_score(s)))
    # campaigns: target account -> set of attack-active day indices. An attack
    # on day d is catchable for a rolling window [d, d+window_days-1] (clamped to
    # the eval span) — auditing the target on any of those days catches the
    # campaign (audited ⇒ detected). window_days=1 = strict (catch only
    # on the attack day, the honest setting for per-day-independent scores).
    max_di = len(days_sorted) - 1
    campaigns: dict[str, set[int]] = {}
    for (p, d) in malicious_keys:
        if d not in day_idx:
            continue
        di = day_idx[d]
        active = campaigns.setdefault(p, set())
        for w in range(window_days):
            if di + w <= max_di:
                active.add(di + w)
    n_camp = len(campaigns)

    out: dict[int, float] = {}
    for k in k_values:
        last_audited: dict[str, int] = {}
        audited_on: dict[int, set[str]] = {}
        for d in days_sorted:
            di = day_idx[d]
            # rank desc by score, tie-break account asc (deterministic)
            rows = sorted(by_day.get(d, []), key=lambda x: (-x[1], x[0]))
            budget = k
            for p, _ in rows:
                if budget <= 0:
                    break
                la = last_audited.get(p)
                if la is not None and (di - la) <= tabu_days:
                    continue  # recently audited — spend budget on someone new
                audited_on.setdefault(di, set()).add(p)
                last_audited[p] = di
                budget -= 1
        detected = sum(
            1 for p, active in campaigns.items()
            if any(p in audited_on.get(di, ()) for di in active)
        )
        out[k] = detected / n_camp if n_camp else float("nan")
    return out
