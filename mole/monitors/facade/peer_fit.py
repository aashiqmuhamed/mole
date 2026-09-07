"""FACADE-inspired level 0 peer-fit monitor (self-contained, numpy-only).

FACADE's insight isn't "flag rare events" — it's **action-context fit**:
does this account's behaviour fit *their peer group*, not the whole org?
A finance user running infra tooling is anomalous even if infra users do
it all day; a global anomaly detector misses that, a peer-conditioned one
catches it.

This is a lightweight, dependency-free realization of that insight: fit a
benign behavioural profile (robust median + scaled-MAD) *per peer group*
over the 18 user-day rollup features, then score a user-day by its
deviation from *its own group's* profile. It is NOT the full learned
two-tower contrastive FACADE (that needs a neural net + is bridged via
audit/facade_export.py for anyone wanting the bit-exact production model) —
it's the same idea at our scale, with zero new dependencies.

level 0: reads only the numeric rollup, never a semantic payload, so
tokens_inspected is always 0.
"""
from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

from ...audit.rollup import UserDayFeatures, rollup
from ...audit.schema import AuditEvent
from ..base import Monitor, Suspicion
from ..classical.anomaly import NUMERIC_FEATURES, _vector

_GLOBAL = "__global__"


class PeerFitMonitor(Monitor):
    """Peer-conditioned anomaly: deviation from the account's group profile."""

    observability_level = 0
    name = "facade.peer_fit"
    Z_CLIP = 8.0

    def __init__(self, peer_of: Mapping[str, str] | None = None,
                 *, top_k: int | None = None) -> None:
        # account -> peer-group key (e.g. team or primary org group).
        self._peer_of = dict(peer_of or {})
        # Aggregation across features: None = mean (FACADE default), which
        # DILUTES a single anomalous feature (a lone public_link in a 100-event
        # day) across ~22 features. top_k = mean of the K most-deviant features —
        # so a low-footprint egress isn't averaged away. This is the knob that
        # decides whether peer-fit can see egress-by-legit-access at all.
        self._top_k = top_k
        # group key -> (median[], scale[])
        self._profiles: dict[str, tuple[list[float], list[float]]] = {}
        self._fitted = False

    def _group_of(self, account: str) -> str:
        return self._peer_of.get(account, _GLOBAL)

    # ── fit ────────────────────────────────────────────────────────────

    def fit(
        self,
        benign_rows: Sequence[UserDayFeatures],
        peer_of: Mapping[str, str] | None = None,
    ) -> "PeerFitMonitor":
        if peer_of is not None:
            self._peer_of = dict(peer_of)
        if not benign_rows:
            raise ValueError("PeerFitMonitor.fit needs ≥1 benign row")

        by_group: dict[str, list[list[float]]] = {}
        for r in benign_rows:
            g = self._group_of(r.account)
            by_group.setdefault(g, []).append(_vector(r))
            # Every row also feeds the global fallback profile.
            by_group.setdefault(_GLOBAL, []).append(_vector(r))

        for g, vecs in by_group.items():
            cols = list(zip(*vecs))
            med = [statistics.median(c) for c in cols]
            scale = [
                max(1.4826 * statistics.median([abs(x - m) for x in c]), 1e-6)
                for c, m in zip(cols, med)
            ]
            self._profiles[g] = (med, scale)
        self._fitted = True
        return self

    def fit_events(
        self,
        benign_events: Sequence[AuditEvent],
        peer_of: Mapping[str, str] | None = None,
    ) -> "PeerFitMonitor":
        return self.fit(rollup(benign_events), peer_of=peer_of)

    # ── score ──────────────────────────────────────────────────────────

    def _row_score(self, udf: UserDayFeatures) -> tuple[float, str, list[tuple[float, str]]]:
        g = self._group_of(udf.account)
        # Fall back to the global profile for an unseen group.
        med, scale = self._profiles.get(g) or self._profiles[_GLOBAL]
        v = _vector(udf)
        per_feat = [
            (min(abs(x - m) / s, self.Z_CLIP), name)
            for x, m, s, name in zip(v, med, scale, NUMERIC_FEATURES)
        ]
        top = sorted(per_feat, reverse=True)[:3]
        if self._top_k:
            kk = sorted((z for z, _ in per_feat), reverse=True)[: self._top_k]
            score = sum(kk) / len(kk) if kk else 0.0
        else:
            score = sum(z for z, _ in per_feat) / len(per_feat) if per_feat else 0.0
        return score, g, top

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
        best = max((self._row_score(r) + (r,) for r in rows), key=lambda t: t[0])
        score, group, top, worst = best
        why = ", ".join(f"{name}={getattr(worst, name)}(z~{z:.1f})" for z, name in top)
        return Suspicion(
            score=score,
            rationale=f"{worst.account} {worst.date} vs peers[{group}]: {why}",
            tokens_inspected=0,
        )


def peers_from_org_yaml(org_yaml_path: str | None = None) -> dict[str, str]:
    """Build an account -> peer-group map from org_template.yaml.

    Peer group = the account's first declared org group (their primary
    affiliation), else their team, else global. Keeps the monitor decoupled
    from the live OrgManager — callers pass the resulting dict to fit().
    """
    import os
    from pathlib import Path

    import yaml

    path = Path(
        org_yaml_path
        or os.environ.get("ORG_TEMPLATE_PATH")
        or Path(__file__).resolve().parents[3] / "bootstrap" / "org_template.yaml"
    )
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    peer_of: dict[str, str] = {}
    for emp in doc.get("accounts") or []:
        eid = emp.get("id")
        if not eid:
            continue
        groups = emp.get("groups") or []
        peer_of[eid] = groups[0] if groups else _GLOBAL
    return peer_of
