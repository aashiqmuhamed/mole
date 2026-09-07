"""Adapter: AuditEvent -> FACADE token-bags + intensities.

Faithful to FACADE's featurization (paper Sec 5; directive.textproto):
  ACTION (resource-history): a doc-access action is featurized by the *accounts
    who accessed the same resource* in the prior `lookback_days`, intensity = a
    dedup'd access count (one count per account per `dedup_hours`). First-access
    (empty-history) actions are DROPPED (paper Sec 7.1).
  CONTEXT (peer graph): per (account, day-snapshot), a temporally-decayed bipartite
    co-access fold -> peer accounts. weight(p->q) = sum over shared resources r of
    decay(p,r)*decay(q,r) (2-hop random walk), half-life decay toward the snapshot.
    Our analog of FACADE's code-review peer graph (we lack code-review; co-resource-
    access + comms is the substitute). Company-wide resources (>= `max_actors`
    distinct accounts) are dropped (paper Sec 7.1).

Tokens are usernames (accounts); the resource itself is never embedded.
"""
from __future__ import annotations

import collections
import math
from typing import Sequence

from ....audit.rollup import _date_of


def _resource(e) -> str:
    args = getattr(e, "args", None) or {}
    return str(getattr(e, "resource_id", None) or args.get("path")
               or args.get("project") or args.get("channel") or "?")


def _ts_seconds(e) -> float:
    """Best-effort epoch seconds from the event's real_ts (monotonic) or ts date."""
    rt = getattr(e, "real_ts", None)
    if isinstance(rt, (int, float)):
        return float(rt)
    d = _date_of(e.ts)
    # fall back to day index * 86400 (ordering preserved within the corpus)
    return _DAY0_ORD(d) * 86400.0


_DAY0 = {}
def _DAY0_ORD(d: str) -> float:
    if d not in _DAY0:
        # ordinal of YYYY-MM-DD; cheap, monotonic
        y, m, day = (int(x) for x in d.split("-"))
        _DAY0[d] = y * 365 + m * 31 + day
    return float(_DAY0[d])


_MISS = object()   # sentinel: distinguishes an un-cached key from a cached ([], []) value


class FacadeV2Featurizer:
    def __init__(self, lookback_days: int = 90, half_life_days: int = 90,
                 dedup_hours: float = 2.0, max_actors: int = 1000, max_peers: int = 200,
                 companywide_accounts: int = 2000, peer_source: str = "org"):
        # peer_source: "org" (org-chart, INDEPENDENT of doc-access — fixes the
        # circularity where co-access peers correlate with the action features),
        # "coaccess" (the temporal bipartite co-access fold), or "org+coaccess".
        self.peer_source = peer_source
        self.lookback = lookback_days * 86400.0
        self.half_life = half_life_days * 86400.0
        self.dedup = dedup_hours * 3600.0
        self.max_actors = max_actors
        self.max_peers = max_peers
        self.companywide = companywide_accounts
        # built in fit():
        self.res_access: dict[str, list[tuple[float, str]]] = {}   # resource -> [(ts, account)]
        self.prin_access: dict[str, list[tuple[float, str]]] = {}  # account -> [(ts, resource)]
        self.org_peers: dict[str, set[str]] = {}
        self.vocab: list[str] = []
        self._companywide: set[str] = set()
        # Optional memoization: action/context features are PURE functions of (event/
        # account) given the fitted state, so during a hyperparameter sweep (fixed
        # featurizer, many models) they can be computed ONCE and reused. Opt-in via
        # enable_cache(); OFF by default so normal single-model use is unchanged.
        self._cache = False
        self._afeat_cache: dict = {}
        self._cfeat_cache: dict = {}

    def enable_cache(self) -> "FacadeV2Featurizer":
        """Turn on action/context feature memoization (see __init__). Idempotent."""
        self._cache = True
        return self

    def fit(self, benign: Sequence) -> "FacadeV2Featurizer":
        res = collections.defaultdict(list)
        prin = collections.defaultdict(list)
        res_daily = collections.defaultdict(lambda: collections.defaultdict(set))  # res -> day -> {prin}
        accounts = set()
        for e in benign:
            r, p, t = _resource(e), e.account, _ts_seconds(e)
            res[r].append((t, p)); prin[p].append((t, r)); accounts.add(p)
            res_daily[r][_date_of(e.ts)].add(p)
        # company-wide resources: ever accessed by >= companywide distinct accounts in a day
        self._companywide = {r for r, days in res_daily.items()
                             if max((len(s) for s in days.values()), default=0) >= self.companywide}
        for r in res: res[r].sort()
        for p in prin: prin[p].sort()
        self.res_access = dict(res); self.prin_access = dict(prin)
        self.vocab = sorted(accounts)
        if "org" in self.peer_source:
            try:
                from ..snn import peers_from_org
                self.org_peers = peers_from_org()
            except Exception:                                  # noqa: BLE001
                self.org_peers = {}
        return self

    def _dedup_count(self, accesses: list[tuple[float, str]], upto: float,
                     decay: bool) -> dict[str, float]:
        """account -> dedup'd (optionally decayed) count within [upto-lookback, upto)."""
        out: dict[str, float] = collections.defaultdict(float)
        last: dict[str, float] = {}
        lo = upto - self.lookback
        for t, who in accesses:
            if t >= upto:
                break
            if t < lo:
                continue
            if who in last and (t - last[who]) < self.dedup:
                continue
            last[who] = t
            w = 0.5 ** ((upto - t) / self.half_life) if decay else 1.0
            out[who] += w
        return out

    def action_features(self, e) -> tuple[list[str], list[float]]:
        """Resource's prior accessors (excl. the actor). Empty -> ([],[]) => DROP."""
        r = _resource(e)
        ck = None
        if self._cache:
            ck = (r, _ts_seconds(e), getattr(e, "account", None))
            hit = self._afeat_cache.get(ck, _MISS)
            if hit is not _MISS:
                return hit
        if r in self._companywide:
            res = ([], [])
        else:
            accesses = self.res_access.get(r, [])
            counts = self._dedup_count(accesses, _ts_seconds(e), decay=False)
            counts.pop(e.account, None)
            if not counts:
                res = ([], [])
            else:
                items = sorted(counts.items(), key=lambda kv: -kv[1])[:self.max_actors]
                res = ([k for k, _ in items], [v for _, v in items])
        if ck is not None:
            self._afeat_cache[ck] = res
        return res

    def context_features(self, account: str, ts: float) -> tuple[list[str], list[float]]:
        """Peer bag. Default = org-chart (INDEPENDENT of the doc-access action
        features, so the contrastive task isn't trivial). Optional co-access fold."""
        ck = None
        if self._cache:
            ck = (account, ts)
            hit = self._cfeat_cache.get(ck, _MISS)
            if hit is not _MISS:
                return hit
        peer_w: dict[str, float] = collections.defaultdict(float)
        if "org" in self.peer_source:
            for q in self.org_peers.get(account, ()):       # static org peers, weight 1.0
                if q != account:
                    peer_w[q] += 1.0
        if "coaccess" in self.peer_source:
            my_res = self._dedup_count(self.prin_access.get(account, []), ts, decay=True)
            for r, w_pr in my_res.items():
                if r in self._companywide:
                    continue
                others = self._dedup_count(self.res_access.get(r, []), ts, decay=True)
                tot = sum(others.values()) or 1.0
                for q, w_qr in others.items():
                    if q != account:
                        peer_w[q] += w_pr * (w_qr / tot)      # 2-hop transition p->r->q
        if not peer_w:
            res: tuple = ([], [])
        else:
            items = sorted(peer_w.items(), key=lambda kv: -kv[1])[:self.max_peers]
            res = ([k for k, _ in items], [v for _, v in items])
        if ck is not None:
            self._cfeat_cache[ck] = res
        return res
