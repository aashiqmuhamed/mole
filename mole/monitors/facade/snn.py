"""Event-level FACADE — faithful two-tower contrastive monitor (PyTorch).

This is the real FACADE design (cf. the original FACADE model), reimplemented standalone
in-bench (no external dep on that repo). Unlike the daily-rollup `peer_fit`, this
scores each ACTION against the acting account's social CONTEXT, at the event
level — which is what removes the concentrated-task-vs-broad-day level 0 false
positive.

Faithful core (v1):
  - **action features**: the set of accounts who normally touch the action's
    resource (a bag of weighted account tokens) + the action-type token. This
    is FACADE's resource-history featurization: "who uses this resource?".
  - **context features**: the acting account's peer set from the org graph
    (group/team co-members) — a simplified bipartite fold.
  - **two-tower SNN**: each side = weighted-mean token embedding (segment
    reduction) -> concat -> SNN (Linear/SELU/AlphaDropout) -> embedding; the
    compatibility score is the dot product.
  - **contrastive, benign-only training**: positive = (action, its own
    account's context); negatives = the action vs OTHER accounts' contexts in
    the minibatch (in-batch negatives). Logistic loss. No attack labels.
  - **anomaly = low compatibility** between an action and the account's context.

Deferred to v2: the directed multi-attribute bipartite 2-hop fold and the exact
pairwise-Huber loss. torch is imported lazily so the rest of the bench stays
numpy-only; install with `pip install -e ./benchmark[facade]`.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ...audit.schema import AuditEvent
from ..base import Monitor, Suspicion

_OOV = "<oov>"
_PAD = "<pad>"


def _day_ord(ts: str) -> int:
    """Day ordinal from an ISO-ish 'YYYY-MM-DD...' timestamp (0 if unparseable).

    Day granularity matches the sim's day structure + the per-(account,day) eval
    episodes; used only for the optional `window_days` recency filter.
    """
    if not ts or len(ts) < 10:
        return 0
    from datetime import date
    try:
        return date(int(ts[0:4]), int(ts[5:7]), int(ts[8:10])).toordinal()
    except ValueError:
        return 0


# ── featurization ────────────────────────────────────────────────────

@dataclass
class FacadeFeaturizer:
    """Turns events into (action-token-bag, context-token-bag) pairs.

    Fit on a benign corpus + the org peer map. action_tokens(e) = accounts who
    touch e's resource (the resource's normal actors); context_tokens(p) = p's
    org peer set. Both are bags of account tokens; the action also carries an
    action-type token.
    """
    peer_of: Mapping[str, set[str]]
    resource_actors: dict[str, set[str]] = field(default_factory=dict)
    vocab: dict[str, int] = field(default_factory=dict)
    atype_vocab: dict[str, int] = field(default_factory=dict)
    # window_days: if set, action_tokens(e) counts only accounts who touched e's
    # resource within the trailing `window_days` (recency), mirroring v2's lookback.
    # NOTE only meaningful with a rolling/streaming featurizer (fit on data up to the
    # scored day); on a static train/test split with a train->test day gap, a small
    # window starves test co-access. None = all-time (the original global behaviour).
    window_days: float | None = None
    resource_times: dict[str, list] = field(default_factory=dict)

    @staticmethod
    def _resource(e: AuditEvent) -> str:
        args = e.args or {}
        return str(e.resource_id or args.get("path") or args.get("project")
                   or args.get("checkpoint_id") or args.get("key") or f"{e.service}:{e.action}")

    @staticmethod
    def _atype(e: AuditEvent) -> str:
        # Egress marker: external email / public-link are the *exfil* actions, and
        # the access-anomaly features can't see them (the insider has legit access;
        # the harm is the data leaving). Tag them so they're a distinct, learnable
        # action-type — unlocks the egress-by-legit-access threats (01/02/06)
        # that a pure resource-co-accessor model structurally misses.
        base = f"{e.service}.{e.action}"
        if getattr(e, "is_external", False) or e.action in ("public_link",):
            base += "|EGRESS"
        return base

    def fit(self, benign: Sequence[AuditEvent]) -> "FacadeFeaturizer":
        res_actors: dict[str, set[str]] = defaultdict(set)
        res_times: dict[str, list] = defaultdict(list)
        accounts: set[str] = set()
        atypes: set[str] = set()
        for e in benign:
            r = self._resource(e)
            res_actors[r].add(e.account)
            accounts.add(e.account)
            atypes.add(self._atype(e))
            if self.window_days is not None:
                res_times[r].append((_day_ord(getattr(e, "ts", "")), e.account))
        accounts |= set(self.peer_of)
        for peers in self.peer_of.values():
            accounts |= set(peers)
        self.resource_actors = {r: set(a) for r, a in res_actors.items()}
        if self.window_days is not None:
            self.resource_times = {r: sorted(v) for r, v in res_times.items()}
        self.vocab = {_PAD: 0, _OOV: 1, **{p: i + 2 for i, p in enumerate(sorted(accounts))}}
        self.atype_vocab = {_OOV: 0, **{t: i + 1 for i, t in enumerate(sorted(atypes))}}
        return self

    def _ids(self, tokens: set[str]) -> list[int]:
        # sorted() is load-bearing for REPRODUCIBILITY: the token bag is a masked
        # mean (order-invariant) BUT _oov_dropout is position-based, so an unsorted
        # set iteration (PYTHONHASHSEED-dependent) makes training drop different
        # tokens run-to-run -> the FACADE headline drifts ~0.27-0.33 across runs.
        # Sorting pins the position order so a fixed seed is fully deterministic.
        return [self.vocab.get(t, 1) for t in sorted(tokens)] or [0]

    def action_tokens(self, e: AuditEvent, *, extra_actors: set[str] | None = None) -> tuple[list[int], int]:
        """(account-token ids who touch this resource, action-type id)."""
        if self.window_days is None:
            actors = set(self.resource_actors.get(self._resource(e), set()))
        else:                                          # recency window: (do - window, do]
            do = _day_ord(getattr(e, "ts", ""))
            lo = do - self.window_days
            actors = {p for (d, p) in self.resource_times.get(self._resource(e), ())
                      if lo < d <= do}
        if extra_actors:
            actors |= extra_actors
        actors.discard(e.account)                    # the actor's own peers carry the signal
        return self._ids(actors), self.atype_vocab.get(self._atype(e), 0)

    def context_tokens(self, account: str) -> list[int]:
        peers = set(self.peer_of.get(account, set()))
        peers.discard(account)
        return self._ids(peers)


def peers_from_org(org_yaml_path: str | None = None) -> dict[str, set[str]]:
    """account -> set of co-members across all org groups (a simple peer fold)."""
    import yaml
    from ...generator.persona_loader import _default_org_yaml
    path = org_yaml_path or _default_org_yaml()
    doc = yaml.safe_load(open(path, encoding="utf-8")) or {}
    peer: dict[str, set[str]] = defaultdict(set)
    for g in doc.get("groups", []):
        members = g.get("members", []) or []
        for m in members:
            peer[m] |= {x for x in members if x != m}
    for dept in doc.get("departments", []):
        for team in dept.get("teams", []):
            members = list(team.get("members", []) or [])
            if team.get("manager"):
                members.append(team["manager"])
            for m in members:
                peer[m] |= {x for x in members if x != m}
    return dict(peer)


# ── two-tower model + contrastive training (torch, lazy) ──────────────

def _build_two_tower(n_accounts: int, n_atypes: int, dim: int):
    import torch
    import torch.nn as nn

    class _SNN(nn.Module):
        """FACADE's self-normalizing MLP: Linear -> SELU -> AlphaDropout -> Linear."""
        def __init__(self, d_in: int, d_hidden: int, d_out: int, p: float = 0.1):
            super().__init__()
            self.l1 = nn.Linear(d_in, d_hidden)
            self.drop = nn.AlphaDropout(p)
            self.l2 = nn.Linear(d_hidden, d_out)

        def forward(self, x):
            x = self.drop(torch.selu(self.l1(x)))
            return self.l2(x)

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.prin = nn.Embedding(n_accounts, dim, padding_idx=0)
            self.atype = nn.Embedding(n_atypes, dim)
            self.action_snn = _SNN(2 * dim, 2 * dim, dim)     # [coactor-bag ; atype]
            self.context_snn = _SNN(dim, 2 * dim, dim)        # [peer-bag]

        @staticmethod
        def _bag(emb, ids, mask):
            # masked mean over a padded [B, L] token-id tensor -> [B, dim]
            v = emb(ids) * mask.unsqueeze(-1)
            denom = mask.sum(1, keepdim=True).clamp(min=1.0)
            return v.sum(1) / denom

        def action_embed(self, a_ids, a_mask, at_ids):
            coactor = self._bag(self.prin, a_ids, a_mask)
            return self.action_snn(torch.cat([coactor, self.atype(at_ids)], dim=-1))

        def context_embed(self, c_ids, c_mask):
            return self.context_snn(self._bag(self.prin, c_ids, c_mask))

    return TwoTower()


def _pad(rows: list[list[int]]):
    # Vectorized: fill a numpy array with per-row SLICE assignment, then one
    # torch.from_numpy. (Was a per-ELEMENT torch assignment in a python double
    # loop -- `ids[i,j]=t` -- which was ~90% of FACADE's runtime: millions of
    # individual tensor index-sets. Same result, ~100x faster.)
    import numpy as np
    import torch
    L = max((len(r) for r in rows), default=1)
    n = len(rows)
    ids = np.zeros((n, L), dtype=np.int64)
    mask = np.zeros((n, L), dtype=np.float32)
    for i, r in enumerate(rows):
        if r:
            ids[i, :len(r)] = r
            mask[i, :len(r)] = 1.0
    return torch.from_numpy(ids), torch.from_numpy(mask)


def _oov_dropout(ids: "Any", mask: "Any", p: float, oov_idx: int = 1) -> "Any":
    """Randomly replace real token ids with the OOV index during training
    (the original FACADE OOV dropout). Without this the OOV
    embedding is never trained, so unseen/cold-start accounts & resources at
    test time map to a garbage vector — the likely cause of shadow-cohort=0.
    Only non-pad positions are dropped; pads stay 0."""
    import torch
    if p <= 0.0:
        return ids
    drop = (torch.rand_like(mask) < p) & (mask > 0)
    return torch.where(drop, torch.full_like(ids, oov_idx), ids)


def _val_auc(model: Any, feat: "FacadeFeaturizer",
             val_events: Sequence[AuditEvent], val_keys: set) -> float:
    """AUROC of per-(account,day) max-anomaly scores vs the val malicious set.
    Mirrors FacadeSNNMonitor.score_episode but sync, for in-loop selection."""
    import torch
    from collections import defaultdict as _dd
    by_pd: dict[tuple, list] = _dd(list)
    for e in val_events:
        by_pd[(getattr(e, "account", ""), (getattr(e, "ts", "") or "")[:10])].append(e)
    keys, scores = [], []
    with torch.no_grad():
        for pd, evs in by_pd.items():
            ep_actors: dict[str, set] = _dd(set)
            for e in evs:
                ep_actors[feat._resource(e)].add(e.account)
            a_rows, at_ids, c_rows = [], [], []
            for e in evs:
                a_list, at = feat.action_tokens(e, extra_actors=ep_actors.get(feat._resource(e)))
                a_rows.append(a_list); at_ids.append(at)
                c_rows.append(feat.context_tokens(e.account))
            a_ids, a_mask = _pad(a_rows); c_ids, c_mask = _pad(c_rows)
            ae = model.action_embed(a_ids, a_mask, torch.tensor(at_ids, dtype=torch.long))
            ce = model.context_embed(c_ids, c_mask)            # batched per episode
            worst = float((1.0 - torch.sigmoid((ae * ce).sum(-1))).max())
            keys.append(pd); scores.append(worst)
    pos = [s for k, s in zip(keys, scores) if k in val_keys]
    neg = [s for k, s in zip(keys, scores) if k not in val_keys]
    if not pos or not neg:
        return 0.0
    wins = sum(1.0 for p in pos for n in neg if p > n) + 0.5 * sum(1.0 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


def train_facade(
    benign: Sequence[AuditEvent],
    *,
    peer_of: Mapping[str, set[str]] | None = None,
    dim: int = 32,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 5e-3,
    seed: int = 0,
    oov_dropout: float = 0.1,
    loss_kind: str = "huber",
    soft_margin: float = 0.05,
    hard_margin: float = 0.02,
    window_days: float | None = None,
    val_events: Sequence[AuditEvent] | None = None,
    val_keys: set | None = None,
) -> "FacadeSNNMonitor":
    """Train the two-tower model contrastively on benign events (in-batch
    negatives: each action vs other accounts' contexts in the minibatch;
    same-account off-diagonals masked). Returns a ready FacadeSNNMonitor.

    Benign-only: callers must pass an is_malicious-filtered set — training on
    attack events as 'benign' teaches the model the attack is normal.

    Val selection: if val_events + val_keys are given, evaluate val AUROC after
    each epoch and KEEP THE BEST-AUC checkpoint (instead of the arbitrary final
    epoch). This is the FACADE val-cutoff idea — without it the fixed-epoch
    model can land anywhere, and FACADE underperformed even z-score."""
    import copy as _copy

    import torch
    torch.manual_seed(seed)
    feat = FacadeFeaturizer(peer_of if peer_of is not None else peers_from_org(),
                            window_days=window_days).fit(benign)

    a_rows, at_ids, c_rows, princ = [], [], [], []
    for e in benign:
        a, at = feat.action_tokens(e)
        a_rows.append(a); at_ids.append(at)
        c_rows.append(feat.context_tokens(e.account)); princ.append(e.account)
    n = len(a_rows)
    if n == 0:
        raise ValueError("no benign events to train on")

    model = _build_two_tower(len(feat.vocab), len(feat.atype_vocab), dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    idx = list(range(n))
    rng = __import__("random").Random(seed)

    do_val = bool(val_events) and val_keys is not None
    best_state, best_auc, best_ep = None, -1.0, -1
    for ep in range(epochs):
        model.train()
        rng.shuffle(idx)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            if len(b) < 2:
                continue
            a_ids, a_mask = _pad([a_rows[i] for i in b])
            c_ids, c_mask = _pad([c_rows[i] for i in b])
            # OOV dropout (train-time only): teaches the OOV embedding so unseen
            # entities at test time aren't garbage. Applied inside model.train().
            a_ids = _oov_dropout(a_ids, a_mask, oov_dropout)
            c_ids = _oov_dropout(c_ids, c_mask, oov_dropout)
            at = torch.tensor([at_ids[i] for i in b], dtype=torch.long)
            ae = model.action_embed(a_ids, a_mask, at)        # [B,dim]
            ce = model.context_embed(c_ids, c_mask)           # [B,dim]
            logits = ae @ ce.t()                              # [B,B] action_i · context_j
            B = len(b)
            # mask same-account off-diagonals (spurious negatives) -- vectorized
            # (was an O(B^2) python double loop). weight[i,j]=0 iff i!=j and same
            # account; diagonal (the positive) stays 1.
            import numpy as _np
            ps = _np.array([princ[i] for i in b])
            same = torch.from_numpy(ps[:, None] == ps[None, :])   # [B,B], incl diagonal
            weight = (~same).float()                               # 0 at same-account pairs
            weight.fill_diagonal_(1.0)                             # restore diagonal (positive) to 1
            if loss_kind == "huber":
                # the FACADE pairwise-Huber margin loss:
                # per query i, positive = logits[i,i], negatives = logits[i,j!=i].
                # Want pos > neg + hard_margin; penalise violation
                # d = neg - pos + hard_margin with a Huber-smoothed hinge (soft_margin
                # transition). O(B^2) direct form (B=128 → trivial), identical loss to
                # their sorted-cumsum O(N log N) version.
                pos = logits.diag().unsqueeze(1)              # [B,1]
                d = logits - pos + hard_margin                # [B,B] neg - pos + hard
                sm = soft_margin
                hinge = torch.where(
                    d <= -sm / 2, torch.zeros_like(d),
                    torch.where(d >= sm / 2, d, (d + sm / 2) ** 2 / (2 * sm)))
                neg_mask = weight.clone()
                neg_mask.fill_diagonal_(0.0)                  # exclude the positive itself
                loss = (hinge * neg_mask).sum() / neg_mask.sum().clamp(min=1.0)
            else:
                target = torch.eye(B)
                loss = (bce(logits, target) * weight).sum() / weight.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        if do_val:
            model.eval()
            auc = _val_auc(model, feat, val_events, val_keys)
            if auc > best_auc:
                best_auc, best_ep = auc, ep
                best_state = _copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  facade: selected epoch {best_ep + 1}/{epochs} by val AUROC={best_auc:.3f}")
    model.eval()
    return FacadeSNNMonitor(model, feat)


# ── monitor ───────────────────────────────────────────────────────────

class FacadeSNNMonitor(Monitor):
    """Event-level FACADE: anomaly = low action-context compatibility. level 0
    (reads only accounts / resources / org graph — no semantic payload)."""
    observability_level = 0
    name = "facade.snn"

    def __init__(self, model: Any, featurizer: FacadeFeaturizer) -> None:
        self.model = model
        self.feat = featurizer

    async def score_episode(self, events: Sequence[AuditEvent], *, budget: Any | None = None) -> Suspicion:
        import torch
        evs = list(events)
        if not evs:
            return Suspicion(0.0, "no events", 0)
        # resource->actors seen within this episode (combined with training history)
        ep_actors: dict[str, set[str]] = defaultdict(set)
        for e in evs:
            ep_actors[self.feat._resource(e)].add(e.account)

        # Featurize ALL events, then ONE batched forward pass (was a per-event,
        # batch-of-1 loop -- the scoring cost). The token bag is a masked MEAN, so
        # padding all events to the episode's max length is numerically identical to
        # per-event padding; the model is in eval() (AlphaDropout = identity). Result
        # is bit-for-bit the same max-pooled anomaly, just far faster.
        a_rows, at_ids, c_rows = [], [], []
        for e in evs:
            a_list, at = self.feat.action_tokens(
                e, extra_actors=ep_actors.get(self.feat._resource(e)))
            a_rows.append(a_list); at_ids.append(at)
            c_rows.append(self.feat.context_tokens(e.account))
        a_ids, a_mask = _pad(a_rows)
        c_ids, c_mask = _pad(c_rows)
        with torch.no_grad():
            ae = self.model.action_embed(a_ids, a_mask, torch.tensor(at_ids, dtype=torch.long))
            ce = self.model.context_embed(c_ids, c_mask)
            compat = torch.sigmoid((ae * ce).sum(-1))        # [N] per-event compatibility
            anomaly = 1.0 - compat                            # [N]
        wi = int(anomaly.argmax())
        e = evs[wi]
        return Suspicion(
            score=float(anomaly[wi]),
            rationale=(f"{e.account} {e.service}.{e.action} "
                       f"{self.feat._resource(e)[:48]} compat={float(compat[wi]):.2f}"),
            tokens_inspected=0,
        )
