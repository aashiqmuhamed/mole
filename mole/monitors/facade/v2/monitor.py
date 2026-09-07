"""FACADE v2 monitor: net assembly + benign-only contrastive training (synthetic
positives) + per-event SF_OMDOT scoring + clustering aggregator g.

Drop-in alongside snn.py: `train_facade_v2(benign)` -> a Monitor with `score_episode`.
"""
from __future__ import annotations

import collections
from typing import Any, Sequence

import torch
import torch.nn as nn

from ...base import Monitor, Suspicion
from ....audit.rollup import _date_of
from ._facade_core import (
    TokenLookupEmbedder, ConcatenateThenSNNTower, compute_scores,
    contrastive_labels_scores, pairwise_huber_loss,
    SF_OMDOT, TR_SOFTPLUS, TR_L2_NORMALIZED, WS_IDENTITY, WN_L2,
)
from .featurizer import FacadeV2Featurizer, _ts_seconds


class FacadeV2Net(nn.Module):
    """Two username-token towers (action=resource accessors, context=peers)."""

    def __init__(self, vocab: list, dim: int = 32, token_dim: int = 16,
                 snn_layers=(24,), oov_dropout: float = 0.05,
                 dropout_tokens: float = 0.05, dropout_neurons: float = 0.05):
        super().__init__()
        self.action_embed = TokenLookupEmbedder(vocab, token_dim, 1, oov_dropout)
        self.context_embed = TokenLookupEmbedder(vocab, token_dim, 1, oov_dropout)
        transforms = [TR_SOFTPLUS, TR_L2_NORMALIZED]
        seg = lambda: [{"name": "u", "embed_dim": token_dim,
                        "weight_scaling": WS_IDENTITY, "weight_normalization": WN_L2}]
        self.action_tower = ConcatenateThenSNNTower(seg(), list(snn_layers), dim,
                                                    dropout_tokens, dropout_neurons, transforms)
        self.context_tower = ConcatenateThenSNNTower(seg(), list(snn_layers), dim,
                                                     dropout_tokens, dropout_neurons, transforms)
        self.scoring = SF_OMDOT

    def embed_contexts(self, ids_pad, intens_pad):     # dense [B, max] -> [B, dim]
        emb = self.context_embed(ids_pad)
        return self.context_tower({"u": {"embeddings": emb, "intensities": intens_pad}})

    def embed_actions(self, ids_flat, intens_flat, lengths):  # ragged -> [n_actions, dim]
        emb = self.action_embed(ids_flat)
        return self.action_tower({"u": {"embeddings": emb, "intensities": intens_flat,
                                        "lengths": lengths}})


def _pad(seqs: list[list[int]]):
    L = max((len(s) for s in seqs), default=1) or 1
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    wt = torch.zeros(len(seqs), L, dtype=torch.float32)
    return ids, wt, L


def _build_examples(benign, feat, net):
    """(account,day) -> (ctx_ids, ctx_intens, [(act_ids, act_intens), ...])."""
    by_pd = collections.defaultdict(list)
    for e in benign:
        by_pd[(e.account, _date_of(e.ts))].append(e)
    examples = []
    a_id, c_id = net.action_embed.to_id, net.context_embed.to_id
    for (p, _day), evs in by_pd.items():
        ts = min(_ts_seconds(e) for e in evs)
        ctoks, cints = feat.context_features(p, ts)
        if not ctoks:
            continue
        actions = []
        for e in evs:
            atoks, aints = feat.action_features(e)
            if atoks:
                actions.append(([a_id(t) for t in atoks], aints))
        if not actions:
            continue
        examples.append(([c_id(t) for t in ctoks], cints, actions, p))
    return examples


def _batch(examples, net, prin_to_int):
    ctx_ids, ctx_wt, _ = _pad([ex[0] for ex in examples])
    for b, ex in enumerate(examples):
        for j, (tid, w) in enumerate(zip(ex[0], ex[1])):
            ctx_ids[b, j] = tid; ctx_wt[b, j] = w
    act_ids_flat, act_wt_flat, lengths, n_per_ex, act_prin = [], [], [], [], []
    for b, ex in enumerate(examples):
        n_per_ex.append(len(ex[2]))
        for tid_list, w_list in ex[2]:
            lengths.append(len(tid_list))
            act_ids_flat += tid_list; act_wt_flat += list(w_list)
            act_prin.append(prin_to_int[ex[3]])
    return (ctx_ids, ctx_wt,
            torch.tensor(act_ids_flat, dtype=torch.long),
            torch.tensor(act_wt_flat, dtype=torch.float32),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(n_per_ex, dtype=torch.long),
            torch.tensor([prin_to_int[ex[3]] for ex in examples], dtype=torch.long),
            torch.tensor(act_prin, dtype=torch.long))


def train_facade_v2(benign: Sequence, *, dim: int = 32, token_dim: int = 16,
                    snn_layers=(24,), epochs: int = 12, batch_size: int = 64,
                    lr: float = 1e-3, weight_decay: float = 0.004,
                    contrastive_per_query: int = 4, soft_margin: float = 0.05,
                    hard_margin: float = 0.02, seed: int = 0,
                    threshold_pct: float = 0.90,
                    featurizer: FacadeV2Featurizer | None = None,
                    prefit_featurizer: FacadeV2Featurizer | None = None) -> "FacadeV2Monitor":
    # prefit_featurizer: an ALREADY-fitted featurizer (optionally cache-enabled) reused
    # across a sweep so the identical featurization isn't recomputed per config. When
    # given, we skip the re-fit. Otherwise fit `featurizer` (or a default) on `benign`.
    torch.manual_seed(seed)
    feat = prefit_featurizer if prefit_featurizer is not None else (featurizer or FacadeV2Featurizer()).fit(benign)
    net = FacadeV2Net(feat.vocab, dim, token_dim, snn_layers)
    examples = _build_examples(benign, feat, net)
    prin_to_int = {p: i for i, p in enumerate(sorted({ex[3] for ex in examples}))}
    if not examples:
        return FacadeV2Monitor(net, feat)
    # materialize LazyLinear before the optimizer sees params
    net.train()
    with torch.no_grad():
        b0 = _batch(examples[:min(8, len(examples))], net, prin_to_int)
        net.embed_contexts(b0[0], b0[1]); net.embed_actions(b0[2], b0[3], b0[4])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    rng = torch.Generator().manual_seed(seed)
    loss_cfg = dict(contrastive_scores_per_query=contrastive_per_query,
                    positive_instances_weight_factor=1.0,
                    soft_margin=soft_margin, hard_margin=hard_margin)
    for _ep in range(epochs):
        order = torch.randperm(len(examples), generator=rng).tolist()
        for i in range(0, len(examples), batch_size):
            batch = [examples[k] for k in order[i:i + batch_size]]
            if len(batch) < 2:
                continue
            (ci, cw, ai, aw, al, npe, _ex_prin, act_prin) = _batch(batch, net, prin_to_int)
            ctx = net.embed_contexts(ci, cw)                 # [B, dim]
            act = net.embed_actions(ai, aw, al)              # [A, dim]
            matching = torch.repeat_interleave(torch.arange(len(batch)), npe)  # [A]
            labels, scores, weights = contrastive_labels_scores(
                queries=act, items=ctx, matching_items=matching, scoring_function=SF_OMDOT,
                contrastive_scores_per_query=loss_cfg["contrastive_scores_per_query"],
                positive_instances_weight_factor=1.0,
                query_compatibility_keys=act_prin,
                item_compatibility_keys=torch.tensor([prin_to_int[ex[3]] for ex in batch]))
            loss = pairwise_huber_loss(labels, scores, weights, soft_margin, hard_margin)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    # Per-action score threshold (FACADE Sec 7.3): only actions scoring above the
    # benign `threshold_pct` quantile are presented to g, so g isn't volume-biased
    # by benign actions. Computed on benign training actions (each vs its own context).
    tau = 0.0
    with torch.no_grad():
        bscores = []
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            if len(batch) < 1:
                continue
            (ci, cw, ai, aw, al, npe, _ep, _ap) = _batch(batch, net, prin_to_int)
            ctx = net.embed_contexts(ci, cw)
            act = net.embed_actions(ai, aw, al)
            matching = torch.repeat_interleave(torch.arange(len(batch)), npe)
            bscores.append(compute_scores(ctx[matching], act, SF_OMDOT))
        if bscores:
            alls = torch.cat(bscores)
            tau = float(torch.quantile(alls, threshold_pct)) if alls.numel() else 0.0
    return FacadeV2Monitor(net, feat, tau=tau)


def _aggregate_g(emb: torch.Tensor, scores: torch.Tensor, delta: float = 0.1) -> float:
    """Aggregator g (paper Sec 6.2.4): greedy cosine clustering -> sum of per-cluster
    max scores. Monotone (more actions never lower g), discounts redundant (cosine>1-delta)
    actions. Approximates HAC with a single-pass threshold."""
    n = emb.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return float(scores[0])
    en = torch.nn.functional.normalize(emb, p=2, dim=1)
    reps: list[int] = []                      # cluster representative indices
    cluster_max: list[float] = []
    order = torch.argsort(scores, descending=True).tolist()   # seed clusters by most-anomalous
    for i in order:
        placed = False
        for c, r in enumerate(reps):
            if float(torch.dot(en[i], en[r])) > (1.0 - delta):
                cluster_max[c] = max(cluster_max[c], float(scores[i]))
                placed = True
                break
        if not placed:
            reps.append(i); cluster_max.append(float(scores[i]))
    return float(sum(cluster_max))


class FacadeV2Monitor(Monitor):
    name = "facade_v2"
    observability_level = 0

    def __init__(self, net: FacadeV2Net, feat: FacadeV2Featurizer, tau: float = 0.0):
        self.net = net
        self.feat = feat
        self.tau = tau   # per-action score threshold before aggregator g (FACADE Sec 7.3)

    async def score_episode(self, events: Sequence, *, budget: Any | None = None) -> Suspicion:
        if not events:
            return Suspicion(0.0, "no events", 0)
        p = events[0].account
        ts = min(_ts_seconds(e) for e in events)
        ctoks, cints = self.feat.context_features(p, ts)
        if not ctoks:
            return Suspicion(0.0, "no peer context", 0)
        c_id, a_id = self.net.context_embed.to_id, self.net.action_embed.to_id
        with torch.no_grad():
            ci = torch.tensor([[c_id(t) for t in ctoks]], dtype=torch.long)
            cw = torch.tensor([cints], dtype=torch.float32)
            ctx = self.net.embed_contexts(ci, cw)             # [1, dim]
            ids_flat, wt_flat, lengths = [], [], []
            for e in events:
                atoks, aints = self.feat.action_features(e)
                if not atoks:                                  # cold-start -> drop
                    continue
                lengths.append(len(atoks))
                ids_flat += [a_id(t) for t in atoks]; wt_flat += list(aints)
            if not lengths:
                return Suspicion(0.0, "no scorable actions", 0)
            act = self.net.embed_actions(torch.tensor(ids_flat, dtype=torch.long),
                                         torch.tensor(wt_flat, dtype=torch.float32),
                                         torch.tensor(lengths, dtype=torch.long))  # [A, dim]
            ctx_rep = ctx.expand(act.shape[0], -1)
            scores = compute_scores(ctx_rep, act, SF_OMDOT)    # [A], higher = more anomalous
            keep = scores > self.tau                            # FACADE Sec 7.3 pre-threshold
            if keep.any():
                g = _aggregate_g(act[keep], scores[keep])
            else:
                g = 0.0
        return Suspicion(float(g), f"facade_v2 g over {int(keep.sum())}/{act.shape[0]} actions", 0)
