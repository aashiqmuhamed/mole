"""Validated FACADE model core — a standalone PyTorch port of the original FACADE model core (
verified ~1e-16 vs Google's TF FACADE). Kept faithful; do not "simplify".

Combines: layers/segment_embedder.py, layers/snn.py, loss/pairwise_huber.py, model.py.
Enum int constants match config_pb2 (WS_*, WN_*, TR_*, SF_*).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPSILON = 1e-6

# ── enums (match Google config_pb2) ─────────────────────────────────────────
WS_IDENTITY, WS_LOG, WS_UNIFORM = 1, 2, 3
WN_L1, WN_L2 = 1, 2
TR_IDENTITY, TR_SIGMOID, TR_SOFTPLUS, TR_SOFTMAX, TR_L2_NORMALIZED = 1, 2, 3, 4, 5
SF_DOT, SF_OMDOT, SF_HARDMIN, SF_SOFTMIN = 1, 2, 3, 4


# ── token embedding lookup ───────────────────────────────────────────────────
class TokenLookupEmbedder(nn.Module):
    """String token -> int id -> embedding. OOV ids are 0..num_oov-1; vocab starts after."""

    def __init__(self, vocabulary: list, dimensions: int, num_oov_indices: int = 1,
                 oov_dropout_rate: float = 0.0):
        super().__init__()
        self.num_oov_indices = max(1, num_oov_indices)
        vocab_size = len(vocabulary) + self.num_oov_indices
        if dimensions <= 0:
            dimensions = int(6 * vocab_size ** 0.25)
        self.embedding = nn.Embedding(vocab_size, dimensions)
        nn.init.trunc_normal_(self.embedding.weight)
        self.oov_dropout = OovDropout(oov_dropout_rate, self.num_oov_indices)
        self._str_to_int = {t: i + self.num_oov_indices for i, t in enumerate(vocabulary)}
        self.dimensions = dimensions

    def to_id(self, token: str) -> int:
        return self._str_to_int.get(token, 0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.oov_dropout(indices))


class OovDropout(nn.Module):
    def __init__(self, dropout_prob: float, num_oov_tokens: int):
        super().__init__()
        if num_oov_tokens < 1:
            raise ValueError("There must be at least one OOV token.")
        if not 0 <= dropout_prob < 1:
            raise ValueError(f"dropout_prob must be in [0,1), got {dropout_prob}")
        self.dropout_prob = dropout_prob
        self.num_oov_tokens = num_oov_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout_prob == 0:
            return x
        mask = torch.rand_like(x, dtype=torch.float32) < self.dropout_prob
        oov = torch.randint(0, self.num_oov_tokens, x.shape, dtype=x.dtype, device=x.device)
        return torch.where(mask, oov, x)


# ── segment embedder (weighted reduction + offset token + L1/L2 norm) ────────
class SegmentEmbedder(nn.Module):
    def __init__(self, embedding_dim: int, weight_scaling: int = WS_IDENTITY,
                 weight_normalization: int = WN_L2, dropout_rate: float = 0.0):
        super().__init__()
        self.weight_scaling = weight_scaling
        self.weight_normalization = weight_normalization
        self.dropout_rate = dropout_rate
        self.offset_token = nn.Parameter(torch.zeros(1, embedding_dim))
        self._offset_token_weight = math.log1p(1.0) if weight_scaling == WS_LOG else 1.0

    def forward(self, embeddings, intensities=None, lengths=None):
        if lengths is None:  # dense [batch, tokens, dim]
            if intensities is None:
                intensities = torch.ones(embeddings.shape[:-1], device=embeddings.device)
            return self._dense(embeddings, intensities)
        if intensities is None:
            intensities = torch.ones(embeddings.shape[0], device=embeddings.device)
        return self._ragged(embeddings, intensities, lengths)

    def _scale(self, w):
        if self.weight_scaling == WS_IDENTITY:
            return w
        if self.weight_scaling == WS_LOG:
            return torch.log1p(w)
        if self.weight_scaling == WS_UNIFORM:
            return torch.ones_like(w)
        raise ValueError(self.weight_scaling)

    def _drop(self, w):
        if self.training and self.dropout_rate > 0:
            w = w * (torch.rand_like(w) > self.dropout_rate).float()
        return w

    def _dense(self, emb, w):
        w = self._drop(self._scale(w)).unsqueeze(-1)
        seg = (emb * w).sum(dim=-2) + self._offset_token_weight * self.offset_token
        if self.weight_normalization == WN_L1:
            norm = w.sum(dim=-2) + self._offset_token_weight
        else:
            norm = torch.sqrt((w ** 2).sum(dim=-2) + self._offset_token_weight ** 2)
        return seg / norm

    def _ragged(self, emb, w, lengths):
        w = self._drop(self._scale(w))
        weighted = emb * w.unsqueeze(-1)
        n_seg = lengths.shape[0]
        seg_ids = torch.repeat_interleave(torch.arange(n_seg, device=lengths.device), lengths)
        dim = emb.shape[-1]
        seg = torch.zeros(n_seg, dim, device=emb.device)
        seg.scatter_add_(0, seg_ids.unsqueeze(1).expand(-1, dim), weighted)
        seg = seg + self._offset_token_weight * self.offset_token
        if self.weight_normalization == WN_L1:
            s = torch.zeros(n_seg, 1, device=emb.device)
            s.scatter_add_(0, seg_ids.unsqueeze(1), w.unsqueeze(1))
            norm = s + self._offset_token_weight
        else:
            s = torch.zeros(n_seg, 1, device=emb.device)
            s.scatter_add_(0, seg_ids.unsqueeze(1), (w ** 2).unsqueeze(1))
            norm = torch.sqrt(s + self._offset_token_weight ** 2)
        return seg / norm


# ── SNN + transforms + scoring ───────────────────────────────────────────────
class _AlphaDropout(nn.Module):
    def __init__(self, rate): super().__init__(); self.rate = rate
    def forward(self, x): return F.alpha_dropout(x, p=self.rate, training=self.training)


class SNN(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], dropout_rate: float):
        super().__init__()
        layers = []
        for size in layer_sizes[:-1]:
            layers += [nn.LazyLinear(size), nn.SELU(), _AlphaDropout(dropout_rate)]
        layers.append(nn.LazyLinear(layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class EmbeddingsTransformation(nn.Module):
    def __init__(self, transformations: Sequence[int]):
        super().__init__(); self.transformations = list(transformations)

    def forward(self, x):
        for tr in self.transformations:
            if tr == TR_IDENTITY: pass
            elif tr == TR_SIGMOID: x = torch.sigmoid(x)
            elif tr == TR_SOFTPLUS: x = F.softplus(x)
            elif tr == TR_SOFTMAX: x = F.softmax(x, dim=1)
            elif tr == TR_L2_NORMALIZED: x = F.normalize(x, p=2, dim=1)
            else: raise ValueError(tr)
        return x


def compute_scores(actions, contexts, scoring_function):
    if scoring_function == SF_DOT:
        return (actions * contexts).sum(dim=1)
    if scoring_function == SF_OMDOT:
        return 1.0 - (actions * contexts).sum(dim=1)
    if scoring_function == SF_HARDMIN:
        return torch.minimum(actions, contexts).sum(dim=1)
    if scoring_function == SF_SOFTMIN:
        return (actions * (contexts / (actions + contexts + _EPSILON))).sum(dim=1)
    raise ValueError(scoring_function)


# ── pairwise Huber + contrastive sampling ────────────────────────────────────
def _fast_pairwise_huber(soft_margin, hard_margin, pos_scores, neg_scores, pos_weights):
    summations = torch.stack([
        torch.cumsum(pos_weights, 0),
        torch.cumsum(pos_weights * pos_scores, 0),
        torch.cumsum(pos_weights * pos_scores ** 2, 0),
    ], dim=1)
    summations = F.pad(summations, (0, 0, 1, 0), value=0.0)
    coefficients = [
        torch.stack([neg_scores + hard_margin, -torch.ones_like(neg_scores),
                     torch.zeros_like(neg_scores)]),
        torch.stack([1 / (2 * soft_margin) * (neg_scores + hard_margin + soft_margin / 2) ** 2,
                     -1 / soft_margin * (neg_scores + hard_margin + soft_margin / 2),
                     1 / (2 * soft_margin) * torch.ones_like(neg_scores)]),
    ]
    boundaries = [None, neg_scores - soft_margin / 2 + hard_margin,
                  neg_scores + soft_margin / 2 + hard_margin]

    def contributions(bounds, coeffs):
        if bounds is None:
            return 0.0
        bix = torch.searchsorted(pos_scores, bounds)
        return torch.einsum('ij,ji->i', summations[bix], coeffs)

    losses = torch.zeros_like(neg_scores)
    for ub, lb, coeffs in zip(boundaries, boundaries[1:], coefficients):
        losses = losses + contributions(lb, coeffs) - contributions(ub, coeffs)
    return losses


def _fast_pairwise_hinge(hard_margin, pos_scores, neg_scores, pos_weights):
    pos_scores = pos_scores - hard_margin
    ix = torch.searchsorted(pos_scores, neg_scores, side='left')
    pos_scores = torch.cat([torch.zeros(1, device=pos_scores.device, dtype=pos_scores.dtype), pos_scores])
    pos_weights = torch.cat([torch.zeros(1, device=pos_weights.device, dtype=pos_weights.dtype), pos_weights])
    return torch.cumsum(pos_weights, 0)[ix] * neg_scores - torch.cumsum(pos_scores * pos_weights, 0)[ix]


def pairwise_huber_loss(labels, scores, weights, soft_margin, hard_margin,
                        norm_push=1.0, lse_scale=0.0, dtype=torch.float64):
    labels, scores, weights = labels.to(dtype), scores.to(dtype), weights.to(dtype)
    pos_mask = labels > 0.5
    pos_scores, neg_scores = scores[pos_mask], scores[~pos_mask]
    pos_w_raw, neg_weights = weights[pos_mask], weights[~pos_mask]
    ix = torch.argsort(pos_scores)
    pos_scores, pos_w_raw = pos_scores[ix], pos_w_raw[ix]
    pos_w_sum, neg_w_sum = pos_w_raw.sum(), neg_weights.sum()
    if pos_w_sum == 0 or neg_w_sum == 0:
        return (scores * 0).sum()
    pos_weights, neg_weights = pos_w_raw / pos_w_sum, neg_weights / neg_w_sum
    if soft_margin > 0.0:
        losses = _fast_pairwise_huber(soft_margin, hard_margin, pos_scores, neg_scores, pos_weights)
    else:
        losses = _fast_pairwise_hinge(hard_margin, pos_scores, neg_scores, pos_weights)
    if lse_scale != 0.0:
        losses = lse_scale * losses + torch.log(neg_weights)
        loss = (1 / lse_scale) * torch.logsumexp(losses, 0)
        return loss if torch.isfinite(loss) else torch.tensor(0.0, device=scores.device, dtype=dtype)
    if norm_push != 1.0:
        losses = (losses + _EPSILON) ** norm_push
    loss = (losses * neg_weights).sum()
    if norm_push != 1.0:
        loss = (loss + _EPSILON) ** (1 / norm_push)
    return loss


def contrastive_labels_scores(queries, items, matching_items, scoring_function,
                              contrastive_scores_per_query, positive_instances_weight_factor,
                              query_compatibility_keys=None, item_compatibility_keys=None):
    """Synthetic-positive contrastive construction (FACADE training mechanism).

    NEGATIVE (label -1) = the true (action, its-own-context) pair (should score LOW
    under SF_OMDOT). POSITIVE (label +1) = action paired with a RANDOM other context
    (a synthetic anomaly, should score HIGH). Spurious positives (same account) are
    zero-weighted. The loss trains true pairs to score below random pairs.
    """
    if query_compatibility_keys is None:
        query_compatibility_keys = matching_items
        item_compatibility_keys = torch.arange(items.shape[0], device=items.device,
                                                dtype=matching_items.dtype)
    neg_scores = compute_scores(queries, items[matching_items], scoring_function)
    all_scores = [neg_scores]
    all_weights = [torch.ones_like(neg_scores)]
    all_labels = [-torch.ones_like(neg_scores)]
    n_items, n_queries = items.shape[0], queries.shape[0]
    base_ix = torch.arange(n_items, device=items.device).repeat((n_queries + n_items - 1) // n_items)
    for _ in range(contrastive_scores_per_query):
        ix = base_ix[torch.randperm(base_ix.shape[0], device=items.device)][:n_queries]
        all_scores.append(compute_scores(queries, items[ix], scoring_function))
        all_labels.append(torch.ones_like(neg_scores))
        allowed = (query_compatibility_keys != item_compatibility_keys[ix]).float()
        all_weights.append(allowed * positive_instances_weight_factor)
    return torch.cat(all_labels), torch.cat(all_scores), torch.cat(all_weights)


# ── two-tower model ───────────────────────────────────────────────────────────
class ConcatenateThenSNNTower(nn.Module):
    def __init__(self, segment_configs: List[dict], snn_layer_sizes: List[int],
                 embedding_dims: int, dropout_tokens: float, dropout_neurons: float,
                 transformations: List[int]):
        super().__init__()
        self.segment_names = []
        self.segment_embedders = nn.ModuleDict()
        for cfg in segment_configs:
            name = cfg['name']
            self.segment_names.append(name)
            self.segment_embedders[name] = SegmentEmbedder(
                embedding_dim=cfg['embed_dim'], weight_scaling=cfg.get('weight_scaling', WS_IDENTITY),
                weight_normalization=cfg.get('weight_normalization', WN_L2), dropout_rate=dropout_tokens)
        self.snn = SNN(snn_layer_sizes + [embedding_dims], dropout_neurons)
        self.transformation = EmbeddingsTransformation(transformations)

    def forward(self, segment_inputs: Dict[str, dict]) -> torch.Tensor:
        parts = []
        for name in self.segment_names:
            inp = segment_inputs[name]
            parts.append(self.segment_embedders[name](
                inp['embeddings'], inp.get('intensities'), inp.get('lengths')))
        return self.transformation(self.snn(torch.cat(parts, dim=-1)))
