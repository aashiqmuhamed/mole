"""FACADE v2 — faithful in-bench replication of Google FACADE (arXiv 2412.06700).

Unlike the simplified `monitors/facade/snn.py` (unweighted token sets, static org
peers, sigmoid-dot, max-pool), this package faithfully reproduces FACADE:
  - intensity-weighted, L2-normalized segment reductions (+ learnable offset token)
  - two-tower SNN MLP -> softplus -> L2-normalize -> SF_OMDOT (= 1 - cosine)
  - benign-only contrastive training with SYNTHETIC POSITIVES (random within-batch)
  - pairwise-Huber loss
  - temporally-decayed resource-history action features + bipartite peer context
  - clustering aggregator g (paper Sec 6.2.4): hierarchical-cluster -> sum of cluster maxes

The validated math (`_facade_core.py`) is a standalone port of the original FACADE model
(a PyTorch port verified ~1e-16 vs Google's TF original). The data adapter
(`featurizer.py`) and monitor wiring (`monitor.py`) are new — they bridge our
`AuditEvent`s to FACADE's `segment_inputs` tensors.
"""
from .monitor import FacadeV2Monitor, train_facade_v2  # noqa: F401
