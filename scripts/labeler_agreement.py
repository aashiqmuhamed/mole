"""Inter-labeler agreement for the transcript-fed ground-truth judge.

How robust is the LLM ground truth across judge models? For every cell that all
labelers judged, report pairwise % agreement + Cohen's kappa (chance-corrected)
on (a) the 4-way outcome and (b) the binary `executed` label. High agreement =>
the ground truth isn't an artifact of one model. Cross-family pairs are the real
test; within-family (e.g. gpt-5.1 vs gpt-5.3) is a weaker upper bound.

Usage:
  python scripts/labeler_agreement.py gpt-5.1=a.json gpt-5.3=b.json DeepSeek=c.json
"""
from __future__ import annotations

import collections
import itertools
import json
import sys


def _load(path: str) -> dict[str, str]:
    ev = json.load(open(path, encoding="utf-8"))["evidence"]
    return {k: v["outcome"] for k, v in ev.items()}


def _agree(a, b, keys, binary):
    f = (lambda x: x == "executed") if binary else (lambda x: x)
    A = [f(a[k]) for k in keys]
    B = [f(b[k]) for k in keys]
    n = len(keys)
    po = sum(x == y for x, y in zip(A, B)) / n
    ca, cb = collections.Counter(A), collections.Counter(B)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(ca) | set(cb))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return po, kappa


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    labs = {}
    for a in argv:
        name, path = a.split("=", 1)
        labs[name] = _load(path)
    common = sorted(set.intersection(*[set(d) for d in labs.values()]))
    print(f"labelers: {list(labs)}   common cells judged by all: {len(common)}")
    print(f"\n{'pair':26s} {'4-way outcome':>18} {'binary executed':>18}")
    print("-" * 64)
    for x, y in itertools.combinations(labs, 2):
        p4, k4 = _agree(labs[x], labs[y], common, binary=False)
        pb, kb = _agree(labs[x], labs[y], common, binary=True)
        print(f"{x + ' vs ' + y:26s} {p4:>7.0%} (k={k4:+.2f}) {pb:>7.0%} (k={kb:+.2f})")
    print("\nexecuted-rate per labeler (calibration, on common cells):")
    for name, d in labs.items():
        ex = sum(d[k] == "executed" for k in common)
        print(f"  {name:12s} executed={ex}/{len(common)} = {ex/len(common):.0%}")
    # biggest disagreement: cells where labelers split on binary executed
    split = [k for k in common
             if len({labs[n][k] == "executed" for n in labs}) > 1]
    print(f"\ncells where labelers DISAGREE on executed: {len(split)}/{len(common)} ({len(split)/len(common):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
