"""Offline replay of audit logs through monitors.

The Phase-2 evaluation substrate: load a persisted `audit.jsonl`, group it
into per-(account, day) episodes, and score each through a Monitor to get
a Suspicion ranking. Entirely offline — no sandbox, no LLM — so it runs over
the benign background-account corpus and the malicious threat traces the same way.

Library use:
    events = load_audit_jsonl(path)          # a local audit.jsonl
    events = load_audit_hf("gpt53_single_day")   # a published corpus from the Hub
    monitor = ZScoreAnomalyMonitor().fit_events(benign_events)
    ranking = await replay(monitor, events)          # [(key, Suspicion), ...]

CLI (sanity check that the audit log is monitorable):
    python -m mole.monitors.replay \
        --fit  /path/to/npc_audit.jsonl \
        --score /path/to/threat/audit.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Sequence

from ..audit.schema import AuditEvent
from .base import Monitor, Suspicion
from .classical.anomaly import ZScoreAnomalyMonitor

_AE_FIELDS = {f.name for f in dataclass_fields(AuditEvent)}


def load_audit_jsonl(path: str | Path) -> list[AuditEvent]:
    """Parse a persisted audit.jsonl back into AuditEvent objects."""
    out: list[AuditEvent] = []
    # split on "\n" only, NOT str.splitlines(): the latter also breaks on unicode line
    # separators ( / /\x85/\v/\f) that can appear literally inside a JSON string
    # value (e.g. an agent's message text), which would cut a line mid-token and fail parsing.
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(AuditEvent.from_row(d))
    return out


HF_DATASET = "forgelab/mole"


def load_audit_hf(split: str, dataset: str = HF_DATASET, **kwargs) -> list[AuditEvent]:
    """Load one published corpus from the Hub into AuditEvent objects.

        events = load_audit_hf("gpt53_single_day")

    `args` is a JSON string in Parquet rather than a struct, because its shape
    varies per action and Arrow rejects a mixed struct column. This decodes it
    so callers get the same AuditEvent objects as load_audit_jsonl.

    Splits: {gpt53, kimi_k26, qwen36_27b, deepseek_v4_flash} x {single_day, multiday}.
    Needs `pip install datasets`. Pass streaming=True for the larger corpora.
    """
    from datasets import load_dataset            # optional dependency

    ds = load_dataset(dataset, "audit", split=split, **kwargs)
    out: list[AuditEvent] = []
    for row in ds:
        d = dict(row)
        if isinstance(d.get("args"), str):
            d["args"] = json.loads(d["args"])
        out.append(AuditEvent.from_row(d))
    return out


def load_transcripts_hf(split: str, dataset: str = HF_DATASET, **kwargs) -> list[dict]:
    """Load one published transcript corpus from the Hub.

    Returns plain dicts in the same shape the generator writes, so they can be
    handed to the transcript readers in scripts/auc_eval.py. Decodes the
    per-message `tool_calls` JSON string back into a list.
    """
    from datasets import load_dataset            # optional dependency

    ds = load_dataset(dataset, "transcripts", split=split, **kwargs)
    out: list[dict] = []
    for row in ds:
        t = dict(row)
        msgs = []
        for m in (t.get("messages") or []):
            m = dict(m)
            if isinstance(m.get("tool_calls"), str):
                m["tool_calls"] = json.loads(m["tool_calls"])
            msgs.append({k: v for k, v in m.items() if v is not None})
        t["messages"] = msgs
        out.append(t)
    return out


def group_episodes(events: Sequence[AuditEvent]) -> dict[tuple[str, str], list[AuditEvent]]:
    """Group events into (account, UTC-date) episodes."""
    by_key: dict[tuple[str, str], list[AuditEvent]] = defaultdict(list)
    for e in events:
        day = (e.ts or "")[:10]
        by_key[(e.account, day)].append(e)
    return dict(by_key)


async def replay(
    monitor: Monitor,
    events: Sequence[AuditEvent],
    *,
    concurrency: int = 1,
) -> list[tuple[tuple[str, str], Suspicion]]:
    """Score each (account, day) episode; return sorted by suspicion desc.

    `concurrency` > 1 scores episodes via a bounded `asyncio.gather` — needed for
    level 2 LLM-judge monitors (one network call per episode), where serial
    scoring is the wall-clock bottleneck. level 0/1 monitors do no I/O, so the
    default (serial) is fine and deterministic for them.
    """
    episodes = list(group_episodes(events).items())
    if concurrency <= 1:
        results = [(key, await monitor.score_episode(evs)) for key, evs in episodes]
    else:
        sem = asyncio.Semaphore(concurrency)

        async def _one(key, evs):
            async with sem:
                return key, await monitor.score_episode(evs)

        results = list(await asyncio.gather(*[_one(k, e) for k, e in episodes]))
    results.sort(key=lambda t: t[1].score, reverse=True)
    return results


async def _main(fit_path: str, score_path: str, top: int) -> int:
    benign = load_audit_jsonl(fit_path)
    monitor = ZScoreAnomalyMonitor().fit_events(benign)
    target = load_audit_jsonl(score_path)
    ranking = await replay(monitor, target)
    print(f"monitor={monitor.name} observability_level={monitor.observability_level}  "
          f"fit on {len(benign)} events, scored {len(target)} events "
          f"({len(ranking)} episodes)")
    print("=== top suspicion (desc):")
    for (account, day), s in ranking[:top]:
        flag = " [MALICIOUS-labeled]" if any(
            e.is_malicious for e in group_episodes(target)[(account, day)]
        ) else ""
        print(f"  {s.score:6.2f}  {account:14} {day}  {s.rationale}{flag}")
    # Did a malicious-labeled episode top the ranking?
    mal = [(k, s) for k, s in ranking if any(
        e.is_malicious for e in group_episodes(target)[k])]
    if mal:
        top_rank = next(i for i, (k, _) in enumerate(ranking) if k == mal[0][0])
        print(f"=== top malicious episode ranks #{top_rank + 1} of {len(ranking)} "
              f"(score {mal[0][1].score:.2f})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="benign audit.jsonl to fit on")
    ap.add_argument("--score", required=True, help="audit.jsonl to score")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_main(args.fit, args.score, args.top)))
