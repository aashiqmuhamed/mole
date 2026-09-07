"""Failed-attempts audit — find tool calls that errored on a *missing resource*.

Reads an audit jsonl (the AuditEvent stream from a sim), filters
`exit_code != 0`, and separates "resource not found / 404 / does not exist"
failures from other (permission/validation) errors. It aggregates the
missing-resource failures by `resource_id` and `(service, action,
resource_id)`.

Why: a high-frequency missing-resource failure is the signature of a
persona/seed DRIFT bug — an insider persona references a target the seeder
never created, so the agent 404s and the attack is silently suppressed. A
one-off failure is usually agent-invented noise (a path the model guessed).

It also checks a WATCH-LIST of resources that were previously broken by seed
drift and are supposed to be fixed now (eval-data-2025Q1/Q2, fine-tune-corpus-v4,
the v17/v18 checkpoints, the C1 cover tickets, q1-interviews-v2, FIN-74). If any
reappears as a missing-resource failure, the seeding fix regressed.

Usage:
    python scripts/failed_attempts_audit.py --audit AUDIT.jsonl
    python scripts/failed_attempts_audit.py --audit AUDIT.jsonl --fail-threshold 3

Exit 0 if no watched target fails and no (service,action,resource) exceeds
--fail-threshold missing-resource failures; 1 otherwise.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

# Error substrings that indicate the TARGET RESOURCE doesn't exist (a seeding
# gap) — as opposed to a permission/validation/conflict error, which is a
# different class. Kept broad; the raw distinct-error report below shows reality
# so a missed pattern is visible.
_MISSING_RE = re.compile(
    r"not found|404|does not exist|no such|could not find|couldn'?t find|"
    r"unknown (?:project|file|checkpoint|issue|secret|user|path|dir|directory|repo)|"
    r"no (?:project|file|checkpoint|issue|secret|user|repo)\b|"
    r"nonexistent|doesn'?t exist|not exist|no .{0,20}named",
    re.IGNORECASE,
)

# Previously-broken targets (persona/seed drift) that the fix is supposed to
# have restored. Specific enough not to false-match. If any shows up as a
# missing-resource failure, the seeding regressed.
_WATCH = [
    "eval-data-2025Q1", "eval-data-2025Q2",
    "fine-tune-corpus-v4",
    "ckpt-llama-finetune-v17", "ckpt-llama-finetune-v18",
    "llama-finetune-v17", "llama-finetune-v18",
    "SRE-110", "DATA-89", "INFRA-301", "INFRA-204",
    "q1-interviews-v2",
    "FIN-74",
]


def _blob(e: dict) -> str:
    return f"{e.get('resource_id','')} {json.dumps(e.get('args',{}))} {e.get('error','')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True, help="path to the audit jsonl")
    ap.add_argument("--fail-threshold", type=int, default=3,
                    help="missing-resource failures on one (service,action,resource) "
                         "at/above this count = likely seeding gap")
    args = ap.parse_args()

    path = Path(args.audit)
    if not path.exists():
        print(f"audit file not found: {path}", file=sys.stderr)
        return 2
    events = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    total = len(events)
    errs = [e for e in events if e.get("exit_code", 0) != 0]
    missing = [e for e in errs if _MISSING_RE.search(e.get("error", "") or "")]
    other = [e for e in errs if e not in missing]

    # A resource that EVER had a successful op (exit_code==0) demonstrably
    # existed in the seeded world — so a later 404 on it is NOT a seeding gap,
    # it's a consume-then-retry (a successful delete already removed it), a
    # collision (a paired insider got there first), or a path-format quirk
    # (trailing slash). Only a resource that NEVER succeeds anywhere is a real
    # seeding gap. We track both the normalized resource_id set (exact) and a
    # lowercased blob of all successful events (for substring/watch checks).
    def _norm(r: str) -> str:
        return (r or "").rstrip("/").lower()
    ok_norm = {_norm(e.get("resource_id", "")) for e in events
               if e.get("exit_code", 0) == 0 and e.get("resource_id")}
    ok_text = " ".join(_blob(e) for e in events
                       if e.get("exit_code", 0) == 0).lower()

    def _existed(resource_id: str) -> bool:
        return _norm(resource_id) in ok_norm

    print(f"audit file:                {path}")
    print(f"total audit events:        {total}")
    print(f"  malicious(insider) events: {sum(1 for e in events if e.get('is_malicious'))}")
    print(f"errored (exit_code!=0):    {len(errs)}")
    print(f"  missing-resource:          {len(missing)}")
    print(f"  other (perm/validation):   {len(other)}")

    # The ground truth: every distinct error string among the errored events, so
    # a regex miss is visible rather than silently dropped.
    print("\n=== distinct error messages among ALL errored events (top 30) ===")
    emsgs = collections.Counter((e.get("error", "") or "(empty)")[:140] for e in errs)
    for msg, n in emsgs.most_common(30):
        print(f"  {n:4d}  {msg}")

    print("\n=== missing-resource failures by resource_id (top 40) ===")
    by_res = collections.Counter(
        (e.get("resource_id", "") or "(no resource_id)") for e in missing)
    for res, n in by_res.most_common(40):
        flag = "   <== >= threshold (SEEDING GAP?)" if n >= args.fail_threshold else ""
        print(f"  {n:4d}  {res}{flag}")

    print("\n=== missing-resource failures by (service, action, resource_id) (top 25) ===")
    agg = collections.Counter(
        (e.get("service", ""), e.get("action", ""), e.get("resource_id", ""))
        for e in missing)
    examples: dict[tuple, str] = {}
    for e in missing:
        k = (e.get("service", ""), e.get("action", ""), e.get("resource_id", ""))
        examples.setdefault(k, (e.get("error", "") or "")[:160])
    gaps = []
    for (svc, act, res), n in agg.most_common(25):
        gap = n >= args.fail_threshold
        if gap:
            gaps.append(((svc, act, res), n))
        print(f"  {n:4d}  {svc}.{act}  res={res!r}{'  <== GAP' if gap else ''}")
        print(f"        e.g.: {examples[(svc, act, res)]}")

    mal_missing = sum(1 for e in missing if e.get("is_malicious"))
    print(f"\nmissing-resource failures on malicious(insider) events: "
          f"{mal_missing}/{len(missing)}")

    print("\n=== WATCH-LIST (previously-broken targets) ===")
    hits = collections.Counter()
    for e in missing:
        blob = _blob(e).lower()
        for w in _WATCH:
            if w.lower() in blob:
                hits[w] += 1
    # A watched target is only REGRESSED if it 404s AND never had a successful
    # op anywhere (genuinely unseeded). If it appears in any exit_code==0 event,
    # the 404s are consume/collision/path artifacts and the seed is intact.
    regressed = {w: n for w, n in hits.items() if w.lower() not in ok_text}
    existed = {w: n for w, n in hits.items() if w.lower() in ok_text}
    for w in sorted(existed):
        print(f"  OK  {w}: had a successful op — {existed[w]} 404(s) are "
              f"consume/collision/path artifacts, not a seeding gap")
    for w in sorted(regressed):
        print(f"  REGRESSED  {w}: {regressed[w]} 404(s) and NEVER any successful op")
    if not hits:
        print("  clean — no watched target appears in any missing-resource failure.")

    # A (service,action,resource) gap is real only if that resource never
    # succeeded anywhere (else it existed and was consumed/collided/mis-pathed).
    real_gaps = [((svc, act, res), n) for (svc, act, res), n in gaps
                 if not _existed(res)]

    print("\n=== VERDICT ===")
    bad = bool(regressed) or bool(real_gaps)
    if regressed:
        print(f"  FAIL: {len(regressed)} watched target(s) regressed: {sorted(regressed)}")
    if real_gaps:
        print(f"  WARN: {len(real_gaps)} (service,action,resource) >= threshold AND "
              f"never-succeeded (possible NEW seeding gap):")
        for (svc, act, res), n in real_gaps:
            print(f"        {n:4d}  {svc}.{act}  {res}")
    if not bad:
        print("  PASS: no watched regressions, no genuinely-missing high-frequency resources.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
