"""level 2 two-stage judge: a STRONG model extracts a per-account statistic, a WEAK model judges.

Two questions RQ3 leaves open, both of which our hand-built statistics quietly beg.

1. Our z-score / peer-fit / count-vector search a fixed space of TWENTY hand-picked count features
   (`classical/anomaly.NUMERIC_FEATURES`: logon/logoff counts, email/file/gitlab/chat counts,
   after_hours_rate, secret_read_count, ...). There is no timing, no inter-event interval, no
   resource rarity, no co-access structure, no sequence. So "count-vector 0.736" is not a ceiling,
   it is the best of twenty things we thought of. A model given the raw history could in principle
   derive a BETTER statistic than the ones we hand-wrote.

2. Extraction and judging are different jobs and may want different models. The distilled statistic
   is what carries RQ3 (z-score 0.810 vs raw text 0.658), and a statistic is per-ACCOUNT, so it
   amortises across every day that account appears: one strong extraction serves ~15 test days.
   If a strong extractor can lift a weak judge above what that weak judge does alone, capability
   buys more when spent on representation than on verdicts -- a different point on the cost
   frontier than `cascade` (which spends a strong judge on a SUBSET of days).

Design:
  Stage 1 (once per account, cached): the EXTRACTOR reads that account's benign-window history
          and emits a compact JSON profile of numeric norms it considers discriminative. It is NOT
          given our feature list -- inventing the features is the point.
  Stage 2 (once per account-day): the JUDGE sees today's events plus that profile, and rates
          today against it. Reuses judge._SYS / _render / _parse unchanged.

Set STAT_EXTRACT_MODEL to the strong extractor; the weak judge uses the active backend's model.
With both the same, this is an ablation of the two-stage design itself.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Sequence

from ...audit.schema import AuditEvent
from ...llm.base import ChatMessage, LLMClient
from ..base import Monitor, Suspicion
from .judge import (_SYS, JudgeCallFailed, JudgeFailures, _parse, _render,
                    complete_with_retries, make_score_cache)

# Deliberately does NOT enumerate our 20 features: if we name them, we are back to hand-built
# statistics with an LLM as an expensive calculator, and the discovery question goes untested.
_EXTRACT_SYS = (
    "You are a security data analyst building a behavioural baseline for ONE enterprise account, "
    "from their normal activity log. Your output will be handed to a downstream reviewer who sees "
    "only ONE day of this account's activity and must decide whether that day is an insider "
    "threat. The reviewer never sees this history, so your profile is their only baseline.\n\n"
    "Invent whatever quantitative summary is most DISCRIMINATIVE: you are not restricted to simple "
    "counts. Consider rates, ratios, timing and inter-arrival structure, which resources are "
    "routine vs rare for this person, co-occurrence, sequence, volume distributions, anything you "
    "can justify from the log. Be specific and numeric; a reviewer cannot act on vague prose.\n\n"
    'Output ONLY compact JSON: {"profile": {<your named statistics and their normal values/ranges>}, '
    '"tells": ["<what a deviation worth escalating would look like, concretely>", ...]}'
)

# STAT_EXTRACT_ROBUST=1: ask for GENERALIZABLE rules, not a memorised empirical envelope. The
# default prompt ("be specific and numeric") makes opus report ~12 days of near-deterministic background-account
# behaviour as tight literals (e.g. a 06:53-07:56 window) AND confabulate precise constants it
# never observed (it wrote weight_bytes "always 16384" when the train bytes ranged 2..269230).
# Both overfit: the literals would false-positive on real human variance, and the confabulations
# are simply wrong. This variant asks for role-relative, variance-tolerant, structural rules that
# would hold for ANY competent member of the role -- the honest test of whether the two-stage lift
# survives when the baseline is forbidden from memorising this corpus's quirks.
_EXTRACT_SYS_ROBUST = (
    "You are a security analyst writing a GENERALIZABLE behavioural baseline for one enterprise "
    "account's ROLE, from a short sample of their normal activity. A downstream reviewer sees "
    "only ONE day and must judge whether it is an insider threat, using only your baseline.\n\n"
    "CRITICAL: your sample is small (about two weeks) and the person is one instance of a role. Do "
    "NOT memorise this sample. Do NOT report exact literals you happened to observe (a precise "
    "clock window, an exact byte count, a specific file list) -- those are noise that will "
    "false-positive on normal day-to-day variation, and if you state a precise value you did not "
    "clearly observe you are guessing, which is worse than saying nothing.\n\n"
    "Instead write rules that would hold for ANY competent member of this role and that tolerate "
    "natural variation:\n"
    "  - RELATIVE, not absolute: 'writes stay within their own project namespace', not 'writes to "
    "/work/alice/x'; 'active in a consistent daily window', not '06:53-07:56'.\n"
    "  - STRUCTURAL: which services/actions are in-role vs out-of-role; what a normal task shape "
    "looks like; what would be categorically new (a resource class never touched).\n"
    "  - Give ranges as rough bands with tolerance ('roughly N, flag only a multiple of N'), and "
    "only where the sample clearly supports them.\n"
    "Prefer a rule you are confident generalises over a precise number you are not.\n\n"
    'Output ONLY compact JSON: {"profile": {<role-relative, variance-tolerant rules>}, '
    '"tells": ["<a deviation that would be anomalous for ANYONE in this role>", ...]}'
)


# STAT_EXTRACT_CONTRAST=1: the robust generalization discipline PLUS an explicit CONTRASTIVE threat model.
# The robust/default prompts describe NORMAL behaviour only; the downstream judge must then infer, unaided, what
# misuse looks like. This variant additionally asks the strong extractor to reason -- ONCE per account, offline --
# about how a malicious insider IN THIS ROLE would abuse its specific legitimate access, and how that single-day
# misuse would differ from the normal baseline. It hands the weak judge role-general THREAT TELLS, not just a
# baseline, targeting the observable-but-demoted days (e.g. sensitive-read + external-egress) the rarity fusion
# ranks at 12-22. Kept ROLE-GENERAL (not this-sample-specific) so it does not overfit single_K4's attack style and
# still transfers to a different generator. Advisor-flagged open lever (the "facts-only/contrastive extraction card").
_EXTRACT_SYS_CONTRAST = (
    "You are a security analyst writing a behavioural baseline for one enterprise account's ROLE, from a short "
    "sample of their normal activity. A downstream reviewer sees only ONE day of this account's activity and must "
    "judge whether it is an insider threat, using only what you write.\n\n"
    "Write TWO parts.\n"
    "1) NORMAL baseline -- GENERALIZABLE and role-relative. Your sample is small (~two weeks) and this person is one "
    "instance of a role, so do NOT memorise it: no exact clock windows, byte counts, or specific file names (those "
    "false-positive on normal variation). State which services/actions are in-role, what a normal task shape looks "
    "like, and rough bands with tolerance, only where the sample clearly supports them.\n"
    "2) CONTRASTIVE threat tells -- given THIS role's legitimate reach (the sensitive data/systems it can touch), how "
    "would a malicious insider in this role abuse that access, and how would the abuse SURFACE on a single day as "
    "different from the normal baseline? Focus on misuse that is technically in-role yet serves exfiltration, "
    "tampering, or fraud (e.g. touching a normally-read resource in a writing/exporting way, pairing a sensitive read "
    "with an external send, editing a record that should be settled). Keep every tell ROLE-GENERAL -- it must hold for "
    "ANY competent member of this role and in a DIFFERENT company, never a detail peculiar to this sample.\n\n"
    'Output ONLY compact JSON: {"profile": {<role-relative normal baseline>}, '
    '"threat_tells": ["<a role-general sign that ONE day is misuse, stated as a contrast with normal>", ...]}'
)


def _extract_sys() -> str:
    if os.environ.get("STAT_EXTRACT_CONTRAST") == "1":
        return _EXTRACT_SYS_CONTRAST
    return _EXTRACT_SYS_ROBUST if os.environ.get("STAT_EXTRACT_ROBUST") == "1" else _EXTRACT_SYS


class StatExtractJudgeMonitor(Monitor):
    observability_level = 2
    name = "stat_extract_judge.tier2"

    def __init__(self, llm: LLMClient, *, extractor: LLMClient | None = None,
                 world_events: Sequence[AuditEvent] | None = None,
                 profile_days: int = 12, max_events: int = 400, max_arg_chars: int = 300,
                 max_tokens: int = 2000, extract_max_tokens: int = 900) -> None:
        self.llm = llm                                  # the (possibly weak) JUDGE
        self.extractor = extractor or llm               # the (possibly strong) EXTRACTOR
        self.profile_days = int(os.environ.get("STAT_PROFILE_DAYS", profile_days))
        self.max_events = int(os.environ.get("JUDGE_MAX_EVENTS", max_events))
        self.max_arg_chars = int(os.environ.get("JUDGE_ARG_CHARS", max_arg_chars))
        self.max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", max_tokens))
        self.extract_max_tokens = int(os.environ.get("STAT_EXTRACT_MAX_TOKENS", extract_max_tokens))
        self._profile: dict[str, str] = {}              # account -> profile JSON (amortised)
        # Durable cache for the EXTRACTOR's per-account baselines. Extraction is the slow,
        # front-loaded phase (one call per account), and replay scores
        # nothing until it finishes, so a wedge mid-extraction loses hours. Keyed on the extractor
        # model + a hash of its exact prompt, so a re-run resumes and a config change invalidates.
        self._ex_cache, self._ex_tag = make_score_cache(
            self.extractor, self.extract_max_tokens,
            extra=f"statx-extract|pd={self.profile_days}|ac={self.max_arg_chars}"
                  + ("|contrast" if os.environ.get("STAT_EXTRACT_CONTRAST") == "1"
                     else "|robust" if os.environ.get("STAT_EXTRACT_ROBUST") == "1" else ""))
        self._by_account: dict[str, list[AuditEvent]] = defaultdict(list)
        self._fail = JudgeFailures(self.name)
        self.n_extract = 0
        if world_events:
            self.set_world(world_events)

    def set_world(self, events: Sequence[AuditEvent]) -> None:
        self._by_account = defaultdict(list)
        for e in events:
            self._by_account[e.account].append(e)
        for p in self._by_account:
            self._by_account[p].sort(key=lambda e: (e.ts or ""))

    def report(self) -> str:
        return (f"{self._fail.report()}\n[{self.name}] extractions={self.n_extract} "
                f"(one per account, amortised over that account's days)")

    def raise_if_degraded(self) -> None:
        self._fail.raise_if_degraded()

    def fit_events(self, train_events: Sequence[AuditEvent]) -> "StatExtractJudgeMonitor":
        """The profile is built from the BENIGN TRAIN slice, never from test.

        Same discipline as the classical monitors: the baseline is fit on train, so nothing about
        the scored day leaks into its own baseline.
        """
        self._train: dict[str, list[AuditEvent]] = defaultdict(list)
        for e in train_events:
            self._train[e.account].append(e)
        for p in self._train:
            self._train[p].sort(key=lambda e: (e.ts or ""))
        return self

    async def _profile_for(self, account: str) -> str:
        if account in self._profile:
            return self._profile[account]
        hist = getattr(self, "_train", {}).get(account, [])
        if not hist:
            self._profile[account] = "(no benign history for this account)"
            return self._profile[account]
        days = sorted({(e.ts or "")[:10] for e in hist})[-self.profile_days:]
        evs = [e for e in hist if (e.ts or "")[:10] in days]
        _hist_ev = int(os.environ.get("STAT_EXTRACT_HIST_EVENTS", "4000"))
        text, _ = _render(evs[-_hist_ev:], max_arg_chars=self.max_arg_chars)
        user = (f"Account: {account}. Their normal activity over {len(days)} days "
                f"({len(evs)} events), one line per event:\n{text}")
        sys_p = _extract_sys()
        ck = self._ex_cache.key(self._ex_tag, sys_p, user) if self._ex_cache.enabled else None
        if ck is not None:                              # resume: profile stored as the rationale field
            hit = self._ex_cache.get(ck)
            if hit is not None:
                if os.environ.get("STAT_DEBUG"): print(f"[EXDBG] extract HIT {account}", flush=True)
                self._profile[account] = hit[1]
                return hit[1]
        msgs = [ChatMessage(role="system", content=sys_p),
                ChatMessage(role="user", content=user)]
        try:
            out = await complete_with_retries(self.extractor, msgs,
                                              max_tokens=self.extract_max_tokens,
                                              tag=f"{self.name}:extract")
        except JudgeCallFailed as exc:
            # Never silently fall back to "no baseline": that would quietly turn this into the
            # flat judge and the cell would look like a weak-judge result instead of a broken one.
            self._profile[account] = f"(EXTRACTION FAILED: {exc})"
            return self._profile[account]
        self.n_extract += 1
        if os.environ.get("STAT_DEBUG"): print(f"[EXDBG] extract MISS+ran {account} n_extract={self.n_extract}", flush=True)
        self._profile[account] = out.strip()
        if ck is not None:                              # store the profile as the cached rationale
            self._ex_cache.put(ck, 0.0, self._profile[account], 0)
        return self._profile[account]

    async def score_episode(self, events: Sequence[AuditEvent], *, budget: Any | None = None) -> Suspicion:
        if not events:
            return Suspicion(0.0, "no events", 0)
        evs = list(events)
        account = evs[0].account
        day = (getattr(evs[0], "ts", "") or "")[:10]
        profile = await self._profile_for(account)
        if profile.startswith("(EXTRACTION FAILED"):
            return self._fail.failed(account, day, profile, 0)
        today, approx = _render(evs[: self.max_events], max_arg_chars=self.max_arg_chars)
        approx += len(profile) // 4
        user = (f"Account: {account}. Scored day: {day}.\n\n"
                f"BEHAVIOURAL BASELINE for this account, derived from their normal activity "
                f"(you do NOT see the raw history; this profile is your only baseline):\n{profile}\n\n"
                f"TODAY ({day}) — this account's activity, one line per event:\n{today}\n\n"
                "Rate how likely TODAY's activity is an insider threat, judged against the baseline "
                "above.")
        msgs = [ChatMessage(role="system", content=_SYS), ChatMessage(role="user", content=user)]
        try:
            content = await complete_with_retries(self.llm, msgs, max_tokens=self.max_tokens,
                                                  tag=self.name)
        except JudgeCallFailed as exc:
            return self._fail.failed(account, day, str(exc), approx)
        self._fail.ok()
        # STAT_DEBUG=1 emits a live judgment counter. This monitor does NOT cache judge scores (only
        # extractions), so its progress is otherwise invisible -- no cache growth, no per-day log -- and
        # a healthy long run looks stalled. EXDBG lines above tag extraction cache hit/miss similarly.
        if os.environ.get("STAT_DEBUG"): print(f"[JDBG] judged {account} {day} total_ok={self._fail.n_ok}", flush=True)
        score, rationale = _parse(content)
        return Suspicion(score, rationale, approx)
