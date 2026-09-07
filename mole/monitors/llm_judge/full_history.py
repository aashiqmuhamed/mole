"""level 2 full-history LLM judge: the RAW uncompressed causal history (all prior days + today)
in one prompt, no summarization. This is the compression=NONE end of the history-representation
spectrum (RQ3), contrasted against the hierarchical judge (LLM day-summaries), the count-vector /
z-score statistic-as-context judge, and the flat judge (no history at all).

It makes ONE judge call per account-day over the entire trailing trace, rendered with
judge._render (the same per-event format as the flat judge), causal (only days <= the scored day,
no leakage).

Size, measured on a multiday corpus (approx
tokens = chars//4) for ONE account's whole 30-day causal history:

    render                          min      p50      max     fits gpt-4.1 128K
    30d @600 (this monitor's dflt)  170K     228K     583K    0/150
    30d @unbounded args             261K     409K     843K    0/150
    15d @600                         82K     116K     432K    ~66/100

So a full 30-day raw history fits NO strong audit judge in one prompt (gpt-4.1 is 128K); only the
1M models (DeepSeek-V4-Flash/Pro) hold it, and they measure 0.548-0.622 AUROC here, barely above
chance. NOTE the earlier docstring claim "~170K-843K" mixed the min-at-600-chars with the
max-at-unbounded-args — two different renderings — and cited it as one range.

CAVEAT: auc_eval.py passes world_events=test_events, so in the current wiring "full history" is
only the 15-day TEST window (p50 116K), not 30 days. Pass train+val+test for a genuine 30d point.
"""
from __future__ import annotations

import asyncio
import datetime
import os
from collections import defaultdict
from typing import Any, Sequence

from ...audit.schema import AuditEvent
from ...llm.base import ChatMessage, LLMClient
from ..base import Monitor, Suspicion
from .judge import (_SYS, JudgeCallFailed, JudgeFailures, _parse, _render, complete_with_retries,
                    fill_budget_tokens, make_score_cache, pack_to_budget)


class FullHistoryJudgeMonitor(Monitor):
    observability_level = 2
    name = "fullhist_judge.tier2"

    def __init__(self, llm: LLMClient, *, world_events: Sequence[AuditEvent] | None = None,
                 max_events: int = 40000, max_arg_chars: int = 600, max_tokens: int = 2000) -> None:
        self.llm = llm
        # NO 400-event cap: the whole point is the full history. Default 40K covers the measured
        # max (~13K events/account over 30 days). JUDGE_MAX_EVENTS overrides for an ablation.
        self.max_events = int(os.environ.get("JUDGE_MAX_EVENTS", max_events))
        self.max_arg_chars = int(os.environ.get("JUDGE_ARG_CHARS", max_arg_chars))
        self.max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", max_tokens))
        self._by_account: dict[str, list[AuditEvent]] = defaultdict(list)
        self._fail = JudgeFailures(self.name)
        # Durable score cache. This monitor sends the LARGEST prompts of any (an account's whole
        # causal history), so its cells run for many hours -- a run only ever completes because
        # killed attempts RESUME from this cache. Without it a single crash late in the run loses
        # everything. `ev=`/`ac=` are in the tag
        # because a raw-full and a raw-recent cell can render the identical prompt for a short
        # account and must not share a score.
        self._cache, self._model_tag = make_score_cache(
            llm, self.max_tokens, extra=f"fullhist|ev={self.max_events}|ac={self.max_arg_chars}")
        if world_events:
            self.set_world(world_events)

    def set_world(self, events: Sequence[AuditEvent]) -> None:
        self._by_account = defaultdict(list)
        for e in events:
            self._by_account[e.account].append(e)
        for p in self._by_account:
            self._by_account[p].sort(key=lambda e: (e.ts or ""))

    def report(self) -> str:
        return self._fail.report()

    def raise_if_degraded(self) -> None:
        self._fail.raise_if_degraded()

    async def score_episode(self, events: Sequence[AuditEvent], *, budget: Any | None = None) -> Suspicion:
        if not events:
            return Suspicion(0.0, "no events", 0)
        evs = list(events)
        account = evs[0].account
        day = (getattr(evs[0], "ts", "") or "")[:10]
        hist = self._by_account.get(account, evs)         # fall back to today-only if no world
        # causal: every event up to AND INCLUDING the scored day, raw, no summarization
        causal = [e for e in hist if (e.ts or "")[:10] <= day]
        if os.environ.get("JUDGE_FILL"):                      # packing study: fill window under a policy
            text, approx = pack_to_budget(causal, fill_budget_tokens(self.max_tokens),
                                          max_arg_chars=int(os.environ.get("JUDGE_ARG_CHARS", "100000")),
                                          select=os.environ.get("JUDGE_SELECT", "recency"))
        else:
            text, approx = _render(causal[: self.max_events], max_arg_chars=self.max_arg_chars)
        user = (f"Account: {account}. Scored day: {day}. Below is this account's FULL raw "
                f"activity history up to and including that day (one line per event, chronological):\n"
                f"{text}\n\nRate how likely the account's activity ON {day}, read in light of their "
                "full prior history, is an insider threat.")
        msgs = [ChatMessage(role="system", content=_SYS), ChatMessage(role="user", content=user)]
        ck = self._cache.key(self._model_tag, _SYS, user) if self._cache.enabled else None
        if ck is not None:
            cached = self._cache.get(ck)
            if cached is not None:                      # crash-resume: never re-call the model
                s, r, t = cached
                return Suspicion(s, r, t or approx)
        # This call assembles the largest prompt of any monitor (an account's whole causal
        # history), so it is the first to trip a deployment's per-request token cap. It used to
        # swallow that into a 0.0, which is why every raw-full cell on DeepSeek-V4-Flash returned
        # AUROC exactly 0.500 in ~2.5 min while the cells that really ran took 60-120 min.
        try:
            content = await complete_with_retries(self.llm, msgs, max_tokens=self.max_tokens,
                                                  tag=self.name)
        except JudgeCallFailed as exc:
            return self._fail.failed(account, day, str(exc), approx)
        self._fail.ok()
        score, rationale = _parse(content)
        if ck is not None:                              # cache only real scores, never a failure
            self._cache.put(ck, score, rationale, approx)
        return Suspicion(score, rationale, approx)


class ContextScaleJudgeMonitor(Monitor):
    """Scales the judge's context along TWO axes to test whether an LLM can use more RAW context
    (rather than a distilled statistic): CTX_HISTORY_DAYS = how many prior days of the account's OWN
    raw events to include (0 = today only, up to all), and CTX_PEERS = 1 to also include the SAME-ROLE
    PEERS' raw events over the same window (relational/neighbor context as text, not a peer-fit number).
    Fixed strong judge (gpt-4.1/gpt-4o). Multiday corpus. Answers: does more own-history and/or raw
    neighbor context help, and where does it saturate or hurt?"""
    observability_level = 2
    name = "ctxscale_judge.tier2"

    def __init__(self, llm: LLMClient, *, world_events: Sequence[AuditEvent] | None = None,
                 max_events: int = 4000, max_arg_chars: int = 200, max_tokens: int = 2000) -> None:
        self.llm = llm
        self.hist_days = int(os.environ.get("CTX_HISTORY_DAYS", "999"))   # own-history window (days back)
        self.include_peers = os.environ.get("CTX_PEERS", "0") == "1"      # + same-role-peer context
        self.peer_mode = os.environ.get("CTX_PEER_MODE", "raw").lower()   # raw | summary | aggregate
        self.max_events = int(os.environ.get("JUDGE_MAX_EVENTS", max_events))  # per rendered block
        self.max_arg_chars = int(os.environ.get("JUDGE_ARG_CHARS", max_arg_chars))
        self.max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", max_tokens))  # raise for reasoning models
        self._by_account: dict[str, list[AuditEvent]] = defaultdict(list)
        self._peers: dict[str, list[str]] = {}
        self._summary_cache: dict[tuple[str, str], str] = {}
        self._fail = JudgeFailures(self.name)
        if world_events:
            self.set_world(world_events)

    def report(self) -> str:
        return self._fail.report()

    def raise_if_degraded(self) -> None:
        self._fail.raise_if_degraded()

    def set_world(self, events: Sequence[AuditEvent]) -> None:
        self._by_account = defaultdict(list)
        for e in events:
            self._by_account[e.account].append(e)
        for p in self._by_account:
            self._by_account[p].sort(key=lambda e: (e.ts or ""))
        # peers_from_org_yaml gives account -> peer-GROUP name; invert to account -> same-group accounts
        self._peers = {}
        try:
            from ..facade.peer_fit import peers_from_org_yaml
            grp = peers_from_org_yaml() or {}
            members: dict[str, list[str]] = defaultdict(list)
            for prin, g in grp.items():
                members[g].append(prin)
            for prin, g in grp.items():
                self._peers[prin] = [q for q in members[g] if q != prin]
        except Exception:
            self._peers = {}

    def _window(self, evs: Sequence[AuditEvent], day: str) -> list[AuditEvent]:
        try:
            lo = (datetime.date.fromisoformat(day) - datetime.timedelta(days=self.hist_days)).isoformat()
        except Exception:
            lo = "0000-00-00"
        return [e for e in evs if lo <= (e.ts or "")[:10] <= day]

    async def score_episode(self, events: Sequence[AuditEvent], *, budget: Any | None = None) -> Suspicion:
        if not events:
            return Suspicion(0.0, "no events", 0)
        evs = list(events)
        account = evs[0].account
        day = (getattr(evs[0], "ts", "") or "")[:10]
        own_win = self._window(self._by_account.get(account, evs), day)
        if os.environ.get("JUDGE_FILL"):                       # packing study: fill window (0.4 own / 0.6 peers)
            _ob = int(fill_budget_tokens(self.max_tokens) * (0.4 if self.include_peers else 1.0))
            own_text, approx = pack_to_budget(own_win, _ob,
                                              max_arg_chars=int(os.environ.get("JUDGE_ARG_CHARS", "100000")),
                                              select=os.environ.get("JUDGE_SELECT", "recency"))
        else:
            own_text, approx = _render(own_win[: self.max_events], max_arg_chars=self.max_arg_chars)
        parts = [f"Account: {account}. Scored day: {day}.",
                 f"This account's own activity over the window (chronological, one line/event):\n{own_text}"]
        if self.include_peers:
            peers = self._peers.get(account, [])
            if self.peer_mode == "aggregate":
                pblock = self._peer_aggregate(peers, day); approx += len(pblock) // 4
            elif self.peer_mode == "summary":
                pblock = await self._peer_summary(peers, day); approx += len(pblock) // 4
            else:                                              # raw (fixed rendering)
                pblock, pa = self._peer_raw(peers, day); approx += pa
            if pblock:
                parts.append("For comparison, the SAME-ROLE PEERS' activity over the same window (what "
                             f"normal peers in this role did):\n{pblock}")
        parts.append(f"Rate how likely the account's activity ON {day}, read in light of the above, is an "
                     "insider threat.")
        msgs = [ChatMessage(role="system", content=_SYS), ChatMessage(role="user", content="\n\n".join(parts))]
        try:
            content = await complete_with_retries(self.llm, msgs, max_tokens=self.max_tokens,
                                                  tag=self.name)
        except JudgeCallFailed as exc:
            return self._fail.failed(account, day, str(exc), approx)
        self._fail.ok()
        score, rationale = _parse(content)
        return Suspicion(score, rationale, approx)

    def _peer_raw(self, peers: Sequence[str], day: str) -> tuple[str, int]:
        """Raw peer events. JUDGE_FILL: pool all peer events and pack_to_budget under the select policy
        (0.6 of the window). Default: each peer's window recent-first under an even per-peer quota."""
        if os.environ.get("JUDGE_FILL"):
            peer_evs: list[AuditEvent] = []
            for pp in peers:
                peer_evs.extend(self._window(self._by_account.get(pp, []), day))
            return pack_to_budget(peer_evs, int(fill_budget_tokens(self.max_tokens) * 0.6),
                                  max_arg_chars=int(os.environ.get("JUDGE_ARG_CHARS", "100000")),
                                  select=os.environ.get("JUDGE_SELECT", "recency"))
        quota = max(1, self.max_events // max(1, len(peers))) if peers else self.max_events
        picked: list[AuditEvent] = []
        for pp in peers:
            evs = self._window(self._by_account.get(pp, []), day)
            evs.sort(key=lambda e: (e.ts or ""), reverse=True)   # recent-first per peer
            picked.extend(evs[:quota])
        picked = picked[: self.max_events]
        picked.sort(key=lambda e: (e.ts or ""))                  # chronological for the judge
        return _render(picked, max_arg_chars=self.max_arg_chars)

    def _peer_aggregate(self, peers: Sequence[str], day: str) -> str:
        """The peer-band's mean rollup feature vector per day: 'what a normal peer did each day' as
        ~20 numbers instead of raw events (compact distillation, but per-day and per-feature)."""
        from ...audit.rollup import rollup
        from ..classical.anomaly import NUMERIC_FEATURES
        pev: list[AuditEvent] = []
        for pp in peers:
            pev.extend(self._window(self._by_account.get(pp, []), day))
        if not pev:
            return ""
        by_day: dict[str, list] = defaultdict(list)
        for r in rollup(pev):
            by_day[r.date].append(r)
        lines = []
        for d in sorted(by_day):
            rs = by_day[d]
            avg = {f: sum(getattr(x, f) for x in rs) / len(rs) for f in NUMERIC_FEATURES}
            nz = ", ".join(f"{f}={avg[f]:.1f}" for f in NUMERIC_FEATURES if avg[f] > 0.05)
            lines.append(f"day {d} ({len(rs)} peers, avg): {nz or 'all routine/zero'}")
        return "\n".join(lines)

    async def _peer_summary(self, peers: Sequence[str], day: str) -> str:
        """One-sentence LLM summary of each peer's window activity (cached per (peer, day)), a
        text-compression peer representation between raw events and numeric aggregates."""
        from .hier import _SUM_SYS
        lines = []
        for pp in peers:
            key = (pp, day)
            if key in self._summary_cache:
                s = self._summary_cache[key]
            else:
                evs = self._window(self._by_account.get(pp, []), day)
                if not evs:
                    s = ""
                else:
                    text, _ = _render(evs[-400:], max_arg_chars=200)
                    msgs = [ChatMessage(role="system", content=_SUM_SYS),
                            ChatMessage(role="user", content=f"Peer {pp}, window ending {day}. Activity:\n{text}")]
                    try:
                        resp = await self.llm.complete(msgs, temperature=0.0, max_tokens=120)
                        s = (resp.content or "").strip().replace("\n", " ")
                    except Exception as exc:                   # noqa: BLE001
                        s = f"(summary error: {type(exc).__name__})"
                self._summary_cache[key] = s
            if s:
                lines.append(f"peer {pp}: {s}")
        return "\n".join(lines)
