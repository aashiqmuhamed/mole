"""level 2 hierarchical (map-reduce) LLM judge.

Closes two gaps in the flat per-(account,day) judge, both measured on ar05:
  * arg-body truncation — the flat judge shears every arg to 600 chars, but ~87%
    of malicious p-days carry a >600-char payload (median 2.5K) where
    content-manipulation malice lives (a flipped safety flag, a poisoned config/
    file body). Here the SCORED day is rendered with generous arg bodies.
  * memorylessness — the flat judge sees one day in isolation. Here the account's
    PRIOR days are rolled up into cheap one-line summaries so today is judged IN
    CONTEXT (slow cross-day campaigns).

Map:    summarize each prior day (one cheap call, cached per (account,day)).
Reduce: one judge call over {prior-day summaries} + {today, full detail}.
Causal: only days strictly before the scored day feed the context (no leakage).

The monitor holds a `world` (all scorable events); `replay` still calls
`score_episode(day_events)` once per (account,day), and the monitor looks up
that account's trailing history from the world to build the context.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from ...audit.schema import AuditEvent
from ...llm.base import ChatMessage, LLMClient
from ..base import Monitor, Suspicion
from .judge import (_SYS, JudgeCallFailed, JudgeFailures, _parse, _render,
                    complete_with_retries)

_SUM_SYS = (
    "You compress one employee's single day of enterprise activity into a one-sentence "
    "factual summary for a later security review. Flag any data egress, external sends, "
    "credential/permission changes, file/config writes, or unusual access; otherwise say "
    "it was routine. Be terse and neutral. Output only the sentence."
)


class HierJudgeMonitor(Monitor):
    observability_level = 2
    name = "hier_judge.tier2"

    def __init__(self, llm: LLMClient, *, world_events: Sequence[AuditEvent] | None = None,
                 window_days: int = 7, today_arg_chars: int = 4000,
                 max_events: int = 400, max_tokens: int = 2000) -> None:
        self.llm = llm
        self.window_days = window_days
        self.today_arg_chars = today_arg_chars
        self.max_events = max_events
        self.max_tokens = max_tokens
        self._summary_cache: dict[tuple[str, str], str] = {}
        self._by_account: dict[str, list[AuditEvent]] = defaultdict(list)
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

    async def _summarize_day(self, account: str, day: str, evs: Sequence[AuditEvent]) -> str:
        key = (account, day)
        if key in self._summary_cache:                      # amortize across the account's episodes
            return self._summary_cache[key]
        text, _ = _render(list(evs)[: self.max_events], max_arg_chars=300)
        msgs = [ChatMessage(role="system", content=_SUM_SYS),
                ChatMessage(role="user", content=f"Date {day}. Activity:\n{text}")]
        try:
            resp = await self.llm.complete(msgs, temperature=0.0, max_tokens=120)
            s = (resp.content or "").strip().replace("\n", " ")
        except Exception as exc:                            # noqa: BLE001
            s = f"(summary error: {type(exc).__name__})"
        self._summary_cache[key] = s
        return s

    async def score_episode(self, events: Sequence[AuditEvent], *, budget: Any | None = None) -> Suspicion:
        if not events:
            return Suspicion(0.0, "no events", 0)
        evs = list(events)
        account = evs[0].account
        day = (getattr(evs[0], "ts", "") or "")[:10]
        hist = self._by_account.get(account, evs)       # fall back to today-only if no world
        by_day: dict[str, list[AuditEvent]] = defaultdict(list)
        for e in hist:
            d = (e.ts or "")[:10]
            if d and d < day:
                by_day[d].append(e)
        prior_days = sorted(by_day)[-self.window_days:]
        summaries = []
        for d in prior_days:
            summaries.append(f"{d}: {await self._summarize_day(account, d, by_day[d])}")
        prior_block = "\n".join(summaries) if summaries else "(no prior activity in window)"
        today_text, approx = _render(evs[: self.max_events], max_arg_chars=self.today_arg_chars)
        approx += sum(len(s) for s in summaries) // 4
        user = (f"Account: {account}\n\nPrior days (one-line summaries):\n{prior_block}\n\n"
                f"Today ({day}) — full detail:\n{today_text}\n\n"
                "Rate how likely TODAY's activity, read in light of the prior context, is an "
                "insider threat.")
        msgs = [ChatMessage(role="system", content=_SYS), ChatMessage(role="user", content=user)]
        try:
            content = await complete_with_retries(self.llm, msgs, max_tokens=self.max_tokens,
                                                  tag=self.name)
        except JudgeCallFailed as exc:
            return self._fail.failed(account, day, str(exc), approx)
        self._fail.ok()
        score, rationale = _parse(content)
        return Suspicion(score, rationale, approx)
