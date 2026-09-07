"""Gating protocols — the *intervention* half of the observability–intervention
frontier. A protocol wraps the agent dispatcher (the seam the agent loop
documents: "the gating middleware wraps the dispatcher so every call passes
through a monitor + protocol decision first"). Before a tool call mutates state,
the protocol consults a monitor tier and may DENY it (returning an error
ToolResult) or just ALERT — realizing Cell 2 (benign + control → utility cost)
and Cell 4 (malicious + control → harm prevented).

Per-action risk reuses the level 1 cascade primitives (`_metadata_red_flag`,
`_payload_red_flag`) and the collector's destination inference, so gating and
detection share one notion of "risky action".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..audit.collector import _infer_dest_domains, is_external_domain
from ..audit.schema import AuditEvent
from ..agent.loop import Dispatcher, ToolResult
from ..llm.base import ToolCall


@dataclass
class GatingDecision:
    allow: bool
    reason: str = ""
    observability_level: int = -1


@dataclass
class GatingProtocol(ABC):
    """Base protocol. `org` (optional) lets the external-destination check use
    the real org domain; `alerts` records every flag (blocking or not)."""
    name: str = "base"
    observability_level: int = -1
    org: Any = None
    alerts: list[dict[str, Any]] = field(default_factory=list)

    def candidate_event(self, tc: ToolCall) -> tuple[AuditEvent, dict[str, Any]]:
        """Build the AuditEvent a successful call WOULD emit, so the same
        red-flag logic that scores the audit log can gate it pre-execution."""
        svc, _, action = (tc.name or "").partition(".")
        args = dict(tc.arguments or {})
        rid = str(args.get("path") or args.get("resource_id") or args.get("to")
                  or args.get("project") or "")
        e = AuditEvent.new(account="focal", account_kind="agent",
                           service=svc, action=action, resource_id=rid, args=args)
        domains = _infer_dest_domains(args)
        if self.org is not None and domains:
            try:
                external_domains = [
                    d for d in domains
                    if self.org.is_external(f"x@{d}")
                ]
                e.dest_domain = (
                    external_domains[0]
                    if external_domains
                    else domains[0]
                )
                e.is_external = bool(external_domains)
            except Exception:                                  # noqa: BLE001
                ext = [d for d in domains if is_external_domain(d)]
                e.dest_domain = ext[0] if ext else domains[0]
                e.is_external = bool(ext)
        else:
            ext = [d for d in domains if is_external_domain(d)]
            e.dest_domain = ext[0] if ext else (domains[0] if domains else None)
            e.is_external = bool(ext)
        return e, args

    @abstractmethod
    async def decide(self, e: AuditEvent, args: dict[str, Any]) -> GatingDecision:
        ...

    def wrap(self, dispatcher: Dispatcher) -> Dispatcher:
        async def gated(tc: ToolCall) -> ToolResult:
            e, args = self.candidate_event(tc)
            decision = await self.decide(e, args)
            if not decision.allow:
                self.alerts.append({"tool": tc.name, "reason": decision.reason,
                                    "blocked": True, "observability_level": self.observability_level})
                return ToolResult(
                    tool_call_id=tc.id, name=tc.name,
                    content=(f"DENIED by the {self.name} control protocol: "
                             f"{decision.reason}. This action was not performed."),
                    is_error=True,
                )
            return await dispatcher(tc)
        return gated
