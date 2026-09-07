"""Unit tests for the source-side realism + validity fixes.

Covers the behaviors added by the regen plan (delightful-floating-gem):
is_external classification, the audit arg-normalizer + channel canon, persona
tool-gating, and rare per-day absence.
"""
from types import SimpleNamespace

from mole.audit.collector import (
    is_external_domain, _normalize_args, _canonical_ts,
)
from mole.agent.tools import _tool_allowed
from mole.generator.member import NPCConfig
from mole.generator.agentic_member import AgenticMember
from mole.generator.persona_loader import Persona


# ── A: is_external + arg-normalizer ───────────────────────────────────

def test_is_external_domain():
    assert is_external_domain("gmail.com") is True
    assert is_external_domain("fair-labs.org") is True
    assert is_external_domain("agentlab.local") is False   # internal — was the bug
    assert is_external_domain("AGENTLAB.LOCAL") is False    # case-insensitive
    assert is_external_domain("") is False
    assert is_external_domain(None) is False


def test_normalize_args_drops_nulls_and_empty_extra():
    out = _normalize_args({"issue_id": "OPS-2", "text": "hi", "sender": None,
                           "body": None, "_extra": {}})
    assert out == {"issue_id": "OPS-2", "text": "hi"}


def test_normalize_args_unwraps_kwargs_wrapper():
    assert _normalize_args({"args": [], "kwargs": {"project": "p", "iid": 5}}) == \
        {"project": "p", "iid": 5}


# ── C: rocketchat channel canonicalization ────────────────────────────

def test_normalize_args_canonicalizes_channel():
    assert _normalize_args({"channel": "alignment"})["channel"] == "#alignment"
    assert _normalize_args({"channel": "#alignment"})["channel"] == "#alignment"


def test_normalize_args_lifts_nonempty_extra():
    # The **_extra catch-all (e.g. submit_eval(suite=, checkpoint_id=)) is lifted
    # to first-class args, None-filtered, without clobbering a named arg.
    out = _normalize_args({"_extra": {"suite": "red-team", "checkpoint_id": "c1",
                                      "x": None}})
    assert out == {"suite": "red-team", "checkpoint_id": "c1"}
    assert _normalize_args({"a": 1, "_extra": {}}) == {"a": 1}        # empty -> dropped
    assert _normalize_args({"account": "alice",
                            "_extra": {"account": "bob"}}) == {"account": "alice"}


# ── ts canonicalization (smoke surfaced Z vs +00:00 in one corpus) ─────

def test_canonical_ts_single_format():
    assert _canonical_ts("2026-04-06T11:06:03+00:00") == "2026-04-06T11:06:03Z"
    assert _canonical_ts("2026-04-06T10:52:03Z") == "2026-04-06T10:52:03Z"
    assert _canonical_ts("2026-04-06T10:52:03.5+00:00") == "2026-04-06T10:52:03.5Z"
    assert _canonical_ts("") == ""


# ── D: MR-id alias (full-lab smoke: 65% of approve/merge_mr failed on `iid`) ──

def test_mr_iid_alias_resolves():
    from mole.agent.tools import _apply_aliases

    def approve_mr(*, project, mr_iid, approver=None):
        ...

    for syn in ("iid", "mr", "mr_id", "merge_request_iid"):
        out = _apply_aliases(approve_mr, {"project": "eval/x", syn: 3})
        assert out == {"project": "eval/x", "mr_iid": 3}, (syn, out)

    # a real mr_iid is never clobbered by a stray synonym
    out = _apply_aliases(approve_mr, {"project": "p", "mr_iid": 5, "iid": 9})
    assert out["mr_iid"] == 5

    # never fires on a tool without mr_iid (e.g. plane create_issue keeps its iid)
    def create_issue(*, project_id, name):
        ...
    out = _apply_aliases(create_issue, {"project_id": "p", "name": "n", "iid": 9})
    assert out == {"project_id": "p", "name": "n", "iid": 9}


# ── F: persona tool-gating ────────────────────────────────────────────

def test_tool_gating_hr_legal_collab_only():
    hr = SimpleNamespace(team="hr", services={})
    assert _tool_allowed(hr, "gitlab", "commit") is False        # no code host
    assert _tool_allowed(hr, "eval_server", "submit_eval") is False
    assert _tool_allowed(hr, "model_registry", "promote") is False
    assert _tool_allowed(hr, "email", "send_email") is True       # collaboration
    assert _tool_allowed(hr, "plane", "create_issue") is True
    assert _tool_allowed(hr, "org", "add_group_member") is True   # HR does access mgmt


def test_tool_gating_gitlab_write_by_permission():
    dev = SimpleNamespace(team="alignment", services={"gitlab": {"permissions": "developer"}})
    rep = SimpleNamespace(team="evaluations", services={"gitlab": {"permissions": "reporter"}})
    assert _tool_allowed(dev, "gitlab", "commit") is True
    assert _tool_allowed(rep, "gitlab", "commit") is False        # reporter: no write
    assert _tool_allowed(rep, "gitlab", "merge_mr") is False
    assert _tool_allowed(rep, "gitlab", "list_files") is True     # read ok
    assert _tool_allowed(rep, "gitlab", "approve_mr") is True     # co-sign ok
    assert _tool_allowed(None, "gitlab", "commit") is True        # insider/focal: full catalog


# ── I: rare per-day absence ───────────────────────────────────────────

def _persona():
    return Persona(id="x.y1", full_name="X Y", email="x.y1@agentlab.local",
                   role="ML Researcher", groups=(), services={}, team="alignment")


def test_absence_yields_empty_day_when_always_absent():
    m = AgenticMember(_persona(), llm=None,
                      config=NPCConfig(sessions_per_day=3, loaf_probability=0.0,
                                       absent_probability=1.0), rng_seed=0)
    assert m.plan_day("2026-04-06") == []


def test_no_absence_when_disabled():
    m = AgenticMember(_persona(), llm=None,
                      config=NPCConfig(sessions_per_day=3, loaf_probability=0.0,
                                       absent_probability=0.0), rng_seed=0)
    assert len(m.plan_day("2026-04-06")) == 3


# ── audit wrapper retries transient lab-connection blips ──

def test_audit_retries_transient_connection_errors():
    # A lab service briefly unreachable (Docker hiccup under load) -> connection
    # refused. The request never reached the service, so retry (side-effect-free)
    # instead of dropping the event. Exactly one event, exit_code 0 after retry.
    import asyncio
    from mole.audit.collector import AuditCollector

    class _M:
        def __init__(self):
            self.calls = 0
        async def fetch(self, *, x=None):
            """Fetch."""
            self.calls += 1
            if self.calls == 1:
                raise ConnectionRefusedError("[WinError 10061] No connection could be made")
            return {"x": x, "calls": self.calls}

    coll = AuditCollector()
    m = _M()
    coll.wrap_manager("svc", m, account_getter=lambda: ("alice", "background_rules_agent"))
    r = asyncio.run(m.fetch(x=1))
    assert r["calls"] == 2            # retried once after the connection error
    assert len(coll.events) == 1      # one emit, not two
    assert coll.events[0].exit_code == 0


# ── audit wrapper must preserve signatures (else schemas + auto-fill break) ──

def test_wrap_manager_preserves_signature():
    # build_tools_for inspects wrapped methods for BOTH the JSON tool schema and
    # actor-param auto-fill. A generic (*args, **kwargs) wrapper hid every param,
    # so the model guessed arg names (iid vs mr_iid) and actor params (user/sender/
    # from_user) were never auto-filled -> read_inbox(None)=[], post_message=401.
    import inspect
    from mole.audit.collector import AuditCollector

    class _M:
        async def post_message(self, *, channel=None, sender=None, text=None):
            """Post."""
            return {}

    coll = AuditCollector()
    m = _M()
    coll.wrap_manager("rocketchat", m, account_getter=lambda: ("system", "system"))
    params = list(inspect.signature(m.post_message).parameters)
    assert params == ["channel", "sender", "text"], params
    assert "sender" in params  # the actor param build_tools_for must see to auto-fill


# ── corpus integrity: emit() under concurrent (threaded) tool dispatch ──

def test_emit_thread_safe_under_concurrency(tmp_path):
    """Tool methods run via ``asyncio.to_thread`` (state/*/manager.py), so several
    threads call ``emit`` at once. Without the collector lock their ``write`` calls
    interleave and splice two events onto one line (corrupt JSONL — surfaced as
    bad_lines>0 in a smoke). Hammer emit() from a thread pool and assert every
    line parses and no event is lost.

    The race only fires under enough GIL handoffs, so we shrink the switch
    interval to force it; otherwise an unlocked emit() could pass by luck
    (measured ~3.5% malformed unlocked at 1e-6, 0% locked).
    """
    import json
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from mole.audit.collector import AuditCollector
    from mole.audit.schema import AuditEvent

    path = tmp_path / "concurrent.jsonl"
    coll = AuditCollector(jsonl_path=path)
    N = 4000

    def one(i):
        coll.emit(AuditEvent.new(
            account=f"p{i % 17}", account_kind="background_rules_agent",
            service="email", action="send_email",
            resource_id=f"msg-{i}", args={"i": i, "body": "x" * 400}))

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(max_workers=48) as ex:
            list(ex.map(one, range(N)))   # list() re-raises any worker exception
    finally:
        sys.setswitchinterval(old_interval)
    coll.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == N, f"expected {N} lines, got {len(lines)}"
    ids = {json.loads(ln)["resource_id"] for ln in lines}  # raises on a spliced line
    assert len(ids) == N              # no dropped / overwritten events
    assert len(coll.events) == N      # in-memory list intact too
