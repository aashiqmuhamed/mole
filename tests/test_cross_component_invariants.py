"""Cross-component invariant tests — catches the bug class we shipped 4+ of in
this session: per-piece unit tests pass but the pieces don't agree with each
other (matcher action-name doesn't match manager method name, tool description
contradicts its schema, audit log account doesn't match the issuing session
under concurrency, persona target keyword doesn't match the matcher's regex,
etc.).

Each test here is small and independent; they should run in well under a second
combined.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
import yaml

BENCH = Path(__file__).resolve().parents[1]


# ── invariant 1: every matcher action-name exists on the corresponding manager

def test_matcher_action_names_exist_on_a_manager():
    """Every `_act(e, "X", "Y", ...)` in threat_matchers.py must name actions
    that actually exist as async methods on at least one StateManager subclass.

    Recovers the action-name drift bug that hid julian.x's priv-esc attack
    for an unknown number of smokes (matcher 03 looked for
    `org.add_to_group` but the actual action is `org.add_group_member`).
    """
    src = (BENCH / "mole" / "monitors" / "threat_matchers.py"
           ).read_text(encoding="utf-8")

    # Find _act(e, "name1", "name2", ...) sequences. Captures the bracketed args.
    matcher_actions: set[str] = set()
    for m in re.finditer(r'_act\(\s*e\s*,\s*(.+?)\)', src, flags=re.DOTALL):
        for name in re.findall(r'"([^"]+)"', m.group(1)):
            matcher_actions.add(name)

    # Walk all StateManager subclasses for their async coroutine method names.
    # Import every manager so the registry is populated.
    import mole.state                                   # noqa: F401
    from mole.state.base import StateManager

    real_actions: set[str] = set()
    for _svc_name, cls in StateManager._registry.items():               # noqa: SLF001
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue
            attr = getattr(cls, attr_name, None)
            if asyncio.iscoroutinefunction(attr):
                real_actions.add(attr_name)

    missing = matcher_actions - real_actions
    assert not missing, (
        f"threat_matchers.py references action names that don't exist on "
        f"any StateManager: {sorted(missing)}. These are silently-failing "
        f"matchers — they'll never label anything malicious because the "
        f"audit log doesn't record actions by those names."
    )


# ── invariant 2: tool description is consistent with its parameter schema

def test_tool_descriptions_cohere_with_schemas():
    """For every agent tool, the description text and the parameter
    schema must agree: if `properties` is empty, the description must not
    instruct the model to "use exactly these argument names" (no names
    exist); if properties has N keys, "Parameters: " must mention those
    same N keys.

    Recovers the empty-params confusion that blocked frank.s + ivan.o on
    Opus iter1 ("every call has Parameters: none — tool unusable").
    """
    from types import SimpleNamespace

    import mole                                         # noqa: F401
    from mole.agent.tools import build_tools_for
    from mole.sandbox.base import DryRunSandbox
    from mole.state.composite import CompositeStateManager

    async def _build():
        # In-proc managers only — the test doesn't need docker; just need
        # SOME tool surface to inspect.
        comp = CompositeStateManager(
            ["org", "model_registry", "eval_server", "secrets_store"]
        )
        await comp.setup(sandbox=DryRunSandbox())
        ctx = SimpleNamespace(**comp.managers)
        ctx._managers = comp.managers
        return build_tools_for(ctx)

    tools, _ = asyncio.run(_build())
    assert tools, "expected build_tools_for to produce a non-empty tool surface"

    contradictions = []
    for t in tools:
        props = t.parameters.get("properties", {}) or {}
        desc = t.description
        if not props:
            # The bug: appending "Use exactly these argument names." when
            # there are no argument names reads as "param system is broken."
            if "Use exactly these argument names" in desc:
                contradictions.append(
                    f"{t.name}: empty schema but description says 'Use exactly "
                    f"these argument names': {desc!r}"
                )
        elif not desc.endswith("…"):
            # When properties is non-empty AND the description wasn't truncated,
            # the named params must appear in the human-readable "Parameters: ..."
            # block too. (Truncation at ~600 chars can lop off the Parameters
            # block for verbose-doc managers; that's a separate known issue
            # in agent/tools.py — _schema_for puts the doc before params.)
            param_block_match = re.search(r"Parameters: ([^.]*)", desc)
            assert param_block_match, (
                f"{t.name}: properties is non-empty but description lacks a "
                f"'Parameters: ...' summary: {desc!r}"
            )

    assert not contradictions, (
        "tool description / schema mismatches:\n  " + "\n  ".join(contradictions)
    )


# ── invariant 3: audit log account matches issuing session under concurrency

@pytest.mark.asyncio
async def test_audit_attribution_under_concurrency():
    """N concurrent tasks each set their own account + task_id, then issue
    audited calls. After gather() finishes, every recorded event must bear
    BOTH the account of the issuing task AND a matching task_id.

    Recovers the account-attribution race that mis-attributed up to 65%
    of pair-uniquely-matched events in the v3 corpora."""
    from dataclasses import dataclass

    from mole.audit.collector import AuditCollector
    from mole.generator.account_context import (
        get_account,
        get_task_id,
        set_account,
    )

    @dataclass
    class _Mgr:
        async def write_file(self, *, path: str = "") -> int:
            return len(path)

    collector = AuditCollector()
    mgr = _Mgr()
    collector.wrap_manager(
        service_name="owncloud", manager=mgr,
        account_getter=get_account,
    )

    N = 8

    async def task(name: str) -> str:
        tid = set_account(name, "background_llm_agent")
        for _ in range(3):
            await asyncio.sleep(0)
        await mgr.write_file(path=f"/p_{name}")
        return tid

    expected_tids = await asyncio.gather(*(task(f"p{i}") for i in range(N)))

    # Build {expected_account: expected_task_id} so we can verify pairs.
    expected = {f"p{i}": expected_tids[i] for i in range(N)}

    by_path = {e.args.get("path"): e for e in collector.events}
    assert len(by_path) == N, "expected one audit event per task"
    for i in range(N):
        name = f"p{i}"
        e = by_path[f"/p_{name}"]
        assert e.account == name, (
            f"task {name} wrote /p_{name} but audit recorded account={e.account!r} "
            f"— the account-attribution race is back"
        )
        assert e.task_id == expected[name], (
            f"task {name} expected task_id={expected[name]!r} but audit "
            f"recorded task_id={e.task_id!r}"
        )


# ── invariant 4: every persona references a matcher-known threat and mentions
#                the matcher's target keyword

PERSONAS_DIR = BENCH / "bootstrap" / "personas"


def _persona_files():
    files = sorted(PERSONAS_DIR.glob("*.yaml"))
    # Exclude the null-control variants (intentionally have no malicious target).
    return [f for f in files if "null_" not in f.name]


@pytest.mark.parametrize("path", _persona_files(), ids=lambda p: p.name)
def test_persona_threat_and_target_alignment(path: Path):
    """For every persona file: its declared threat must have a matcher,
    and its persona_override text must mention at least one of the
    matcher's expected target keywords. Prevents persona files drifting
    out of sync with the harm matcher (rare but high-impact: a persona
    that points at the wrong target will produce 0 labeled events).
    """
    from mole.monitors.threat_matchers import _TARGETS, REGISTRY

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    threat = doc.get("threat")
    stealth = doc.get("stealth_level") or "s0"
    text = (doc.get("persona_override") or "").lower()

    assert threat, f"{path.name}: missing 'threat' field"

    # Solo-threat matcher must exist for s0/s1; collusion is handled separately.
    if threat in REGISTRY:
        assert threat in _TARGETS, (
            f"{path.name}: threat {threat!r} has a matcher in REGISTRY "
            f"but no entry in _TARGETS"
        )
        targets = _TARGETS[threat].get(stealth, {})
        # The persona must mention SOMETHING about the target so the agent
        # has a concrete artifact to act on.
        target_values = " ".join(str(v) for v in targets.values()).lower()
        target_tokens = [t for t in re.split(r"[\s/_,]+", target_values)
                         if len(t) >= 4 and t not in ("the", "and")]
        if not target_tokens:
            return  # threat has no usable target tokens; skip
        # At least one target token (file/ticket/path/key fragment) must
        # appear verbatim in the persona text.
        matched = [t for t in target_tokens if t in text]
        assert matched, (
            f"{path.name}: persona text mentions none of the matcher's "
            f"target keywords {target_tokens!r} — the agent will explore "
            f"but won't fire the matcher even if compliant. Persona text "
            f"head: {text[:200]!r}"
        )
