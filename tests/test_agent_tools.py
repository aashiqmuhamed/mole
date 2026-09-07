"""Unit tests for agent/tools.py — auto-built tool catalog + dispatcher."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from mole.agent.tools import build_tools_for
from mole.llm import ToolCall


class _FakeMail:
    """Stand-in state manager for testing reflection-based tool building."""

    async def send(self, *, to: str, subject: str, body: str = "") -> str:
        """Send an email message to a single recipient."""
        return f"id:{to}:{subject}"

    async def read_inbox(self, *, max_count: int = 10) -> list[dict]:
        """Read the latest messages from the inbox."""
        return [{"i": i} for i in range(min(max_count, 3))]

    async def _private(self) -> None:
        """Private — not exposed."""

    async def setup(self, *, sandbox) -> None:
        """Lifecycle, not a tool."""

    async def cleanup(self) -> None:
        """Lifecycle, not a tool."""

    def sync_helper(self) -> int:
        """Sync, not exposed."""
        return 1


class _FakeGit:
    async def commit(self, *, project: str, branch: str, path: str,
                     content: str, message: str) -> dict:
        """Create or update one file in one commit."""
        return {"id": "abc", "project": project, "path": path}


class _NeedsActor:
    async def submit(self, *, account: str, item: str) -> dict:
        """Submit an item as the acting account."""
        return {"account": account, "item": item}


def _ctx(**managers) -> object:
    """Build a stub context exposing `_managers`."""
    ctx = SimpleNamespace()
    ctx._managers = managers
    for name, mgr in managers.items():
        setattr(ctx, name, mgr)
    return ctx


# ── schema generation ────────────────────────────────────────────────


def test_tool_names_are_service_dot_action():
    ctx = _ctx(email=_FakeMail(), gitlab=_FakeGit())
    tools, _ = build_tools_for(ctx)
    names = {t.name for t in tools}
    assert "email.send" in names
    assert "email.read_inbox" in names
    assert "gitlab.commit" in names


def test_lifecycle_and_private_methods_are_not_exposed():
    ctx = _ctx(email=_FakeMail())
    tools, _ = build_tools_for(ctx)
    names = {t.name for t in tools}
    assert "email.setup" not in names
    assert "email.cleanup" not in names
    assert "email._private" not in names
    assert "email.sync_helper" not in names


def test_tool_description_pulls_from_docstring():
    ctx = _ctx(email=_FakeMail())
    tools, _ = build_tools_for(ctx)
    send = next(t for t in tools if t.name == "email.send")
    # Description leads with the docstring, then appends an explicit parameter list so
    # the model uses our exact argument names instead of memorized real-API names.
    assert send.description.startswith("Send an email message to a single recipient.")
    assert "Parameters:" in send.description
    assert "Use exactly these argument names." in send.description


def test_tool_parameters_capture_required_and_optional():
    ctx = _ctx(email=_FakeMail())
    tools, _ = build_tools_for(ctx)
    send = next(t for t in tools if t.name == "email.send")
    props = send.parameters["properties"]
    assert set(props.keys()) == {"to", "subject", "body"}
    # `to` and `subject` are required (no default); `body` defaults to "".
    assert "to" in send.parameters["required"]
    assert "subject" in send.parameters["required"]
    assert "body" not in send.parameters["required"]


def test_actor_parameters_are_optional_in_schema():
    ctx = _ctx(svc=_NeedsActor())
    tools, _ = build_tools_for(ctx)
    submit = next(t for t in tools if t.name == "svc.submit")
    assert "item" in submit.parameters["required"]
    assert "account" not in submit.parameters["required"]


def test_parameter_types_map_to_json_schema():
    ctx = _ctx(email=_FakeMail())
    tools, _ = build_tools_for(ctx)
    read = next(t for t in tools if t.name == "email.read_inbox")
    assert read.parameters["properties"]["max_count"]["type"] == "integer"


def test_empty_managers_yields_empty_catalog():
    ctx = _ctx()
    tools, _ = build_tools_for(ctx)
    assert tools == []


# ── dispatcher routing ───────────────────────────────────────────────


def test_dispatcher_routes_to_named_method():
    ctx = _ctx(email=_FakeMail())
    _, dispatcher = build_tools_for(ctx)
    result = asyncio.run(dispatcher(ToolCall(
        id="tc-1", name="email.send",
        arguments={"to": "bob@example.com", "subject": "hi"},
    )))
    assert result.tool_call_id == "tc-1"
    assert result.is_error is False
    assert result.content == "id:bob@example.com:hi"


def test_actor_param_autofilled_when_null_or_absent():
    # The model very often emits the actor param as explicit null (the schema marks
    # it optional/auto-filled). Auto-fill must inject the calling account for BOTH
    # absent and null — setdefault used to keep the null, losing the acting identity,
    # which silently broke read_inbox/post_message/send_email in the v1 corpus.
    ctx = _ctx(svc=_NeedsActor())
    ctx._account_box = {"account": "alice.kim"}
    _, dispatcher = build_tools_for(ctx)

    def call(args):
        r = asyncio.run(dispatcher(ToolCall(id="x", name="svc.submit", arguments=args)))
        assert r.is_error is False, r.content
        return json.loads(r.content)["account"]

    assert call({"account": None, "item": "a"}) == "alice.kim"   # explicit null
    assert call({"item": "b"}) == "alice.kim"                      # absent
    assert call({"account": "bob.li", "item": "c"}) == "bob.li"  # explicit value preserved


def test_actor_param_autofilled_from_contextvar_when_no_box():
    # The generator sets the account via the per-task ContextVar (set_account),
    # NOT ctx._account_box (only the focal orchestrator sets the box). The
    # auto-fill must read the ContextVar too, else the whole sim interaction layer
    # runs with no identity -> read_inbox(user=None)=[], post_message(sender=None)=401.
    from mole.generator.account_context import set_account
    ctx = _ctx(svc=_NeedsActor())   # deliberately NO _account_box
    _, dispatcher = build_tools_for(ctx)
    try:
        set_account("carol.x", "background_llm_agent")
        r = asyncio.run(dispatcher(ToolCall(id="x", name="svc.submit",
                                            arguments={"account": None, "item": "a"})))
        assert json.loads(r.content)["account"] == "carol.x", r.content
    finally:
        set_account("system", "system")   # don't leak into other tests


def test_int_typed_args_coerced_from_strings():
    # Now that wrapped methods expose real signatures, schemas show param types
    # and the model passes int params as strings ("10"). Coerce digit-strings so
    # e.g. read_inbox's `[:max_count]` slice doesn't "slice indices must be int".
    class _M:
        async def fetch(self, *, count: int = 10, name: str = ""):
            """Fetch."""
            return {"count": count, "ctype": type(count).__name__, "name": name}
    ctx = _ctx(svc=_M())
    _, dispatcher = build_tools_for(ctx)
    r = asyncio.run(dispatcher(ToolCall(id="x", name="svc.fetch",
                                        arguments={"count": "10", "name": "5"})))
    out = json.loads(r.content)
    assert out["count"] == 10 and out["ctype"] == "int", out   # coerced
    assert out["name"] == "5"                                   # str param untouched


def test_dispatcher_serializes_dict_results_as_json():
    ctx = _ctx(gitlab=_FakeGit())
    _, dispatcher = build_tools_for(ctx)
    result = asyncio.run(dispatcher(ToolCall(
        id="tc-2", name="gitlab.commit",
        arguments={"project": "p", "branch": "main", "path": "f.py",
                   "content": "x", "message": "m"},
    )))
    assert result.is_error is False
    parsed = json.loads(result.content)
    assert parsed["id"] == "abc"
    assert parsed["path"] == "f.py"


def test_dispatcher_unknown_tool_returns_error():
    ctx = _ctx(email=_FakeMail())
    _, dispatcher = build_tools_for(ctx)
    result = asyncio.run(dispatcher(ToolCall(
        id="tc-3", name="ghost.action", arguments={},
    )))
    assert result.is_error is True
    assert "Unknown tool" in result.content


def test_dispatcher_typeerror_for_bad_args():
    ctx = _ctx(email=_FakeMail())
    _, dispatcher = build_tools_for(ctx)
    # Missing the required `subject` kwarg.
    result = asyncio.run(dispatcher(ToolCall(
        id="tc-4", name="email.send",
        arguments={"to": "bob@example.com"},
    )))
    assert result.is_error is True
    assert "TypeError" in result.content


def test_dispatcher_exception_from_method_surfaces_as_tool_error():
    class _Raises:
        async def boom(self) -> None:
            """Always raises."""
            raise ValueError("nope")

    ctx = _ctx(svc=_Raises())
    _, dispatcher = build_tools_for(ctx)
    result = asyncio.run(dispatcher(ToolCall(id="tc-5", name="svc.boom", arguments={})))
    assert result.is_error is True
    assert "ValueError" in result.content
    assert "nope" in result.content


def test_dispatcher_handles_non_dict_arguments_gracefully():
    ctx = _ctx(email=_FakeMail())
    _, dispatcher = build_tools_for(ctx)
    # If the LLM somehow passes a list instead of a dict, we still get a
    # clean error rather than a crash.
    bad = ToolCall(id="tc-6", name="email.read_inbox", arguments=[])  # type: ignore[arg-type]
    result = asyncio.run(dispatcher(bad))
    # Empty-dict args → defaults apply, so this should actually succeed.
    assert result.is_error is False


def test_dispatcher_autofills_actor_from_context_account_box():
    ctx = _ctx(svc=_NeedsActor())
    ctx._account_box = {"account": "bob.li", "kind": "agent"}
    _, dispatcher = build_tools_for(ctx)
    result = asyncio.run(dispatcher(ToolCall(
        id="tc-7", name="svc.submit", arguments={"item": "eval"},
    )))
    assert result.is_error is False
    assert json.loads(result.content) == {"account": "bob.li", "item": "eval"}


# ── serialize-offload identity (tools.py: await to_thread(_stringify, ...)) ───
# The dispatcher serializes tool results OFF the event-loop thread so a giant
# result can't block the asyncio loop for minutes. That must be byte-identical to
# the old inline serialize, or the per-result cap and the corpus would change.


def test_offloaded_stringify_is_byte_identical_to_inline():
    import datetime

    from mole.agent.tools import _stringify

    cases = [
        {"a": 1, "t": datetime.datetime(2026, 7, 4, 12, 0)},   # datetime via default=str
        {1: "x", 2: [3, 4, {"nested": True}]},                  # non-str keys
        b"hello-bytes",                                         # bytes
        None,
        "already a string",
        {"big": [{"i": i, "s": "x" * 200, "u": "café-中"} for i in range(5000)]},
        [{"to": f"u{i}@agentlab.local", "subject": "Re: " + "z" * 300} for i in range(3000)],
    ]
    for obj in cases:
        inline = _stringify(obj)
        offloaded = asyncio.run(asyncio.to_thread(_stringify, obj))
        assert offloaded == inline, f"offload mismatch for {type(obj).__name__}"


def test_dispatcher_content_identical_for_large_result():
    from mole.agent.tools import _stringify

    big = [{"i": i, "body": "line " * 100} for i in range(4000)]

    class _Big:
        async def dump(self) -> list:
            """Return a large inbox-like dump."""
            return big

    ctx = _ctx(svc=_Big())
    _, dispatcher = build_tools_for(ctx)
    result = asyncio.run(dispatcher(ToolCall(id="tc-big", name="svc.dump", arguments={})))
    assert result.is_error is False
    assert result.content == _stringify(big)   # byte-identical to a direct serialize


# ── raw-result cap: bound giant results BEFORE serialize (no full-encode balloon) ─
def test_shrink_is_content_identical_for_normal_results():
    import datetime

    from mole.agent.tools import _shrink_bounded

    cases = [
        {"a": 1, "b": "x", "c": [1, 2, {"d": True, "e": None}]},
        {1: "int-key", 2: [3, 4, {"nested": True}]},                 # non-str keys
        [{"i": i, "s": "row" * 5, "u": "café-中"} for i in range(200)],
        {"t": datetime.datetime(2026, 7, 4, 12, 0), "n": 3.5, "z": None},
        (1, 2, (3, [4, 5])),                                          # tuples -> arrays
    ]
    for obj in cases:
        # under budget the shrink is content-preserving, so the serialize is identical
        assert json.dumps(_shrink_bounded(obj), default=str) == json.dumps(obj, default=str)


def test_shrink_bounds_giant_list_and_stops_early():
    from mole.agent.tools import _shrink_bounded

    giant = [{"id": i, "blob": "x" * 500} for i in range(20_000)]   # full serialize ~10 MB
    shrunk = _shrink_bounded(giant, budget=50_000)                  # tiny budget for the test
    s = json.dumps(shrunk)
    assert len(s) < 200_000                       # bounded, nowhere near the full ~10 MB
    assert "truncated" in s                       # truncation marker present
    assert '"id":0' in s.replace(" ", "")         # head preserved (first row visible)
    assert len(shrunk) < 2_000                     # only ~budget worth of items kept, not 20k


def test_shrink_bounds_giant_string_value():
    from mole.agent.tools import _shrink_bounded

    out = _shrink_bounded({"huge": "y" * 5_000_000}, budget=100_000, max_str=50_000)
    assert len(out["huge"]) <= 50_000 + 64
    assert out["huge"].startswith("y")
    assert "truncated" in out["huge"]


def test_stringify_bounds_pathological_result_fast():
    import time

    from mole.agent.tools import _stringify, _SHRINK_BUDGET

    giant = {"rows": [{"id": i, "blob": "x" * 800} for i in range(200_000)]}  # full ~160 MB
    t0 = time.perf_counter()
    s = _stringify(giant)
    dt = time.perf_counter() - t0
    assert dt < 10.0                              # would be many seconds+ if it encoded all 200k rows
    assert len(s) <= _SHRINK_BUDGET + 5_000_000   # bounded, not the full ~160 MB
    assert "truncated" in s


def test_shrink_never_raises_on_reference_cycle():
    from mole.agent.tools import _shrink_bounded, _stringify

    d = {}
    d["self"] = d                                 # reference cycle
    out = _shrink_bounded(d, max_depth=5)         # bounded by max_depth, must not hang/raise
    assert out is not None
    assert isinstance(_stringify({"x": 1}), str)  # serializer still healthy afterward
