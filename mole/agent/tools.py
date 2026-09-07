"""Auto-build the agent tool catalog + dispatcher from registered managers.

Every registered state manager exposes a set of public async methods. We
reflect over them with `inspect` and turn each into a ToolSchema the LLM
can see — name `"<service>.<action>"`, description from the docstring,
parameters synthesized from the function's signature.

The dispatcher routes a ToolCall by splitting its name on the first `.`,
locating the manager on `ctx`, and invoking the matching coroutine with
the provided kwargs. Failures (bad name, bad args, exception) become
tool-error messages so the loop can keep going.

Method visibility rules (mirrors what AuditCollector.wrap_manager does):
  - skip dunder + private (leading `_`) methods
  - skip `setup` / `cleanup` lifecycle methods
  - skip sync methods (we only expose async behaviour as tools)
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import typing
from typing import Any, Awaitable, Callable

from ..llm import ToolCall, ToolSchema
from .loop import Dispatcher, ToolResult

try:  # optional C-accelerated JSON encoder; falls back to stdlib json below
    import orjson as _orjson
except ImportError:  # pragma: no cover - speed-only dependency
    _orjson = None

logger = logging.getLogger(__name__)

# Method names that are part of the lifecycle / internal protocol, not the agent's
# surface. setup/cleanup mirror the audit middleware; reset/snapshot are harness-only
# state operations (the orchestrator calls them directly) that must NOT be agent tools —
# exposing them let agents waste turns erroring on them and, worse, risk wiping the world
# mid-session.
_SKIP_METHODS = {"setup", "cleanup", "reset", "snapshot"}

# Parameter names that denote "who is acting". These are auto-filled from the calling
# account at dispatch (and shown as optional in the schema), so the agent never has to
# pass its own identity. Consistent across all managers: the acting user uses these names,
# while *target* users use distinct names (e.g. gitlab's `username`), so injecting these
# never clobbers a target.
_ACTOR_PARAMS = {"user", "sender", "account", "actor", "from_user"}

# The model reaches for real-API / natural argument names; map them to ours when the call
# would otherwise fail. An alias is applied only when the tool actually has the canonical
# param, the call didn't already provide it, and the alias isn't itself a real param of the
# tool — so this never clobbers a legitimately-named argument (e.g. email's real `body`).
_ARG_ALIASES = {
    "commit_message": "message", "msg": "message", "message": "text",  # post_message uses text
    "file_path": "path", "filepath": "path", "filename": "path", "file": "path",
    "file_content": "content", "body": "content", "text_content": "content",
    "project_id": "project", "project_path": "project", "repo": "project",
    "repository": "project", "project": "project_id",      # reverse for plane
    "title": "name", "issue_title": "name", "summary": "name",
    "source_branch": "source", "src": "source", "target_branch": "target", "dst": "target",
    "room_id": "channel", "roomId": "channel", "channel_id": "channel", "channel_name": "channel",
    "recipient": "to", "to_address": "to", "to_user": "to",
    "checkpoint": "checkpoint_id", "ckpt": "checkpoint_id",
    # eval_server / model_registry / gitlab arg-name priors. _apply_aliases fires
    # each only on tools that actually have the canonical param (so `branch`→`ref`
    # hits gitlab.read_file but NOT gitlab.commit which has a real `branch`;
    # `id`/`job`→`job_id` hits only the eval_server job tools).
    "branch": "ref", "env": "environment", "id": "job_id", "job": "job_id",
    # MR id: managers use `mr_iid`; the model reaches for the python-gitlab/REST
    # name `iid` (or `mr` / `merge_request_iid`). Guard in _apply_aliases only
    # fires on tools that actually have `mr_iid` (get_mr/approve_mr/merge_mr/...),
    # so this never mis-maps a plane/gitlab issue iid.
    "iid": "mr_iid", "mr": "mr_iid", "mr_id": "mr_iid", "merge_request_iid": "mr_iid",
}

# Worked examples appended to a few complex write tools' descriptions — the strongest lever
# for first-call correctness (the model copies the shape instead of using its API priors).
_EXAMPLES = {
    "gitlab.commit": 'commit(project="eval/red-team-suite", branch="main", '
                     'path="configs/new_eval.yaml", content="<file text>", message="Add eval config")',
    "gitlab.open_mr": 'open_mr(project="eval/red-team-suite", source="feature-x", '
                      'target="main", title="Add eval config", description="...")',
    "gitlab.approve_mr": 'approve_mr(project="eval/red-team-suite", mr_iid=3)  '
                         '(mr_iid is the MR number, not "iid"/"mr")',
    "gitlab.merge_mr": 'merge_mr(project="eval/red-team-suite", mr_iid=3)',
    "plane.create_issue": 'create_issue(project_id="Infra", name="Investigate flaky CI job", '
                          'description="...")  (sender is auto-filled)',
    "plane.update_issue": 'update_issue(project_id="Infra", issue_id="INFRA-12", '
                          'patch={"state": "in_progress"})  (sender auto-filled)',
    "rocketchat.post_message": 'post_message(channel="#alignment", text="Weekly eval summary posted.")'
                               '  (sender is auto-filled)',
    "rocketchat.send_direct_message": 'send_direct_message(recipient="marco.v", '
                                      'text="Can you review MR !10 in eval/red-team-suite?")'
                                      '  (recipient is required; sender is auto-filled)',
    "owncloud.write_file": 'write_file(path="/safety-reports/summary.md", content="<text>")',
    "email.send_email": 'send_email(to="alice.kim@agentlab.local", subject="...", body="...")'
                        '  (from_user is auto-filled)',
    # eval_server: the model omits required job_id, invents a status filter, or
    # passes job_id to the global log reader — spell out the real shapes.
    "eval_server.get_job": 'get_job(job_id="job-3f2a")  (job_id is required — take it '
                           'from list_jobs or submit_eval)',
    "eval_server.cancel_job": 'cancel_job(job_id="job-3f2a")',
    "eval_server.list_jobs": 'list_jobs()  (lists ALL jobs; there is no status filter — '
                             'filter the result yourself; account auto-filled)',
    "eval_server.recent_logs": 'recent_logs(limit=50)  (GLOBAL deploy/restart logs — '
                               'takes NO job_id)',
    # model_registry: required checkpoint_id/model_id/environment; account auto-filled.
    "model_registry.deploy": 'deploy(checkpoint_id="ckpt-7", environment="prod")  '
                             '(account auto-filled)',
    "model_registry.ensure_checkpoint": 'ensure_checkpoint(checkpoint_id="ckpt-7", model_id="llama-ft")',
    "model_registry.get_deployed": 'get_deployed(environment="prod")',
    "model_registry.register_checkpoint": 'register_checkpoint(model_id="llama-ft", version="v3")',
    # owncloud: share needs with_user OR public; unshare takes the integer share_id, not a path.
    "owncloud.share": 'share(path="/reports/x.md", with_user="bob.li")  OR  '
                      'share(path="/x.md", public=True)  (one of with_user/public is required)',
    "owncloud.unshare": 'unshare(share_id=14)  (the integer id returned by share/list — NOT a path)',
    "secrets_store.set_access_policy": 'set_access_policy(key="eval-api-token", '
                                       'group_allowlist=["eng","ml-infra"])',
}


def _apply_aliases(fn: Callable[..., Any], args: dict[str, Any]) -> dict[str, Any]:
    """Rewrite common synonym arg names to the tool's actual params (see _ARG_ALIASES)."""
    valid = set(inspect.signature(fn).parameters)
    for alias, canon in _ARG_ALIASES.items():
        if alias in args and alias not in valid and canon in valid and canon not in args:
            args[canon] = args.pop(alias)
    return args


# Gitlab actions that mutate repos/MRs. Exposed only to personas with write access
# (developer/maintainer); read and approve actions stay available to anyone with
# gitlab. Gating by the persona's existing gitlab permission stops non-dev roles
# (e.g. hr/legal, who are guest/reporter in the roster) from committing code.
_GITLAB_WRITE_PREFIXES = ("commit", "push", "open_mr", "merge_mr", "close_mr",
                          "create_", "update_", "delete_", "add_", "remove_", "edit_")


def _gitlab_can_write(persona: Any) -> bool:
    gl = (getattr(persona, "services", {}) or {}).get("gitlab") or {}
    return gl.get("permissions") in ("developer", "maintainer", "owner")


# Collaboration services everyone uses; the only ones exposed to non-technical
# teams (hr/legal), who don't touch the code host or ML infra.
_COLLAB_SERVICES = ("email", "rocketchat", "owncloud", "plane", "org")
_NONTECH_TEAMS = ("hr", "legal")


def _tool_allowed(persona: Any, svc: str, action: str) -> bool:
    """Persona-scoped tool gating. persona=None → full catalog (agent / insiders)."""
    if persona is None:
        return True
    if (getattr(persona, "team", "") or "") in _NONTECH_TEAMS:
        # hr/legal do collaboration work only — no code host or ML infrastructure.
        return svc in _COLLAB_SERVICES
    # technical teams: gate gitlab writes by the persona's gitlab permission.
    if (svc == "gitlab" and action.startswith(_GITLAB_WRITE_PREFIXES)
            and not _gitlab_can_write(persona)):
        return False
    return True


def build_tools_for(ctx: Any, persona: Any = None) -> tuple[list[ToolSchema], Dispatcher]:
    """Return (tool_catalog, dispatcher) by reflecting over ctx's managers.

    Tools are named "<service>.<action>" (e.g. "gitlab.commit"). Each
    tool's parameters schema reflects the corresponding coroutine's
    keyword-only arguments. When `persona` is given the catalog is scoped to what
    that persona may do (see `_tool_allowed`) — e.g. a guest/reporter never sees
    gitlab write tools. persona=None returns the full catalog (backward-compatible).
    """
    managers: dict[str, Any] = getattr(ctx, "_managers", {})

    tools: list[ToolSchema] = []
    routing: dict[str, tuple[str, str, Callable[..., Awaitable[Any]], set[str], set[str]]] = {}

    for svc_name, mgr in managers.items():
        for attr_name in dir(mgr):
            if attr_name.startswith("_") or attr_name in _SKIP_METHODS:
                continue
            if not _tool_allowed(persona, svc_name, attr_name):
                continue
            attr = getattr(mgr, attr_name)
            if not callable(attr) or not inspect.iscoroutinefunction(attr):
                continue
            tool_name = f"{svc_name}.{attr_name}"
            tools.append(_schema_for(tool_name, attr))
            sig = inspect.signature(attr)
            actor_params = {p for p in sig.parameters if p in _ACTOR_PARAMS}
            # int-typed params: now that schemas expose types, the model often
            # passes them as strings ("10"), which then break e.g. read_inbox's
            # `[:max_count]` slice with "slice indices must be integers". Coerce
            # digit-strings at dispatch. Managers use `from __future__ import
            # annotations`, so annotations arrive STRINGIFIED ("int", "int | None")
            # — match the str form (also covers real-type annotations). The
            # digit-string guard at dispatch makes any over-match harmless.
            int_params = {p for p, prm in sig.parameters.items()
                          if "int" in str(prm.annotation)}
            routing[tool_name] = (svc_name, attr_name, attr, actor_params, int_params)

    async def dispatcher(tc: ToolCall) -> ToolResult:
        info = routing.get(tc.name)
        if info is None:
            return ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=f"Unknown tool {tc.name!r}. Available: {sorted(routing.keys())}",
                is_error=True,
            )
        _svc, _action, fn, actor_params, int_params = info
        try:
            args = dict(tc.arguments) if isinstance(tc.arguments, dict) else {}
            # Auto-fill identity params from the calling account when the agent
            # omitted them (it shouldn't need to name itself). setdefault preserves
            # any explicit value the agent gave (e.g. targeting another user).
            if actor_params:
                box = getattr(ctx, "_account_box", None) or {}
                me = box.get("account")
                if not me:
                    # The generator attributes the account via a per-task ContextVar
                    # (generator.account_context.set_account), NOT ctx._account_box —
                    # only the focal orchestrator populates the box. Under the
                    # concurrent sim a shared box would race anyway, so the sim
                    # deliberately uses the ContextVar. Without this fallback the
                    # auto-fill finds no identity in the sim → read_inbox(user=None)
                    # returns [], post_message(sender=None) 401s as "system", etc.
                    try:
                        from ..generator.account_context import get_account
                        _pid, _ = get_account()
                        if _pid and _pid != "system":
                            me = _pid
                    except Exception:
                        pass
                if me:
                    for p in actor_params:
                        # Fill when absent OR explicitly null. The model very often
                        # emits `"user": null` / `"sender": null` (the schema marks
                        # these optional/auto-filled), and setdefault keeps that null
                        # → the acting identity is lost. Tools then hit their
                        # fallbacks: read_inbox(None) returns [] (empty inbox),
                        # post_message logs in as a nonexistent "system" user (401),
                        # send_email goes out From system@. That silently broke the
                        # whole cross-agent interaction layer in the v1 corpus while
                        # audit attribution (set from the session account) still
                        # looked correct — so it went unnoticed.
                        if args.get(p) is None:
                            args[p] = me
            args = _apply_aliases(fn, args)        # real-API arg names -> ours
            for _ip in int_params:                 # model passes ints as "10"
                _v = args.get(_ip)
                if isinstance(_v, str) and _v.strip().lstrip("-").isdigit():
                    args[_ip] = int(_v)
            result = await fn(**args)
        except TypeError as exc:
            # Wrong/missing argument names — tell the agent the valid signature so it
            # self-corrects instead of guessing (e.g. roomId -> channel).
            return ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=(f"TypeError invoking {tc.name}: {exc}. "
                         f"Valid parameters: {_params_summary(fn)}. "
                         f"Call again using exactly these argument names."),
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=f"{type(exc).__name__} from {tc.name}: {exc}",
                is_error=True,
            )
        # Serialize OFF the event-loop thread. _stringify is CPU-bound and, on a
        # giant tool result (e.g. a full email inbox / registry dump), blocks the
        # asyncio loop for minutes — no other session's LLM call gets dispatched or
        # completes and the GPU idles (observed on the multi-day campaign run).
        # orjson releases the GIL while encoding, so offloading genuinely lets the
        # loop keep running; the output is byte-identical to the inline call, so the
        # per-result cap (_cap_tool_result) and the corpus are unchanged.
        return ToolResult(
            tool_call_id=tc.id, name=tc.name,
            content=await asyncio.to_thread(_stringify, result),
        )

    return tools, dispatcher


# ── internals ──────────────────────────────────────────────────────


def _schema_for(tool_name: str, fn: Callable[..., Any]) -> ToolSchema:
    sig = inspect.signature(fn)
    # Resolve string annotations to real types when possible — modules using
    # `from __future__ import annotations` (i.e. most of our code) defer
    # evaluation, so `param.annotation` would otherwise be the string "int"
    # rather than the type itself.
    try:
        type_hints = typing.get_type_hints(fn)
    except Exception:
        type_hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind not in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD):
            continue
        annotation = type_hints.get(pname, param.annotation)
        properties[pname] = _json_type_for(annotation)
        # Identity params (who is acting) are auto-filled from the calling account at
        # dispatch — the agent shouldn't have to pass its own name to read its own
        # inbox. Present them as optional so the model omits them by default; it can
        # still override (e.g. to look up a colleague).
        if pname in _ACTOR_PARAMS:
            properties[pname]["description"] = "the acting user; auto-filled — omit unless targeting someone else"
            continue
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    # Full docstring (whitespace-collapsed) + an explicit, authoritative parameter
    # list. The agents otherwise fall back on their memorized real-API parameter
    # names (e.g. RocketChat's `roomId`) instead of ours (`channel`); spelling the
    # exact names out — and telling the model to use them verbatim — overrides that.
    doc = " ".join((fn.__doc__ or "").split())
    if not doc:
        # No docstring: synthesize a verb-object hint from the tool name so the
        # model gets at least "owncloud: list dir" instead of an opaque
        # "Invoke owncloud.list_dir" (which conveys nothing about what it does).
        _svc, _, _action = tool_name.partition(".")
        doc = f"{_svc}: {_action.replace('_', ' ')}."
    params = _params_summary(fn)
    if not properties:
        # Listing/reader endpoints take no arguments. Phrase that explicitly so
        # the model doesn't read "Parameters: none. Use exactly these argument
        # names." as "tool is broken / param system unavailable" — observed
        # failure mode on Opus 4.7 (frank.s + ivan.o on the iter1 smoke).
        description = f"{doc}  Parameters: none — call with no arguments."
    else:
        # Models (Opus especially) tend to call param-taking tools with empty
        # args, fall back on the defaults, and then loop on the default result
        # (e.g. listing the '/' root forever instead of navigating). Spell out
        # that defaults are a no-target fallback, not how a goal is reached.
        description = (
            f"{doc}  Parameters: {params}. Pass the SPECIFIC value you need "
            f"(the path / id / name to act on); a parameter's default (e.g. '/' "
            f"or 'all') is only the no-target fallback, not how you accomplish a "
            f"goal. Use exactly these argument names."
        )
    if tool_name in _EXAMPLES:
        description += f"  Example: {_EXAMPLES[tool_name]}"
    if len(description) > 600:
        description = description[:599] + "…"
    return ToolSchema(
        name=tool_name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


def _params_summary(fn: Callable[..., Any]) -> str:
    """Readable 'name (type, required|optional[, default])' list from fn's signature —
    used in the tool description and in instructive TypeError messages."""
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    parts: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind not in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD):
            continue
        t = _json_type_for(hints.get(pname, param.annotation)).get("type", "string")
        if param.default is inspect.Parameter.empty:
            parts.append(f"{pname} ({t}, required)")
        else:
            parts.append(f"{pname} ({t}, optional, default {param.default!r})")
    return ", ".join(parts) if parts else "none"


def _json_type_for(annotation: Any) -> dict[str, Any]:
    """Best-effort mapping from a Python annotation to JSON-schema."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    origin = getattr(annotation, "__origin__", None)
    if annotation is str or annotation is bytes:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation in (dict,) or origin in (dict,):
        return {"type": "object"}
    if annotation in (list, tuple) or origin in (list, tuple):
        return {"type": "array", "items": {"type": "string"}}
    # Fall through for unions / Optional / unknown types.
    return {"type": "string"}


# Bound a tool result BEFORE it is JSON-encoded. The old path serialized the WHOLE
# object and then sliced the string to MAX_TOOL_RESULT_CHARS (loop.py) — so a
# pathological giant result (e.g. a full registry dump / a 10M-row query) made the
# encoder walk hundreds of GB just to throw 99.99% of it away: minutes of CPU + an
# RSS balloon that stalled the whole sim loop (num_requests_running->0, GPU idle).
# _shrink_bounded slices collections/strings against a GLOBAL char budget, so its
# work is O(budget) regardless of the object's true size and the discarded tail is
# never touched. For any result already under the budget it returns content EQUAL to
# the input (same items, order, keys) so the serialization is byte-identical and
# normal results are untouched; only genuinely huge results are truncated. That is
# fine: the transcript already caps the tool result to 16k chars for the model/judge,
# and ground truth is the transcript-fed LLM judge, not the byte-gated rule oracle.
_SHRINK_BUDGET = 16_000_000        # ~16 MB of content; serializes in ~tens of ms
_SHRINK_MAX_STR = 8_000_000
_SHRINK_MAX_DEPTH = 40


def _shrink_bounded(result: Any, budget: int = _SHRINK_BUDGET,
                    max_str: int = _SHRINK_MAX_STR, max_depth: int = _SHRINK_MAX_DEPTH) -> Any:
    remaining = [budget]

    def rec(o: Any, depth: int) -> Any:
        if remaining[0] <= 0 or depth > max_depth:
            return "...[truncated: result too large to serialize]"
        if isinstance(o, str):
            if len(o) <= max_str and len(o) <= remaining[0]:
                remaining[0] -= len(o)
                return o
            take = max(0, min(max_str, remaining[0]))
            remaining[0] = 0
            return o[:take] + f"...[+{len(o) - take} chars truncated]"
        if isinstance(o, (bytes, bytearray)):
            take = max(0, min(max_str, remaining[0]))
            b = bytes(o[:take])
            remaining[0] -= len(b)
            try:
                s = b.decode("utf-8")
            except UnicodeDecodeError:
                s = b.hex()
            return s if len(o) <= len(b) else s + f"...[+{len(o) - len(b)} bytes truncated]"
        if isinstance(o, dict):
            out: dict = {}
            for k, v in o.items():
                if remaining[0] <= 0:
                    out["...[truncated]"] = f"+{len(o) - len(out)} more keys"
                    break
                remaining[0] -= 4
                out[k] = rec(v, depth + 1)
            return out
        if isinstance(o, (list, tuple)):
            out_list: list = []
            for i, v in enumerate(o):
                if remaining[0] <= 0:
                    try:
                        extra = len(o) - i
                    except Exception:
                        extra = None
                    out_list.append(f"...[+{extra} more items truncated]" if extra is not None
                                    else "...[more items truncated]")
                    break
                remaining[0] -= 2
                out_list.append(rec(v, depth + 1))
            return out_list
        remaining[0] -= 16
        return o

    try:
        return rec(result, 0)
    except Exception:
        # Never let shrinking break serialization — fall back to the raw object and
        # let _stringify's own orjson->stdlib->repr chain handle it.
        return result


def _stringify(result: Any) -> str:
    """Serialise a tool's return value for the LLM to read."""
    import json
    if result is None:
        return "null"
    if isinstance(result, str):
        return result
    if isinstance(result, (bytes, bytearray)):
        try:
            return result.decode("utf-8")
        except UnicodeDecodeError:
            return result.hex()
    result = _shrink_bounded(result)  # bound giant dict/list results BEFORE encoding
    # Hot path: dominates sim CPU on large tool results (stdlib json.dumps with a
    # per-object default= callback blocks the event loop and starves the GPU).
    # orjson is a C encoder with native datetime/UUID handling — same data, much
    # faster. Fall back to stdlib json (then repr) when orjson is absent or on
    # edge cases it rejects (e.g. NaN/Inf), so behaviour is otherwise unchanged.
    if _orjson is not None:
        try:
            return _orjson.dumps(
                result, default=str, option=_orjson.OPT_NON_STR_KEYS
            ).decode("utf-8")
        except Exception:
            pass
    try:
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception:
        return repr(result)
