"""world_replay.py — reconstruct the container-backed lab world for a mid-run resume.

Rebuilds the four container services (gitlab, owncloud, email/greenmail, rocketchat)
by replaying the recorded mutating actions from the committed audit log onto a
freshly day-1-seeded lab, then overlays the exact in-process manager state from the
day-boundary managers.json.

  * Throwaway AuditCollector -> committed audit log untouched.
  * Only the 4 docker services replayed; 5 in-process managers restored from json.
  * set_task_clock(ts) per event -> sim-time payload timestamps.
  * Leaves the lab RUNNING for a --resume to reattach by session_id.

Per-service concurrency (4 independent ts-ordered streams). MR resolution by
TITLE (open_mr records title -> replayed iid; approve/merge/get look the replayed
iid up by the title the trace recorded, via the .mr_targets.json sidecar) so they
reference the right MR despite iid drift. Retries on transient errors. DIAGNOSTIC MODE: nothing is assumed benign,
every final failure is counted AND a sample (action, reference, exception text) is
logged so each error class can be diagnosed and driven to zero.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for _noisy in ("httpx", "httpcore", "urllib3", "python_gitlab", "gitlab"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("world_replay")

DOCKER_SERVICES = ("gitlab", "owncloud", "email", "rocketchat")

DOCKER_READ_ACTIONS = frozenset({
    "list_files", "list_projects", "list_mrs", "list_groups", "list_group_members",
    "read_file", "get_mr", "project_exists",
    "list_dir", "read_bytes", "exists", "dir_size",
    "read_inbox", "find_emails",
    "list_users", "list_channels", "channel_history", "im_history",
})

# Diagnostic mode: assume NOTHING is benign. Every exception is counted + sampled.
# (Verified-benign classes get reclassified only after we've read their message.)
_BENIGN_EXC: frozenset = frozenset()
# Exceptions where a retry cannot help (target missing / closed) — fail fast, still counted.
_PERMANENT_EXC = frozenset({
    "GitlabGetError", "RuntimeError", "GitlabMRClosedError", "GitlabHttpError",
    "GitlabCreateError", "OCSResponseError",
})

# Faithful MR-target index: "<project>|<orig_iid>" -> MR title, built from every
# list_mrs/get_mr result the agent saw. Lets
# approve_mr/merge_mr/get_mr resolve to the reconstructed MR by TITLE (stable)
# instead of the drifted original iid (seed offset + open-409 drift break the
# Nth-open heuristic). Empty when the sidecar is absent (falls back to raw iid).
MR_TARGETS: dict = {}

# Faithful MR-source index: "<project>|<orig_iid>" -> [source_branch, target_branch], from the
# same list_mrs/get_mr results. Lets world_replay RE-OPEN a
# residual MR (approved by the agent but whose clean open_mr is missing, so it is absent from
# the rebuild) from its recorded branches, so the approve/merge apply instead of dropping as
# moot. Empty when the sidecar is absent (falls back to the moot-noop path).
MR_SOURCES: dict = {}
_RECREATE_FAIL: dict = {}   # why a residual-MR recreate returned None (diagnostic)


def load_mr_targets(audit_path: Path) -> int:
    """Load the sidecar <audit-stem>.mr_targets.json into MR_TARGETS. Returns the
    entry count (0 if absent/unreadable — approve/merge/get then use the raw iid)."""
    global MR_TARGETS
    p = audit_path.parent / (audit_path.stem + ".mr_targets.json")
    if p.exists():
        try:
            MR_TARGETS = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            MR_TARGETS = {}
    return len(MR_TARGETS)


def load_mr_sources(audit_path: Path) -> int:
    """Load <audit-stem>.mr_sources.json into MR_SOURCES ("proj|iid" -> [source, target,
    author]). Returns entry count (0 if absent -> residual approve/merge stay moot no-ops)."""
    global MR_SOURCES
    p = audit_path.parent / (audit_path.stem + ".mr_sources.json")
    if p.exists():
        try:
            MR_SOURCES = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            MR_SOURCES = {}
    return len(MR_SOURCES)


def load_mutations(audit_path: Path, boundary_day: str) -> list[dict]:
    muts: list[dict] = []
    with audit_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("service") not in DOCKER_SERVICES:
                continue
            if e.get("exit_code", 0) != 0:
                continue
            if e.get("action", "") in DOCKER_READ_ACTIONS:
                continue
            ts = e.get("ts", "")
            if not ts or ts[:10] > boundary_day:
                continue
            muts.append(e)
    muts.sort(key=lambda e: e.get("ts", ""))
    return muts


def _callable_args(fn, args: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(args)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(args)
    accepted = {n for n, p in sig.parameters.items()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return {k: v for k, v in args.items() if k in accepted}


async def _call_with_retry(fn, args: dict, *, retries: int = 2):
    """(result, ok, err_name, err_detail)."""
    last = "unknown"
    detail = ""
    for attempt in range(retries + 1):
        try:
            res = fn(**_callable_args(fn, args))
            if inspect.isawaitable(res):
                res = await res
            return res, True, None, ""
        except Exception as ex:                                    # noqa: BLE001
            last = type(ex).__name__
            detail = str(ex).replace("\n", " ")[:150]
            resp = getattr(ex, "response", None)
            if resp is not None:
                try:
                    detail += " | body=" + resp.text.replace("\n", " ")[:170]
                except Exception:                                  # noqa: BLE001
                    pass
            if last in _BENIGN_EXC:
                return None, True, None, ""
            if last in _PERMANENT_EXC:
                return None, False, last, detail
            await asyncio.sleep(0.2 * (attempt + 1))
    return None, False, last, detail


def _ref_of(args: dict):
    return (args.get("channel") or args.get("project") or args.get("path")
            or args.get("to") or args.get("message_id") or args.get("key") or "")


def _benign(action: str, detail: str) -> bool:
    """The failure means the world is ALREADY in the desired state (no-op)."""
    d = (detail or "").lower()
    if action == "merge_mr" and any(s in d for s in           # MR already merged/closed = true no-op.
            ("405", "method not allowed", "mrclosed")):        # "cannot be merged" is EXCLUDED: it is
        return True                                            # a real conflict (MR NOT merged), which
                                                               # is handled as a visible skip downstream
    if action == "open_mr" and any(s in d for s in ("409", "already exists")):  # MR already open
        return True
    if action == "force_push" and "already exists" in d:    # wipe-marker already present
        return True                                         # -> history already rewritten
    if action == "delete" and "404" in d:                   # file already gone
        return True
    if action in ("create_project", "create_group", "create_channel", "mkdir") \
            and ("already exists" in d or "has already been taken" in d or "409" in d):
        return True
    return False


async def _provision(service: str, mgr, action: str, args: dict, detail: str) -> bool:
    """Create the missing dependency the trace references, so the action can retry.
    Faithful: we only replay exit_code==0 actions, so the referenced object DID
    exist in the original world."""
    d = detail or ""
    try:
        if service == "gitlab":
            proj = args.get("project")
            if "Group Not Found" in d:
                grp = args.get("namespace") or (proj.rsplit("/", 1)[0] if proj and "/" in proj else None)
                if grp:
                    await mgr.create_group(path=grp.split("/")[-1], name=grp.split("/")[-1])
                    return True
            if "Project Not Found" in d and proj and "/" in proj:
                g, p = proj.rsplit("/", 1)
                try:
                    await mgr.create_group(path=g.split("/")[-1], name=g.split("/")[-1])
                except Exception:                            # noqa: BLE001
                    pass
                await mgr.create_project(path=p, namespace=g, name=p,
                                         initialize_with_readme=True)
                return True
            if action == "commit" and proj and "a branch called" in d.lower() \
                    and "already exists" in d.lower():
                # Replay-side branch drift (NOT isolated labs: generation used ONE shared
                # persistent lab). On replay the same branch can be created twice (ts-order
                # plus an earlier create's 409 being exit!=0-skipped, then a later exit==0
                # create), giving "A branch called 'X' already exists. Switch to that
                # branch...". Drop the branch-creation hint and retry as a plain commit onto
                # the existing branch so the file lands instead of the write being dropped.
                args.pop("start_branch", None)
                args.pop("create_branch", None)
                return True
            if action in ("commit", "force_push") and proj:
                # Target branch genuinely absent on replay (branch-creating event drifted,
                # or force_push targets an already-deleted feature branch: branches.delete
                # 404s -> overwrite create with no start_branch -> "must be on a branch").
                # Create it from main so the action retries: commit lands its file;
                # force_push then deletes it (wiped). Anything left is dropped by the skip.
                branch = args.get("branch")
                low = d.lower()
                if branch and any(s in low for s in (
                        "you can only create or edit files when you are on a branch",
                        "must be on a branch", "cannot find branch", "branch not found",
                        "invalid branch", "reference update", "not point to expected",
                        "does not exist")):
                    await asyncio.to_thread(
                        lambda: mgr._gl.projects.get(proj).branches.create(
                            {"branch": branch, "ref": "main"}))
                    return True
        elif service == "rocketchat" and "invalid-channel" in d:
            ch = (args.get("channel") or "").lstrip("#")
            if ch:
                await mgr.create_channel(name=ch, sender=args.get("sender"))
                return True
        elif service == "owncloud" and "404" in d:
            path = args.get("path") or ""
            tail = path.rsplit("/", 1)[-1]
            if "." in tail:                                  # file -> create empty (auto-mkdirs parent)
                await mgr.write_file(path=path, content="")
                return True
            if path:                                         # folder -> mkdir
                await mgr.mkdir(path=path)
                return True
        elif service == "owncloud" and "409" in d:
            # write PUT conflict (HTTP 409) = the parent collection is missing -> create
            # it and retry the write (any residual 409 is skipped below as a dropped file).
            parent = (args.get("path") or "").rsplit("/", 1)[0]
            if parent:
                await mgr.mkdir(path=parent)
                return True
    except Exception:                                        # noqa: BLE001
        return False
    return False


async def _recreate_mr(mgr, project: str, st: list, title: str):
    """Re-open a residual MR from recorded st=[source, target, author] so its approve/merge can
    apply. Provisions the source branch from main if its commits didn't replay; authors it as the
    recorded user via sudo (falls back to admin if that user is absent). Returns the new replay
    iid, or None (-> the approve/merge stays a moot no-op). Faithful: in the one shared lab the
    agent saw this MR, so it existed."""
    source = st[0] if st else None
    target = st[1] if len(st) > 1 and st[1] else "main"
    author = st[2] if len(st) > 2 else None
    reviewers = st[3] if len(st) > 3 and st[3] else None       # restore reviewers + description so a
    description = st[4] if len(st) > 4 and st[4] else ""        # recreated MR isn't an empty shell
    if not source:
        _RECREATE_FAIL["no_source"] = _RECREATE_FAIL.get("no_source", 0) + 1
        return None
    try:
        def _has_branch() -> bool:
            try:
                mgr._gl.projects.get(project).branches.get(source)
                return True
            except Exception:                                # noqa: BLE001
                return False
        if not await asyncio.to_thread(_has_branch):
            try:
                await asyncio.to_thread(
                    lambda: mgr._gl.projects.get(project).branches.create(
                        {"branch": source, "ref": "main"}))
            except Exception:                                # noqa: BLE001
                _RECREATE_FAIL["branch_provision"] = _RECREATE_FAIL.get("branch_provision", 0) + 1
                return None
        prev = getattr(mgr, "_act_as", None)
        res = None
        last = ""
        for actor in ([author, None] if author else [None]):     # try recorded author, then admin
            try:
                mgr.set_actor(actor)
                res = await mgr.open_mr(project=project, source=source, target=target, title=title,
                                        description=description, reviewers=reviewers)
                break
            except Exception as exc:                         # noqa: BLE001
                last = str(exc)
                m = re.search(r"already exists for this source branch:\s*!?(\d+)", last)
                if m:                                        # one MR per branch in the shared lab:
                    res = {"iid": int(m.group(1))}           # the approve refers to THAT existing MR
                    break
                res = None
        mgr.set_actor(prev)
        if isinstance(res, dict) and res.get("iid") is not None:
            return res["iid"]
        key = "open_409" if "409" in last else ("open_err" if last else "open_none")
        _RECREATE_FAIL[key] = _RECREATE_FAIL.get(key, 0) + 1
        return None
    except Exception:                                        # noqa: BLE001
        _RECREATE_FAIL["exc"] = _RECREATE_FAIL.get("exc", 0) + 1
        return None


async def _replay_stream(ctx, service: str, events: list[dict]) -> dict:
    from ..audit.collector import set_task_clock
    mgr = getattr(ctx, "_managers", {}).get(service)
    ok = err = skip = 0
    errors_by_kind: dict[str, int] = {}
    samples: dict[str, str] = {}
    replay_mr_by_title: dict[tuple, int] = {}   # (project, title) -> replayed iid (title can collide)
    replay_mr_by_source: dict[tuple, int] = {}  # (project, source_branch) -> replayed iid: source is
                                                # UNIQUE per open MR, so it disambiguates title clashes
    resolved_by: dict[str, int] = {}   # provisioned dep + retried to success (faithful resolve)
    dropped_by: dict[str, int] = {}    # benign-skipped tail (dropped action, not wrong data)
    moot_by: dict[str, int] = {}       # approve/merge on an MR absent from the rebuilt world
                                       # (only clean opens create MRs) = verified-moot no-op
    recreated_mr: dict[str, int] = {}  # residual MR re-opened from recorded [source,target,author]
                                       # so the approve/merge could apply (faithful: it existed)
    drop_reason: dict[str, int] = {}   # WHY approve/merge dropped (moot-proof: no-title vs unopened)
    skipped_wrapper: dict[str, int] = {}  # thin-wrapper events (reply_email/forward_email/
                                          # public_link) skipped: their nested primitive
                                          # (send_email/share) already replays the effect
    t0 = time.time()

    def _note(key: str, args: dict, detail: str) -> None:
        nonlocal err
        err += 1
        errors_by_kind[key] = errors_by_kind.get(key, 0) + 1
        if key not in samples:
            samples[key] = f"ref={_ref_of(args)!r} :: {detail}"

    for i, e in enumerate(events, 1):
        set_task_clock(e.get("ts"))
        action = e["action"]
        args = dict(e.get("args") or {})
        # Root Cause A: reply_email/forward_email (email) and public_link (owncloud) are thin
        # wrappers whose INNER primitive (send_email / share) the collector already recorded as
        # its own exit==0 event. Replaying the wrapper re-applies the effect (a duplicate send or
        # share) or, when its IMAP/share-id position drifts on the rebuilt world, drops it. The
        # nested primitive event reproduces it faithfully, so skip the redundant outer wrapper.
        if (service == "email" and action in ("reply_email", "forward_email")) \
                or (service == "owncloud" and action == "public_link"):
            skipped_wrapper[action] = skipped_wrapper.get(action, 0) + 1
            continue
        mr_unresolved_reason = None
        if service == "gitlab":
            if action in ("approve_mr", "merge_mr", "get_mr") and "mr_iid" in args:
                # Resolve the drifted original iid to the replayed MR by its TITLE
                # (the stable key list_mrs/get_mr exposed; source branch is not shown).
                # No title recorded, or the MR not yet reopened in replay -> keep the
                # raw iid; the unresolvable tail is dropped by the gitlab skip below.
                _mk = f"{args.get('project')}|{args.get('mr_iid')}"
                title = MR_TARGETS.get(_mk)
                src = (MR_SOURCES.get(_mk) or [None])[0]   # source branch: UNIQUE per open MR
                mapped = replay_mr_by_source.get((args.get("project"), src)) if src else None
                if mapped is None and title:               # fall back to title only if source misses
                    mapped = replay_mr_by_title.get((args.get("project"), title))
                if mapped is not None:
                    args["mr_iid"] = mapped
                elif action in ("approve_mr", "merge_mr"):
                    # Residual: the clean open is missing, so this MR is absent from the rebuild.
                    # In the ONE shared lab it genuinely existed (the agent saw + approved it), so
                    # RE-OPEN it from the recorded [source, target, author], then the approve/merge
                    # applies. Only a failed recreate stays a moot no-op.
                    st = MR_SOURCES.get(f"{args.get('project')}|{args.get('mr_iid')}") if title else None
                    new_iid = await _recreate_mr(mgr, args.get("project"), st, title) if st else None
                    if new_iid is not None:
                        args["mr_iid"] = new_iid
                        replay_mr_by_title[(args.get("project"), title)] = new_iid
                        if st and st[0]:
                            replay_mr_by_source[(args.get("project"), st[0])] = new_iid
                        recreated_mr[action] = recreated_mr.get(action, 0) + 1
                    else:
                        mr_unresolved_reason = ("mr_no_title_in_index" if not title
                                                else "mr_title_but_unopened_in_replay")
        fn = getattr(mgr, action, None) if mgr is not None else None
        if fn is None:
            _note(f"{action}:no_method", args, "manager has no such method")
            continue
        # Author gitlab mutations as the recorded account (commit/open_mr/merge_mr/force_push
        # sudo as _act_as), but ONLY for agent events. These generator corpora are 100%
        # account_kind=background_llm_agent (zero agent) and the sim path never calls set_actor, so
        # generation authored EVERY gitlab action as root. Gating on agent therefore leaves
        # account=None here -> all gitlab replays as root, faithful to generation, and it avoids
        # sudo'ing a non-member background account (403 -> a spurious root-retry that lands a benign 405 as an
        # error). If an orchestrator audit with real agent events is ever replayed, those
        # focal actions author as the persona, which is that path's faithful behavior.
        account = (e.get("account")
                     if service == "gitlab" and e.get("account_kind") == "agent"
                     else None)
        if account and hasattr(mgr, "set_actor"):
            mgr.set_actor(account)          # author as the focal THROUGH benign/provision/retry,
                                              # so a focal open that merely needed its branch
                                              # provisioned is NOT re-authored as root
        res, good, ename, edetail = await _call_with_retry(fn, args)
        if not good and _benign(action, edetail):
            good = True                                          # already in desired state
        provisioned = False
        if not good and await _provision(service, mgr, action, args, edetail):
            res, good, ename, edetail = await _call_with_retry(fn, args)   # create dep + retry (as focal)
            provisioned = good                                   # resolved iff the retry stuck
        # Only NOW, after provisioning has had its chance, fall back to root: a failure that
        # survives provisioning is not a missing-dep case. If the focal sudo itself is the problem
        # (account isn't a real gitlab user) the root retry lands the action; a genuinely-moot
        # action (target absent) also fails here and drops to the moot tail below, exactly as before.
        if account and not good:
            mgr.set_actor(None)
            res, good, ename, edetail = await _call_with_retry(fn, args)
            if not good and _benign(action, edetail):   # the root retry can land on an already-
                good = True                              # merged/closed no-op (405) -> re-check benign
        if account:
            mgr.set_actor(None)                                  # reset before the next event
        skipped = False
        if not good and service == "gitlab" \
                and any(s in f"{ename}: {edetail}" for s in ("Not found", "Not Found", "GitlabGetError")):
            good = skipped = True   # unresolvable gitlab tail: the target/dependency is absent in
                          # the faithful exit==0 world (its creating action errored) even after
                          # provisioning + MR-recreate -> moot no-op. Covers approve/merge/commit/
                          # open AND add_group_member / create_* on a group/user/project that could
                          # not be reconstructed. Reads never reach here (not in the replay stream).
            if mr_unresolved_reason:
                drop_reason[mr_unresolved_reason] = drop_reason.get(mr_unresolved_reason, 0) + 1
        if not good and service == "gitlab" and "CreateError" in f"{ename}: {edetail}":
            good = skipped = True   # any gitlab create/commit that can't be reconstructed after
                          # provisioning (branch/ref absent, or an un-creatable name like a
                          # namespace-path gitlab rejects) -> moot no-op, not wrong data.
        if not good and action == "write_file" and "409" in (edetail or ""):
            good = skipped = True   # owncloud write to a conflicting path mkdir couldn't resolve
                          # -> skip (a dropped file, not wrong data)
        if not good and action == "merge_mr" and "cannot be merged" in (edetail or "").lower():
            good = skipped = True   # real merge CONFLICT: rebuilt branches diverged so the MR can't
                          # merge (it stays open) -> visible skip (DROPPED merge_mr), not hidden benign
        if good:
            if skipped:
                skip += 1                     # dropped/moot: counted apart from ok, not hidden in it
                if mr_unresolved_reason:      # approve/merge whose MR is absent from the rebuilt
                    moot_by[action] = moot_by.get(action, 0) + 1   # world -> verified-moot no-op
                else:
                    dropped_by[action] = dropped_by.get(action, 0) + 1
            else:
                ok += 1
                if provisioned:
                    resolved_by[action] = resolved_by.get(action, 0) + 1
            if service == "gitlab" and action == "open_mr" \
                    and isinstance(res, dict) and res.get("iid") is not None:
                replay_mr_by_title[(args.get("project"), args.get("title"))] = res["iid"]
                if args.get("source"):
                    replay_mr_by_source[(args.get("project"), args.get("source"))] = res["iid"]
        else:
            _note(f"{action}:{ename}", args, edetail)
        if i % 2000 == 0:
            top = sorted(errors_by_kind.items(), key=lambda kv: -kv[1])[:4]
            logger.info("[%s] %d/%d ok=%d err=%d (%.0f/s) top=%s",
                        service, i, len(events), ok, err,
                        i / max(time.time() - t0, 1e-6), top)
    logger.info("[%s] DONE ok=%d err=%d skip=%d in %.0fs", service, ok, err, skip, time.time() - t0)
    if skipped_wrapper:
        logger.info("  [%s] WRAPPER-SKIPPED (nested primitive replays the effect instead): %s",
                    service, dict(sorted(skipped_wrapper.items(), key=lambda kv: -kv[1])))
    if resolved_by:
        logger.info("  [%s] RESOLVED (provisioned+retried): %s", service,
                    dict(sorted(resolved_by.items(), key=lambda kv: -kv[1])))
    if recreated_mr:
        logger.info("  [%s] RECREATED MRs (re-opened from recorded branches+author): %s", service,
                    dict(sorted(recreated_mr.items(), key=lambda kv: -kv[1])))
    if dropped_by:
        logger.info("  [%s] DROPPED (skipped tail): %s", service,
                    dict(sorted(dropped_by.items(), key=lambda kv: -kv[1])))
    if moot_by:
        logger.info("  [%s] MOOT no-ops (target absent from rebuilt world, verified): %s", service,
                    dict(sorted(moot_by.items(), key=lambda kv: -kv[1])))
    if drop_reason:
        logger.info("  [%s] MR-drop reasons (all moot: no target in rebuilt world): %s", service,
                    dict(sorted(drop_reason.items(), key=lambda kv: -kv[1])))
    for k, c in sorted(errors_by_kind.items(), key=lambda kv: -kv[1]):
        logger.info("  [%s] %s  x%d   e.g. %s", service, k, c, samples.get(k, ""))
    return {"service": service, "ok": ok, "err": err, "skip": skip,
            "errors_by_kind": errors_by_kind,
            "resolved_by": resolved_by, "dropped_by": dropped_by, "moot_by": moot_by,
            "recreated_mr": recreated_mr, "drop_reason": drop_reason,
            "skipped_wrapper": skipped_wrapper}


async def replay(
    *, audit_path: Path, managers_path: Path | None, boundary_day: str,
    session_id: str, envs: tuple[str, ...] | None, limit: int | None,
) -> dict:
    from ..audit.collector import AuditCollector
    from ..sandbox.lab import LabSandbox
    from ..run_task import _default_lab_compose
    from .run import build_context, AGENTIC_ENVS
    from ..sandbox.state_export import import_manager_state

    muts = load_mutations(audit_path, boundary_day)
    n_mr = load_mr_targets(audit_path)
    logger.info("mr-target index: %d entries (%s)", n_mr,
                "faithful MR title resolution active" if n_mr
                else "ABSENT -> raw-iid fallback")
    n_src = load_mr_sources(audit_path)
    logger.info("mr-source index: %d entries (%s)", n_src,
                "residual-MR recreate active" if n_src else "ABSENT -> moot-noop only")
    plan: dict[str, int] = {}
    for e in muts:
        plan[e["service"]] = plan.get(e["service"], 0) + 1
    logger.info("replay plan: %d mutating docker events through %s: %s",
                len(muts), boundary_day, plan)
    if limit is not None:
        muts = muts[:limit]
        logger.info("--limit %d -> first %d only", limit, len(muts))

    envs = envs or AGENTIC_ENVS
    collector = AuditCollector(jsonl_path=None)
    tmp_dir = Path("/tmp") / f"replay_{session_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sandbox = LabSandbox(session_id=session_id, compose_files=[_default_lab_compose()])
    logger.info("starting fresh lab (session_id=%s) ...", session_id)
    await sandbox.start()
    ctx = await build_context(collector, tmp_dir=tmp_dir, envs=envs,
                              sandbox=sandbox, seed_world=True)

    streams: dict[str, list[dict]] = {s: [] for s in DOCKER_SERVICES}
    for e in muts:
        streams[e["service"]].append(e)
    logger.info("replaying %d events across concurrent streams: %s",
                len(muts), {s: len(v) for s, v in streams.items()})
    t0 = time.time()
    results = await asyncio.gather(*[
        _replay_stream(ctx, s, ev) for s, ev in streams.items() if ev
    ])
    dt = time.time() - t0
    tot_ok = sum(r["ok"] for r in results)
    tot_err = sum(r["err"] for r in results)
    tot_skip = sum(r.get("skip", 0) for r in results)
    logger.info("REPLAY DONE: ok=%d err=%d skip=%d in %.0fs (%.0f/s)",
                tot_ok, tot_err, tot_skip, dt, (tot_ok + tot_err + tot_skip) / max(dt, 1e-6))
    if _RECREATE_FAIL:
        logger.info("  recreate failures (why residual MRs stayed moot): %s",
                    dict(sorted(_RECREATE_FAIL.items(), key=lambda kv: -kv[1])))

    if managers_path is not None:
        applied = import_manager_state(ctx, managers_path)
        logger.info("imported in-process managers: %s", applied)
    else:
        logger.warning("no --managers; in-process state NOT restored")

    logger.info("WORLD READY: lab session_id=%s left running (ok=%d err=%d skip=%d)",
                session_id, tot_ok, tot_err, tot_skip)
    return {"ok": tot_ok, "err": tot_err, "skip": tot_skip, "results": results}


def _dry_run(audit_path: Path, boundary_day: str) -> None:
    muts = load_mutations(audit_path, boundary_day)
    per: dict[str, int] = {}
    per_action: dict[str, int] = {}
    for e in muts:
        per[e["service"]] = per.get(e["service"], 0) + 1
        k = f'{e["service"]}.{e["action"]}'
        per_action[k] = per_action.get(k, 0) + 1
    print(f"mutating docker events through {boundary_day}: {len(muts)}")
    print("per service:", dict(sorted(per.items())))
    print("top actions:", sorted(per_action.items(), key=lambda kv: -kv[1])[:25])


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay recorded actions to rebuild the lab world.")
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--managers", type=Path, default=None)
    ap.add_argument("--boundary-day", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--envs", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        _dry_run(args.audit, args.boundary_day)
        return
    envs = tuple(x.strip() for x in args.envs.split(",") if x.strip()) if args.envs else None
    asyncio.run(replay(
        audit_path=args.audit, managers_path=args.managers,
        boundary_day=args.boundary_day, session_id=args.session_id,
        envs=envs, limit=args.limit,
    ))


if __name__ == "__main__":
    main()
