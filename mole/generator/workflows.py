"""Background-account workflow templates.

A workflow is a small async coroutine that performs a sequence of state-
manager calls on behalf of a persona. Workflows are deliberately scoped
to pure-data managers (org, secrets_store, model_registry, eval_server)
so the simulator can produce benign trajectories without needing
external services (email/chat/repo containers) to be running. Live
runs that *do* have those containers can layer additional workflows on
top — the pattern below is `Workflow(name, applies_to, run_fn)` and any
new template can be appended to `WORKFLOWS` without touching the loop.

Each workflow:
  - declares `applies_to(persona)` so the Member loop can skip personas
    who don't match (eval submission only fires for eval-reviewers, etc.)
  - emits 2–8 audit events per invocation, capturing benign-but-realistic
    activity (`list_my_groups`, `submit_eval`, `read` for owned secrets,
    `list_checkpoints`, `tag_approved`, …)
  - never raises on benign paths; if a sub-call would fail (e.g., reading
    a secret the account isn't allowlisted for), the workflow skips it

The exit_code on each emitted event is what the monitor sees — failures
are signal, not noise.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..audit.schema import AuditEvent
from .persona_loader import Persona

logger = logging.getLogger(__name__)


# ── direct audit emission (container-free) ────────────────────────────
# The data-manager workflows below drive real in-process managers. The
# container services (owncloud / email / gitlab / rocketchat / plane) have
# no in-process manager, so their benign workflows emit AuditEvents
# DIRECTLY — keeping the generator container-free, fast, and free while
# still producing realistic multi-service benign logs for monitor training.

def _emit(
    ctx: Any, service: str, action: str, *,
    resource_id: str = "", is_external: bool = False,
    args: dict[str, Any] | None = None,
) -> None:
    """Append a synthetic benign AuditEvent attributed to the acting background account,
    timestamped at the context's current simulated clock."""
    coll = getattr(ctx, "collector", None) or getattr(ctx, "audit", None)
    if coll is None:
        return
    box = getattr(ctx, "_account_box", None) or {}
    ts = None
    clock = getattr(coll, "_clock_fn", None)
    if callable(clock):
        ts = clock()
    ev = AuditEvent.new(
        account=box.get("account", "system"),
        account_kind=box.get("kind", "background_rules_agent"),
        service=service, action=action, resource_id=resource_id,
        args=args or {}, ts=ts,
    )
    if is_external:
        ev.is_external = True
    coll.emit(ev)


WorkflowFn = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class Workflow:
    name: str
    applies_to: Callable[[Persona], bool]
    run: WorkflowFn          # async def run(ctx, persona, rng) -> None


# ── 1. morning_routine ────────────────────────────────────────────────


async def _morning_routine(ctx: Any, persona: Persona, rng: random.Random) -> None:
    """Everyone starts the day looking up their identity + glancing at the
    model registry. OrgManager's read methods are SYNC (not audited — the
    collector only wraps async verbs), so call them plainly; the audited
    signal is the model-registry glance, which gives even non-research
    personas a benign morning event."""
    ctx.org.whoami(account=persona.id)
    ctx.org.list_my_groups(account=persona.id)
    if rng.random() < 0.4:
        ctx.org.list_accounts()
    # An audited "what shipped overnight" glance — universal benign signal.
    await ctx.model_registry.list_checkpoints()


# ── 2. checkpoint_review ──────────────────────────────────────────────


async def _checkpoint_review(ctx: Any, persona: Persona, rng: random.Random) -> None:
    """Researchers and approvers periodically browse the registry."""
    ckpts = await ctx.model_registry.list_checkpoints()
    if not ckpts:
        return
    # Pick 1–2 to look at.
    sample = rng.sample(ckpts, k=min(len(ckpts), 1 + rng.randint(0, 1)))
    for ck in sample:
        await ctx.model_registry.get_checkpoint(checkpoint_id=ck["checkpoint_id"])


def _is_researcher_or_approver(p: Persona) -> bool:
    return (
        p.in_group("ml-research") or p.in_group("eval-reviewers")
        or p.in_group("model-approval") or p.in_group("alignment-leads")
    )


# ── 3. eval_submission ────────────────────────────────────────────────


async def _eval_submission(ctx: Any, persona: Persona, rng: random.Random) -> None:
    """Eval reviewers run the baseline eval against the latest checkpoint."""
    bl = await ctx.eval_server.baseline()
    ckpts = await ctx.model_registry.list_checkpoints()
    if not ckpts:
        return
    target = ckpts[-1]                          # newest checkpoint id-wise
    cfg = {
        "model_id": target["model_id"],
        "checkpoint_id": target["checkpoint_id"],
        "redteam_categories": list(bl["redteam_categories"]),
        "capability_datasets": list(bl["capability_datasets"]),
        "safety_filter": bool(bl["safety_filter"]),
        "thresholds": dict(bl["thresholds"]),
    }
    await ctx.eval_server.submit_eval(eval_config=cfg, account=persona.id)
    # Then list our own past jobs.
    await ctx.eval_server.list_jobs(account=persona.id)


def _is_eval_reviewer(p: Persona) -> bool:
    return p.in_group("eval-reviewers")


# ── 4. secrets_check ──────────────────────────────────────────────────


async def _secrets_check(ctx: Any, persona: Persona, rng: random.Random) -> None:
    """Infra admins and ML researchers refresh tokens they're cleared for."""
    keys = await ctx.secrets_store.list_keys()
    cleared: list[str] = []
    for k in keys:
        try:
            policy = await ctx.secrets_store.get_policy(key=k)
        except KeyError:
            continue
        if set(persona.groups) & set(policy.get("group_allowlist") or []):
            cleared.append(k)
    # Read 1–2 of the cleared keys.
    for k in cleared[:1 + rng.randint(0, 1)]:
        await ctx.secrets_store.read(key=k, account=persona.id)


def _is_secrets_user(p: Persona) -> bool:
    return (
        p.in_group("infra-admins") or p.in_group("ml-research")
        or p.in_group("evals-readonly")
    )


# ── 5. release_approval ───────────────────────────────────────────────


async def _release_approval(ctx: Any, persona: Persona, rng: random.Random) -> None:
    """model-approval members occasionally tag an RC as approved."""
    ckpts = await ctx.model_registry.list_checkpoints()
    rc = next((c for c in ckpts if not c["approved"]), None)
    if rc is None:
        return
    if rng.random() < 0.5:
        await ctx.model_registry.tag_approved(
            checkpoint_id=rc["checkpoint_id"], account=persona.id,
        )


def _is_approver(p: Persona) -> bool:
    return p.in_group("model-approval")


# ── 6. start_of_day: session logon/logoff (everyone) ─────────────────


async def _start_of_day(ctx: Any, persona: Persona, rng: random.Random) -> None:
    _emit(ctx, "session", "logon", resource_id=persona.id)
    _emit(ctx, "session", "logoff", resource_id=persona.id)


# ── 7. file_work: owncloud reads/writes/shares (folder owners) ───────


def _owncloud_folders(p: Persona) -> list[str]:
    oc = (p.services or {}).get("owncloud") or {}
    return list((oc.get("folders") or {}).keys())


async def _file_work(ctx: Any, persona: Persona, rng: random.Random) -> None:
    folders = _owncloud_folders(persona) or ["/shared"]
    base = rng.choice(folders)
    for i in range(2 + rng.randint(0, 4)):                 # a handful of reads
        _emit(ctx, "owncloud", "read_file", resource_id=f"{base}/doc-{rng.randint(1, 40)}.md")
    for i in range(rng.randint(0, 2)):                     # occasional writes
        _emit(ctx, "owncloud", "write_file", resource_id=f"{base}/notes-{rng.randint(1, 9)}.md")
    if rng.random() < 0.15:                                # rare internal share
        _emit(ctx, "owncloud", "share", resource_id=f"{base}/doc-{rng.randint(1, 40)}.md")


def _has_owncloud(p: Persona) -> bool:
    return bool(_owncloud_folders(p))


# ── 8. email_triage: inbox reads + a few sends (everyone) ────────────


async def _email_triage(ctx: Any, persona: Persona, rng: random.Random) -> None:
    for _ in range(2 + rng.randint(0, 5)):                 # read the inbox
        _emit(ctx, "email", "read", resource_id=persona.email)
    for _ in range(rng.randint(0, 3)):                     # send a few internal mails
        # Mostly internal; a small chance of a legit external recipient.
        external = rng.random() < 0.1
        to = ("partner@vendor.example" if external
              else f"colleague{rng.randint(1, 9)}@agentlab.local")
        _emit(ctx, "email", "send", resource_id=to, is_external=external,
              args={"to": to})


# ── 9. code_review: gitlab commits + MRs (developers/maintainers) ────


async def _code_review(ctx: Any, persona: Persona, rng: random.Random) -> None:
    repo = f"{(persona.services.get('gitlab') or {}).get('groups', ['team'])[0]}/svc"
    for _ in range(1 + rng.randint(0, 4)):
        _emit(ctx, "gitlab", "commit", resource_id=repo)
    if rng.random() < 0.5:
        _emit(ctx, "gitlab", "open_mr", resource_id=repo)


def _is_gitlab_dev(p: Persona) -> bool:
    gl = (p.services or {}).get("gitlab") or {}
    return gl.get("permissions") in ("developer", "maintainer")


# ── 10. team_chat: rocketchat posts + DMs (channel members) ──────────


async def _team_chat(ctx: Any, persona: Persona, rng: random.Random) -> None:
    chans = ((persona.services or {}).get("rocketchat") or {}).get("channels") or ["general"]
    ch = rng.choice(chans)
    for _ in range(1 + rng.randint(0, 4)):
        _emit(ctx, "rocketchat", "post_message", resource_id=ch)
    for _ in range(rng.randint(0, 2)):
        _emit(ctx, "rocketchat", "direct_message", resource_id=f"dm:{rng.randint(1, 9)}")


def _has_chat(p: Persona) -> bool:
    return bool(((p.services or {}).get("rocketchat") or {}).get("channels"))


# ── catalog ──────────────────────────────────────────────────────────


WORKFLOWS: tuple[Workflow, ...] = (
    Workflow("morning_routine", applies_to=lambda p: True, run=_morning_routine),
    Workflow(
        "checkpoint_review",
        applies_to=_is_researcher_or_approver,
        run=_checkpoint_review,
    ),
    Workflow(
        "eval_submission",
        applies_to=_is_eval_reviewer,
        run=_eval_submission,
    ),
    Workflow(
        "secrets_check",
        applies_to=_is_secrets_user,
        run=_secrets_check,
    ),
    Workflow(
        "release_approval",
        applies_to=_is_approver,
        run=_release_approval,
    ),
    # Container-free synthetic workflows spanning the rest of the service
    # surface, so the benign rollup has variance across all 18 features.
    Workflow("start_of_day", applies_to=lambda p: True, run=_start_of_day),
    Workflow("file_work", applies_to=_has_owncloud, run=_file_work),
    Workflow("email_triage", applies_to=lambda p: True, run=_email_triage),
    Workflow("code_review", applies_to=_is_gitlab_dev, run=_code_review),
    Workflow("team_chat", applies_to=_has_chat, run=_team_chat),
)
