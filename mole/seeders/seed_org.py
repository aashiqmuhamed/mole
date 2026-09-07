"""Populate live services from org_template.yaml.

Phase 1 seed scope (idempotent — re-running is safe):
  - email     : warm every account's inbox by sending one welcome email
                to themselves. GreenMail auto-creates the mailbox on
                first connection, so this stands the user up for IMAP
                reads later.
  - owncloud  : create every folder named in any account's
                services.owncloud.folders map. ACL granting is left
                to per-task setup since per-folder permissions differ
                across threats.
  - rocketchat: create every channel named in any account's
                services.rocketchat.channels list. Uses the admin
                token captured by RocketChatManager.setup(). Treats
                "name already in use" as success.
  - gitlab    : create every GitLab group named in any account's
                services.gitlab.groups list; create a user account
                per account; add each user to their listed groups.
                The lab's GitLab image ships with its own default user
                pool — we add OUR org_template.yaml personas alongside,
                not replacing.

Pure-data managers (org, secrets_store, model_registry, eval_server)
self-load from their own yaml files at manager.setup() time, so
seed_org skips them.

Per-threat project provisioning (e.g. models/llama-finetune for
threat 06) lives in the threat's task.seed(ctx) function — that
runs immediately after this common seed.

Public API:
    await seed_org(ctx)                       # uses default yaml
    await seed_org(ctx, org_yaml=Path(...))   # custom yaml
    await seed_org(ctx, only={"email"})       # restrict scope
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger(__name__)


def _default_org_yaml() -> Path:
    env = os.environ.get("ORG_TEMPLATE_PATH")
    if env:
        return Path(env)
    # seeders/seed_org.py → seeders → mole → benchmark
    return Path(__file__).resolve().parents[2] / "bootstrap" / "org_template.yaml"


async def seed_org(
    ctx: Any,
    *,
    org_yaml: str | Path | None = None,
    only: Iterable[str] | None = None,
) -> dict[str, int]:
    """Seed every live backend from `org_yaml`. Returns per-backend counts.

    `only` is a set of backend names to restrict to (e.g. {"email"}).
    Defaults to all known backends.
    """
    path = Path(org_yaml or _default_org_yaml())
    if not path.exists():
        raise FileNotFoundError(f"org template YAML not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    accounts: list[dict[str, Any]] = doc.get("accounts") or []
    targets = (
        set(only) if only is not None
        else {"email", "owncloud", "rocketchat", "gitlab"}
    )
    counts: dict[str, int] = {}

    if "email" in targets:
        counts["email"] = await _seed_email(ctx, accounts)
    if "owncloud" in targets:
        counts["owncloud"] = await _seed_owncloud(ctx, accounts)
    if "rocketchat" in targets:
        counts["rocketchat"] = await _seed_rocketchat(ctx, accounts)
    if "gitlab" in targets:
        counts["gitlab"] = await _seed_gitlab(ctx, accounts)

    logger.info("seed_org: %s", counts)
    return counts


# ── email ────────────────────────────────────────────────────────────


# Shared team mailboxes a real org routes cross-functional mail to. Seeded as real
# GreenMail mailboxes so agent mail to procurement@/finance@/etc. lands somewhere valid
# instead of black-holing (mirrors _STANDARD_CHANNELS for rocketchat).
_STANDARD_EMAIL_ALIASES = (
    "procurement", "finance", "accounts-payable", "payroll", "benefits", "hr",
    "recruiting", "security", "it-help", "it", "facilities", "legal", "privacy",
    "compliance", "vendor-management", "travel",
)


async def _seed_email(ctx: Any, employees: list[dict[str, Any]]) -> int:
    """Send a one-line welcome from admin to each account — creates
    the mailbox in GreenMail. Idempotent: a duplicate welcome is harmless."""
    mgr = getattr(ctx, "email", None)
    if mgr is None:
        logger.debug("seed_org.email: no email backend on ctx; skipping")
        return 0
    sent = 0
    for emp in employees:
        addr = emp.get("email")
        if not addr:
            continue
        try:
            await mgr.send_email(
                from_user="admin@agentlab.local",
                to=addr,
                subject="Welcome",
                body=f"Welcome to the lab, {emp.get('full_name', emp.get('id', ''))}.",
            )
            sent += 1
        except Exception as exc:                          # noqa: BLE001
            logger.warning("seed_org.email: failed for %s: %s", addr, exc)
    # Stand up the shared role mailboxes so cross-functional agent mail lands valid.
    domain = getattr(mgr, "_domain", None) or "agentlab.local"
    for alias in _STANDARD_EMAIL_ALIASES:
        addr = f"{alias}@{domain}"
        try:
            await mgr.send_email(
                from_user="admin@agentlab.local", to=addr, subject="Mailbox active",
                body=f"Shared mailbox {addr} is active.")
            sent += 1
        except Exception as exc:                          # noqa: BLE001
            logger.warning("seed_org.email: alias %s failed: %s", addr, exc)
    return sent


# ── owncloud ─────────────────────────────────────────────────────────


async def _seed_owncloud(ctx: Any, employees: list[dict[str, Any]]) -> int:
    """Ensure every folder named in any account's owncloud.folders exists."""
    mgr = getattr(ctx, "owncloud", None)
    if mgr is None:
        logger.debug("seed_org.owncloud: no owncloud backend on ctx; skipping")
        return 0
    folders: set[str] = set()
    for emp in employees:
        svc = (emp.get("services") or {}).get("owncloud") or {}
        for path in (svc.get("folders") or {}):
            folders.add(path)
    created = 0
    for path in sorted(folders):
        try:
            already = await mgr.exists(path=path)
        except Exception:                                 # noqa: BLE001
            already = False
        if already:
            continue
        try:
            await mgr.mkdir(path=path, recursive=True)
            created += 1
        except Exception as exc:                          # noqa: BLE001
            logger.warning("seed_org.owncloud: mkdir %s failed: %s", path, exc)
    return created


# ── rocketchat ───────────────────────────────────────────────────────


# Standard company-wide channels that exist in any real org regardless of who is a
# member (some stay quiet — that long tail is itself realistic). Layered on top of the
# per-persona channels so the workspace isn't just the six work channels.
_STANDARD_CHANNELS = (
    "announcements", "random", "help", "incidents", "hiring",
    "security", "social", "watercooler",
    # Functional channels agents reach for by role — a real AI lab has these, so a
    # post to #hr-ops lands in a real channel instead of 400-ing. Grounding keeps each
    # agent mostly on its own subset (services.rocketchat.channels); create_channel
    # covers anything genuinely new.
    "hr-ops", "hr-requests", "benefits-questions", "hiring-coordination", "people-ops",
    "people-managers", "managers", "legal-requests", "legal-inbox", "legal-team",
    "privacy", "privacy-compliance", "procurement", "procurement-legal", "sales-legal",
    "eval-ops", "ml-evals", "ml-training", "ml-updates",
)


async def _seed_rocketchat(ctx: Any, employees: list[dict[str, Any]]) -> int:
    """Ensure every channel named in any account's rocketchat.channels exists.

    Uses the admin token captured by RocketChatManager.setup(). If
    setup() ran with skip_admin_login=True, the admin token will be
    empty and we skip silently.
    """
    mgr = getattr(ctx, "rocketchat", None)
    if mgr is None:
        logger.debug("seed_org.rocketchat: no rocketchat backend on ctx; skipping")
        return 0
    if not getattr(mgr, "_admin_token", None):
        logger.debug("seed_org.rocketchat: admin token missing; skipping")
        return 0
    # Disable RocketChat's password-complexity policy BEFORE creating users.
    # Our convention is password == username (e.g. "bob.li"), which the default
    # policy rejects with HTTP 400 (too short / no uppercase / digit / symbol),
    # making every seeded login — and thus the chat-based announce/changelog/
    # notify oracles — unreachable. One admin settings flip keeps the simple
    # convention working.
    for setting, value in (
        ("Accounts_Password_Policy_Enabled", False),
        ("Accounts_RegistrationForm", "Disabled"),     # admin-only user creation
        # RocketChat enables email 2FA by default, which 401s every login with
        # "totp-required" — disable it so seeded password==username logins work.
        ("Accounts_TwoFactorAuthentication_Enabled", False),
        ("Accounts_TwoFactorAuthentication_By_Email_Enabled", False),
        # RC's REST rate limiter caps users.create at ~10 calls/min by default;
        # past ~50 accounts the loop stalls (httpx 15s timeout fires on every
        # subsequent call, none of which ever clear). Disabling the limiter
        # globally is safe in a single-session sandbox where the only caller
        # is the admin seed pass + the simulated users. Without this, seeding
        # a 150-account org hangs around account #48-58 (e.g. yuki.v49).
        ("API_Enable_Rate_Limiter", False),
    ):
        try:
            with mgr._client_factory(base_url=mgr._base_url, timeout=15.0) as c:
                r = c.post(f"/api/v1/settings/{setting}",
                           json={"value": value}, headers=mgr._admin_token)
                r.raise_for_status()
        except Exception as exc:                              # noqa: BLE001
            logger.warning("seed_org.rocketchat: could not set %s: %s", setting, exc)
    # Register each account as a RocketChat user (password == username),
    # which is the credential RocketChatManager._token_for assumes. Without
    # this the agent / background accounts get 401 on login and can't post — the
    # team-notified / changelog / announce oracles become unreachable even
    # for a perfect agent. (Found via the oracle-reachability probe.)
    created_users = 0
    for emp in employees:
        svc = (emp.get("services") or {}).get("rocketchat") or {}
        username = svc.get("username") or emp.get("id")
        if not username:
            continue
        payload = {
            "username": username,
            "name": emp.get("full_name") or username,
            "email": emp.get("email") or f"{username}@agentlab.local",
            "password": username,                 # matches _token_for(login(user, user))
            "verified": True,
            "requirePasswordChange": False,
            "joinDefaultChannels": False,
        }
        try:
            with mgr._client_factory(base_url=mgr._base_url, timeout=15.0) as c:
                resp = c.post("/api/v1/users.create", json=payload, headers=mgr._admin_token)
                resp.raise_for_status()
            created_users += 1
        except Exception as exc:                          # noqa: BLE001
            msg = str(exc).lower()
            if not ("already in use" in msg or "already-in-use" in msg or "duplicate" in msg):
                logger.warning("seed_org.rocketchat: create user %s failed: %s", username, exc)
                continue
            # User already exists (e.g. persistent volume, or a prior seed with a
            # different password) — RESET its password to the convention so
            # _token_for(login(user, user)) works. Otherwise stale users 401.
            try:
                with mgr._client_factory(base_url=mgr._base_url, timeout=15.0) as c:
                    info = c.get("/api/v1/users.info", params={"username": username},
                                 headers=mgr._admin_token)
                    info.raise_for_status()
                    uid = ((info.json() or {}).get("user") or {}).get("_id")
                    if uid:
                        upd = c.post("/api/v1/users.update",
                                     json={"userId": uid,
                                           "data": {"password": username,
                                                    "requirePasswordChange": False}},
                                     headers=mgr._admin_token)
                        upd.raise_for_status()
                        created_users += 1
            except Exception as exc2:                     # noqa: BLE001
                logger.warning("seed_org.rocketchat: reset password for %s failed: %s",
                               username, exc2)

    channels: set[str] = set(_STANDARD_CHANNELS)
    for emp in employees:
        svc = (emp.get("services") or {}).get("rocketchat") or {}
        for ch in (svc.get("channels") or []):
            channels.add(ch)
    existing: set[str] = set()
    try:
        for ch in await mgr.list_channels():
            existing.add(ch.get("name") or "")
    except Exception as exc:                              # noqa: BLE001
        logger.warning("seed_org.rocketchat: list_channels failed: %s", exc)

    created = 0
    for name in sorted(channels - existing):
        try:
            with mgr._client_factory(base_url=mgr._base_url, timeout=15.0) as c:
                resp = c.post(
                    "/api/v1/channels.create",
                    json={"name": name},
                    headers=mgr._admin_token,
                )
                resp.raise_for_status()
            created += 1
        except Exception as exc:                          # noqa: BLE001
            msg = str(exc).lower()
            if "name-invalid-or-already-in-use" in msg or "already" in msg:
                continue
            logger.warning("seed_org.rocketchat: create %s failed: %s", name, exc)
    return created + created_users


# ── gitlab ───────────────────────────────────────────────────────────


async def _seed_gitlab(ctx: Any, employees: list[dict[str, Any]]) -> int:
    """Create the union of GitLab groups + per-account user accounts +
    group memberships from org_template.yaml. Returns total entities
    created (groups + users + memberships).

    Idempotent: GitLabManager.create_user / create_group / add_group_member
    return existing entities when re-called.
    """
    mgr = getattr(ctx, "gitlab", None)
    if mgr is None:
        logger.debug("seed_org.gitlab: no gitlab backend on ctx; skipping")
        return 0

    # Collect groups from per-account gitlab.groups lists.
    group_names: set[str] = set()
    for emp in employees:
        svc = (emp.get("services") or {}).get("gitlab") or {}
        for g in (svc.get("groups") or []):
            if isinstance(g, str) and g:
                group_names.add(g)

    created = 0
    for name in sorted(group_names):
        try:
            await mgr.create_group(path=name, name=name)
            created += 1
        except Exception as exc:                          # noqa: BLE001
            logger.warning("seed_org.gitlab: create_group %s failed: %s", name, exc)

    # Create users.
    for emp in employees:
        svc = (emp.get("services") or {}).get("gitlab") or {}
        username = svc.get("username")
        if not username:
            continue
        try:
            await mgr.create_user(
                username=username,
                email=emp.get("email") or f"{username}@agentlab.local",
                name=emp.get("full_name") or username,
            )
            created += 1
        except Exception as exc:                          # noqa: BLE001
            logger.warning(
                "seed_org.gitlab: create_user %s failed: %s", username, exc,
            )

    # Add memberships.
    _PERM_ACCESS = {
        "guest": 10, "reporter": 20, "developer": 30,
        "maintainer": 40, "owner": 50,
    }
    for emp in employees:
        svc = (emp.get("services") or {}).get("gitlab") or {}
        username = svc.get("username")
        if not username:
            continue
        access = _PERM_ACCESS.get(
            str(svc.get("permissions") or "developer").lower(), 30,
        )
        for g in (svc.get("groups") or []):
            try:
                await mgr.add_group_member(
                    group=g, username=username, access_level=access,
                )
                created += 1
            except Exception as exc:                      # noqa: BLE001
                logger.warning(
                    "seed_org.gitlab: add %s to %s failed: %s",
                    username, g, exc,
                )
    return created
