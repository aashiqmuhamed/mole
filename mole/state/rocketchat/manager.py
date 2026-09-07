"""RocketChatManager — HTTP REST wrapper for the RocketChat container.

Talks to a RocketChat instance (default container exposes API on port
3000) running in the per-session sandbox. Reached from the host on
sandbox.ports[3000].

Auth model. RocketChat issues `X-Auth-Token` + `X-User-Id` per account
on successful login. We obtain one token per agent-acting-as-account
on first method call and cache it for the session. The admin account
(seeded by bootstrap/seed_org.py + the lab's docker-compose stack) doubles as a fallback
for list_users / list_channels.

Method surface (covers threats 03 priv-esc social engineering, 04
fraud chat coordination, 05 sabotage public messaging, plus all background-account
chatter):

  post_message(sender, channel, text)            → message record
  channel_history(channel, limit=50)             → [message, ...]
  list_channels()                                → [channel, ...]
  send_direct_message(sender, recipient, text)   → message record
  im_history(user, peer, limit=50)               → [message, ...]
  list_users()                                   → [user, ...]
  set_topic(sender, channel, topic)              → updated channel

NEEDS_SANDBOX defaults to True; tests override config["host"]/["port"]
and inject `_client_factory` to bypass real HTTP.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable

import httpx

from ..base import StateManager

logger = logging.getLogger(__name__)


def _stamp_sim(obj: dict[str, Any]) -> dict[str, Any]:
    """Overwrite a RocketChat message/channel's wall-clock time fields (``ts``,
    ``_updatedAt``, ``lm``) with the current simulated time so payloads agents
    read back carry sim-time. RocketChat's container clock can't be set and its
    API takes no ts on POST, so remap on the way out. No-op when no session clock
    is set (keeps the original value).
    """
    from ...audit.collector import get_current_sim_time
    sim = get_current_sim_time()
    if sim and isinstance(obj, dict):
        for _k in ("ts", "_updatedAt", "lm"):
            if _k in obj:
                obj[_k] = sim
    return obj


@StateManager.register("rocketchat")
class RocketChatManager(StateManager):
    DEFAULT_PORT = 3000
    DEFAULT_HOST = "127.0.0.1"
    # "admin" is in RocketChat 5.x's reserved-username blocklist, so the
    # initial-admin insert silently fails. Compose seeds "rcadmin" instead.
    ADMIN_USER = "rcadmin"
    ADMIN_PASS = "TheAgentCompany"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._host: str = ""
        self._port: int = 0
        self._base_url: str = ""
        self._admin_token: dict[str, str] = {}        # {X-Auth-Token, X-User-Id}
        self._user_tokens: dict[str, dict[str, str]] = {}
        # httpx.Client factory kept on the instance so tests can swap fakes.
        self._client_factory: Callable[..., httpx.Client] = httpx.Client

    async def setup(self, *, sandbox) -> None:
        host = (
            self.config.get("host")
            or os.environ.get("ROCKETCHAT_HOST")
            or self.DEFAULT_HOST
        )
        port = (
            self.config.get("port")
            or int(os.environ.get("ROCKETCHAT_PORT", "0") or 0)
            or (sandbox.ports or {}).get(self.DEFAULT_PORT)
        )
        if port is None:
            raise RuntimeError(
                f"rocketchat backend needs port {self.DEFAULT_PORT} in "
                f"sandbox.ports or env"
            )
        self._host = host
        self._port = int(port)
        self._base_url = f"http://{self._host}:{self._port}"
        admin_user = self.config.get("admin_user") or os.environ.get(
            "ROCKETCHAT_ADMIN_USER", self.ADMIN_USER,
        )
        admin_pass = self.config.get("admin_pass") or os.environ.get(
            "ROCKETCHAT_ADMIN_PASS", self.ADMIN_PASS,
        )
        if self.config.get("skip_admin_login"):
            logger.info("rocketchat manager up at %s (admin login skipped)", self._base_url)
            return
        # Admin login during setup() races RocketChat's admin-user
        # materialisation AND its API warmup (which can disconnect mid-handshake
        # for a while after the container reports healthy). Give it a generous
        # budget — ~90s — so a slow boot doesn't abort the whole sim; the default
        # 8×3s≈24s lost the race under load. Once RC is ready this returns at once.
        self._admin_token = await asyncio.to_thread(
            self._login, admin_user, admin_pass,
            retry_on_401=True, max_attempts=30, retry_sleep_s=3.0,
        )
        logger.info("rocketchat manager up at %s; admin logged in", self._base_url)

    async def cleanup(self) -> None:
        # No long-lived sessions; per-call clients are short-lived.
        self._admin_token = {}
        self._user_tokens = {}

    async def reset(self, *, sandbox) -> None:
        """Scrub channels/messages/users the agent could have created.

        Strategy: hit the admin REST API to:
          1. Delete every channel that's not in the seeded list (channels
             the agent created during the run).
          2. Clear messages in seeded channels (chat.cleanRoomMessages).
          3. Drop the user-token cache so per-user logins re-authenticate
             cleanly on the next run.

        The RocketChat admin user itself stays — its credentials were
        baked into the image at install time. We never delete it.

        Failure mode: if a delete fails (e.g., a channel is mid-write),
        we log and continue — Sandbox.reset_service('rocketchat') is
        the heavier escape hatch the pool can fall back to.
        """
        seeded_channels = set(
            self.config.get("seeded_channels") or [
                "general", "alignment", "incidents", "infra",
                "capabilities", "releases", "ops",
            ]
        )
        if not self._admin_token:
            logger.debug("rocketchat.reset: no admin token; skipping API scrub")
        else:
            try:
                existing = await self.list_channels()
            except Exception as exc:                       # noqa: BLE001
                logger.warning("rocketchat.reset: list_channels failed: %s", exc)
                existing = []
            for ch in existing:
                name = ch.get("name") or ""
                room_id = ch.get("_id") or ""
                if not name:
                    continue
                if name in seeded_channels:
                    # 1a — clear message history in seeded channels (keep
                    # the channel itself)
                    if room_id:
                        try:
                            await asyncio.to_thread(
                                self._post_json, "/api/v1/rooms.cleanHistory",
                                self._admin_token, {
                                    "roomId": room_id,
                                    "latest": "9999-12-31T23:59:59.999Z",
                                    "oldest": "0001-01-01T00:00:00.000Z",
                                },
                            )
                        except Exception as exc:           # noqa: BLE001
                            logger.warning(
                                "rocketchat.reset: cleanHistory %s failed: %s",
                                name, exc,
                            )
                    continue
                # 1b — agent-created channel: delete entirely
                try:
                    await asyncio.to_thread(
                        self._post_json, "/api/v1/channels.delete",
                        self._admin_token, {"roomName": name},
                    )
                except Exception as exc:                   # noqa: BLE001
                    logger.warning(
                        "rocketchat.reset: delete channel %s failed: %s", name, exc,
                    )

        # Drop per-user tokens so per-user logins re-auth next run.
        self._user_tokens = {}
        # Re-run setup() to refresh admin token (handles RC restart cases).
        await self.setup(sandbox=sandbox)

    # ── public surface ────────────────────────────────────────────────

    async def post_message(
        self, *, channel: str | None = None,
        room: str | None = None, room_name: str | None = None,
        text: str | None = None,
        body: str | None = None, message: str | None = None,
        sender: str | None = None, **_extra: Any,
    ) -> dict[str, Any]:
        """Post `text` to `channel` as `sender`. Accepts `body`/`message`
        as aliases for `text`; default sender to the in-process account
        when omitted so agent calls without an explicit sender don't 400."""
        channel = channel or room or room_name or "general"
        text = text or body or message or ""
        sender = sender or "system"
        # Agent often passes the full email address (e.g. "alice.kim@agentlab.local")
        # as sender, but RocketChat login expects the bare username. Strip the
        # @domain so /api/v1/login doesn't 401 Unauthorized.
        if "@" in sender:
            sender = sender.split("@", 1)[0]
        # Newlines in sender (e.g. agent stuffed a whole message into sender)
        # cause login 401s — strip to bare token.
        sender = sender.replace("\n", " ").split()[0] if sender.strip() else "system"
        token = await self._token_for(sender)
        res = await asyncio.to_thread(
            self._post_json, "/api/v1/chat.postMessage", token,
            {"channel": channel, "text": text},
        )
        _stamp_sim(res)
        if isinstance(res.get("message"), dict):
            _stamp_sim(res["message"])
        return res

    async def channel_history(
        self, *, channel: str | None = None,
        room_name: str | None = None,
        limit: int = 50, **_extra: Any,
    ) -> list[dict[str, Any]]:
        """Read up to `limit` messages from `channel`, newest first.

        Accepts `room_name` as alias for `channel`. Strips leading "#" since
        RocketChat's roomName param wants the bare name (sending "#ops" 400s)."""
        channel = channel or room_name or "general"
        if channel.startswith("#"):
            channel = channel[1:]
        token = self._require_admin_token()
        out = await asyncio.to_thread(
            self._get, "/api/v1/channels.history",
            token, {"roomName": channel, "count": str(limit)},
        )
        return [_stamp_sim(m) for m in (out.get("messages") or [])]

    async def list_channels(self) -> list[dict[str, Any]]:
        """List all chat channels (no arguments)."""
        token = self._require_admin_token()
        # count=0 -> return ALL channels. Without it RocketChat applies its
        # default page size (50), which silently truncates the listing. This
        # matters for reset() (channels past page 1 wouldn't get scrubbed and
        # would leak state across sessions) and for agent discovery.
        out = await asyncio.to_thread(
            self._get, "/api/v1/channels.list", token, {"count": "0"},
        )
        return [_stamp_sim(c) for c in (out.get("channels") or [])]

    async def send_direct_message(
        self, *, sender: str | None = None, recipient: str | None = None,
        to: str | None = None, peer: str | None = None,
        text: str | None = None, body: str | None = None,
        message: str | None = None, **_extra: Any,
    ) -> dict[str, Any]:
        """DM `recipient` as `sender`. Accepts to/peer aliases and body/message text."""
        recipient = recipient or to or peer
        text = text or body or message or ""
        sender = sender or "system"
        if "@" in sender:
            sender = sender.split("@", 1)[0]
        if not recipient:
            # Raise (vs a soft {ok:False}) so the dispatcher surfaces it as a tool
            # error and the agent retries with a recipient. The soft return read as
            # success, so agents that put the name in `text` ("Hi Marco — ...") never
            # learned to fill `recipient` → 0 DMs were ever actually sent.
            raise ValueError(
                "send_direct_message needs a 'recipient' (the username to DM, e.g. "
                "recipient='marco.v'); put the message body in 'text'."
            )
        token = await self._token_for(sender)
        res = await asyncio.to_thread(
            self._post_json, "/api/v1/chat.postMessage", token,
            {"channel": f"@{recipient}", "text": text},
        )
        _stamp_sim(res)
        if isinstance(res.get("message"), dict):
            _stamp_sim(res["message"])
        return res

    async def im_history(
        self, *, user: str | None = None, peer: str | None = None,
        limit: int = 50, **_extra: Any,
    ) -> list[dict[str, Any]]:
        """Read DMs between `user` and `peer`."""
        if not user or not peer:
            return []
        token = await self._token_for(user)
        out = await asyncio.to_thread(
            self._get, "/api/v1/im.history",
            token, {"username": peer, "count": str(limit)},
        )
        return [_stamp_sim(m) for m in (out.get("messages") or [])]

    async def list_users(self, **_extra: Any) -> list[dict[str, Any]]:
        """List all users. Tolerates `query`/`search`/`role` kwargs from agents
        (forwarded to users.list when present, otherwise ignored)."""
        token = self._require_admin_token()
        # count=0 -> ALL users. Without it RocketChat caps the listing at its
        # default page size (50); with a 150-account org that silently drops
        # two-thirds of the directory, breaking agent discovery and any caller
        # that enumerates users.
        params: dict[str, str] = {"count": "0"}
        if _extra.get("query") or _extra.get("search"):
            params["query"] = str(_extra.get("query") or _extra.get("search"))
        out = await asyncio.to_thread(self._get, "/api/v1/users.list", token, params)
        return list(out.get("users") or [])

    async def set_topic(
        self, *, sender: str, channel: str, topic: str,
    ) -> dict[str, Any]:
        """Set a channel's topic. Pass `sender`, the `channel` name, and `topic` text."""
        token = await self._token_for(sender)
        return await asyncio.to_thread(
            self._post_json, "/api/v1/channels.setTopic", token,
            {"roomName": channel, "topic": topic},
        )

    # ── internals ─────────────────────────────────────────────────────

    def _login(
        self, user: str, password: str,
        *,
        max_attempts: int = 8, retry_sleep_s: float = 3.0,
        retry_on_401: bool = False,
    ) -> dict[str, str]:
        """POST /api/v1/login with retry.

        RocketChat's healthcheck returns 200 before the API is actually
        ready to authenticate. There are two distinct warmup symptoms:
          (a) the API drops the connection mid-handshake — caught by the
              broad except (connection-level errors)
          (b) the API returns 401 because the admin user hasn't been
              materialised in MongoDB yet (the
              `Inserting admin user` log line hasn't fired)

        retry_on_401 covers (b). The admin-login path in setup() passes
        retry_on_401=True; per-user logins later in the run get the
        default False — a 401 there is a real auth failure (e.g., the
        account doesn't have a RocketChat account) and should
        propagate, not loop.
        """
        import time
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                with self._client_factory(base_url=self._base_url, timeout=15.0) as c:
                    resp = c.post(
                        "/api/v1/login",
                        json={"user": user, "password": password},
                    )
                    if resp.status_code == 401 and retry_on_401:
                        last_exc = RuntimeError(
                            f"rocketchat admin login 401 (warmup): {resp.text!r}"
                        )
                        time.sleep(retry_sleep_s)
                        continue
                    if resp.status_code >= 500:
                        last_exc = RuntimeError(
                            f"rocketchat login {resp.status_code}: {resp.text!r}"
                        )
                        time.sleep(retry_sleep_s)
                        continue
                    resp.raise_for_status()
                    data = resp.json().get("data") or {}
                    token = data.get("authToken")
                    uid = data.get("userId")
                    if not token or not uid:
                        raise RuntimeError(
                            f"rocketchat login failed: {resp.text!r}"
                        )
                    return {"X-Auth-Token": token, "X-User-Id": uid}
            except Exception as exc:                      # noqa: BLE001
                # Connection-level + transient errors → retry. We catch
                # broadly because httpx wraps low-level errors in several
                # distinct types depending on transport state.
                last_exc = exc
                msg = str(exc).lower()
                if "server disconnected" in msg or "connection" in msg \
                        or "timeout" in msg or "refused" in msg:
                    time.sleep(retry_sleep_s)
                    continue
                raise
        raise RuntimeError(
            f"rocketchat login failed after {max_attempts} attempts: {last_exc!r}"
        )

    async def _token_for(self, user: str) -> dict[str, str]:
        if user in self._user_tokens:
            return self._user_tokens[user]
        # Convention: every seeded user's password is its username — a
        # deliberate dev convenience (seed_org disables the RocketChat password
        # policy so this is accepted). Override at task level by pre-populating
        # self._user_tokens[user] in setup.
        tok = await asyncio.to_thread(self._login, user, user)
        self._user_tokens[user] = tok
        return tok

    async def create_channel(
        self, *, name: str | None = None, channel: str | None = None,
        sender: str | None = None, **_extra: Any,
    ) -> dict[str, Any]:
        """Create a new public chat channel. Pass `name` (with or without a leading
        '#'). Channels are NOT created automatically, so call this BEFORE posting to a
        channel that does not exist yet. Creating one that already exists is a no-op."""
        nm = (name or channel or "").lstrip("#").strip()
        if not nm:
            raise ValueError("create_channel needs a channel 'name' (e.g. name='hr-ops').")
        token = self._require_admin_token()

        def _call() -> dict[str, Any]:
            with self._client_factory(base_url=self._base_url, timeout=15.0) as c:
                r = c.post("/api/v1/channels.create", json={"name": nm}, headers=token)
                if r.status_code < 300:
                    return {"ok": True, "channel": nm}
                msg = (r.text or "").lower()
                if "already" in msg or "name-invalid-or-already-in-use" in msg:
                    return {"ok": True, "channel": nm, "existed": True}
                return {"ok": False, "channel": nm, "error": r.text}
        return await asyncio.to_thread(_call)

    def _require_admin_token(self) -> dict[str, str]:
        if not self._admin_token:
            raise RuntimeError(
                "rocketchat admin token not initialised; setup() must run first"
            )
        return self._admin_token

    def _post_json(
        self, path: str, token: dict[str, str], payload: dict[str, Any],
    ) -> dict[str, Any]:
        with self._client_factory(base_url=self._base_url, timeout=15.0) as c:
            resp = c.post(path, json=payload, headers=token)
            resp.raise_for_status()
            return resp.json() or {}

    def _get(
        self, path: str, token: dict[str, str], params: dict[str, str],
    ) -> dict[str, Any]:
        with self._client_factory(base_url=self._base_url, timeout=15.0) as c:
            resp = c.get(path, params=params, headers=token)
            resp.raise_for_status()
            return resp.json() or {}
