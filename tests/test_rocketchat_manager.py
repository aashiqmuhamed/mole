"""Unit tests for RocketChatManager.

We swap in a fake httpx.Client via _client_factory so no live RocketChat
container is required. An opt-in integration test
(ROCKETCHAT_INTEGRATION=1) hits a real instance.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from mole.sandbox.base import DryRunSandbox
from mole.state.base import StateManager
from mole.state.rocketchat.manager import RocketChatManager


# ── fake httpx client ─────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, json_data: dict[str, Any], status: int = 200) -> None:
        self._json = json_data
        self.status_code = status
        self.text = repr(json_data)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def json(self) -> dict[str, Any]:
        return self._json


class _FakeClient:
    """Records every call; returns scripted JSON keyed by HTTP path."""

    def __init__(
        self,
        *,
        login_token: tuple[str, str] = ("tok-admin", "u-admin"),
        scripted: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.scripted = scripted or {}
        self.login_token = login_token
        self.calls: list[dict[str, Any]] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def _resolve(self, path: str, default: dict[str, Any]) -> dict[str, Any]:
        for prefix, body in self.scripted.items():
            if path.startswith(prefix):
                return body
        return default

    def post(self, path: str, json: dict | None = None,
             headers: dict | None = None) -> _FakeResponse:
        self.calls.append({"method": "POST", "path": path,
                           "json": json, "headers": dict(headers or {})})
        if path == "/api/v1/login":
            return _FakeResponse({
                "data": {
                    "authToken": self.login_token[0],
                    "userId": self.login_token[1],
                },
            })
        body = self._resolve(path, {"success": True, "echo": json})
        return _FakeResponse(body)

    def get(self, path: str, params: dict | None = None,
            headers: dict | None = None) -> _FakeResponse:
        self.calls.append({"method": "GET", "path": path,
                           "params": dict(params or {}),
                           "headers": dict(headers or {})})
        body = self._resolve(path, {"success": True})
        return _FakeResponse(body)


def _client_factory_for(client: _FakeClient):
    def _factory(*args, **kwargs):
        return client
    return _factory


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_client() -> _FakeClient:
    return _FakeClient(scripted={
        "/api/v1/channels.history": {"messages": [
            {"_id": "m1", "msg": "hi"}, {"_id": "m2", "msg": "bye"},
        ]},
        "/api/v1/channels.list": {"channels": [
            {"_id": "c1", "name": "general"}, {"_id": "c2", "name": "incidents"},
        ]},
        "/api/v1/users.list": {"users": [
            {"_id": "u1", "username": "alice.kim"},
            {"_id": "u2", "username": "bob.li"},
        ]},
        "/api/v1/im.history": {"messages": [
            {"_id": "d1", "msg": "ping"},
        ]},
    })


@pytest.fixture
def manager(fake_client: _FakeClient) -> RocketChatManager:
    mgr = RocketChatManager(config={"host": "127.0.0.1", "port": 13000})
    mgr._client_factory = _client_factory_for(fake_client)
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    return mgr


# ── registration + setup ──────────────────────────────────────────────


def test_rocketchat_is_registered():
    assert "rocketchat" in StateManager._registry
    assert StateManager._registry["rocketchat"] is RocketChatManager


def test_setup_logs_in_admin(manager, fake_client):
    """The first POST must be /api/v1/login with admin credentials."""
    posts = [c for c in fake_client.calls if c["method"] == "POST"]
    assert posts[0]["path"] == "/api/v1/login"
    assert posts[0]["json"]["user"] == "rcadmin"
    assert manager._admin_token == {"X-Auth-Token": "tok-admin", "X-User-Id": "u-admin"}


def test_setup_uses_sandbox_port_when_no_config():
    sandbox = DryRunSandbox(ports={3000: 54000})
    mgr = RocketChatManager(config={"skip_admin_login": True})
    asyncio.run(mgr.setup(sandbox=sandbox))
    assert mgr._port == 54000


def test_setup_raises_when_port_missing():
    mgr = RocketChatManager(config={"skip_admin_login": True})
    with pytest.raises(RuntimeError, match="port"):
        asyncio.run(mgr.setup(sandbox=DryRunSandbox(ports={})))


def test_setup_skip_admin_login_leaves_token_empty():
    mgr = RocketChatManager(config={"port": 13000, "skip_admin_login": True})
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    assert mgr._admin_token == {}


# ── post_message ──────────────────────────────────────────────────────


def test_post_message_logs_in_as_sender_and_posts(manager, fake_client):
    asyncio.run(manager.post_message(
        sender="alice.kim", channel="general", text="hello",
    ))
    posts = [c for c in fake_client.calls if c["method"] == "POST"]
    # First is admin login (during setup); second is alice's login; third is
    # the actual chat.postMessage.
    alice_login = [c for c in posts if c["path"] == "/api/v1/login"
                   and c["json"]["user"] == "alice.kim"]
    assert len(alice_login) == 1
    chat = [c for c in posts if c["path"] == "/api/v1/chat.postMessage"]
    assert len(chat) == 1
    assert chat[0]["json"] == {"channel": "general", "text": "hello"}


def test_post_message_caches_user_token(manager, fake_client):
    """Sending twice as the same user only logs in once."""
    asyncio.run(manager.post_message(sender="alice.kim", channel="x", text="1"))
    asyncio.run(manager.post_message(sender="alice.kim", channel="x", text="2"))
    alice_logins = [c for c in fake_client.calls
                    if c["method"] == "POST" and c["path"] == "/api/v1/login"
                    and c["json"]["user"] == "alice.kim"]
    assert len(alice_logins) == 1


# ── channel_history / list_channels ───────────────────────────────────


def test_channel_history_returns_messages(manager):
    out = asyncio.run(manager.channel_history(channel="general", limit=10))
    assert [m["_id"] for m in out] == ["m1", "m2"]


def test_channel_history_passes_limit_to_api(manager, fake_client):
    asyncio.run(manager.channel_history(channel="general", limit=25))
    gets = [c for c in fake_client.calls
            if c["method"] == "GET" and c["path"] == "/api/v1/channels.history"]
    assert gets and gets[0]["params"] == {"roomName": "general", "count": "25"}


def test_channel_history_uses_admin_token(manager, fake_client):
    asyncio.run(manager.channel_history(channel="general"))
    gets = [c for c in fake_client.calls
            if c["method"] == "GET" and c["path"] == "/api/v1/channels.history"]
    assert gets[0]["headers"]["X-Auth-Token"] == "tok-admin"


def test_list_channels_returns_channel_records(manager):
    out = asyncio.run(manager.list_channels())
    assert [c["name"] for c in out] == ["general", "incidents"]


def test_list_channels_requests_count_zero(manager, fake_client):
    """Regression: without count=0 RocketChat caps the listing at its default
    page size (50), silently truncating channels (-> reset() leaves stale
    channels, agent discovery sees a partial directory)."""
    asyncio.run(manager.list_channels())
    gets = [c for c in fake_client.calls
            if c["method"] == "GET" and c["path"] == "/api/v1/channels.list"]
    assert gets and gets[-1]["params"].get("count") == "0"


def test_list_users_requests_count_zero(manager, fake_client):
    """Regression: same truncation as channels — with a 150-account org the
    default page size drops two-thirds of the directory."""
    asyncio.run(manager.list_users())
    gets = [c for c in fake_client.calls
            if c["method"] == "GET" and c["path"] == "/api/v1/users.list"]
    assert gets and gets[-1]["params"].get("count") == "0"


def test_list_channels_raises_if_admin_skipped(fake_client):
    mgr = RocketChatManager(config={"port": 13000, "skip_admin_login": True})
    mgr._client_factory = _client_factory_for(fake_client)
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    with pytest.raises(RuntimeError, match="admin token"):
        asyncio.run(mgr.list_channels())


# ── direct messages ───────────────────────────────────────────────────


def test_direct_message_targets_at_recipient(manager, fake_client):
    asyncio.run(manager.send_direct_message(
        sender="alice.kim", recipient="bob.li", text="hello",
    ))
    dms = [c for c in fake_client.calls
           if c["method"] == "POST" and c["path"] == "/api/v1/chat.postMessage"
           and c["json"]["channel"] == "@bob.li"]
    assert len(dms) == 1
    assert dms[0]["json"]["text"] == "hello"


def test_im_history_uses_sender_token(manager, fake_client):
    out = asyncio.run(manager.im_history(user="alice.kim", peer="bob.li", limit=5))
    assert [m["_id"] for m in out] == ["d1"]
    gets = [c for c in fake_client.calls
            if c["method"] == "GET" and c["path"] == "/api/v1/im.history"]
    assert gets[0]["params"] == {"username": "bob.li", "count": "5"}
    # alice's token, not admin's.
    assert gets[0]["headers"]["X-Auth-Token"] == "tok-admin"
    # (Our fake login always returns tok-admin regardless of user. We assert
    # the alice login was issued instead — that proves the right token *path*.)
    alice_login = [c for c in fake_client.calls
                   if c["method"] == "POST" and c["path"] == "/api/v1/login"
                   and c["json"]["user"] == "alice.kim"]
    assert len(alice_login) == 1


# ── users + topic ─────────────────────────────────────────────────────


def test_list_users_returns_user_records(manager):
    out = asyncio.run(manager.list_users())
    assert [u["username"] for u in out] == ["alice.kim", "bob.li"]


def test_set_topic_posts_to_channels_set_topic(manager, fake_client):
    asyncio.run(manager.set_topic(
        sender="alice.kim", channel="incidents", topic="all-hands at 4pm",
    ))
    setters = [c for c in fake_client.calls
               if c["method"] == "POST" and c["path"] == "/api/v1/channels.setTopic"]
    assert setters and setters[0]["json"] == {
        "roomName": "incidents", "topic": "all-hands at 4pm",
    }


# ── opt-in integration ───────────────────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("ROCKETCHAT_INTEGRATION") != "1",
    reason="set ROCKETCHAT_INTEGRATION=1 (+ ROCKETCHAT_HOST/PORT) to run live",
)
def test_real_rocketchat_round_trip():
    mgr = RocketChatManager()
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    # Posts to #general; reads it back. Requires the seeded admin account.
    asyncio.run(mgr.post_message(
        sender="admin", channel="general", text="itb-smoke",
    ))
    msgs = asyncio.run(mgr.channel_history(channel="general", limit=5))
    assert any("itb-smoke" in m.get("msg", "") for m in msgs)
