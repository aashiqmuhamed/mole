"""Unit tests for OwnCloudManager.

Mocks pyocclient so they run without a live OwnCloud. An opt-in integration
test (gated on OWNCLOUD_INTEGRATION=1) hits a real container.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mole.sandbox.base import DryRunSandbox
from mole.state.base import StateManager
from mole.state.owncloud.manager import OwnCloudManager


def _make_mocked_oc() -> MagicMock:
    oc = MagicMock(name="owncloud.Client")
    oc.login.return_value = None
    return oc


@pytest.fixture
def manager():
    mgr = OwnCloudManager(config={"endpoint": "http://oc.test",
                                   "user": "admin", "password": "admin"})
    sandbox = DryRunSandbox(ports={80: 33333})
    fake = SimpleNamespace(Client=MagicMock(return_value=_make_mocked_oc()))
    with patch.dict("sys.modules", {"owncloud": fake}):
        asyncio.run(mgr.setup(sandbox=sandbox))
    return mgr


# ── registration + setup ──────────────────────────────────────────────


def test_owncloud_manager_is_registered():
    assert "owncloud" in StateManager._registry
    assert StateManager._registry["owncloud"] is OwnCloudManager


def test_setup_uses_explicit_endpoint(manager):
    assert manager._endpoint == "http://oc.test"
    assert manager._user == "admin"


def test_setup_falls_back_to_sandbox_port_when_no_endpoint():
    sandbox = DryRunSandbox(ports={80: 44444})
    mgr = OwnCloudManager()
    fake = SimpleNamespace(Client=MagicMock(return_value=_make_mocked_oc()))
    with patch.dict("sys.modules", {"owncloud": fake}):
        asyncio.run(mgr.setup(sandbox=sandbox))
    assert "44444" in mgr._endpoint
    assert mgr._endpoint.startswith("http://127.0.0.1:")


def test_setup_raises_when_no_endpoint_and_no_port():
    mgr = OwnCloudManager()
    fake = SimpleNamespace(Client=MagicMock(return_value=_make_mocked_oc()))
    with patch.dict("sys.modules", {"owncloud": fake}):
        with pytest.raises(RuntimeError, match="not in sandbox.ports"):
            asyncio.run(mgr.setup(sandbox=DryRunSandbox(ports={})))


def test_setup_raises_when_login_fails():
    mgr = OwnCloudManager(config={"endpoint": "http://x", "user": "u", "password": "wrong"})
    failing = MagicMock()
    failing.login.side_effect = RuntimeError("401")
    fake = SimpleNamespace(Client=MagicMock(return_value=failing))
    with patch.dict("sys.modules", {"owncloud": fake}):
        with pytest.raises(RuntimeError, match="could not log in"):
            asyncio.run(mgr.setup(sandbox=DryRunSandbox()))


# ── files ─────────────────────────────────────────────────────────────


def test_exists_true_when_file_info_succeeds(manager):
    manager._oc.file_info.return_value = SimpleNamespace(name="x")
    assert asyncio.run(manager.exists(path="/a")) is True
    manager._oc.file_info.assert_called_with("/a")


def test_exists_false_when_file_info_raises(manager):
    manager._oc.file_info.side_effect = Exception("404")
    assert asyncio.run(manager.exists(path="/missing")) is False


def test_mkdir_passes_path_through(manager):
    asyncio.run(manager.mkdir(path="/safety-reports"))
    manager._oc.mkdir.assert_called_with("/safety-reports")


def test_list_dir_normalizes_fileinfo(manager):
    f1 = SimpleNamespace(name="a.md", path="/safety-reports/a.md",
                         file_type="file", size=512, last_modified="2026-04-06")
    f2 = SimpleNamespace(name="subdir", path="/safety-reports/subdir",
                         file_type="dir", size=0, last_modified="2026-04-05")
    manager._oc.list.return_value = [f1, f2]
    out = asyncio.run(manager.list_dir(path="/safety-reports"))
    assert out == [
        {"name": "a.md", "path": "/safety-reports/a.md", "type": "file",
         "size": 512, "last_modified": "2026-04-06"},
        {"name": "subdir", "path": "/safety-reports/subdir", "type": "dir",
         "size": 0, "last_modified": "2026-04-05"},
    ]


def test_read_file_returns_str(manager):
    manager._oc.get_file_contents.return_value = b"hello"
    out = asyncio.run(manager.read_file(path="/a.txt"))
    assert out == "hello"
    manager._oc.get_file_contents.assert_called_with("/a.txt")


def test_read_bytes_returns_raw(manager):
    manager._oc.get_file_contents.return_value = b"\x00\x01\x02"
    out = asyncio.run(manager.read_bytes(path="/blob.bin"))
    assert out == b"\x00\x01\x02"


def test_read_file_handles_str_payload(manager):
    """Some pyocclient versions return str rather than bytes."""
    manager._oc.get_file_contents.return_value = "already a string"
    out = asyncio.run(manager.read_file(path="/text"))
    assert out == "already a string"


def test_write_file_encodes_str_to_utf8(manager):
    asyncio.run(manager.write_file(path="/x.txt", content="hello"))
    manager._oc.put_file_contents.assert_called_once()
    args = manager._oc.put_file_contents.call_args.args
    assert args[0] == "/x.txt"
    assert args[1] == b"hello"


def test_write_file_passes_bytes_through(manager):
    asyncio.run(manager.write_file(path="/y.bin", content=b"\xde\xad"))
    args = manager._oc.put_file_contents.call_args.args
    assert args[1] == b"\xde\xad"


def test_delete_passes_path_through(manager):
    asyncio.run(manager.delete(path="/training-data/old.csv"))
    manager._oc.delete.assert_called_with("/training-data/old.csv")


def test_delete_retries_with_added_trailing_slash(manager):
    """WebDAV addresses a directory WITH a trailing slash; an agent deleting
    '/dir' (no slash) 404s. delete() must retry with the slash added so either
    form works for a directory (regression for the threat-05 collision 404s
    where wei.k10 used '.../Q2/' and succeeded but elena.t129 used '.../Q2')."""
    calls: list[str] = []
    def fake(p):
        calls.append(p)
        if not p.endswith("/"):
            raise Exception("HTTP error: 404")
    manager._oc.delete.side_effect = fake
    asyncio.run(manager.delete(path="/training-data/eval-data-2025Q2"))
    assert calls == ["/training-data/eval-data-2025Q2",
                     "/training-data/eval-data-2025Q2/"]


def test_delete_retries_with_removed_trailing_slash(manager):
    """Reverse direction: a resource addressed as '/x/' 404s; retry without it."""
    calls: list[str] = []
    def fake(p):
        calls.append(p)
        if p.endswith("/"):
            raise Exception("HTTP error: 404")
    manager._oc.delete.side_effect = fake
    asyncio.run(manager.delete(path="/reports/notes.txt/"))
    assert calls == ["/reports/notes.txt/", "/reports/notes.txt"]


def test_delete_reraises_original_error_when_both_forms_fail(manager):
    """A genuinely missing resource (404 in both forms) surfaces the ORIGINAL
    (as-given) error, not the retry's — so the message points at the caller's
    path, not the toggled one."""
    def always_fail(p):
        raise Exception(f"HTTP error: 404 on {p}")
    manager._oc.delete.side_effect = always_fail
    with pytest.raises(Exception, match=r"eval-data-2025Q9$"):
        asyncio.run(manager.delete(path="/training-data/eval-data-2025Q9"))


# ── shares ────────────────────────────────────────────────────────────


def test_share_with_user_passes_permissions(manager):
    manager._oc.share_file_with_user.return_value = SimpleNamespace(
        share_id=42, share_with="bob.li",
    )
    out = asyncio.run(manager.share(
        path="/safety-reports/x.md", with_user="bob.li", permissions=1,
    ))
    manager._oc.share_file_with_user.assert_called_with(
        "/safety-reports/x.md", "bob.li", perms=1,
    )
    assert out["share_id"] == 42
    assert out["public"] is False
    assert out["permissions"] == 1
    assert out["share_with"] == "bob.li"


def test_share_public_uses_link_call(manager):
    manager._oc.share_file_with_link.return_value = SimpleNamespace(
        share_id=99, link="http://oc.test/s/abcdef",
    )
    out = asyncio.run(manager.share(path="/safety-reports/x.md", public=True))
    manager._oc.share_file_with_link.assert_called_with("/safety-reports/x.md")
    assert out["public"] is True
    assert out["url"] == "http://oc.test/s/abcdef"


def test_share_without_user_and_not_public_raises(manager):
    with pytest.raises(ValueError, match="with_user or public"):
        asyncio.run(manager.share(path="/x"))


def test_update_share_passes_permissions(manager):
    out = asyncio.run(manager.update_share(share_id=42, permissions=15))
    manager._oc.update_share.assert_called_with(42, perms=15)
    assert out == {"share_id": 42, "permissions": 15}


def test_unshare_passes_share_id(manager):
    asyncio.run(manager.unshare(share_id=42))
    manager._oc.delete_share.assert_called_with(42)


def test_public_link_returns_url(manager):
    manager._oc.share_file_with_link.return_value = SimpleNamespace(
        share_id=7, link="http://oc.test/s/xyz",
    )
    url = asyncio.run(manager.public_link(path="/safety-reports/x.md"))
    assert url == "http://oc.test/s/xyz"


# ── cleanup ───────────────────────────────────────────────────────────


def test_cleanup_logs_out(manager):
    asyncio.run(manager.cleanup())
    manager._oc is None or manager._oc.logout.assert_called_once()
    assert manager._oc is None


# ── opt-in integration test (real OwnCloud) ───────────────────────────


@pytest.mark.skipif(
    os.environ.get("OWNCLOUD_INTEGRATION") != "1",
    reason="set OWNCLOUD_INTEGRATION=1 (and OWNCLOUD_ENDPOINT/USER/PASSWORD) to run",
)
def test_real_owncloud_roundtrip():
    mgr = OwnCloudManager()
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    try:
        asyncio.run(mgr.write_file(path="/itb-smoke.txt", content="hello"))
        out = asyncio.run(mgr.read_file(path="/itb-smoke.txt"))
        assert out == "hello"
    finally:
        try:
            asyncio.run(mgr.delete(path="/itb-smoke.txt"))
        except Exception:
            pass
        asyncio.run(mgr.cleanup())


def test_list_dir_normalizes_trailing_slash(manager):
    """Regression: pyocclient list() returns nothing for '/dir/' but works for
    '/dir'. A trailing slash silently yielded an empty listing, making dir_size
    report 0 and threat-05's data-destruction harm oracle unreachable."""
    manager._oc.list.return_value = []
    asyncio.run(manager.list_dir(path="/training-data/eval-data-2025/"))
    manager._oc.list.assert_called_once_with("/training-data/eval-data-2025")
    # Root stays "/".
    manager._oc.list.reset_mock()
    asyncio.run(manager.list_dir(path="/"))
    manager._oc.list.assert_called_once_with("/")
