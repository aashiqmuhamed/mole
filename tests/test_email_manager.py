"""Unit tests for EmailManager — mocks smtplib/imaplib so no GreenMail needed.

An opt-in integration test (EMAIL_INTEGRATION=1) hits a real container.
"""
from __future__ import annotations

import asyncio
import email
import os
from email.message import EmailMessage
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mole.sandbox.base import DryRunSandbox
from mole.state.base import StateManager
from mole.state.email.manager import EmailManager


# ── helpers / fixtures ────────────────────────────────────────────────


def _smtp_factory() -> MagicMock:
    """Return a MagicMock standing in for `smtplib.SMTP(host, port)`.

    smtplib.SMTP is used as a context manager (`with smtplib.SMTP(...) as s`).
    """
    smtp_instance = MagicMock(name="smtp_instance")
    smtp_factory = MagicMock(name="smtp_factory")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=smtp_instance)
    cm.__exit__ = MagicMock(return_value=False)
    smtp_factory.return_value = cm
    smtp_factory.instance = smtp_instance        # convenience handle for assertions
    return smtp_factory


def _imap_factory(messages: list[bytes] | None = None) -> MagicMock:
    """MagicMock for `imaplib.IMAP4(host, port)`. Yields a context manager."""
    messages = messages or []
    imap = MagicMock(name="imap_instance")
    imap.login.return_value = ("OK", [b"Logged in"])
    imap.select.return_value = ("OK", [str(len(messages)).encode()])
    # search returns ("OK", [b"1 2 3 ..."])
    if messages:
        imap.search.return_value = ("OK", [b" ".join(str(i + 1).encode() for i in range(len(messages)))])
    else:
        imap.search.return_value = ("OK", [b""])

    def fetch_side(mid: bytes, *_args, **_kw):
        try:
            idx = int(mid.decode("ascii")) - 1
            return ("OK", [(b"placeholder", messages[idx])])
        except (ValueError, IndexError):
            return ("NO", [None])
    imap.fetch.side_effect = fetch_side
    imap.store.return_value = ("OK", [b"Flagged"])
    imap.expunge.return_value = ("OK", [b"Done"])

    factory = MagicMock(name="imap_factory")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=imap)
    cm.__exit__ = MagicMock(return_value=False)
    factory.return_value = cm
    factory.instance = imap
    return factory


def _raw_message(*, from_addr: str, to_addr: str, subject: str, body: str) -> bytes:
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Message-ID"] = f"<{subject.replace(' ', '_')}@test>"
    msg.set_content(body)
    return msg.as_bytes()


@pytest.fixture
def manager():
    mgr = EmailManager(config={
        "host": "127.0.0.1",
        "smtp_port": 12500,
        "imap_port": 12501,
    })
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    mgr._smtp_factory = _smtp_factory()
    mgr._imap_factory = _imap_factory()
    return mgr


# ── registration + setup ──────────────────────────────────────────────


def test_email_manager_is_registered():
    assert "email" in StateManager._registry
    assert StateManager._registry["email"] is EmailManager


def test_setup_uses_explicit_config_ports(manager):
    assert manager._smtp_port == 12500
    assert manager._imap_port == 12501


def test_setup_falls_back_to_sandbox_ports():
    sandbox = DryRunSandbox(ports={3025: 55001, 3143: 55002})
    mgr = EmailManager()
    asyncio.run(mgr.setup(sandbox=sandbox))
    assert mgr._smtp_port == 55001
    assert mgr._imap_port == 55002


def test_setup_raises_when_ports_missing():
    mgr = EmailManager()
    with pytest.raises(RuntimeError, match="SMTP"):
        asyncio.run(mgr.setup(sandbox=DryRunSandbox(ports={})))


# ── send ──────────────────────────────────────────────────────────────


def test_send_email_calls_smtp_with_message(manager):
    out = asyncio.run(manager.send_email(
        from_user="alice.kim@agentlab.local",
        to="bob.li@agentlab.local",
        subject="hello",
        body="hi there",
    ))
    smtp = manager._smtp_factory.instance
    smtp.send_message.assert_called_once()
    sent = smtp.send_message.call_args.args[0]
    assert sent["From"] == "alice.kim@agentlab.local"
    assert sent["To"] == "bob.li@agentlab.local"
    assert sent["Subject"] == "hello"
    assert "hi there" in sent.get_content()
    assert out["to"] == ["bob.li@agentlab.local"]
    assert out["bytes"] > 0


def test_send_email_supports_multiple_recipients_and_cc(manager):
    asyncio.run(manager.send_email(
        from_user="a@x.com",
        to=["b@x.com", "c@x.com"],
        cc="d@x.com",
        subject="multi", body="body",
    ))
    smtp = manager._smtp_factory.instance
    sent = smtp.send_message.call_args.args[0]
    assert sent["To"] == "b@x.com, c@x.com"
    assert sent["Cc"] == "d@x.com"


def test_send_email_attaches_files(manager):
    asyncio.run(manager.send_email(
        from_user="a@x.com", to="b@x.com",
        subject="weights", body="see attached",
        attachments=[
            {"filename": "weights.bin", "content": b"\x00\x01\x02",
             "maintype": "application", "subtype": "octet-stream"},
        ],
    ))
    smtp = manager._smtp_factory.instance
    sent = smtp.send_message.call_args.args[0]
    parts = list(sent.iter_attachments())
    assert len(parts) == 1
    assert parts[0].get_filename() == "weights.bin"
    assert parts[0].get_payload(decode=True) == b"\x00\x01\x02"


# ── receive ───────────────────────────────────────────────────────────


def test_read_inbox_returns_newest_first(manager):
    messages = [
        _raw_message(from_addr="a@x.com", to_addr="bob.li@agentlab.local",
                     subject="first", body="m1"),
        _raw_message(from_addr="b@x.com", to_addr="bob.li@agentlab.local",
                     subject="second", body="m2"),
        _raw_message(from_addr="c@x.com", to_addr="bob.li@agentlab.local",
                     subject="third", body="m3"),
    ]
    manager._imap_factory = _imap_factory(messages)

    out = asyncio.run(manager.read_inbox(user="bob.li", max_count=10))
    assert [m["subject"] for m in out] == ["third", "second", "first"]
    # Body parses out cleanly.
    assert "m3" in out[0]["body"]


def test_read_inbox_respects_max_count(manager):
    messages = [
        _raw_message(from_addr=f"u{i}@x.com", to_addr="bob@x.com",
                     subject=f"s{i}", body=f"b{i}")
        for i in range(5)
    ]
    manager._imap_factory = _imap_factory(messages)

    out = asyncio.run(manager.read_inbox(user="bob.li", max_count=2))
    assert len(out) == 2
    # Still newest-first.
    assert [m["subject"] for m in out] == ["s4", "s3"]


def test_read_inbox_empty_when_no_messages(manager):
    manager._imap_factory = _imap_factory([])
    out = asyncio.run(manager.read_inbox(user="bob.li"))
    assert out == []


def test_find_emails_filters_by_subject_substring(manager):
    messages = [
        _raw_message(from_addr="alice@x.com", to_addr="bob@x.com",
                     subject="Weekly status update", body="..."),
        _raw_message(from_addr="mallory@gmail.com", to_addr="bob@x.com",
                     subject="re: side gig opportunity", body="$50k"),
        _raw_message(from_addr="alice@x.com", to_addr="bob@x.com",
                     subject="Eval harness refactor", body="please do it"),
    ]
    manager._imap_factory = _imap_factory(messages)

    out = asyncio.run(manager.find_emails(user="bob.li",
                                           subject_contains="harness"))
    assert len(out) == 1
    assert "harness" in out[0]["subject"].lower()


def test_find_emails_filters_by_from_substring(manager):
    messages = [
        _raw_message(from_addr="alice@agentlab.local", to_addr="bob@x.com",
                     subject="x", body="."),
        _raw_message(from_addr="mallory@gmail.com", to_addr="bob@x.com",
                     subject="y", body="."),
    ]
    manager._imap_factory = _imap_factory(messages)

    out = asyncio.run(manager.find_emails(user="bob.li", from_contains="gmail.com"))
    assert len(out) == 1
    assert "gmail" in out[0]["from"]


# ── clear / forward ───────────────────────────────────────────────────


def test_clear_folder_deletes_all_messages(manager):
    messages = [_raw_message(from_addr=f"u{i}@x.com", to_addr="bob@x.com",
                              subject=f"s{i}", body=".") for i in range(3)]
    manager._imap_factory = _imap_factory(messages)

    n = asyncio.run(manager.clear_folder(user="bob.li"))
    assert n == 3
    manager._imap_factory.instance.expunge.assert_called_once()


def test_forward_email_resends_with_fwd_prefix(manager):
    messages = [_raw_message(from_addr="alice@x.com", to_addr="bob@x.com",
                              subject="Original subject", body="see this")]
    manager._imap_factory = _imap_factory(messages)
    # Re-bind SMTP so we capture the forwarded send too.
    manager._smtp_factory = _smtp_factory()

    asyncio.run(manager.forward_email(
        user="bob.li", message_id="1", to="dan@x.com",
    ))
    smtp = manager._smtp_factory.instance
    smtp.send_message.assert_called_once()
    forwarded = smtp.send_message.call_args.args[0]
    assert forwarded["Subject"].startswith("Fwd: ")
    assert "see this" in forwarded.get_content()
    assert forwarded["To"] == "dan@x.com"


# ── opt-in integration ───────────────────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("EMAIL_INTEGRATION") != "1",
    reason="set EMAIL_INTEGRATION=1 (and EMAIL_HOST/PORTS) to run against live GreenMail",
)
def test_real_email_round_trip():
    mgr = EmailManager()
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    sent = asyncio.run(mgr.send_email(
        from_user="alice@agentlab.local",
        to="bob.li@agentlab.local",
        subject="itb-smoke",
        body="hello from the integration test",
    ))
    assert sent["bytes"] > 0
    received = asyncio.run(mgr.find_emails(
        user="bob.li", subject_contains="itb-smoke",
    ))
    assert any("hello from" in m["body"] for m in received)
