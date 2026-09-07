"""Lab-service *payloads* carry SIMULATED time, not the backend container's wall-clock.

The audit log was already sim-stamped (via the `_sim_clock_var` contextvar), but the
payloads agents read back leaked real wall-clock time — RocketChat message `ts`,
OwnCloud `last_modified`, the email `Date`, and the secrets rotation timestamp — which
agents then echoed into content they authored (real dates showing up in plane issues /
gitlab commits even though those managers stamp nothing). These tests pin each fixed
adapter to the per-task sim clock via `get_current_sim_time`.
"""
from __future__ import annotations

import asyncio
import contextlib
from email.message import EmailMessage
from types import SimpleNamespace

from mole.audit.collector import (
    get_current_sim_time,
    reset_task_clock,
    set_task_clock,
)

SIM = "2026-04-06T09:00:00Z"
REAL = "2026-06-21T00:00:00.000Z"   # a "today" wall-clock stamp the container would emit


@contextlib.contextmanager
def sim_clock(ts: str = SIM):
    tok = set_task_clock(ts)
    try:
        yield ts
    finally:
        reset_task_clock(tok)


# ── the shared helper ─────────────────────────────────────────────────


def test_get_current_sim_time_reads_task_clock():
    with sim_clock(SIM):
        assert get_current_sim_time() == SIM


def test_get_current_sim_time_canonicalizes_offset_to_Z():
    with sim_clock("2026-04-06T09:00:00+00:00"):
        assert get_current_sim_time() == "2026-04-06T09:00:00Z"


def test_get_current_sim_time_none_without_clock():
    # All clock-setting in these tests goes through sim_clock (which resets), so
    # outside any block the contextvar is back to its None default.
    assert get_current_sim_time() is None


# ── RocketChat: remap on read ─────────────────────────────────────────


def test_rocketchat_stamp_sim_remaps_time_fields():
    from mole.state.rocketchat.manager import _stamp_sim
    with sim_clock(SIM):
        msg = _stamp_sim({
            "ts": REAL, "_updatedAt": REAL, "lm": REAL,
            "msg": "hi", "u": {"username": "wei.k10"},
        })
    assert msg["ts"] == SIM
    assert msg["_updatedAt"] == SIM
    assert msg["lm"] == SIM
    assert msg["msg"] == "hi"                      # non-time fields untouched
    assert msg["u"] == {"username": "wei.k10"}


def test_rocketchat_stamp_sim_noop_without_clock():
    from mole.state.rocketchat.manager import _stamp_sim
    msg = _stamp_sim({"ts": REAL})                 # no clock set
    assert msg["ts"] == REAL                        # kept; never wall-clock-substituted


# ── OwnCloud: remap last_modified ─────────────────────────────────────


def test_owncloud_fileinfo_uses_sim_time():
    from mole.state.owncloud.manager import _fileinfo_summary
    info = SimpleNamespace(name="report.md", path="/report.md",
                           file_type="file", size=42, last_modified="2026-06-21 00:00:00")
    with sim_clock(SIM):
        out = _fileinfo_summary(info)
    assert out["last_modified"] == SIM
    assert out["name"] == "report.md"               # other fields intact


def test_owncloud_fileinfo_falls_back_without_clock():
    from mole.state.owncloud.manager import _fileinfo_summary
    info = SimpleNamespace(name="x", path="/x", file_type="file", size=1,
                           last_modified="2026-06-21 00:00:00")
    out = _fileinfo_summary(info)                   # no clock
    assert out["last_modified"] == "2026-06-21 00:00:00"


# ── secrets_store: write sim-time ─────────────────────────────────────


def test_secrets_rotate_stamps_sim_time(tmp_path):
    from mole.sandbox.base import DryRunSandbox
    from mole.state.secrets_store.manager import SecretsStoreManager
    p = tmp_path / "secrets.yaml"
    p.write_text("secrets:\n  wandb_prod:\n    value: w\n    policy: {}\n", encoding="utf-8")
    mgr = SecretsStoreManager(config={"path": str(p)})
    asyncio.run(mgr.setup(sandbox=DryRunSandbox()))
    with sim_clock(SIM):
        res = asyncio.run(mgr.rotate(key="wandb_prod"))
    assert res["sha_changed_after"] == SIM


# ── email: Date on send, sim-fallback on read ─────────────────────────


def test_email_build_message_sets_sim_date():
    from mole.state.email.manager import EmailManager
    mgr = EmailManager(config={})
    with sim_clock(SIM):
        msg = mgr._build_message(from_user="a@x", to="b@x", subject="hi",
                                 body="hello", cc=None, attachments=None)
    assert msg["Date"] is not None
    assert "06 Apr 2026" in msg["Date"]             # the sim day, RFC-2822
    assert "Jun 2026" not in msg["Date"]


def test_email_summarise_falls_back_to_sim_ts_when_no_date():
    from mole.state.email.manager import _summarise_message
    m = EmailMessage()
    m["From"] = "a@x"; m["To"] = "b@x"; m["Subject"] = "hi"
    m.set_content("body")                            # no Date header
    with sim_clock(SIM):
        out = _summarise_message(m.as_bytes(), message_id="1")
    assert out["ts"] == SIM


def test_email_summarise_keeps_existing_date():
    from mole.state.email.manager import _summarise_message
    m = EmailMessage()
    m["From"] = "a@x"; m["To"] = "b@x"; m["Subject"] = "hi"
    m["Date"] = "Mon, 06 Apr 2026 09:00:00 +0000"    # message's own send-time Date
    m.set_content("body")
    with sim_clock("2026-04-10T00:00:00Z"):          # reader's clock is a different day
        out = _summarise_message(m.as_bytes(), message_id="1")
    assert "06 Apr 2026" in out["ts"]                # keeps the message Date, not reader clock
