"""Unit tests for the disk-full write-retry helper (audit/io_retry.py).

A full shared PVC surfaces write/flush/fsync as OSError ENOSPC(28) / EDQUOT(122).
``retry_on_disk_full`` turns that into a live stall (retry until space frees)
instead of an uncaught crash; any other OSError propagates unchanged.
"""
from __future__ import annotations

import errno

import pytest

from mole.audit import io_retry


def test_retries_enospc_then_succeeds(monkeypatch):
    monkeypatch.setattr(io_retry.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        return "ok"

    assert io_retry.retry_on_disk_full(fn, what="t") == "ok"
    assert calls["n"] == 3


def test_retries_edquot_then_succeeds(monkeypatch):
    monkeypatch.setattr(io_retry.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise OSError(122, "Disk quota exceeded")   # Linux EDQUOT
        return 42

    assert io_retry.retry_on_disk_full(fn, what="t") == 42
    assert calls["n"] == 2


def test_non_disk_full_oserror_propagates(monkeypatch):
    monkeypatch.setattr(io_retry.time, "sleep", lambda s: None)

    def fn():
        raise OSError(errno.EACCES, "permission denied")

    with pytest.raises(OSError) as exc:
        io_retry.retry_on_disk_full(fn)
    assert exc.value.errno == errno.EACCES


def test_is_disk_full_classification():
    assert io_retry.is_disk_full(OSError(errno.ENOSPC, "x")) is True
    assert io_retry.is_disk_full(OSError(122, "x")) is True          # EDQUOT
    assert io_retry.is_disk_full(OSError(errno.EACCES, "x")) is False
    assert io_retry.is_disk_full(ValueError("x")) is False


def test_collector_emit_survives_transient_disk_full(tmp_path, monkeypatch):
    """Integration: the audit collector's per-event flush stalls-and-recovers on a
    transient ENOSPC (the Part A wiring) instead of propagating an uncaught OSError
    that would kill the sim. Verifies the real write path, not just the helper."""
    from mole.audit.collector import AuditCollector
    from mole.audit.schema import AuditEvent
    monkeypatch.setattr(io_retry.time, "sleep", lambda s: None)

    c = AuditCollector(jsonl_path=tmp_path / "audit.jsonl")

    class _FlakyFile:
        """Wraps the collector's real file; its first `fail` flushes raise ENOSPC."""
        def __init__(self, real, fail):
            self._real, self._n, self._fail = real, 0, fail

        def write(self, s):
            return self._real.write(s)

        def flush(self):
            self._n += 1
            if self._n <= self._fail:
                raise OSError(errno.ENOSPC, "No space left on device")
            return self._real.flush()

        def fileno(self):
            return self._real.fileno()

        def close(self):
            return self._real.close()

    c._fp = _FlakyFile(c._fp, fail=3)
    ev = AuditEvent.new(account="alice", account_kind="background_rules_agent",
                        service="gitlab", action="commit", ts="2026-04-06T09:00:00Z")
    c.emit(ev)          # must NOT raise despite 3 consecutive ENOSPC flushes
    c.close()
    on_disk = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip()
    assert on_disk and "commit" in on_disk, "event should be durable once the stall cleared"
