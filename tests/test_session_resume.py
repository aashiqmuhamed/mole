"""Tests for session-level (bucket-granular) warm resume (generator.run --session-resume).

Covers: the day-plan serialize/restore round-trip and _today_keys reconstruction
(so the attack schedule is preserved without re-running plan_day under a diverged
RNG); the flag being strictly opt-in (default resume is unchanged); and an
end-to-end crash mid-day -> resume-from-bucket that lands past the last bucket
boundary instead of discarding the whole in-progress day.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from mole.generator import resume_state
from mole.generator.run import (
    _restore_day_plan,
    _serialize_day_plan,
    simulate,
)


# ── day-plan serialize / restore (schedule preservation) ─────────────────────


def test_serialize_restore_day_plan_round_trip():
    alice = SimpleNamespace(persona=SimpleNamespace(id="alice"))
    bob = SimpleNamespace(persona=SimpleNamespace(id="bob"))
    members = [alice, bob]
    day_plan = [
        (datetime(2026, 4, 6, 9, 0), alice, "session_0"),
        (datetime(2026, 4, 6, 9, 15), bob, "session_0"),
        (datetime(2026, 4, 6, 11, 0), alice, "session_3"),
        (datetime(2026, 4, 6, 14, 0), bob, "session_2"),
    ]
    saved = _serialize_day_plan(day_plan)
    assert saved == [
        ["2026-04-06T09:00:00", "alice", "session_0"],
        ["2026-04-06T09:15:00", "bob", "session_0"],
        ["2026-04-06T11:00:00", "alice", "session_3"],
        ["2026-04-06T14:00:00", "bob", "session_2"],
    ]
    restored = _restore_day_plan(saved, members)
    assert restored == day_plan                       # same tuples, sorted by slot time
    # _today_keys restored per member (plan_day sets these; resume must too).
    assert alice._today_keys == ["session_0", "session_3"]
    assert bob._today_keys == ["session_0", "session_2"]


def test_restore_day_plan_orders_today_keys_by_slot_index():
    """_today_keys is the attack-slot pool that plan_day builds in slot-index order.
    Even if a member's keys appear out of index order in the (globally time-sorted)
    saved plan, restore must return them in session-index order to reproduce the
    deterministic attack-slot pick."""
    alice = SimpleNamespace(persona=SimpleNamespace(id="alice"))
    saved = [
        ["2026-04-06T09:00:00", "alice", "session_5"],
        ["2026-04-06T10:00:00", "alice", "session_1"],
        ["2026-04-06T11:00:00", "alice", "session_2"],
    ]
    _restore_day_plan(saved, [alice])
    assert alice._today_keys == ["session_1", "session_2", "session_5"]


# ── flag is strictly opt-in ──────────────────────────────────────────────────


def test_session_resume_off_writes_no_in_progress_checkpoint(tmp_path):
    """Default (no --session-resume): no in_progress checkpoint is ever written, so
    day-granular resume behaviour is byte-for-byte unchanged."""
    out = tmp_path / "npc.jsonl"
    asyncio.run(simulate(n_personas=3, n_days=2, out_path=out, seed=0,
                         start_date="2026-04-06", concurrency=4, bucket_minutes=15))
    m = resume_state.load(out)
    assert m is not None
    assert resume_state.in_progress_record(m) is None
    assert [d["day"] for d in m["days"]] == ["2026-04-06", "2026-04-07"]


# ── end-to-end: crash mid-day, then bucket-granular resume ────────────────────


def test_session_resume_resumes_from_bucket_not_day(tmp_path, monkeypatch):
    """With --session-resume, a crash partway through a day leaves an in_progress
    checkpoint; the resume re-enters that day from the next bucket (not day 1) and
    completes, keeping the audit consistent (valid JSON, both days committed)."""
    out = tmp_path / "npc.jsonl"
    kw = dict(n_personas=5, n_days=2, out_path=out, seed=7,
              start_date="2026-04-06", concurrency=4, bucket_minutes=15,
              session_resume=True)

    # Inject a crash right after day-1 bucket 1 finishes (its gather is done and on
    # disk, but the checkpoint advancing next_bucket_index to 2 doesn't get written).
    real_record = resume_state.record_in_progress

    def boom(manifest, audit_path, *, day, next_bucket_index, **k):
        if day == "2026-04-06" and next_bucket_index == 2:
            raise RuntimeError("injected mid-day crash")
        return real_record(manifest, audit_path, day=day,
                           next_bucket_index=next_bucket_index, **k)

    monkeypatch.setattr(resume_state, "record_in_progress", boom)
    with pytest.raises(RuntimeError, match="injected mid-day crash"):
        asyncio.run(simulate(**kw))
    monkeypatch.undo()

    # Crash state: an in_progress checkpoint for day 1, no committed day yet.
    m = resume_state.load(out)
    ip = resume_state.in_progress_record(m)
    assert ip is not None, "session-resume should have written a mid-day checkpoint"
    assert ip["day"] == "2026-04-06" and ip["next_bucket_index"] == 1
    assert resume_state.last_completed_day(m) is None
    lines_at_crash = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # Resume with --session-resume: re-enters day 1 from bucket 1, finishes the run.
    asyncio.run(simulate(resume=True, **kw))

    m2 = resume_state.load(out)
    assert resume_state.in_progress_record(m2) is None            # cleared on day commit
    assert [d["day"] for d in m2["days"]] == ["2026-04-06", "2026-04-07"]

    on_disk = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Every audit line is intact JSON — the mid-bucket truncation left no torn tail.
    dates = {json.loads(ln).get("ts", "")[:10] for ln in on_disk}
    assert {"2026-04-06", "2026-04-07"} <= dates                  # both days present
    assert len(on_disk) > len(lines_at_crash)                     # resume made progress
