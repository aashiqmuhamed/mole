"""Tests for generator/resume_state.py — the day-boundary resume manifest.

Covers the crash-safety contract: the manifest names only fully-committed days,
resume truncates a partial/garbage tail back to the right boundary, and the
--restore-world target is the last day with a COMPLETE world snapshot.
"""
from __future__ import annotations

from mole.generator import resume_state as rs


def _seed(tmp_path):
    audit = tmp_path / "sim.jsonl"
    trans = tmp_path / "sim.transcripts.jsonl"
    audit.write_text("e1\ne2\ne3\n", encoding="utf-8")
    trans.write_text("t1\nt2\n", encoding="utf-8")
    m = rs.new_manifest(
        audit_path=audit, transcript_path=trans, start_date="2026-04-06",
        n_days=7, seed=0, session_id="sess-x")
    return audit, trans, m


def test_record_load_round_trip(tmp_path):
    audit, trans, m = _seed(tmp_path)
    rs.record_day(m, audit, day="2026-04-06",
                  audit_bytes=audit.stat().st_size, transcript_bytes=trans.stat().st_size,
                  snapshot_ok=True, managers_ok=True,
                  snapshot_dir=str(tmp_path / "snap1"),
                  managers_path=str(tmp_path / "snap1" / "managers.json"))
    loaded = rs.load(audit)
    assert loaded is not None
    assert rs.last_completed_day(loaded) == "2026-04-06"


def test_truncate_drops_partial_and_garbage_tail(tmp_path):
    audit, trans, m = _seed(tmp_path)
    rs.record_day(m, audit, day="2026-04-06",
                  audit_bytes=audit.stat().st_size, transcript_bytes=trans.stat().st_size,
                  snapshot_ok=True, managers_ok=True,
                  snapshot_dir=str(tmp_path / "s1"),
                  managers_path=str(tmp_path / "s1" / "managers.json"))
    # A crash appends a garbage tail after the committed boundary.
    with audit.open("a", encoding="utf-8") as f:
        f.write("GARBAGE\n")
    with trans.open("a", encoding="utf-8") as f:
        f.write("G1\nG2\n")

    m2 = rs.load(audit)
    ta, tt = rs.truncate_outputs(m2, audit, trans)
    assert ta > 0 and tt > 0
    assert audit.read_text(encoding="utf-8") == "e1\ne2\ne3\n"
    assert trans.read_text(encoding="utf-8") == "t1\nt2\n"


def test_restore_target_skips_incomplete_boundary(tmp_path):
    """--restore-world must target the last day whose snapshot AND managers
    export both succeeded — not a later day with only a partial boundary."""
    audit, trans, m = _seed(tmp_path)
    # day 1: complete boundary
    rs.record_day(m, audit, day="2026-04-06",
                  audit_bytes=audit.stat().st_size, transcript_bytes=trans.stat().st_size,
                  snapshot_ok=True, managers_ok=True,
                  snapshot_dir=str(tmp_path / "s1"),
                  managers_path=str(tmp_path / "s1" / "managers.json"))
    with audit.open("a", encoding="utf-8") as f:
        f.write("e4\ne5\n")
    # day 2: snapshot FAILED -> not a restorable boundary
    rs.record_day(m, audit, day="2026-04-07",
                  audit_bytes=audit.stat().st_size, transcript_bytes=trans.stat().st_size,
                  snapshot_ok=False, managers_ok=True)

    m2 = rs.load(audit)
    assert rs.last_completed_day(m2) == "2026-04-07"        # default resume
    rr = rs.last_restorable_record(m2)                      # --restore-world
    assert rr is not None and rr["day"] == "2026-04-06"

    # Truncating to the restore boundary rewinds past day 2's (inconsistent) data.
    rs.truncate_to_record(rr, audit, trans)
    assert audit.read_text(encoding="utf-8") == "e1\ne2\ne3\n"


def test_record_day_is_idempotent_on_rerun(tmp_path):
    """Re-recording a day (a re-run day on resume) replaces, never duplicates."""
    audit, trans, m = _seed(tmp_path)
    for ok in (False, True):
        rs.record_day(m, audit, day="2026-04-06",
                      audit_bytes=10, transcript_bytes=5,
                      snapshot_ok=ok, managers_ok=ok,
                      snapshot_dir=str(tmp_path / "s") if ok else None,
                      managers_path=str(tmp_path / "s" / "managers.json") if ok else None)
    m2 = rs.load(audit)
    assert [d["day"] for d in m2["days"]] == ["2026-04-06"]
    assert m2["days"][0]["snapshot_ok"] is True


def test_missing_manifest_loads_as_none(tmp_path):
    assert rs.load(tmp_path / "absent.jsonl") is None


def test_in_progress_record_clear_and_finer_truncation(tmp_path):
    """The mid-day (bucket) checkpoint round-trips, coexists with committed days,
    truncates a partial tail FINER than a day boundary, and clears on commit."""
    audit, trans, m = _seed(tmp_path)
    # Commit day 1 at the current offsets.
    rs.record_day(m, audit, day="2026-04-06",
                  audit_bytes=audit.stat().st_size, transcript_bytes=trans.stat().st_size,
                  snapshot_ok=False, managers_ok=True)
    # Day 2 in progress: two more buckets land on disk...
    with audit.open("a", encoding="utf-8") as f:
        f.write("e4\ne5\n")
    with trans.open("a", encoding="utf-8") as f:
        f.write("t3\n")
    bucket = (audit.stat().st_size, trans.stat().st_size)
    rs.record_in_progress(m, audit, day="2026-04-07", next_bucket_index=1,
                          audit_bytes=bucket[0], transcript_bytes=bucket[1],
                          day_plan=[["2026-04-07T09:00:00", "alice", "session_0"]])
    # ...then a partial next bucket is written and the run dies.
    with audit.open("a", encoding="utf-8") as f:
        f.write("e6_partial\n")
    with trans.open("a", encoding="utf-8") as f:
        f.write("t4_partial\n")

    m2 = rs.load(audit)
    ip = rs.in_progress_record(m2)
    assert ip is not None and ip["day"] == "2026-04-07" and ip["next_bucket_index"] == 1
    assert ip["day_plan"] == [["2026-04-07T09:00:00", "alice", "session_0"]]
    # Committed day 1 still present alongside the in-progress day-2 checkpoint.
    assert rs.last_completed_day(m2) == "2026-04-06"
    # Truncate to the bucket boundary — finer than the day-1 boundary, keeps day 2's
    # committed buckets, drops only the partial tail.
    rs.truncate_to_record(ip, audit, trans)
    assert audit.read_text(encoding="utf-8") == "e1\ne2\ne3\ne4\ne5\n"
    assert trans.read_text(encoding="utf-8") == "t1\nt2\nt3\n"
    # Clearing drops the checkpoint (a committed day supersedes it).
    rs.clear_in_progress(m2, audit)
    assert rs.in_progress_record(rs.load(audit)) is None
