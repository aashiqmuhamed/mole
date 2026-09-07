"""Unit tests for per-run trace persistence."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mole.persistence import (
    RunManifest, TraceWriter, default_traces_base, new_run_id,
)


def test_new_run_id_is_unique_and_sortable():
    ids = {new_run_id() for _ in range(10)}
    assert len(ids) == 10  # all unique
    # Each starts with YYYYMMDDTHHMMSSZ which sorts chronologically.
    for i in ids:
        assert "T" in i and i.endswith("Z" + i.split("Z", 1)[1])


def test_default_traces_base_uses_env_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("MOLE_TRACE_DIR", str(tmp_path / "custom"))
    assert default_traces_base() == tmp_path / "custom"


def test_default_traces_base_falls_back_when_env_missing(monkeypatch):
    monkeypatch.delenv("MOLE_TRACE_DIR", raising=False)
    # No env set → local ./traces default.
    assert default_traces_base() == Path("traces")


def test_trace_writer_creates_per_run_directory(tmp_path):
    tw = TraceWriter(traces_base=tmp_path, task_id="06_weight_exfil")
    assert tw.run_dir.exists()
    assert tw.run_dir.parent.name == "06_weight_exfil"
    assert tw.run_dir.name == tw.run_id


def test_trace_writer_honors_explicit_run_id(tmp_path):
    tw = TraceWriter(traces_base=tmp_path, task_id="t", run_id="my-fixed-id")
    assert tw.run_dir.name == "my-fixed-id"


def test_write_system_prompt_writes_utf8_file(tmp_path):
    tw = TraceWriter(traces_base=tmp_path, task_id="t")
    tw.write_system_prompt("hello unicode → ✓")
    assert tw.system_prompt.read_text(encoding="utf-8") == "hello unicode → ✓"


def test_append_transcript_writes_jsonl_entries(tmp_path):
    tw = TraceWriter(traces_base=tmp_path, task_id="t")
    tw.append_transcript(stage_id="stage0", role="user", content="hi")
    tw.append_transcript(
        stage_id="stage0", role="assistant", content="hello",
        tool_calls=[{"name": "list_projects", "args": {}}],
    )
    lines = tw.transcript_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    e0 = json.loads(lines[0])
    e1 = json.loads(lines[1])
    assert e0["role"] == "user" and e0["content"] == "hi"
    assert e1["tool_calls"] == [{"name": "list_projects", "args": {}}]


def test_write_manifest_serializes_all_fields(tmp_path):
    tw = TraceWriter(traces_base=tmp_path, task_id="t", run_id="r0")
    m = RunManifest(
        run_id="r0", task_id="t", task_name="T",
        adversary="a0_naive", is_malicious=True,
        focal_account="bob.li",
        llm_backend="anthropic", llm_model="claude-opus-4-7",
        temperature=0.3, max_turns_per_stage=30,
        sim_start_iso="2026-04-06T09:00:00Z",
        started_at_iso="2026-04-06T09:00:00Z",
        score=0.42,
        rubric_summary={"utility.pr_opened": True, "harm.any": False},
    )
    tw.write_manifest(m)
    j = json.loads(tw.manifest_path.read_text(encoding="utf-8"))
    assert j["adversary"] == "a0_naive"
    assert j["is_malicious"] is True
    assert j["score"] == 0.42
    assert j["rubric_summary"] == {"utility.pr_opened": True, "harm.any": False}


def test_append_index_writes_one_jsonl_line_per_call(tmp_path):
    tw1 = TraceWriter(traces_base=tmp_path, task_id="t", run_id="r1")
    tw2 = TraceWriter(traces_base=tmp_path, task_id="t", run_id="r2")
    m1 = RunManifest(
        run_id="r1", task_id="t", task_name="T", adversary=None,
        is_malicious=False, focal_account="bob.li",
        llm_backend="anthropic", llm_model="claude-opus-4-7",
        temperature=0.3, max_turns_per_stage=30,
        sim_start_iso="t", started_at_iso="t", score=0.5,
    )
    m2 = RunManifest(**{**m1.__dict__, "run_id": "r2", "score": 0.7,
                        "adversary": "a0_naive", "is_malicious": True})
    tw1.append_index(m1)
    tw2.append_index(m2)
    lines = (tmp_path / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(l) for l in lines]
    assert rows[0]["run_id"] == "r1" and rows[0]["score"] == 0.5
    assert rows[1]["run_id"] == "r2" and rows[1]["adversary"] == "a0_naive"


def test_two_writers_for_same_task_get_different_run_dirs(tmp_path):
    """Concurrent or sequential runs of the same threat must not
    overwrite each other's traces — that's the whole point of run_id."""
    a = TraceWriter(traces_base=tmp_path, task_id="06_weight_exfil")
    b = TraceWriter(traces_base=tmp_path, task_id="06_weight_exfil")
    assert a.run_dir != b.run_dir


def test_audit_jsonl_path_is_under_run_dir(tmp_path):
    """audit.jsonl must land in the run_dir so the AuditCollector's
    flushed events don't collide with another concurrent run."""
    tw = TraceWriter(traces_base=tmp_path, task_id="t")
    assert tw.audit_jsonl.parent == tw.run_dir
