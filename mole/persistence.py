"""Per-run trace persistence.

Every live threat run produces a self-contained directory:

    <traces_base>/<task_id>/<run_id>/
        manifest.json       — threat / adversary / seed / model / score metadata
        audit.jsonl         — every state-manager call the agent or background accounts made
        agent_transcript.jsonl — LLM messages in/out across stages
        system_prompt.txt   — rendered system prompt (with adversary overlay if any)
        result.json         — the TaskResult dict the orchestrator returns

Plus an index line appended to <traces_base>/index.jsonl per run for
cross-run discovery without scanning the tree.

run_id format: `YYYYMMDDTHHMMSSZ-<uuid8>` — sorts chronologically AND
collision-resistant across parallel runs.

`traces_base` resolution:
  1. Constructor arg
  2. $MOLE_TRACE_DIR env
  3. ./traces (local default)
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    """Stable, chronologically-sortable run id with collision resistance."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}-{suffix}"


def default_traces_base() -> Path:
    """Resolve where traces should land: ``$MOLE_TRACE_DIR`` if set, else ``./traces``."""
    env = os.environ.get("MOLE_TRACE_DIR")
    if env:
        return Path(env)
    return Path("traces")


@dataclass
class RunManifest:
    """Metadata for a single live run."""
    run_id: str
    task_id: str
    task_name: str
    adversary: str | None
    is_malicious: bool
    focal_account: str
    llm_backend: str
    llm_model: str
    temperature: float
    max_turns_per_stage: int
    sim_start_iso: str
    started_at_iso: str
    completed_at_iso: str = ""
    execution_time_s: float = 0.0
    score: float | None = None
    rubric_summary: dict[str, bool] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "adversary": self.adversary,
            "is_malicious": self.is_malicious,
            "focal_account": self.focal_account,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_turns_per_stage": self.max_turns_per_stage,
            "sim_start_iso": self.sim_start_iso,
            "started_at_iso": self.started_at_iso,
            "completed_at_iso": self.completed_at_iso,
            "execution_time_s": self.execution_time_s,
            "score": self.score,
            "rubric_summary": dict(self.rubric_summary),
            "error": self.error,
        }


class TraceWriter:
    """Owns the per-run directory and append-only writes to its files.

    Construction creates the directory. open_for_writes() can be used
    in a context-manager but isn't required — individual write methods
    are safe to call in any order.
    """

    def __init__(
        self,
        *,
        traces_base: Path | None = None,
        task_id: str,
        run_id: str | None = None,
    ) -> None:
        base = traces_base or default_traces_base()
        self.run_id = run_id or new_run_id()
        self.task_id = task_id
        self.run_dir = Path(base) / task_id / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = Path(base) / "index.jsonl"

    # ── paths individual writers use ───────────────────────────────

    @property
    def audit_jsonl(self) -> Path:
        return self.run_dir / "audit.jsonl"

    @property
    def transcript_jsonl(self) -> Path:
        return self.run_dir / "agent_transcript.jsonl"

    @property
    def system_prompt(self) -> Path:
        return self.run_dir / "system_prompt.txt"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def result_path(self) -> Path:
        return self.run_dir / "result.json"

    # ── writes ─────────────────────────────────────────────────────

    def write_system_prompt(self, prompt: str) -> None:
        self.system_prompt.write_text(prompt, encoding="utf-8")

    def append_transcript(
        self,
        *,
        stage_id: str,
        role: str,
        content: str | list,
        tool_calls: list | None = None,
        tool_results: list | None = None,
    ) -> None:
        """Append one entry to agent_transcript.jsonl. role ∈
        {system, user, assistant, tool}; content + tool_calls +
        tool_results are dumped JSON-safely."""
        entry = {
            "stage": stage_id,
            "role": role,
            "content": content if isinstance(content, str) else _safe_json(content),
            "tool_calls": _safe_json(tool_calls) if tool_calls else None,
            "tool_results": _safe_json(tool_results) if tool_results else None,
        }
        with self.transcript_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def write_manifest(self, manifest: RunManifest) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )

    def write_result(self, result: dict) -> None:
        self.result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def append_index(self, manifest: RunManifest) -> None:
        """Append one line to the top-level index.jsonl. Cheap cross-run
        discovery — `jq` / `awk` over this file to find runs by
        (threat, adversary, score) without scanning every directory."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "run_id": manifest.run_id,
                "task_id": manifest.task_id,
                "adversary": manifest.adversary,
                "is_malicious": manifest.is_malicious,
                "model": manifest.llm_model,
                "started_at_iso": manifest.started_at_iso,
                "score": manifest.score,
                "error": manifest.error or "",
                "path": str(self.run_dir),
            }) + "\n")


def _safe_json(obj: Any) -> Any:
    """Convert nested Python objects to a JSON-safe form."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return _safe_json(vars(obj))
    return repr(obj)
