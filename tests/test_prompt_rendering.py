"""Every threat PROMPT must fully render — no leftover `{placeholder}`.

This would have caught the 2026-05-23 bug where threat 04's PROMPT
shipped literal `{ticket_id}` / `{sim_month}` / `{recon_csv_path}`
braces (render_prompt only filled `{persona}`/`{today}`), so the agent
saw the raw placeholders and stalled asking the user for the values.
The fix made render_prompt expand scalar METADATA keys too; this test
pins that every threat's prompt resolves with its own METADATA.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from mole.agent.prompts import render_prompt

TASKS_ROOT = Path(__file__).resolve().parent.parent / "tasks"

# Matches a leftover placeholder, including indexed forms the agent would
# see as garbage: {ticket_id}, {rotated_secrets[0]}, {recon_csv_path}.
_PLACEHOLDER = re.compile(r"\{[a-z_][a-z0-9_]*(?:\[\d+\])?\}")


def _threat_dirs() -> list[Path]:
    return [
        d for d in sorted(TASKS_ROOT.iterdir())
        if d.is_dir()
        and not d.name.startswith(".")
        and not d.name.startswith("_")
        and d.name != "00_persona_lookup"
        and (d / "task.py").exists()
    ]


def _load(task_dir: Path):
    spec = importlib.util.spec_from_file_location(f"_pr_{task_dir.name}", task_dir / "task.py")
    mod = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return mod


@pytest.mark.parametrize("task_dir", _threat_dirs(), ids=lambda d: d.name)
def test_prompt_renders_without_leftover_placeholders(task_dir: Path):
    mod = _load(task_dir)
    prompt = getattr(mod, "PROMPT", "")
    metadata = getattr(mod, "METADATA", {})
    rendered = render_prompt(
        prompt,
        focal_account=metadata.get("focal_account"),
        sim_start_iso="2026-04-27T09:00:00Z",
        metadata=metadata,
    )
    leftover = sorted(set(_PLACEHOLDER.findall(rendered)))
    # Also catch the collapse-to-"{" failure mode: an indexed placeholder
    # whose key wasn't a list renders as a bare "{" (first char of the
    # preserved "{key}" string). A correctly rendered prompt has no
    # unbalanced "{" left.
    assert "{" not in rendered, (
        f"{task_dir.name}: rendered prompt still contains a stray '{{' — "
        f"likely an indexed placeholder whose METADATA value isn't a list."
    )
    assert not leftover, (
        f"{task_dir.name}: unrendered placeholder(s) {leftover} — either the "
        f"PROMPT references a key not in METADATA, or the name differs from "
        f"the METADATA key."
    )


def test_render_prompt_survives_unbalanced_brace_in_value():
    """A METADATA value containing a stray/JSON brace must not crash the
    render (str.format_map would raise ValueError). render_prompt should
    return the last good pass instead."""
    out = render_prompt(
        "config looks like {example} today",
        focal_account=None,
        metadata={"example": '{"k": "v"}'},   # contains unbalanced-for-format braces
    )
    # Doesn't raise; the literal value is substituted in the first pass.
    assert "today" in out
