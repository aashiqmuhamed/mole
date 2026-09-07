"""Persona loader — turns org_template.yaml rows into background-account Personas.

A Persona is the immutable role + service-membership data the generator
needs at runtime. It's a flat projection of the org row, with
groups, services, MBTI, and interests pulled forward so workflows can
gate behaviour on them (e.g., only `model-approval` members run the
release-approval workflow).

Holdouts (`holdout: true` in the org yaml) are excluded by default —
they're reserved for the L1P-O generalisation split and must never
appear in monitor training, so they should not generate benign
trajectories either.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Persona:
    """One background account's role + identity. Frozen so it's safe to share across coroutines."""
    id: str
    full_name: str
    email: str
    role: str
    groups: tuple[str, ...]
    services: dict[str, Any]
    mbti: str = ""
    interests: tuple[str, ...] = ()
    persona_text: str = ""
    operating_style: str = ""     # free-text per-agent working style (diversity)
    team: str = ""
    manager: str = ""
    is_manager: bool = False
    cohort: str = ""              # focal | background | holdout

    def in_group(self, group_id: str) -> bool:
        return group_id in self.groups


def _default_org_yaml() -> Path:
    env = os.environ.get("ORG_TEMPLATE_PATH")
    if env:
        return Path(env)
    # generator/persona_loader.py → generator → mole → benchmark
    return Path(__file__).resolve().parents[2] / "bootstrap" / "org_template.yaml"


def load_personas(
    org_yaml_path: str | Path | None = None,
    *,
    include_holdouts: bool = False,
) -> list[Persona]:
    """Read org yaml and return Personas.

    By default, holdouts are dropped. Pass `include_holdouts=True` only
    when intentionally inflating the personality pool (e.g., when the
    test split is generated separately).
    """
    path = Path(org_yaml_path or _default_org_yaml())
    if not path.exists():
        raise FileNotFoundError(f"org template YAML not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    # Build team / manager lookup so each persona carries its hierarchy.
    team_of: dict[str, str] = {}
    manager_of: dict[str, str] = {}
    managers: set[str] = set()
    for dept in doc.get("departments") or []:
        for team in dept.get("teams") or []:
            tid = team.get("id", "")
            mgr = team.get("manager")
            if mgr:
                team_of[mgr] = tid
                managers.add(mgr)
            for member in team.get("members") or []:
                team_of[member] = tid
                if mgr:
                    manager_of[member] = mgr

    personas: list[Persona] = []
    for emp in doc.get("accounts") or []:
        if emp.get("holdout") and not include_holdouts:
            continue
        eid = emp["id"]
        personas.append(Persona(
            id=eid,
            full_name=emp.get("full_name", ""),
            email=emp.get("email", ""),
            role=emp.get("role", ""),
            groups=tuple(emp.get("groups") or []),
            services=dict(emp.get("services") or {}),
            mbti=emp.get("mbti", ""),
            interests=tuple(emp.get("interests") or []),
            persona_text=emp.get("persona_text", ""),
            operating_style=emp.get("operating_style", ""),
            team=team_of.get(eid, ""),
            manager=manager_of.get(eid, ""),
            is_manager=eid in managers,
            cohort=emp.get("cohort", ""),
        ))
    return personas
