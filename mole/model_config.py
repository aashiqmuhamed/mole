"""Load Mole's model configuration (``configs/models.yaml``).

Replaces the legacy deployment registry. Provides per-model tuning (context
window, request concurrency, agent turn cap) and the paper role -> model mapping,
keyed by generic model ids / OpenRouter slugs rather than internal deployment
strings. Override the config path with ``MOLE_MODELS_CONFIG``.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


def _config_path() -> Path:
    override = os.environ.get("MOLE_MODELS_CONFIG")
    return Path(override) if override else _DEFAULT_PATH


@functools.lru_cache(maxsize=None)
def load_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {"defaults": {}, "models": {}, "roles": {}}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _defaults() -> dict[str, Any]:
    return load_config().get("defaults", {}) or {}


def model_meta(model_id: str) -> dict[str, Any]:
    """Config entry for a model id. Matches on the config key, the
    ``openrouter_slug``, or the ``paper_name``. Returns ``{}`` if unknown."""
    if not model_id:
        return {}
    models = load_config().get("models", {}) or {}
    if model_id in models:
        return models[model_id]
    for meta in models.values():
        if model_id in (meta.get("openrouter_slug"), meta.get("paper_name")):
            return meta
    return {}


def context_window_for(model_id: str, default: int = 128000) -> int:
    """Context-window size (tokens) for a model, used to budget the semantic
    monitor's transcript affordance."""
    return int(model_meta(model_id).get(
        "context_tokens", _defaults().get("context_tokens", default)))


def recommended_concurrency(model_id: str, default: int = 8) -> int:
    """Suggested request concurrency for a model. A generic starting point; tune
    to your provider's rate limits (e.g. ``VLLM_CONCURRENCY`` for self-hosting)."""
    return int(model_meta(model_id).get(
        "concurrency", _defaults().get("concurrency", default)))


def recommended_max_turns(model_id: str, default: int = 30) -> int:
    """Per-account-day agent turn cap for a model (verbose models need more)."""
    return int(model_meta(model_id).get(
        "max_turns", _defaults().get("max_turns", default)))


def slug_for(model_id: str) -> str | None:
    """OpenRouter slug for a model id, if configured."""
    return model_meta(model_id).get("openrouter_slug")


def role_models(role: str) -> list[str]:
    """Model ids configured for a role (``generators`` / ``monitors`` / ``judges``)."""
    return list(load_config().get("roles", {}).get(role, []) or [])
