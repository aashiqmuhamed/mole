"""Construct an LLMClient by name from environment variables.

Three provider-agnostic backends are supported:
  - ``openrouter`` (default): hosted models via openrouter.ai. Set
    ``OPENROUTER_API_KEY`` and ``OPENROUTER_MODEL`` (an OpenRouter slug, e.g.
    ``deepseek/deepseek-v4``). This is the reproduction path used by
    ``configs/models.yaml``.
  - ``vllm``: any local OpenAI-compatible endpoint (``VLLM_BASE_URL``,
    ``VLLM_MODEL``), e.g. a self-hosted open-weight model.
  - ``anthropic``: the Anthropic API directly (``ANTHROPIC_API_KEY``,
    ``ANTHROPIC_MODEL``).
"""
from __future__ import annotations

import os
from typing import Literal

from .base import LLMClient
from .retry import wrap_with_retry

BackendName = Literal["openrouter", "vllm", "anthropic"]


def build_llm(backend: BackendName | str | None = None) -> LLMClient:
    """Return a configured LLMClient. Reads ``LLM_BACKEND`` if ``backend`` is
    None (default ``openrouter``).

    The returned client is wrapped in a RetryingLLMClient by default so transient
    429/5xx/``model_error`` failures don't kill a run. Disable with ``LLM_RETRY=0``.
    """
    name = (backend or os.environ.get("LLM_BACKEND", "openrouter")).strip().lower()

    if name in {"openrouter", "open_router", "or"}:
        from .openrouter import from_env as _or_from_env
        return wrap_with_retry(_or_from_env())
    if name in {"vllm", "local_vllm", "local"}:
        from .vllm import from_env as _vllm_from_env
        return wrap_with_retry(_vllm_from_env())
    if name in {"anthropic", "claude"}:
        from .anthropic import from_env
        return wrap_with_retry(from_env())
    raise ValueError(
        f"Unknown LLM backend: {name!r}. Supported: openrouter, vllm, anthropic."
    )
