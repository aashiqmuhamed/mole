"""State-backend registration aggregator.

Each submodule's own __init__.py applies a `@StateManager.register("<name>")`
decorator to its concrete manager class. Importing this package transitively
imports each submodule, registering every backend.

Imports are lazily tolerant: a missing backend dependency (e.g. python-gitlab
not installed in a dev env that doesn't need GitLab) logs a debug message and
continues, rather than crashing the entire benchmark import.
"""
from __future__ import annotations

import logging

from .base import StateManager  # re-export

logger = logging.getLogger(__name__)

_OPTIONAL_BACKENDS = (
    "email",
    "gitlab",
    "owncloud",
    "rocketchat",
    "plane",
    "eval_server",
    "model_registry",
    "secrets_store",
    "org",
)

for _name in _OPTIONAL_BACKENDS:
    try:
        __import__(f"{__name__}.{_name}")
    except ImportError as exc:
        # Backend not implemented yet, or its deps not installed in this env.
        logger.debug("Skipping backend %r: %s", _name, exc)

__all__ = ["StateManager"]
