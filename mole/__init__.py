"""mole — observability-intervention benchmark for AI-lab insiders.

Importing this package is what registers every state backend via decorator
side-effects, so it must be imported before any task is loaded. Use the
wrapper entry `python -m mole.run_task` to guarantee that.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Side-effect import: registering ctx-backends happens here.
from . import state  # noqa: F401
