"""LLM source-layer — provider-agnostic chat completion + tool calling.

See `llm/base.py` for the LLMClient Protocol. Concrete backends (select via
`LLM_BACKEND`, default `openrouter`):
  - openrouter:  hosted models via openrouter.ai (OPENROUTER_API_KEY / OPENROUTER_MODEL)
  - vllm:        any local OpenAI-compatible endpoint (VLLM_BASE_URL / VLLM_MODEL)
  - anthropic:   the Anthropic API directly (ANTHROPIC_API_KEY / ANTHROPIC_MODEL)
"""
from __future__ import annotations

from .base import (
    ChatMessage,
    ChatResponse,
    LLMClient,
    TokenUsage,
    ToolCall,
    ToolSchema,
)

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "LLMClient",
    "TokenUsage",
    "ToolCall",
    "ToolSchema",
]
