"""OpenRouter backend: constructs from env, errors cleanly without a key, and
tolerates the malformed `{}`-prefixed tool-call args DeepSeek models emit.

No live call is made (that needs a real OPENROUTER_API_KEY + spend); these lock
in the construction contract + the tolerant parser that keeps one bad tool call
from killing a session at turn 1.
"""
import pytest

from mole.llm import openrouter


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError) as e:
        openrouter.from_env()
    assert "OPENROUTER_API_KEY" in str(e.value)


def test_constructs_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
    c = openrouter.from_env()
    assert c.backend == "openrouter"
    assert c.model_id == "deepseek/deepseek-v4-flash"
    assert c._base_url == "https://openrouter.ai/api/v1"


def test_default_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert openrouter.from_env().model_id == "deepseek/deepseek-v4-pro"


def test_tolerant_tool_arg_parse():
    # DeepSeek emits an empty {} prefix before the real object; a bare json.loads
    # would raise "Extra data" (non-retryable) and abort the whole session.
    assert openrouter._parse_tool_args('{}{"city": "Paris"}') == {"city": "Paris"}
    assert openrouter._parse_tool_args('{"a": 1}') == {"a": 1}
    assert openrouter._parse_tool_args("not json at all") == {}
    assert openrouter._parse_tool_args({"already": "dict"}) == {"already": "dict"}
    assert openrouter._parse_tool_args(None) == {}


def test_factory_dispatch(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    from mole.llm.factory import build_llm
    w = build_llm()
    # wrapped in the retry client; the inner client is our OpenRouterClient
    assert w is not None
