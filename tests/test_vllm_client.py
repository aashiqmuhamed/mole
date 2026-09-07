"""Unit tests for the local vLLM (OpenAI-compatible) backend.

The hot path POSTs the request via the SDK's low-level
``client.post("/chat/completions", cast_to=ChatCompletion, body=...)`` to skip
the per-call ``maybe_transform`` (a recursive pure-Python walk of the whole
request that, under the GIL, starves the event loop and collapses server-side
concurrency). These tests pin that the bypass is behaviourally identical to the
old ``client.chat.completions.create(**kwargs)`` path:

  1. the request JSON is byte-identical (incl. the VLLM_DISABLE_THINKING path), and
  2. a full round-trip parses the response the same way, including the
     dotted-tool-name sanitize-out / reverse-on-parse kimi workaround.

Mocks the HTTP transport so no server, key, or network is needed.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion

from mole.llm.base import ChatMessage, ToolSchema
from mole.llm.vllm import VLLMClient


def _mock_openai(handler):
    """An OpenAI client whose transport is a MockTransport (never hits the network)."""
    return OpenAI(base_url="http://mock/v1", api_key="EMPTY",
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


_CANNED_EMPTY = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.mark.parametrize("disable_thinking", [False, True])
def test_post_bypass_sends_identical_request_to_create(disable_thinking):
    """The low-level ``.post(body=...)`` must send byte-identical request JSON to the
    old ``.chat.completions.create(**kwargs)`` path, including the
    VLLM_DISABLE_THINKING ``extra_body`` -> top-level-body merge."""
    captured: dict = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_CANNED_EMPTY)

    messages = [
        {"role": "system", "content": "you are alice"},
        {"role": "user", "content": "list the repos"},
    ]
    tools = [{"type": "function", "function": {
        "name": "gitlab_list_projects", "description": "list projects",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": []}}}]
    base = {"model": "m", "messages": messages, "temperature": 0.0,
            "tools": tools, "max_tokens": 128, "seed": 7}

    # old path: .create(**kwargs) with extra_body (SDK merges extra_body into the body)
    create_kwargs = dict(base)
    if disable_thinking:
        create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    _mock_openai(handler).chat.completions.create(**create_kwargs)
    body_create = captured["body"]

    # new path: raw .post with extra_body content merged directly into the body
    body = dict(base)
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    _mock_openai(handler).post("/chat/completions", cast_to=ChatCompletion, body=body)
    body_post = captured["body"]

    assert body_create == body_post


def test_complete_round_trips_request_and_response():
    """Drive the real VLLMClient.complete() on the .post path: the dotted tool name is
    sanitized to underscores on the wire and restored on the way back, arguments are
    JSON-parsed, usage + finish_reason map through, and content is preserved."""
    canned = {
        "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "deepseek-v4-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "on it",
            "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "gitlab_list_projects", "arguments": "{\"q\": \"infra\"}"}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
    }
    captured: dict = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=canned)

    client = VLLMClient(base_url="http://mock/v1", model="deepseek-v4-flash")
    client._clients = [_mock_openai(handler)]

    resp = asyncio.run(client.complete(
        [ChatMessage(role="system", content="you are alice"),
         ChatMessage(role="user", content="list infra repos")],
        tools=[ToolSchema(name="gitlab.list_projects", description="list projects",
                          parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": []})],
        temperature=0.0, max_tokens=100, seed=7,
    ))

    # request side: dotted name -> underscore on the wire; scalars threaded through
    assert captured["body"]["tools"][0]["function"]["name"] == "gitlab_list_projects"
    assert captured["body"]["model"] == "deepseek-v4-flash"
    assert captured["body"]["seed"] == 7
    assert captured["body"]["max_tokens"] == 100

    # response side: parsed identically, dotted name restored, args parsed, usage mapped
    assert resp.content == "on it"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "gitlab.list_projects"
    assert resp.tool_calls[0].arguments == {"q": "infra"}
    assert resp.usage.input_tokens == 123
    assert resp.usage.output_tokens == 45
    assert resp.finish_reason == "tool_calls"
    assert resp.backend == "vllm"
    assert resp.model_id == "deepseek-v4-flash"
