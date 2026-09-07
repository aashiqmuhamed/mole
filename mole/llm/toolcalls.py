"""Tool-call parsing helpers shared across OpenAI-compatible backends.

Some open models (notably Kimi K2.x) occasionally emit tool calls as a
Claude-style XML ``<function_calls>`` block inside the assistant message content
instead of populating the structured ``tool_calls`` field. These helpers parse
that block into :class:`ToolCall` objects and strip it from the transcript. They
are pure regex + ``ToolCall`` with no provider coupling, so every backend reuses
them.
"""
from __future__ import annotations

import re
from typing import Any

from .base import ToolCall

_XML_INVOKE_RE = re.compile(
    r'<invoke\s+name="([^"]+)"\s*(?:/>|>(.*?)</invoke>)',
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>',
    re.DOTALL,
)
_XML_FUNCTION_CALLS_BLOCK_RE = re.compile(
    r'\s*<function_calls>.*?</function_calls>\s*',
    re.DOTALL,
)


def _parse_xml_function_calls(
    content: str, sanitized_map: dict[str, str],
) -> list[ToolCall]:
    """Parse Claude-style XML function calls embedded in ``content``.

    Some models (e.g. Kimi K2.6) sometimes emit tool calls as::

        <function_calls>
          <invoke name="gitlab_list_projects"></invoke>
          <invoke name="plane_list_issues">
            <parameter name="project_id">INFRA</parameter>
          </invoke>
        </function_calls>

    instead of populating the structured ``tool_calls`` field. This parser
    extracts each invoke into a ToolCall -- including its parameters -- and
    reverses the sanitized->canonical name mapping (e.g. ``gitlab_list_projects``
    -> ``gitlab.list_projects``) so the dispatcher routes to the right manager
    method. Returns an empty list if no ``<invoke>`` tag is found.
    """
    out: list[ToolCall] = []
    for i, m in enumerate(_XML_INVOKE_RE.finditer(content)):
        name_raw = m.group(1)
        body = m.group(2) or ""
        args: dict[str, Any] = {}
        for pm in _XML_PARAM_RE.finditer(body):
            args[pm.group(1)] = pm.group(2).strip()
        canonical = sanitized_map.get(name_raw, name_raw)
        out.append(ToolCall(id=f"xml-{i}", name=canonical, arguments=args))
    return out


def _strip_xml_function_calls(content: str) -> str:
    """Remove the ``<function_calls>...</function_calls>`` block from content so
    the transcript / judge prompt don't double-count what was meant as a tool
    call. Leaves any preceding prose intact."""
    return _XML_FUNCTION_CALLS_BLOCK_RE.sub("", content).rstrip()
