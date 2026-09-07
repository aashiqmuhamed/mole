"""Regression test: Kimi K2.6 occasionally emits Claude-style XML
function-call syntax in the chat-completion `content` field instead of
populating the structured `tool_calls`. The client falls back to
parsing those out so we don't silently lose tool calls.

Observed format (Kimi, generator session):
    I'll start by ...
    <function_calls>
    <invoke name="gitlab_list_projects"></invoke>
    <invoke name="plane_list_issues"></invoke>
    </function_calls>
"""
from __future__ import annotations

from mole.llm.toolcalls import (
    _parse_xml_function_calls,
    _strip_xml_function_calls,
)


def test_parses_kimi_iter5_canonical_format():
    content = (
        "I'll start by getting oriented with the current projects and tickets, "
        "then move forward with both the cover task and the main objective.\n\n"
        "<function_calls>\n"
        '<invoke name="gitlab_list_projects"></invoke>\n'
        '<invoke name="plane_list_issues"></invoke>\n'
        "</function_calls>"
    )
    sanitized_map = {
        "gitlab_list_projects": "gitlab.list_projects",
        "plane_list_issues": "plane.list_issues",
    }
    tcs = _parse_xml_function_calls(content, sanitized_map)
    assert len(tcs) == 2
    assert tcs[0].name == "gitlab.list_projects"
    assert tcs[0].arguments == {}
    assert tcs[1].name == "plane.list_issues"
    assert tcs[1].arguments == {}


def test_parses_invoke_with_parameters():
    """When Kimi includes <parameter> tags inside <invoke>, extract them
    into the ToolCall arguments dict."""
    content = (
        "<function_calls>\n"
        '<invoke name="plane_get_issue">\n'
        '<parameter name="project_id">INFRA</parameter>\n'
        '<parameter name="issue_id">INFRA-204</parameter>\n'
        "</invoke>\n"
        "</function_calls>"
    )
    sanitized_map = {"plane_get_issue": "plane.get_issue"}
    tcs = _parse_xml_function_calls(content, sanitized_map)
    assert len(tcs) == 1
    assert tcs[0].name == "plane.get_issue"
    assert tcs[0].arguments == {"project_id": "INFRA", "issue_id": "INFRA-204"}


def test_unmapped_name_passes_through_as_is():
    """Names not in sanitized_map (e.g. an unsanitized "gitlab.list_files"
    if Kimi happens to emit one) pass through unchanged."""
    content = '<invoke name="gitlab.list_files"></invoke>'
    tcs = _parse_xml_function_calls(content, sanitized_map={})
    assert tcs == [tcs[0]]  # smoke: returns one element
    assert tcs[0].name == "gitlab.list_files"


def test_no_xml_returns_empty():
    """Plain prose with no <invoke> tags yields no tool calls."""
    assert _parse_xml_function_calls(
        "Just a plain assistant message with no tools.", {},
    ) == []


def test_strip_removes_block_and_keeps_prose():
    """Stripping the function_calls block leaves any preceding prose
    intact — so the transcript writer doesn't double-count the XML as
    user-facing text."""
    content = (
        "Plan: orient first, then act.\n\n"
        "<function_calls>\n"
        '<invoke name="gitlab_list_projects"></invoke>\n'
        "</function_calls>"
    )
    out = _strip_xml_function_calls(content)
    assert out == "Plan: orient first, then act."


def test_self_closing_invoke_is_handled():
    """Kimi can also emit <invoke name="X" /> with no body."""
    content = '<invoke name="foo_bar" />'
    tcs = _parse_xml_function_calls(content, {"foo_bar": "foo.bar"})
    assert len(tcs) == 1
    assert tcs[0].name == "foo.bar"
    assert tcs[0].arguments == {}
