"""Native completion parsing preserves tools while rejecting incomplete syntax."""

import pytest

from exp.common.tasks import ToolSchema
from exp.optimize.claas.backends.verl.decoding import (
    HermesCompletionDecoder,
    Qwen35CompletionDecoder,
    TextCompletionDecoder,
)


def test_hermes_decodes_tools_and_omits_reasoning() -> None:
    """Reasoning stays out of executable actions and call IDs remain stable."""
    text = (
        "<think>private reasoning</think>"
        '<tool_call>{"name":"search","arguments":{"q":"claim"}}</tool_call>'
    )
    decoder = HermesCompletionDecoder()
    action = decoder.decode(text, "req")
    assert action.content is None
    assert action.tool_calls[0].arguments == {"q": "claim"}
    assert action == decoder.decode(text, "req")
    assert action.tool_calls[0].call_id != decoder.decode(text, "another").tool_calls[0].call_id


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call",
        "</tool_call",
        "Plan: <tool_call ",
        '<tool_call>{"name":"search","arguments":{}}</tool_call',
        '<tool_call>{"name":"search"}',
        "<think>unfinished",
        '<tool_call>{"name":"search","arguments":[]}</tool_call>',
    ],
)
def test_malformed_tools_fail_closed(text: str) -> None:
    """Partial or unsupported tool syntax cannot turn silently into a successful action."""
    with pytest.raises(ValueError):
        HermesCompletionDecoder().decode(text, "req")


def test_text_decoder_is_explicit() -> None:
    """A caller deliberately selecting ordinary text receives the exact text."""
    text = "  leading and trailing  "
    assert TextCompletionDecoder().decode(text, "req").content == text


def test_qwen35_native_xml_uses_tool_types_without_guessing() -> None:
    """The official Qwen3.5 format preserves string-like numbers and JSON containers."""

    tool = ToolSchema(
        name="search",
        description="Search",
        input_schema={
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "count": {"type": "integer"},
                "filters": {"type": "object"},
            },
            "required": ["q"],
        },
    )
    text = (
        "reasoning</think><tool_call>\n<function=search>\n"
        "<parameter=q>\n123\n</parameter>\n"
        "<parameter=count>\n3\n</parameter>\n"
        '<parameter=filters>\n{"site":"news"}\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    action = Qwen35CompletionDecoder().decode(text, "request", (tool,))
    assert action.tool_calls[0].arguments == {"q": "123", "count": 3, "filters": {"site": "news"}}
    with pytest.raises(ValueError, match="native function"):
        Qwen35CompletionDecoder().decode(
            '<tool_call>{"name":"search","arguments":{}}</tool_call>', "r", (tool,)
        )
