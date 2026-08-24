"""Tests for the shared native Gemini generateContent payload builders."""

import pytest

from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall, ToolChoice
from exp.common.tasks import ToolSchema
from exp.runtime.models.providers.gemini_requests import (
    gemini_generate_request,
    gemini_model_path,
)


def test_gemini_model_path_strips_the_optional_wire_prefix() -> None:
    """Prefixed and bare model identifiers resolve to one route segment."""
    assert gemini_model_path("models/gemini-2.5-pro") == "gemini-2.5-pro"
    assert gemini_model_path("gemini-2.5-pro") == "gemini-2.5-pro"


def test_gemini_generate_request_builds_contents_tools_and_generation() -> None:
    """System text, tool schemas, tool choice, and sampling land in native shape."""
    request = ModelRequest(
        messages=(
            ModelMessage(role="system", content="be terse"),
            ModelMessage(role="user", content="hi"),
        ),
        tools=(ToolSchema(name="lookup", description="find", input_schema={"type": "object"}),),
        tool_choice=ToolChoice(name="lookup"),
        temperature=0.4,
        maximum_output_tokens=64,
    )
    payload = gemini_generate_request("gemini-2.5-pro", request)

    assert payload["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert payload["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "lookup",
                    "description": "find",
                    "parametersJsonSchema": {"type": "object"},
                }
            ]
        }
    ]
    assert payload["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["lookup"]}
    }
    assert payload["generationConfig"] == {"temperature": 0.4, "maxOutputTokens": 64}


def test_gemini_generate_request_links_tool_results_to_prior_calls() -> None:
    """Tool results resolve their function name from the preceding assistant call."""
    call = ToolCall(call_id="call-1", name="lookup", arguments={"q": "x"})
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="go"),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(tool_calls=(call,)),
            ),
            ModelMessage(role="tool", content="answer", tool_call_id="call-1"),
        ),
    )
    payload = gemini_generate_request("gemini-2.5-pro", request)

    contents = payload["contents"]
    assert isinstance(contents, list)
    assert contents[1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "lookup", "args": {"q": "x"}}}],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "lookup", "response": {"content": "answer"}}}],
    }


def test_gemini_generate_request_rejects_an_unlinked_tool_result() -> None:
    """A tool result without its preceding call fails instead of dropping linkage."""
    request = ModelRequest(
        messages=(ModelMessage(role="tool", content="orphan", tool_call_id="missing"),),
    )
    with pytest.raises(ValueError, match="no preceding tool-call name"):
        gemini_generate_request("gemini-2.5-pro", request)
