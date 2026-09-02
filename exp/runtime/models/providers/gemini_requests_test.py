"""Tests for the shared native Gemini generateContent payload builders."""

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall, ToolChoice
from exp.common.tasks import ToolSchema
from exp.runtime.models.providers.gemini_requests import (
    GEMINI_THOUGHT_SIGNATURE_BYPASS,
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


def test_gemini_generate_request_omits_unproven_top_k_and_logprobs() -> None:
    """Gemini discovery must prove model-specific top-k/logprob support first."""
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hi"),),
        top_p=0.8,
        top_k=20,
        logprobs=True,
        top_logprobs=5,
    )
    payload = gemini_generate_request("gemini-2.5-pro", request, supports_logprobs=True)
    generation = cast("dict[str, object]", payload["generationConfig"])
    assert generation["topP"] == 0.8
    assert "topK" not in generation
    assert "responseLogprobs" not in generation
    assert "logprobs" not in generation


def test_gemini_generate_request_adds_stop_and_json_schema_when_requested() -> None:
    """Shared Gemini builders preserve gateway stop and structured-output controls."""
    request = ModelRequest(messages=(ModelMessage(role="user", content="hi"),))
    schema: JsonObject = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    payload = gemini_generate_request(
        "gemini-2.5-pro",
        request,
        stop_sequences=("DONE",),
        response_json_schema=schema,
    )

    generation = cast("dict[str, object]", payload["generationConfig"])
    assert generation["stopSequences"] == ["DONE"]
    assert generation["responseMimeType"] == "application/json"
    assert generation["responseJsonSchema"] == schema


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
        "parts": [
            {
                "functionCall": {"name": "lookup", "args": {"q": "x"}},
                "thoughtSignature": GEMINI_THOUGHT_SIGNATURE_BYPASS,
            }
        ],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "lookup", "response": {"content": "answer"}}}],
    }


def test_gemini_generate_request_replays_every_function_call_with_the_bypass_signature() -> None:
    """Each replayed function call carries Gemini's documented placeholder signature.

    Gemini 3 rejects a follow-up turn (HTTP 400, ``missing a thought_signature``)
    when a replayed ``functionCall`` part has no signature, and the gateway's
    public surfaces cannot carry the real one back from the client.
    """
    calls = (
        ToolCall(call_id="call-1", name="write", arguments={"path": "a"}),
        ToolCall(call_id="call-2", name="read", arguments={"path": "b"}),
    )
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="go"),
            ModelMessage(
                role="assistant",
                content="working",
                assistant_action=AssistantAction(tool_calls=calls),
            ),
            ModelMessage(role="tool", content="ok", tool_call_id="call-1"),
            ModelMessage(role="tool", content="ok", tool_call_id="call-2"),
        ),
    )
    payload = gemini_generate_request("gemini-3-flash-preview", request)

    contents = payload["contents"]
    assert isinstance(contents, list)
    model_turn = contents[1]
    assert isinstance(model_turn, dict)
    parts = model_turn["parts"]
    assert isinstance(parts, list)
    assert parts[0] == {"text": "working"}
    for part in parts[1:]:
        assert isinstance(part, dict)
        assert "functionCall" in part
        assert part["thoughtSignature"] == "skip_thought_signature_validator"


def test_gemini_generate_request_rejects_an_unlinked_tool_result() -> None:
    """A tool result without its preceding call fails instead of dropping linkage."""
    request = ModelRequest(
        messages=(ModelMessage(role="tool", content="orphan", tool_call_id="missing"),),
    )
    with pytest.raises(ValueError, match="no preceding tool-call name"):
        gemini_generate_request("gemini-2.5-pro", request)
