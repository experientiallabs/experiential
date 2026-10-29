"""Tests for the shared native Gemini generateContent payload builders."""

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall, ToolChoice
from exp.common.models.content import DocumentContentPart, ImageContentPart, TextContentPart
from exp.common.models.model import ReasoningEffort
from exp.common.tasks import ToolSchema
from exp.runtime.models.providers.gemini_requests import (
    GEMINI_MISSING_TOOL_RESULT,
    GEMINI_THOUGHT_SIGNATURE_BYPASS,
    gemini_generate_request,
    gemini_model_path,
)


@pytest.mark.parametrize("maximum", (None, 1, 8_192, 128_000))
def test_gateway_style_gemini_builder_preserves_omission_or_exact_cap(maximum: int | None) -> None:
    """Gateway calls opt out of the non-gateway model client's explicit default."""
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hi"),), maximum_output_tokens=maximum
    )
    payload = gemini_generate_request("gemini-2.5-pro", request, default_maximum_output_tokens=None)
    generation = payload["generationConfig"]
    assert isinstance(generation, dict)
    if maximum is None:
        assert "maxOutputTokens" not in generation
    else:
        assert generation["maxOutputTokens"] == maximum


def test_model_client_gemini_default_remains_explicit() -> None:
    """The non-gateway execution budget is unchanged by gateway omission policy."""
    request = ModelRequest(messages=(ModelMessage(role="user", content="hi"),))
    payload = gemini_generate_request("gemini-2.5-pro", request)
    assert payload["generationConfig"] == {"maxOutputTokens": 4096}


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
    when a replayed ``functionCall`` part has no signature, so a turn whose
    calls carry none (the caller dropped them, or the history came from another
    provider) replays every call on the placeholder.
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


def _function_responses(content: JsonObject) -> list[tuple[str, object]]:
    parts = cast("list[JsonObject]", content["parts"])
    responses = [cast("JsonObject", part["functionResponse"]) for part in parts]
    return [
        (cast("str", response["name"]), cast("JsonObject", response["response"])["content"])
        for response in responses
    ]


def _parallel_calls() -> AssistantAction:
    return AssistantAction(
        tool_calls=(
            ToolCall(call_id="call-1", name="write", arguments={"path": "a"}),
            ToolCall(call_id="call-2", name="read", arguments={"path": "b"}),
            ToolCall(call_id="call-3", name="list", arguments={}),
        )
    )


def test_parallel_tool_results_answer_their_model_turn_in_one_user_content() -> None:
    """One functionResponse per functionCall, in ONE content, in call order.

    Gemini refuses "the number of function response parts is equal to the
    number of function call parts" when each tool message becomes its own
    content (51 production requests across about 20 organizations, 2026-10-18).
    """
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="go"),
            ModelMessage(role="assistant", assistant_action=_parallel_calls()),
            ModelMessage(role="tool", content="r2", tool_call_id="call-2"),
            ModelMessage(role="tool", content="r1", tool_call_id="call-1"),
            ModelMessage(role="tool", content="r3", tool_call_id="call-3"),
        ),
    )
    contents = cast(
        "list[JsonObject]", gemini_generate_request("gemini-3.5-flash", request)["contents"]
    )

    assert [content["role"] for content in contents] == ["user", "model", "user"]
    assert _function_responses(contents[2]) == [("write", "r1"), ("read", "r2"), ("list", "r3")]


def test_user_text_between_parallel_results_follows_the_grouped_responses() -> None:
    """A client reminder between two results no longer splits the answer turn."""
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="go"),
            ModelMessage(role="assistant", assistant_action=_parallel_calls()),
            ModelMessage(role="tool", content="r1", tool_call_id="call-1"),
            ModelMessage(role="user", content="reminder"),
            ModelMessage(role="tool", content="r2", tool_call_id="call-2"),
            ModelMessage(role="tool", content="r3", tool_call_id="call-3"),
            ModelMessage(role="assistant", content="done"),
        ),
    )
    contents = cast(
        "list[JsonObject]", gemini_generate_request("gemini-3.5-flash", request)["contents"]
    )

    assert [content["role"] for content in contents] == ["user", "model", "user", "user", "model"]
    assert _function_responses(contents[2]) == [("write", "r1"), ("read", "r2"), ("list", "r3")]
    assert contents[3] == {"role": "user", "parts": [{"text": "reminder"}]}


def test_an_unanswered_parallel_call_gets_the_placeholder_once_a_later_turn_exists() -> None:
    """An interrupted call is answered with the placeholder so the turn stays countable."""
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="go"),
            ModelMessage(role="assistant", assistant_action=_parallel_calls()),
            ModelMessage(role="tool", content="r1", tool_call_id="call-1"),
            ModelMessage(role="user", content="stop, do something else"),
        ),
    )
    contents = cast(
        "list[JsonObject]", gemini_generate_request("gemini-3.5-flash", request)["contents"]
    )

    assert [content["role"] for content in contents] == ["user", "model", "user", "user"]
    assert _function_responses(contents[2]) == [
        ("write", "r1"),
        ("read", GEMINI_MISSING_TOOL_RESULT),
        ("list", GEMINI_MISSING_TOOL_RESULT),
    ]


def test_a_trailing_model_turn_awaiting_results_is_left_unanswered() -> None:
    """No later turn means nothing for Gemini to count; no placeholder is invented."""
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="go"),
            ModelMessage(role="assistant", assistant_action=_parallel_calls()),
        ),
    )
    contents = cast(
        "list[JsonObject]", gemini_generate_request("gemini-3.5-flash", request)["contents"]
    )

    assert [content["role"] for content in contents] == ["user", "model"]


def test_gemini_generate_request_replays_a_signed_turn_exactly_as_issued() -> None:
    """A carried signature returns on its call; Gemini's unsigned sibling stays unsigned.

    Gemini signs only the first call of a parallel batch, so the turn replays
    in the shape Google issued it rather than mixing in the placeholder.
    """
    calls = (
        ToolCall(
            call_id="call-1",
            name="get_weather",
            arguments={"city": "Paris"},
            provider_thought_signature="c2lnbmF0dXJlLWE=",
        ),
        ToolCall(call_id="call-2", name="get_weather", arguments={"city": "Tokyo"}),
    )
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="go"),
            ModelMessage(role="assistant", assistant_action=AssistantAction(tool_calls=calls)),
            ModelMessage(role="tool", content="sunny", tool_call_id="call-1"),
            ModelMessage(role="tool", content="sunny", tool_call_id="call-2"),
        ),
    )
    payload = gemini_generate_request("gemini-3-flash-preview", request)

    contents = payload["contents"]
    assert isinstance(contents, list)
    assert contents[1] == {
        "role": "model",
        "parts": [
            {
                "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                "thoughtSignature": "c2lnbmF0dXJlLWE=",
            },
            {"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}},
        ],
    }


def test_gemini_generate_request_rejects_an_unlinked_tool_result() -> None:
    """A tool result without its preceding call fails instead of dropping linkage."""
    request = ModelRequest(
        messages=(ModelMessage(role="tool", content="orphan", tool_call_id="missing"),),
    )
    with pytest.raises(ValueError, match="no preceding tool-call name"):
        gemini_generate_request("gemini-2.5-pro", request)


def test_gemini_generate_request_inlines_documents_in_caller_order() -> None:
    """PDF parts ride the user turn as ``inline_data`` at their positions among text."""
    pdf = "JVBERi0xLjQKJSBtaW5pbWFsIHBkZgo="
    request = ModelRequest(
        messages=(
            ModelMessage(
                role="user",
                content="first second",
                content_parts=(
                    TextContentPart(text="first "),
                    DocumentContentPart(data=pdf, name="a.pdf"),
                    TextContentPart(text="second"),
                    DocumentContentPart(data="JVBERi0xLjcK"),
                ),
            ),
        ),
    )
    payload = gemini_generate_request("gemini-2.5-pro", request)
    assert payload["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "first "},
                {"inline_data": {"mime_type": "application/pdf", "data": pdf}},
                {"text": "second"},
                {"inline_data": {"mime_type": "application/pdf", "data": "JVBERi0xLjcK"}},
            ],
        }
    ]


def test_gemini_folds_a_mid_conversation_system_turn_into_user_text_in_place() -> None:
    """systemInstruction hoists only the leading run; a later instruction keeps its position.

    Claude Code on the Chat wire injects a system turn after the first user
    turn and after every tool result; the gateway used to refuse the whole
    route for it. The text now rides as user content where the caller put it.
    """
    payload = gemini_generate_request(
        "gemini-2.5-pro",
        ModelRequest(
            messages=(
                ModelMessage(role="system", content="be terse"),
                ModelMessage(role="system", content="answer in English"),
                ModelMessage(role="user", content="hi"),
                ModelMessage(role="system", content="# Environment\nPlatform: linux"),
                ModelMessage(role="assistant", content="hello"),
                ModelMessage(role="system", content="<total_tokens>1</total_tokens>"),
                ModelMessage(role="user", content="go"),
            ),
            tools=(),
        ),
    )
    assert payload["systemInstruction"] == {
        "parts": [{"text": "be terse"}, {"text": "answer in English"}]
    }
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "hi\n\n# Environment\nPlatform: linux"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
        {"role": "user", "parts": [{"text": "<total_tokens>1</total_tokens>"}]},
        {"role": "user", "parts": [{"text": "go"}]},
    ]


def test_assistant_image_history_preserves_bytes_and_uses_signature_bypass() -> None:
    """Caller-owned chat image history is rebuilt in Gemini's model role."""
    image = ImageContentPart(media_type="image/png", data="iVBORw0KGgo=")
    request = ModelRequest(
        messages=(ModelMessage(role="assistant", content="", content_parts=(image,)),)
    )
    payload = gemini_generate_request("gemini-3.1-flash-image", request)
    assert payload["contents"] == [
        {
            "role": "model",
            "parts": [
                {
                    "inline_data": {"mime_type": "image/png", "data": "iVBORw0KGgo="},
                    "thoughtSignature": GEMINI_THOUGHT_SIGNATURE_BYPASS,
                }
            ],
        }
    ]


def test_a_reasoning_route_with_no_effort_asks_only_for_thoughts() -> None:
    """Thought summaries are requested even when no thinking level is pinned."""
    request = ModelRequest(messages=(ModelMessage(role="user", content="hi"),))
    payload = gemini_generate_request(
        "gemini-2.5-pro",
        request,
        supports_reasoning=True,
        default_maximum_output_tokens=None,
    )
    generation = cast("JsonObject", payload["generationConfig"])
    assert generation["thinkingConfig"] == {"includeThoughts": True}


def test_a_non_reasoning_route_sends_no_thinking_config() -> None:
    """A route without reasoning support never carries a thinking config."""
    request = ModelRequest(messages=(ModelMessage(role="user", content="hi"),))
    payload = gemini_generate_request(
        "gemini-2.5-pro",
        request,
        reasoning_effort="high",
        default_maximum_output_tokens=None,
    )
    generation = cast("JsonObject", payload.get("generationConfig", {}))
    assert "thinkingConfig" not in generation


def test_the_display_kill_switch_stops_asking_for_thoughts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EXP_GATEWAY_REASONING_DISPLAY=0`` sends no thinking config without an effort."""
    monkeypatch.setenv("EXP_GATEWAY_REASONING_DISPLAY", "0")
    request = ModelRequest(messages=(ModelMessage(role="user", content="hi"),))
    payload = gemini_generate_request(
        "gemini-2.5-pro",
        request,
        supports_reasoning=True,
        default_maximum_output_tokens=None,
    )
    assert "thinkingConfig" not in cast("JsonObject", payload["generationConfig"])


def _thinking_config(model_id: str, effort: str | None) -> object:
    """The thinkingConfig one Gemini model receives for one caller effort."""
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hi"),),
        reasoning_effort=cast("ReasoningEffort | None", effort),
    )
    payload = gemini_generate_request(model_id, request, supports_reasoning=True)
    generation = payload["generationConfig"]
    assert isinstance(generation, dict)
    return generation.get("thinkingConfig")


@pytest.mark.parametrize(
    ("model_id", "effort", "expected"),
    [
        # Gemini 2.5 has no thinkingLevel (both AI Studio and Vertex answer
        # 400), so every effort rides Google's OpenAI-compatible budgets.
        ("gemini-2.5-pro", "low", {"thinkingBudget": 1024, "includeThoughts": True}),
        ("gemini-2.5-pro", "medium", {"thinkingBudget": 8192, "includeThoughts": True}),
        ("gemini-2.5-pro", "high", {"thinkingBudget": 24576, "includeThoughts": True}),
        ("gemini-2.5-pro", None, {"includeThoughts": True}),
        ("gemini-2.5-flash", "none", {"thinkingBudget": 0}),
        ("gemini-2.5-flash", "low", {"thinkingBudget": 1024, "includeThoughts": True}),
        ("gemini-2.5-flash", None, {"includeThoughts": True}),
        # Flash-Lite thinks only when budgeted; Vertex refuses includeThoughts
        # on a request that leaves thinking off.
        ("gemini-2.5-flash-lite", None, None),
        ("gemini-2.5-flash-lite", "none", {"thinkingBudget": 0}),
        ("gemini-2.5-flash-lite", "low", {"thinkingBudget": 1024, "includeThoughts": True}),
        (
            "publishers/google/models/gemini-2.5-flash-lite",
            "high",
            {
                "thinkingBudget": 24576,
                "includeThoughts": True,
            },
        ),
        # Later families keep thinkingLevel; a zero budget only switches off.
        ("gemini-3-flash-preview", "low", {"thinkingLevel": "LOW", "includeThoughts": True}),
        ("gemini-3-flash-preview", None, {"includeThoughts": True}),
        ("gemini-4-argon", "none", {"thinkingBudget": 0}),
    ],
)
def test_gemini_thinking_controls_follow_each_family(
    model_id: str, effort: str | None, expected: object
) -> None:
    """Each family gets the one thinking control it accepts, live-probed 2026-10-19."""
    assert _thinking_config(model_id, effort) == expected


def test_gemini_2_5_pro_refuses_thinking_off() -> None:
    """2.5 Pro only works in thinking mode, so ``none`` fails before dispatch."""
    with pytest.raises(ValueError, match="none"):
        _thinking_config("gemini-2.5-pro", "none")


def test_no_thought_summaries_are_requested_when_display_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reasoning display kill switch leaves only the thinking control."""
    monkeypatch.setenv("EXP_GATEWAY_REASONING_DISPLAY", "0")
    assert _thinking_config("gemini-2.5-flash", "medium") == {"thinkingBudget": 8192}
    assert _thinking_config("gemini-2.5-flash", None) is None
