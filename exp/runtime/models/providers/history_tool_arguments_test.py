"""Each wire receives a replayed call's raw text or its disclosed object stand-in."""

from __future__ import annotations

import json

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolArgumentsTreatment, ToolCall
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.gateway.surfaces import GatewayApiSurface
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.history_tool_arguments import (
    TREATMENT_ORDER,
    history_tool_argument_disclosures,
    opaque_tool_arguments_wire,
    treatment_disclosure,
)

_RAW = '{"path": "src/ma'
_STAND_IN: JsonObject = {"path": "src/ma"}

_OPENAI = GatewayWireProfile(
    dialect="openai_compatible", url="https://api.openai.com/v1/chat/completions"
)
_AZURE = GatewayWireProfile(
    dialect="openai_compatible",
    url="https://unit.openai.azure.com/openai/deployments/gpt/chat/completions",
)
_OPENAI_RESPONSES = GatewayWireProfile(
    dialect="openai_responses", url="https://api.openai.com/v1/responses"
)
_VLLM = GatewayWireProfile(
    dialect="openai_compatible", url="http://10.0.0.7:8000/v1/chat/completions"
)
_OPENROUTER = GatewayWireProfile(
    dialect="openai_compatible", url="https://openrouter.ai/api/v1/chat/completions"
)
_OPENROUTER_RESPONSES = GatewayWireProfile(
    dialect="openai_responses", url="https://openrouter.ai/api/v1/responses"
)
_ANTHROPIC = GatewayWireProfile(
    dialect="anthropic_messages",
    url="https://api.anthropic.com/v1/messages",
    model_id="claude-sonnet-4-6",
    maximum_output_tokens=128_000,
)
_GEMINI = GatewayWireProfile(
    dialect="gemini_generate_content", url="https://gemini.test", model_id="gemini-3-pro"
)
_BEDROCK = GatewayWireProfile(
    dialect="bedrock_converse_stream",
    url="https://bedrock-runtime.us-east-1.amazonaws.com",
    model_id="anthropic.claude-sonnet-4-6",
)


def _request(
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
    treatment: ToolArgumentsTreatment | None = "truncated_json",
) -> GatewayRequest:
    """A history replaying one call whose text was cut off, then its result."""
    call = (
        ToolCall(
            call_id="call_1",
            name="read",
            arguments=_STAND_IN,
            raw_arguments=_RAW,
            arguments_treatment=treatment,
        )
        if treatment is not None
        else ToolCall(call_id="call_1", name="read", arguments={"path": "a"})
    )
    return GatewayRequest(
        surface=surface,
        stream=True,
        include_usage=True,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(role="assistant", tool_calls=(call,)),
            GatewayMessage(role="tool", tool_call_id="call_1", content="done"),
            GatewayMessage(role="user", content="next"),
        ),
    )


def _chat_arguments(payload: JsonObject) -> JsonValue:
    messages = payload["messages"]
    assert isinstance(messages, list)
    (assistant,) = [m for m in messages if isinstance(m, dict) and m.get("tool_calls")]
    assert isinstance(assistant, dict)
    calls = assistant["tool_calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    function = calls[0]["function"]
    assert isinstance(function, dict)
    return function["arguments"]


def _responses_arguments(payload: JsonObject) -> JsonValue:
    items = payload["input"]
    assert isinstance(items, list)
    (call,) = [i for i in items if isinstance(i, dict) and i.get("type") == "function_call"]
    assert isinstance(call, dict)
    return call["arguments"]


@pytest.mark.parametrize("profile", (_OPENAI, _AZURE))
def test_openai_chat_wires_take_the_raw_text_byte_exact(profile: GatewayWireProfile) -> None:
    """OpenAI's own API treats ``arguments`` as opaque: nothing rewritten, nothing disclosed."""
    assert opaque_tool_arguments_wire(profile)
    assert _chat_arguments(dialect_stream_payload(profile, _request())) == _RAW
    assert history_tool_argument_disclosures(profile, _request()) == ()


def test_openai_responses_wire_takes_the_raw_text_byte_exact() -> None:
    """The same holds for OpenAI's own Responses ``function_call`` item."""
    request = _request(GatewayApiSurface.RESPONSES)
    assert _responses_arguments(dialect_stream_payload(_OPENAI_RESPONSES, request)) == _RAW
    assert history_tool_argument_disclosures(_OPENAI_RESPONSES, request) == ()


@pytest.mark.parametrize("profile", (_VLLM, _OPENROUTER))
def test_other_chat_wires_take_the_stand_in_as_one_object(profile: GatewayWireProfile) -> None:
    """A server that decodes history arguments receives text that decodes to one object."""
    assert not opaque_tool_arguments_wire(profile)
    sent = _chat_arguments(dialect_stream_payload(profile, _request()))
    assert isinstance(sent, str) and json.loads(sent) == _STAND_IN
    assert history_tool_argument_disclosures(profile, _request()) == (
        "messages.tool_calls.function.arguments->repaired(truncated_json)",
    )


def test_a_foreign_responses_origin_takes_the_stand_in() -> None:
    """OpenRouter's Responses wire is a string wire, but not OpenAI's opaque one."""
    request = _request(GatewayApiSurface.RESPONSES)
    sent = _responses_arguments(dialect_stream_payload(_OPENROUTER_RESPONSES, request))
    assert isinstance(sent, str) and json.loads(sent) == _STAND_IN
    assert history_tool_argument_disclosures(_OPENROUTER_RESPONSES, request) == (
        "input.arguments->repaired(truncated_json)",
    )


def test_anthropic_tool_use_input_is_the_stand_in_object() -> None:
    """``tool_use.input`` must be an object."""
    payload = dialect_stream_payload(_ANTHROPIC, _request())
    messages = payload["messages"]
    assert isinstance(messages, list)
    blocks = [
        block
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    assert [block["input"] for block in blocks] == [_STAND_IN]
    assert history_tool_argument_disclosures(_ANTHROPIC, _request()) == (
        "messages.tool_calls.function.arguments->repaired(truncated_json)",
    )


def test_gemini_function_call_args_are_the_stand_in_object() -> None:
    """``functionCall.args`` must be an object."""
    payload = dialect_stream_payload(_GEMINI, _request())
    contents = payload["contents"]
    assert isinstance(contents, list)
    args = [
        part["functionCall"]["args"]
        for content in contents
        if isinstance(content, dict)
        for part in content.get("parts", [])
        if isinstance(part, dict) and isinstance(part.get("functionCall"), dict)
    ]
    assert args == [_STAND_IN]
    assert history_tool_argument_disclosures(_GEMINI, _request()) != ()


def test_bedrock_tool_use_input_is_the_stand_in_object() -> None:
    """Converse ``toolUse.input`` must be a document object."""
    payload = dialect_stream_payload(_BEDROCK, _request())
    messages = payload["messages"]
    assert isinstance(messages, list)
    inputs = [
        block["toolUse"]["input"]
        for message in messages
        if isinstance(message, dict)
        for block in message.get("content", [])
        if isinstance(block, dict) and isinstance(block.get("toolUse"), dict)
    ]
    assert inputs == [_STAND_IN]
    assert history_tool_argument_disclosures(_BEDROCK, _request()) != ()


@pytest.mark.parametrize("profile", (_OPENAI, _VLLM, _ANTHROPIC))
def test_an_untreated_history_is_untouched_and_undisclosed(profile: GatewayWireProfile) -> None:
    """Valid object arguments never gain a disclosure on any wire."""
    assert history_tool_argument_disclosures(profile, _request(treatment=None)) == ()


def test_every_treatment_has_one_disclosure_per_surface() -> None:
    """The disclosure names the public field of the surface the caller used."""
    chat = [treatment_disclosure(GatewayApiSurface.CHAT_COMPLETIONS, t) for t in TREATMENT_ORDER]
    responses = [treatment_disclosure(GatewayApiSurface.RESPONSES, t) for t in TREATMENT_ORDER]
    assert chat == [
        "messages.tool_calls.function.arguments->decoded(double_encoded_json)",
        "messages.tool_calls.function.arguments->repaired(truncated_json)",
        "messages.tool_calls.function.arguments->wrapped(non_object_json)",
        "messages.tool_calls.function.arguments->wrapped(invalid_json)",
    ]
    assert responses == [
        "input.arguments->decoded(double_encoded_json)",
        "input.arguments->repaired(truncated_json)",
        "input.arguments->wrapped(non_object_json)",
        "input.arguments->wrapped(invalid_json)",
    ]
