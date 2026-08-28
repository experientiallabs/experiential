"""Contracts for Fireworks reasoning history and exact reasoning efforts."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    OpaqueReasoningContentBlock,
    ToolCall,
)
from exp.common.tasks import ToolSchema
from exp.runtime.models.providers.errors import (
    ProviderParameterError,
    UnsupportedReasoningEffortError,
)
from exp.runtime.models.providers.fireworks import (
    decode_reasoning_content,
    encode_reasoning_content,
    fireworks_reasoning_effort,
    fireworks_reasoning_efforts,
    is_fireworks_base_url,
    normalized_fireworks_default,
    prepare_model_reasoning_history,
    reasoning_content_route_sha256,
)
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_MODEL_IDS = (
    "accounts/fireworks/models/deepseek-v4-flash-0731",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/kimi-k2p7-code",
)
_ROUTE_SHA256 = "a" * 64
_OTHER_ROUTE_SHA256 = "b" * 64


def _snapshot(model_id: str, *, connection_sha256: str = _ROUTE_SHA256) -> ModelSnapshot:
    """Build one exact Fireworks model identity."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fireworks",
        model_id=model_id,
        revision="fixture-revision",
        capabilities_sha256="c" * 64,
        connection_sha256=connection_sha256,
    )


def _tool_call(call_id: str = "call-weather") -> ToolCall:
    """Build one deterministic tool call."""
    return ToolCall(call_id=call_id, name="weather", arguments={"city": "Zürich"})


def _tool_schema() -> ToolSchema:
    """Build the matching tool schema."""
    return ToolSchema(
        name="weather",
        description="Read weather",
        input_schema={"type": "object"},
    )


def _action(*, route_sha256: str, content: str, call_id: str) -> AssistantAction:
    """Build one assistant tool action with opaque Fireworks history."""
    return AssistantAction(
        tool_calls=(_tool_call(call_id),),
        provider_reasoning=(
            OpaqueReasoningContentBlock(
                route_sha256=route_sha256,
                content=content,
            ),
        ),
    )


def _tool_response(*, reasoning_content: str, call_id: str = "call-weather") -> JsonHttpResponse:
    """Build one completed Fireworks tool-call response."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "reasoning_content": reasoning_content,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Zürich"}',
                                },
                            }
                        ],
                    },
                }
            ]
        },
    )


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_non_streaming_agent_history_round_trips_only_on_the_issuing_route(
    model_id: str,
) -> None:
    """DeepSeek, GLM, and Kimi preserve opaque history through a tool continuation."""
    raw_reasoning = f"private reasoning for {model_id}\nwith: punctuation"
    transport = ScriptedJsonTransport(
        [
            _tool_response(reasoning_content=raw_reasoning),
            JsonHttpResponse(
                status_code=200,
                body={"choices": [{"message": {"content": "sunny"}}]},
            ),
        ]
    )
    snapshot = _snapshot(model_id)
    client = OpenAICompatibleClient(
        model=snapshot,
        api_key="fake-key",
        base_url=_FIREWORKS_BASE_URL,
        transport=transport,
        supports_reasoning=True,
    )

    first = client.complete(
        ModelRequest(
            messages=(ModelMessage(role="user", content="Weather?"),),
            tools=(_tool_schema(),),
        )
    )
    route_sha256 = reasoning_content_route_sha256(snapshot)
    assert first.output.provider_reasoning == (
        OpaqueReasoningContentBlock(
            route_sha256=route_sha256,
            content=raw_reasoning,
        ),
    )

    second = client.complete(
        ModelRequest(
            messages=(
                ModelMessage(role="user", content="Weather?"),
                ModelMessage(role="assistant", assistant_action=first.output),
                ModelMessage(role="tool", content="sunny", tool_call_id="call-weather"),
            ),
            tools=(_tool_schema(),),
        )
    )

    assert second.output.content == "sunny"
    first_payload = transport.requests[0].payload
    assert "reasoning_history" not in first_payload
    second_payload = transport.requests[1].payload
    assert second_payload["reasoning_history"] == "interleaved"
    messages = cast("list[JsonObject]", second_payload["messages"])
    assert messages[1]["reasoning_content"] == raw_reasoning


def test_active_history_rejects_other_fireworks_routes_and_other_providers() -> None:
    """An active carrier cannot cross a model revision, connection, or provider route."""
    action = _action(
        route_sha256=_OTHER_ROUTE_SHA256,
        content="provider-private",
        call_id="call-one",
    )
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="Use a tool"),
            ModelMessage(role="assistant", assistant_action=action),
            ModelMessage(role="tool", content="done", tool_call_id="call-one"),
        )
    )
    fireworks = OpenAICompatibleClient(
        model=_snapshot(_MODEL_IDS[0]),
        api_key="fake-key",
        base_url=_FIREWORKS_BASE_URL,
        transport=ScriptedJsonTransport(),
    )
    generic = OpenAICompatibleClient(
        model=_snapshot(_MODEL_IDS[0]),
        api_key="fake-key",
        base_url="https://compatible.example/v1",
        transport=ScriptedJsonTransport(),
    )

    for client in (fireworks, generic):
        with pytest.raises(ProviderParameterError) as raised:
            client.complete(request)
        assert raised.value.param == "messages.reasoning_content"


def test_old_reasoning_is_stripped_at_the_next_user_boundary() -> None:
    """Interleaved history never replays a carrier from an earlier user turn."""
    messages = (
        ModelMessage(role="user", content="First turn"),
        ModelMessage(
            role="assistant",
            assistant_action=_action(
                route_sha256=_ROUTE_SHA256,
                content="old private reasoning",
                call_id="call-old",
            ),
        ),
        ModelMessage(role="tool", content="old result", tool_call_id="call-old"),
        ModelMessage(role="user", content="Second turn"),
    )

    prepared, active = prepare_model_reasoning_history(
        messages,
        route_sha256=_OTHER_ROUTE_SHA256,
    )

    assert not active
    assert prepared[1].assistant_action is not None
    assert prepared[1].assistant_action.provider_reasoning == ()


def test_active_multi_step_history_requires_every_tool_result() -> None:
    """Two interleaved tool rounds preserve both carriers and reject an incomplete round."""
    messages = (
        ModelMessage(role="user", content="First turn"),
        ModelMessage(
            role="assistant",
            assistant_action=_action(
                route_sha256=_ROUTE_SHA256,
                content="first reasoning",
                call_id="call-one",
            ),
        ),
        ModelMessage(role="tool", content="one", tool_call_id="call-one"),
        ModelMessage(
            role="assistant",
            assistant_action=_action(
                route_sha256=_ROUTE_SHA256,
                content="second reasoning",
                call_id="call-two",
            ),
        ),
        ModelMessage(role="tool", content="two", tool_call_id="call-two"),
    )

    prepared, active = prepare_model_reasoning_history(
        messages,
        route_sha256=_ROUTE_SHA256,
    )
    assert active
    assert prepared == messages

    with pytest.raises(ProviderParameterError, match="completed tool continuation"):
        prepare_model_reasoning_history(
            messages[:-1],
            route_sha256=_ROUTE_SHA256,
        )


def test_reasoning_carrier_round_trip_is_byte_exact_and_closed() -> None:
    """The carrier preserves arbitrary provider text and rejects non-gateway strings."""
    block = OpaqueReasoningContentBlock(
        route_sha256=_ROUTE_SHA256,
        content="line one:\nZürich \N{SNOWMAN}",
    )

    assert decode_reasoning_content(encode_reasoning_content(block)) == block
    for invalid in ("raw provider text", "x-experiential-fireworks-reasoning-v1:"):
        with pytest.raises(ValueError):
            decode_reasoning_content(invalid)


def test_fireworks_endpoint_detection_rejects_spoofed_or_legacy_roots() -> None:
    """Only the exact HTTPS Fireworks inference root enables provider history."""
    assert is_fireworks_base_url(_FIREWORKS_BASE_URL)
    assert is_fireworks_base_url(f"{_FIREWORKS_BASE_URL}/")
    assert not is_fireworks_base_url("http://api.fireworks.ai/inference/v1")
    assert not is_fireworks_base_url("https://api.fireworks.ai.evil.test/inference/v1")
    assert not is_fireworks_base_url("https://api.fireworks.ai/models")
    assert not is_fireworks_base_url("https://api.fireworks.ai/inference/v1?route=other")


def test_route_identity_changes_with_model_revision_and_connection() -> None:
    """Every identity component that can change provider semantics changes the carrier route."""
    original = _snapshot(_MODEL_IDS[0])
    revision = original.model_copy(update={"revision": "other-revision"})
    connection = original.model_copy(update={"connection_sha256": "d" * 64})
    model = original.model_copy(update={"model_id": _MODEL_IDS[1]})

    hashes = {
        reasoning_content_route_sha256(snapshot)
        for snapshot in (original, revision, connection, model)
    }
    assert len(hashes) == 4


@pytest.mark.parametrize(
    ("model_id", "expected"),
    (
        (_MODEL_IDS[0], ("none", "high", "max")),
        (_MODEL_IDS[1], ("none", "high", "max")),
        (_MODEL_IDS[2], ("none", "low", "medium", "high", "max")),
    ),
)
def test_advertised_efforts_include_only_semantically_distinct_values(
    model_id: str,
    expected: tuple[str, ...],
) -> None:
    """Accepted aliases that Fireworks silently collapses are not advertised."""
    assert fireworks_reasoning_efforts(model_id) == expected
    assert (
        fireworks_reasoning_efforts(
            model_id,
            explicit_efforts=("none", "low", "medium", "high", "xhigh", "max"),
        )
        == expected
    )
    for effort in expected:
        assert fireworks_reasoning_effort(model_id, effort) == effort


@pytest.mark.parametrize(
    ("model_id", "effort"),
    (
        (_MODEL_IDS[0], "low"),
        (_MODEL_IDS[0], "medium"),
        (_MODEL_IDS[0], "xhigh"),
        (_MODEL_IDS[1], "low"),
        (_MODEL_IDS[1], "medium"),
        (_MODEL_IDS[1], "xhigh"),
        (_MODEL_IDS[2], "xhigh"),
    ),
)
def test_collapsed_or_undocumented_caller_efforts_are_rejected(
    model_id: str,
    effort: str,
) -> None:
    """A caller never receives a different semantic tier from the one requested."""
    with pytest.raises(UnsupportedReasoningEffortError):
        fireworks_reasoning_effort(model_id, effort)


@pytest.mark.parametrize(
    ("model_id", "configured", "normalized"),
    (
        (_MODEL_IDS[0], "low", "high"),
        (_MODEL_IDS[0], "medium", "high"),
        (_MODEL_IDS[0], "xhigh", "max"),
        (_MODEL_IDS[1], "low", "high"),
        (_MODEL_IDS[1], "medium", "high"),
        (_MODEL_IDS[1], "xhigh", "max"),
    ),
)
def test_legacy_operator_defaults_normalize_to_truthful_representatives(
    model_id: str,
    configured: str,
    normalized: str,
) -> None:
    """Existing catalog defaults resolve to the effort Fireworks actually executes."""
    assert normalized_fireworks_default(model_id, configured) == normalized
    client = OpenAICompatibleClient(
        model=_snapshot(model_id),
        api_key="fake-key",
        base_url=_FIREWORKS_BASE_URL,
        supports_reasoning=True,
        reasoning_effort=configured,
    )
    assert client.gateway_wire_profile().reasoning_effort == normalized


def test_undocumented_kimi_xhigh_default_is_not_invented_or_normalized() -> None:
    """Kimi xhigh remains invalid because Fireworks documents no equivalent tier."""
    model_id = _MODEL_IDS[2]
    assert normalized_fireworks_default(model_id, "xhigh") == "xhigh"
    client = OpenAICompatibleClient(
        model=_snapshot(model_id),
        api_key="fake-key",
        base_url=_FIREWORKS_BASE_URL,
        transport=ScriptedJsonTransport(),
        supports_reasoning=True,
        reasoning_effort="xhigh",
    )

    with pytest.raises(UnsupportedReasoningEffortError):
        client.complete(ModelRequest(messages=(ModelMessage(role="user", content="hello"),)))
