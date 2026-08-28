"""Tests for canonical serving and model-client adapters."""

from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelFinishReason,
    ModelResponse,
    ModelSnapshot,
    OpaqueReasoningContentBlock,
    OperationEconomics,
    ToolCall,
    ToolChoice,
    Usage,
)
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEventKind,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.openai_protocol.model_adapter import model_request, model_response_events


def test_model_request_preserves_messages_tools_and_limits() -> None:
    """The shared adapter preserves every model-client field it can represent."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="developer", content="rules"),
            GatewayMessage(
                role="assistant",
                tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={"q": 1}),),
            ),
            GatewayMessage(role="tool", tool_call_id="call-1", content="result"),
        ),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                description="Look up one value.",
                parameters={"type": "object"},
            ),
        ),
        tool_choice=GatewayNamedToolChoice(name="lookup"),
        temperature=0.25,
        top_p=1.0,
        reasoning_effort="high",
        maximum_output_tokens=128,
    )

    adapted = model_request(request)

    assert [message.role for message in adapted.messages] == ["system", "assistant", "tool"]
    assert adapted.messages[1].assistant_action is not None
    assert adapted.messages[1].assistant_action.tool_calls[0].arguments == {"q": 1}
    assert isinstance(adapted.tool_choice, ToolChoice)
    assert adapted.tool_choice.name == "lookup"
    assert adapted.temperature == 0.25
    assert adapted.top_p == 1.0
    assert adapted.reasoning_effort == "high"
    assert adapted.maximum_output_tokens == 128


def test_model_response_events_preserve_tool_bytes_usage_and_terminal() -> None:
    """A buffered response becomes one lossless ordered canonical event stream."""
    response = ModelResponse(
        output=AssistantAction(
            content="done",
            tool_calls=(
                ToolCall(
                    call_id="call-1",
                    name="lookup",
                    arguments={"q": 1},
                    raw_arguments='{ "q": 1 }',
                ),
            ),
        ),
        model=ModelSnapshot(
            provider="openai",
            model_id="gpt-test",
            capabilities_sha256="0" * 64,
            connection_sha256="1" * 64,
            billing_source=BillingSource.CUSTOMER_MANAGED,
        ),
        economics=OperationEconomics(
            usage=Usage(input_tokens=3, output_tokens=2, cached_input_tokens=1)
        ),
        finish_reason=ModelFinishReason.LENGTH,
    )

    events = model_response_events(response)

    assert [event.kind for event in events] == [
        GatewayEventKind.TEXT_DELTA,
        GatewayEventKind.TOOL_CALL_STARTED,
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        GatewayEventKind.TOOL_CALL_COMPLETED,
        GatewayEventKind.USAGE,
        GatewayEventKind.INCOMPLETE,
    ]
    assert events[2].raw_arguments_delta == '{ "q": 1 }'
    assert events[4].usage is not None
    assert events[4].usage.cached_input_tokens == 1


def test_model_response_events_preserve_route_bound_fireworks_reasoning() -> None:
    """Buffered provider reasoning enters the same canonical path as native streaming."""
    route_sha256 = "a" * 64
    response = ModelResponse(
        output=AssistantAction(
            tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),),
            provider_reasoning=(
                OpaqueReasoningContentBlock(
                    route_sha256=route_sha256,
                    content="provider private",
                ),
            ),
        ),
        model=ModelSnapshot(
            provider="fireworks",
            model_id="accounts/fireworks/models/deepseek-v4-flash-0731",
            capabilities_sha256="0" * 64,
            connection_sha256="1" * 64,
            billing_source=BillingSource.CUSTOMER_MANAGED,
        ),
        economics=OperationEconomics(),
    )

    events = model_response_events(response)

    assert events[0].kind == GatewayEventKind.REASONING_CONTENT_DELTA
    assert events[0].reasoning_content_route_sha256 == route_sha256
    assert events[0].text_delta == "provider private"
    assert events[-1].kind == GatewayEventKind.COMPLETED
