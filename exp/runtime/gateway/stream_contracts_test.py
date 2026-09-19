"""Tests for stream outcome contracts."""

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.stream_contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRefusalReason,
    GatewayUsage,
)


def test_gateway_failure_carries_an_optional_bounded_refusal_reason() -> None:
    """The refusal reason is an optional typed field that round-trips through
    the contract's JSON serialization and defaults to absent."""
    bare = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="provider stream failed",
    )
    assert bare.refusal_reason is None

    refused = GatewayFailure(
        failure_class=GatewayFailureClass.REFUSAL,
        safe_message="provider refused the request: cybersecurity policy",
        refusal_reason=GatewayRefusalReason.CYBER_POLICY,
    )
    assert refused.refusal_reason is GatewayRefusalReason.CYBER_POLICY
    restored = GatewayFailure.model_validate(refused.model_dump(mode="json"))
    assert restored.refusal_reason is GatewayRefusalReason.CYBER_POLICY
    # The enum members are exactly the closed vocabulary shared with the engine.
    assert {reason.value for reason in GatewayRefusalReason} == {
        "cyber_policy",
        "cbrn",
        "content_policy",
        "recitation",
        "data_inspection",
        "unspecified",
    }


def test_decision_rejection_evidence_is_strict_and_not_serialized() -> None:
    """Internal settlement evidence defaults safe and never changes public event identity."""
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED, safe_message="provider rejected the request"
    )
    event = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=failure)
    rejected = GatewayEvent(
        kind=GatewayEventKind.FAILED,
        sequence_number=0,
        failure=failure,
        decision_provider_rejected=True,
    )
    assert event.decision_provider_rejected is False
    assert rejected.decision_provider_rejected is True
    assert event.model_dump_json() == rejected.model_dump_json()
    with pytest.raises(ValidationError):
        GatewayEvent.model_validate({**event.model_dump(), "decision_provider_rejected": "true"})


@pytest.mark.parametrize("length", [257, 65_536])
def test_stream_started_event_preserves_long_tool_id(length: int) -> None:
    """Provider tool IDs fit the same bound on output as on replay."""
    event = GatewayEvent(
        kind=GatewayEventKind.TOOL_CALL_STARTED,
        sequence_number=0,
        tool_call_index=0,
        tool_call_id="x" * length,
        tool_name="terminal",
    )
    assert event.tool_call_id == "x" * length


def test_web_search_requests_ride_on_usage_but_never_make_usage_alone() -> None:
    """The gateway-executed search count defaults to zero, is never negative, and needs a carrier.

    The existing rule stands: usage is token totals or invoked tool names. The
    count is a per-attempt billing meter of the gateway's own work, not a
    provider meter and not a token subset, so it rides on either shape and is
    rejected on its own exactly as an empty usage was before the field existed.
    """
    assert GatewayUsage(input_tokens=1, output_tokens=1).web_search_requests == 0
    with_tokens = GatewayUsage(input_tokens=1, output_tokens=1, web_search_requests=3)
    assert with_tokens.web_search_requests == 3
    tools_only = GatewayUsage(tool_names=("web_search",), web_search_requests=2)
    assert tools_only.web_search_requests == 2 and not tools_only.has_token_counts
    with pytest.raises(ValidationError, match="token totals or invoked tool names"):
        GatewayUsage(web_search_requests=2)
    with pytest.raises(ValidationError):
        GatewayUsage(input_tokens=1, output_tokens=1, web_search_requests=-1)


def test_tool_search_requests_ride_on_usage_but_never_make_usage_alone() -> None:
    """The gateway-executed tool-search count mirrors ``web_search_requests`` exactly.

    It defaults to zero, is never negative, rides on token-bearing or tool-only
    usage, and is rejected on its own because a bare count is not usage.
    """
    assert GatewayUsage(input_tokens=1, output_tokens=1).tool_search_requests == 0
    with_tokens = GatewayUsage(input_tokens=1, output_tokens=1, tool_search_requests=3)
    assert with_tokens.tool_search_requests == 3
    tools_only = GatewayUsage(tool_names=("tool_search",), tool_search_requests=2)
    assert tools_only.tool_search_requests == 2 and not tools_only.has_token_counts
    both = GatewayUsage(
        input_tokens=1, output_tokens=1, web_search_requests=1, tool_search_requests=2
    )
    assert (both.web_search_requests, both.tool_search_requests) == (1, 2)
    with pytest.raises(ValidationError, match="token totals or invoked tool names"):
        GatewayUsage(tool_search_requests=2)
    with pytest.raises(ValidationError):
        GatewayUsage(input_tokens=1, output_tokens=1, tool_search_requests=-1)
