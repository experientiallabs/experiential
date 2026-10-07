"""Complete classifier subjects differ even when public request serialization does not."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.subjects import observation_subject_bytes
from exp.runtime.gateway.reasoning_blocks import (
    OpaqueReasoningContentBlock,
    SealedReasoningContentBlock,
)
from exp.runtime.gateway.tool_contracts import GatewayToolDefinition


def _changed_subject(field: str) -> tuple[GatewayRequest, GatewayRequest]:
    """Return two valid classifier subjects differing only in a privately carried field."""
    message = GatewayMessage(role="assistant", content="visible")
    request = GatewayRequest(surface=GatewayApiSurface.MESSAGES, messages=(message,))
    if field == "reasoning":
        before = SealedReasoningContentBlock(carrier="sealed", deployment_hint="fixture")
        after = OpaqueReasoningContentBlock(route_sha256="a" * 64, content="recovered")
        return (
            request.model_copy(
                update={"messages": (message.model_copy(update={"provider_reasoning": (before,)}),)}
            ),
            request.model_copy(
                update={"messages": (message.model_copy(update={"provider_reasoning": (after,)}),)}
            ),
        )
    if field == "input_examples":
        tool = GatewayToolDefinition(name="fixture", parameters={"type": "object"})
        return (
            request.model_copy(update={"tools": (tool,)}),
            request.model_copy(
                update={
                    "tools": (
                        tool.model_copy(update={"input_examples": ({"private": "example"},)}),
                    )
                }
            ),
        )
    if field == "provider_anthropic_blocks":
        return request, request.model_copy(
            update={
                "messages": (
                    message.model_copy(
                        update={field: ({"type": "thinking", "thinking": "private"},)}
                    ),
                )
            }
        )
    if field == "provider_text_blocks":
        return request, request.model_copy(
            update={
                "messages": (
                    message.model_copy(
                        update={
                            field: (
                                {
                                    "type": "text",
                                    "text": "visible",
                                    "cache_control": {"type": "ephemeral"},
                                },
                            )
                        }
                    ),
                )
            }
        )
    return request, request.model_copy(update={field: {"private": "value"}})


@pytest.mark.parametrize(
    "field",
    [
        "context_management",
        "provider_output_config",
        "reasoning",
        "input_examples",
        "provider_anthropic_blocks",
        "provider_text_blocks",
    ],
)
def test_complete_subject_includes_fields_excluded_from_wire_serialization(field: str) -> None:
    """Private provider history, options and tool metadata remain distinct to observers."""
    before, after = _changed_subject(field)
    assert canonical_json_bytes(before) == canonical_json_bytes(after)
    assert observation_subject_bytes(before) != observation_subject_bytes(after)
    assert observation_subject_bytes(after) == observation_subject_bytes(
        after.model_copy(deep=True)
    )


def test_complete_subject_budget_includes_private_payload_bytes() -> None:
    """An opaque private field cannot evade the retained observation byte budget."""
    before, _ = _changed_subject("context_management")
    after = before.model_copy(update={"context_management": {"private": "x" * 10_000}})
    assert len(observation_subject_bytes(after)) > len(observation_subject_bytes(before)) + 10_000
