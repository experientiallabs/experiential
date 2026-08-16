"""Bedrock Converse request and response translation tests."""

from __future__ import annotations

import pytest

from wmo.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    ToolCall,
    ToolChoice,
)
from wmo.common.tasks import ToolSchema
from wmo.runtime.models.providers.bedrock_converse import converse_request, converse_response
from wmo.runtime.models.providers.errors import ProviderResponseError


def _snapshot() -> ModelSnapshot:
    """Build an immutable Bedrock identity fixture."""
    return ModelSnapshot(
        provider="bedrock",
        model_id="us.anthropic.claude-sonnet-4-5",
        capabilities_sha256="a" * 64,
        connection_sha256="a" * 64,
    )


def _request() -> ModelRequest:
    """Build a visible transcript containing an earlier tool call and result."""
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content="You are precise."),
            ModelMessage(role="user", content="Create a ticket."),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(
                        ToolCall(
                            call_id="call-old",
                            name="create_ticket",
                            arguments={"priority": "normal"},
                        ),
                    )
                ),
            ),
            ModelMessage(role="tool", content="created", tool_call_id="call-old"),
        ),
        tools=(
            ToolSchema(
                name="create_ticket",
                description="Create one support ticket.",
                input_schema={"type": "object"},
            ),
        ),
        tool_choice=ToolChoice(name="create_ticket"),
        temperature=0.1,
        maximum_output_tokens=256,
    )


def test_converse_request_preserves_tool_ids_and_named_choice() -> None:
    """Converse keeps exact tool-use IDs and forwards named tool choice."""
    payload = converse_request("us.anthropic.claude-sonnet-4-5", _request())

    assert payload["modelId"] == "us.anthropic.claude-sonnet-4-5"
    assert payload["system"] == [{"text": "You are precise."}]
    assert payload["inferenceConfig"] == {"maxTokens": 256, "temperature": 0.1}
    tool_config = payload["toolConfig"]
    assert isinstance(tool_config, dict)
    assert tool_config["toolChoice"] == {"tool": {"name": "create_ticket"}}
    messages = payload["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]
    tool_result = messages[2]
    assert isinstance(assistant, dict)
    assert isinstance(tool_result, dict)
    assistant_content = assistant["content"]
    result_content = tool_result["content"]
    assert isinstance(assistant_content, list)
    assert isinstance(result_content, list)
    tool_use = assistant_content[0]
    result_block = result_content[0]
    assert isinstance(tool_use, dict)
    assert isinstance(result_block, dict)
    tool_use_block = tool_use["toolUse"]
    result_payload = result_block["toolResult"]
    assert isinstance(tool_use_block, dict)
    assert isinstance(result_payload, dict)
    assert tool_use_block["toolUseId"] == "call-old"
    assert tool_result["role"] == "user"
    assert result_payload["toolUseId"] == "call-old"


def test_converse_response_normalizes_cache_legs_without_double_counting() -> None:
    """Converse inputTokens exclude cache legs, so read and write are added once."""
    response = converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "done"},
                        {
                            "toolUse": {
                                "toolUseId": "call-new",
                                "name": "create_ticket",
                                "input": {"priority": "urgent"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "cacheReadInputTokens": 6,
                "cacheWriteInputTokens": 2,
            },
        },
        configured_model=_snapshot(),
        latency_seconds=0.5,
    )

    assert response.finish_reason == ModelFinishReason.COMPLETED
    assert response.output.content == "done"
    assert response.output.tool_calls[0].call_id == "call-new"
    assert response.economics.usage is not None
    assert response.economics.usage.input_tokens == 18
    assert response.economics.usage.cached_input_tokens == 6
    assert response.economics.usage.cache_write_input_tokens == 2


def test_converse_response_maps_length_and_rejects_unsupported_blocks() -> None:
    """max_tokens becomes length, and unknown content blocks fail closed."""
    length = converse_response(
        {
            "output": {"message": {"content": [{"text": "partial"}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 3, "outputTokens": 2},
        },
        configured_model=_snapshot(),
        latency_seconds=0.1,
    )
    assert length.finish_reason == ModelFinishReason.LENGTH
    with pytest.raises(ProviderResponseError, match="unsupported block"):
        converse_response(
            {
                "output": {"message": {"content": [{"image": {"format": "png"}}]}},
                "stopReason": "end_turn",
            },
            configured_model=_snapshot(),
            latency_seconds=0.1,
        )
    with pytest.raises(ProviderResponseError, match="not supported"):
        converse_response(
            {
                "output": {"message": {"content": [{"text": "blocked"}]}},
                "stopReason": "content_filtered",
            },
            configured_model=_snapshot(),
            latency_seconds=0.1,
        )
