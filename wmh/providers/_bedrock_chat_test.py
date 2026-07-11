"""Tests for structured Bedrock Converse translation."""

from typing import cast

import pytest
from llm_waterfall import ChatRequest

from wmh.providers._bedrock_chat import converse_request, converse_response


def test_converse_round_trip_preserves_tools_results_and_usage() -> None:
    request = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "use tools"},
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "a.txt"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "required",
            "max_completion_tokens": 1024,
        }
    )

    wire = converse_request(request, "model-id")

    assert wire["modelId"] == "model-id"
    assert wire["inferenceConfig"] == {"maxTokens": 1024}
    assert wire["system"] == [{"text": "use tools"}]
    tool_config = wire["toolConfig"]
    assert isinstance(tool_config, dict)
    assert cast("dict[str, object]", tool_config)["toolChoice"] == {"any": {}}

    response = converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "checking"},
                        {
                            "toolUse": {
                                "toolUseId": "call-2",
                                "name": "bash",
                                "input": {"command": "pwd"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 42, "outputTokens": 7},
        },
        "model-id",
    )

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].function.name == "bash"
    assert response.token_usage().input_tokens == 42


@pytest.mark.parametrize("stop_reason", ["content_filtered", "guardrail_intervened"])
def test_converse_response_preserves_filtered_stops(stop_reason: str) -> None:
    """Blocked Bedrock turns cannot look like successful assistant stops."""
    response = converse_response(
        {
            "output": {"message": {"role": "assistant", "content": []}},
            "stopReason": stop_reason,
            "usage": {"inputTokens": 5, "outputTokens": 0},
        },
        "model-id",
    )

    assert response.choices[0].finish_reason == "content_filter"
