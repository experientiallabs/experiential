"""Tests for the Anthropic structured tool-calling translation."""

from typing import cast

from llm_waterfall import ChatRequest

from wmo.providers._anthropic_chat import messages_request, messages_response


def _request() -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "Be exact."},
                {"role": "user", "content": "Inspect the repo."},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_details": [
                        {
                            "type": "reasoning.encrypted",
                            "data": '[{"type":"thinking","thinking":"look","signature":"sig"}]',
                            "id": "call-1",
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "a.py"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "max_completion_tokens": 2048,
        }
    )


def test_messages_request_preserves_tools_results_and_signed_thinking() -> None:
    payload = messages_request(
        _request(),
        "claude-sonnet-5",
        reasoning_effort="high",
    )

    assert payload["model"] == "claude-sonnet-5"
    assert payload["system"] == "Be exact."
    assert payload["max_tokens"] == 2048
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high"}
    assert payload["tool_choice"] == {"type": "any"}
    messages = payload["messages"]
    assert isinstance(messages, list)
    second = cast("dict[str, object]", messages[1])
    content = second["content"]
    assert isinstance(content, list)
    assert content[0] == {
        "type": "thinking",
        "thinking": "look",
        "signature": "sig",
    }
    assert cast("dict[str, object]", content[1])["type"] == "tool_use"
    third = cast("dict[str, object]", messages[2])
    result_content = third["content"]
    assert isinstance(result_content, list)
    assert cast("dict[str, object]", result_content[0])["type"] == "tool_result"


def test_messages_response_preserves_tool_call_usage_and_thinking() -> None:
    response = messages_response(
        {
            "model": "claude-sonnet-5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "inspect", "signature": "sig-2"},
                {
                    "type": "tool_use",
                    "id": "call-2",
                    "name": "bash",
                    "input": {"command": "pwd"},
                },
            ],
            "usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 10,
                "output_tokens": 30,
            },
        }
    )

    assert response.model == "claude-sonnet-5"
    assert response.choices[0].finish_reason == "tool_calls"
    calls = response.choices[0].message.tool_calls
    assert calls is not None
    assert calls[0].id == "call-2"
    assert calls[0].function.name == "bash"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 150
    assert response.usage.completion_tokens == 30
    assert response.usage.model_extra == {
        "prompt_tokens_details": {
            "cached_tokens": 40,
            "cache_write_tokens": 10,
        }
    }
    extras = response.choices[0].message.model_extra
    assert extras is not None
    details = extras["reasoning_details"]
    assert isinstance(details, list)
    assert details[0]["id"] == "call-2"
