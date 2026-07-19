"""Tests for provider-neutral structured Responses API translation."""

import json
from typing import Literal

import pytest
from llm_waterfall import ChatRequest

from wmh.providers._responses_common import (
    responses_request,
    responses_response,
)


def test_responses_request_translates_exact_pi_first_turn() -> None:
    """The pi bridge request becomes a native, non-streaming Responses call."""
    request = ChatRequest.model_validate(
        {
            "model": "worker",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": [{"type": "text", "text": "go"}]},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "store": False,
            "max_completion_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "parallel_tool_calls": True,
            "tool_choice": "required",
            "reasoning": {"effort": "minimal"},
            "service_tier": "flex",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "read",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                        "strict": False,
                    },
                }
            ],
        }
    )

    payload = responses_request(
        request,
        "gpt-5.5-deployment",
        reasoning_effort="high",
        service_tier="priority",
    )

    assert payload == {
        "model": "gpt-5.5-deployment",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "go"},
        ],
        "stream": False,
        "store": False,
        "max_output_tokens": 4096,
        "parallel_tool_calls": True,
        "tool_choice": "required",
        "reasoning": {"effort": "high"},
        "include": ["reasoning.encrypted_content"],
        "service_tier": "priority",
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                "strict": False,
            }
        ],
    }
    assert "messages" not in payload
    assert "max_completion_tokens" not in payload
    assert "stream_options" not in payload
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_responses_request_translates_tool_result_turn_with_exact_call_id() -> None:
    """A stateless replay preserves the assistant call and matching tool result."""
    request = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "developer", "content": "be concise"},
                {"role": "user", "content": [{"type": "text", "text": "go"}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-abc",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"x"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "contents", "tool_call_id": "call-abc"},
            ],
            "max_tokens": 512,
            "store": True,
            "temperature": 0.25,
            "top_p": 0.8,
            "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        }
    )

    payload = responses_request(request, "gpt-5.5")

    assert payload == {
        "model": "gpt-5.5",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "developer", "content": "be concise"},
            {"role": "user", "content": "go"},
            {
                "type": "function_call",
                "call_id": "call-abc",
                "name": "read_file",
                "arguments": '{"path":"x"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-abc",
                "output": "contents",
            },
        ],
        "stream": False,
        "store": True,
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": 512,
        "temperature": 0.25,
        "top_p": 0.8,
        "tool_choice": {"type": "function", "name": "read_file"},
    }


def test_responses_response_preserves_text_multiple_tools_usage_and_tier() -> None:
    """Native Responses output maps back to one OpenAI-compatible chat choice."""
    response = responses_response(
        {
            "id": "resp-1",
            "model": "gpt-5.5-2026-05-01",
            "status": "completed",
            "service_tier": "priority",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [],
                    "encrypted_content": "ciphertext-1",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Checking "},
                        {"type": "output_text", "text": "both."},
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call-a",
                    "name": "read_file",
                    "arguments": '{"path":"a"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "call_id": "call-b",
                    "name": "read_file",
                    "arguments": '{"path":"b"}',
                    "status": "completed",
                },
            ],
            "usage": {
                "input_tokens": 321,
                "output_tokens": 45,
                "total_tokens": 366,
                "input_tokens_details": {"cached_tokens": 12},
                "output_tokens_details": {"reasoning_tokens": 20},
            },
        }
    )

    assert response.model == "gpt-5.5-2026-05-01"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 321
    assert response.usage.completion_tokens == 45
    assert response.usage.model_extra == {
        "total_tokens": 366,
        "input_tokens_details": {"cached_tokens": 12},
        "output_tokens_details": {"reasoning_tokens": 20},
    }
    assert response.wire_payload()["service_tier"] == "priority"
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "Checking both."
    assert choice.message.tool_calls is not None
    assert [call.id for call in choice.message.tool_calls] == ["call-a", "call-b"]
    assert [call.function.name for call in choice.message.tool_calls] == [
        "read_file",
        "read_file",
    ]
    assert [call.function.arguments for call in choice.message.tool_calls] == [
        '{"path":"a"}',
        '{"path":"b"}',
    ]
    assert choice.message.reasoning_details is not None
    [detail] = choice.message.reasoning_details
    assert detail.id == "call-a"
    envelope = json.loads(detail.data)
    assert envelope["format"] == "openai.responses.output.v1"
    assert envelope["provider"] == "openai_responses"
    assert envelope["model"] == "gpt-5.5"
    assert envelope["output"] == [
        {
            "type": "reasoning",
            "id": "reasoning-1",
            "summary": [],
            "encrypted_content": "ciphertext-1",
        },
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Checking "},
                {"type": "output_text", "text": "both."},
            ],
        },
        {
            "type": "function_call",
            "call_id": "call-a",
            "name": "read_file",
            "arguments": '{"path":"a"}',
            "status": "completed",
        },
        {
            "type": "function_call",
            "call_id": "call-b",
            "name": "read_file",
            "arguments": '{"path":"b"}',
            "status": "completed",
        },
    ]


@pytest.mark.parametrize("reserved_name", ["prompt_tokens", "completion_tokens"])
def test_responses_response_rejects_reserved_usage_aliases(reserved_name: str) -> None:
    """Native usage aliases cannot overwrite translated canonical counters."""
    with pytest.raises(ValueError, match="reserved ChatUsage field"):
        responses_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    reserved_name: 0,
                },
            }
        )


def test_responses_response_rejects_partial_usage() -> None:
    with pytest.raises(ValueError, match="usage counters"):
        responses_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {"input_tokens": 1},
            }
        )


def test_ordered_encrypted_reasoning_round_trips_through_pi_chat_history() -> None:
    """A stateless second turn replays every ordered reasoning item through Pi once."""
    first = responses_response(
        {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [],
                    "encrypted_content": "ciphertext-1",
                },
                {
                    "type": "reasoning",
                    "id": "reasoning-2",
                    "summary": [{"type": "summary_text", "text": "second"}],
                    "encrypted_content": "ciphertext-2",
                },
                {
                    "type": "function_call",
                    "call_id": "call-a",
                    "name": "read_file",
                    "arguments": '{"path":"a"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "call_id": "call-b",
                    "name": "read_file",
                    "arguments": '{"path":"b"}',
                    "status": "completed",
                },
            ],
        },
        "gpt-5.5",
    )
    wire_choices = first.wire_payload()["choices"]
    assert isinstance(wire_choices, list)
    wire_choice = wire_choices[0]
    assert isinstance(wire_choice, dict)
    assistant = wire_choice["message"]
    assert isinstance(assistant, dict)
    reasoning_details = assistant["reasoning_details"]
    assert isinstance(reasoning_details, list)
    assert len(reasoning_details) == 1
    detail = reasoning_details[0]
    assert isinstance(detail, dict)
    data = detail["data"]
    assert isinstance(data, str)
    assert [item["id"] for item in json.loads(data)["output"] if item["type"] == "reasoning"] == [
        "reasoning-1",
        "reasoning-2",
    ]
    request = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "inspect"},
                assistant,
                {"role": "tool", "tool_call_id": "call-a", "content": "A"},
                {"role": "tool", "tool_call_id": "call-b", "content": "B"},
            ],
            "store": False,
        }
    )

    payload = responses_request(request, "gpt-5.5", reasoning_effort="high")

    assert payload["input"] == [
        {"role": "user", "content": "inspect"},
        {
            "type": "reasoning",
            "id": "reasoning-1",
            "summary": [],
            "encrypted_content": "ciphertext-1",
        },
        {
            "type": "reasoning",
            "id": "reasoning-2",
            "summary": [{"type": "summary_text", "text": "second"}],
            "encrypted_content": "ciphertext-2",
        },
        {
            "type": "function_call",
            "call_id": "call-a",
            "name": "read_file",
            "arguments": '{"path":"a"}',
            "status": "completed",
        },
        {
            "type": "function_call",
            "call_id": "call-b",
            "name": "read_file",
            "arguments": '{"path":"b"}',
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": "call-a", "output": "A"},
        {"type": "function_call_output", "call_id": "call-b", "output": "B"},
    ]
    assert payload["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize(
    ("origin_provider", "replay_provider"),
    [("azure", "openai_responses"), ("openai_responses", "azure")],
)
def test_encrypted_reasoning_cannot_replay_across_response_providers(
    origin_provider: Literal["azure", "openai_responses"],
    replay_provider: Literal["azure", "openai_responses"],
) -> None:
    first = responses_response(
        {
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [],
                    "encrypted_content": "provider-bound-ciphertext",
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                    "status": "completed",
                },
            ],
        },
        "gpt-5.5",
        snapshot_provider=origin_provider,
    )
    assistant = first.choices[0].message.model_dump(mode="json", exclude_none=True)
    request = ChatRequest.model_validate({"messages": [assistant]})

    with pytest.raises(ValueError, match="invalid or foreign Responses reasoning envelope"):
        responses_request(
            request,
            "gpt-5.5",
            reasoning_effort="high",
            snapshot_provider=replay_provider,
        )


def test_duplicate_parallel_reasoning_details_fail_closed() -> None:
    """A second opaque snapshot could reorder or replace provider-bound state."""
    reasoning = {
        "type": "reasoning",
        "id": "reasoning-1",
        "summary": [],
        "encrypted_content": "ciphertext-1",
    }
    data = json.dumps(reasoning, separators=(",", ":"), sort_keys=True)
    request = ChatRequest.model_validate(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": "call-b",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                    ],
                    "reasoning_details": [
                        {"type": "reasoning.encrypted", "id": "call-a", "data": data},
                        {"type": "reasoning.encrypted", "id": "call-b", "data": data},
                    ],
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="exactly one reasoning detail"):
        responses_request(request, "gpt-5.5", reasoning_effort="high")


def test_encrypted_snapshot_preserves_native_message_phase_and_item_order() -> None:
    native_output = [
        {
            "type": "reasoning",
            "id": "reasoning-1",
            "summary": [],
            "encrypted_content": "ciphertext-1",
            "status": "completed",
        },
        {
            "type": "message",
            "id": "msg-1",
            "role": "assistant",
            "phase": "commentary",
            "status": "completed",
            "content": [{"type": "output_text", "text": "I will inspect."}],
        },
        {
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
            "status": "completed",
        },
    ]
    first = responses_response(
        {
            "model": "gpt-5.5-2026-04-23",
            "status": "completed",
            "output": native_output,
        },
        "gpt-5.5",
    )
    assistant = first.choices[0].message.model_dump(mode="json", exclude_none=True)
    request = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "inspect"},
                assistant,
                {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
            ]
        }
    )

    payload = responses_request(request, "gpt-5.5", reasoning_effort="high")

    assert payload["input"] == [
        {"role": "user", "content": "inspect"},
        *native_output,
        {"type": "function_call_output", "call_id": "call-1", "output": "contents"},
    ]


def test_encrypted_snapshot_rejects_tampered_tool_call_and_foreign_model() -> None:
    first = responses_response(
        {
            "model": "gpt-5.5-2026-04-23",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [],
                    "encrypted_content": "ciphertext-1",
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                    "status": "completed",
                },
            ],
        },
        "gpt-5.5",
    )
    assistant = first.choices[0].message.model_dump(mode="json", exclude_none=True)
    tampered = json.loads(json.dumps(assistant))
    tampered["tool_calls"][0]["function"]["arguments"] = '{"path":"secrets"}'

    with pytest.raises(ValueError, match="does not match its signed Responses snapshot"):
        responses_request(
            ChatRequest.model_validate({"messages": [tampered]}),
            "gpt-5.5",
            reasoning_effort="high",
        )
    with pytest.raises(ValueError, match="belongs to a different model"):
        responses_request(
            ChatRequest.model_validate({"messages": [assistant]}),
            "gpt-5.4",
            reasoning_effort="high",
        )


@pytest.mark.parametrize("status", [None, "queued", "in_progress", "cancelled", "unknown"])
def test_non_completed_responses_fail_closed(status: str | None) -> None:
    with pytest.raises(ValueError, match="non-completed status"):
        responses_response({"status": status, "output": []})


def test_failed_response_includes_provider_diagnostics() -> None:
    with pytest.raises(ValueError, match="code=bad_request, message=invalid tool"):
        responses_response(
            {
                "status": "failed",
                "error": {"code": "bad_request", "message": "invalid tool"},
                "output": [],
            }
        )


def test_incomplete_response_never_exposes_a_partial_tool_call() -> None:
    with pytest.raises(ValueError, match="incomplete response: max_output_tokens"):
        responses_response(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "write_file",
                        "arguments": "{",
                        "status": "incomplete",
                    }
                ],
            }
        )


def test_completed_response_rejects_an_incomplete_output_item() -> None:
    with pytest.raises(ValueError, match="output with status 'incomplete'"):
        responses_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "write_file",
                        "arguments": "{",
                        "status": "incomplete",
                    }
                ],
            }
        )


def test_completed_response_requires_recognized_output() -> None:
    with pytest.raises(ValueError, match="without a message or function call"):
        responses_response({"status": "completed", "output": []})
