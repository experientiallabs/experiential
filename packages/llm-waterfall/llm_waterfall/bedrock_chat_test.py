"""Tests for lossless Bedrock Converse translation."""

from __future__ import annotations

import json
from typing import cast

import pytest

from llm_waterfall.bedrock_chat import (
    BedrockResponseTranslationError,
    bedrock_converse_client_tool_names,
    bedrock_converse_request,
    bedrock_converse_response,
)
from llm_waterfall.types import ChatRequest, ResponseTranslationFailure

_MODEL = "us.anthropic.claude-opus-4-6-v1"


def _tool_request() -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "inspect"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "read one file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "max_completion_tokens": 4096,
        }
    )


def test_advertised_tool_name_membership_does_not_reject_duplicate_entries() -> None:
    assert bedrock_converse_client_tool_names(
        {
            "toolConfig": {
                "tools": [
                    {"toolSpec": {"name": "read_file"}},
                    {"toolSpec": {"name": "read_file"}},
                ]
            }
        }
    ) == frozenset({"read_file"})


def test_adaptive_reasoning_uses_official_converse_request_fields() -> None:
    wire = bedrock_converse_request(_tool_request(), _MODEL, reasoning_effort="max")

    assert wire["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "max"},
    }
    inference = wire["inferenceConfig"]
    assert isinstance(inference, dict)
    assert "thinking" not in inference
    assert "effort" not in inference


@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "read_file"}}],
)
def test_adaptive_reasoning_rejects_forced_tool_choice(tool_choice: object) -> None:
    request_data = _tool_request().model_dump(mode="json")
    request_data["tool_choice"] = tool_choice
    request = ChatRequest.model_validate(request_data)

    with pytest.raises(ValueError, match="only auto or none"):
        bedrock_converse_request(request, _MODEL, reasoning_effort="max")


@pytest.mark.parametrize("tool_choice", [None, "auto", "none"])
def test_adaptive_reasoning_omits_sampling_and_allows_only_unforced_tool_modes(
    tool_choice: str | None,
) -> None:
    request_data = _tool_request().model_dump(mode="json")
    request_data.update(
        {
            "tool_choice": tool_choice,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 10,
        }
    )

    wire = bedrock_converse_request(
        ChatRequest.model_validate(request_data),
        _MODEL,
        reasoning_effort="max",
    )

    assert wire["inferenceConfig"] == {"maxTokens": 4096}
    assert "top_p" not in wire
    assert "top_k" not in wire
    if tool_choice == "none":
        assert "toolConfig" not in wire
    else:
        tool_config = wire["toolConfig"]
        assert isinstance(tool_config, dict)
        assert "toolChoice" not in tool_config


def test_signed_reasoning_and_redacted_content_replay_byte_for_byte() -> None:
    content = [
        {"reasoningContent": {"reasoningText": {"text": "", "signature": "signed-token"}}},
        {"reasoningContent": {"redactedContent": b"\x00\xffsecret"}},
        {"text": "I will inspect it."},
        {
            "toolUse": {
                "toolUseId": "call-1",
                "name": "read_file",
                "input": {"path": "README.md"},
                "type": "tool_use",
            }
        },
    ]
    response = bedrock_converse_response(
        {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 5, "outputTokens": 7},
        },
        _MODEL,
        advertised_client_tools=frozenset({"read_file"}),
    )
    assistant = response.choices[0].message.model_dump(mode="json", exclude_none=True)
    replay = ChatRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "inspect"},
                assistant,
                {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
            ],
            "tools": _tool_request().model_dump(mode="json")["tools"],
        }
    )

    wire = bedrock_converse_request(replay, _MODEL, reasoning_effort="max")

    messages = wire["messages"]
    assert isinstance(messages, list)
    replayed_assistant = cast("dict[str, object]", messages[1])
    assert replayed_assistant["role"] == "assistant"
    assert replayed_assistant["content"] == content
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "call-1",
                    "content": [{"text": "contents"}],
                }
            }
        ],
    }


def test_parallel_tool_calls_replay_one_ordered_signed_snapshot() -> None:
    content = [
        {"reasoningContent": {"reasoningText": {"text": "check both", "signature": "signature"}}},
        {"toolUse": {"toolUseId": "call-a", "name": "read_file", "input": {"path": "a"}}},
        {"toolUse": {"toolUseId": "call-b", "name": "read_file", "input": {"path": "b"}}},
    ]
    response = bedrock_converse_response(
        {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        },
        _MODEL,
    )
    message = response.choices[0].message

    assert message.reasoning_details is not None
    assert len(message.reasoning_details) == 1
    assert message.reasoning_details[0].id == "call-a"
    request = ChatRequest.model_validate(
        {
            "messages": [message.model_dump(mode="json", exclude_none=True)],
        }
    )
    wire = bedrock_converse_request(request, _MODEL, reasoning_effort="max")
    assert wire["messages"] == [{"role": "assistant", "content": content}]


def test_missing_usage_remains_distinguishable_from_reported_zero() -> None:
    response = bedrock_converse_response(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
            "stopReason": "end_turn",
        },
        _MODEL,
    )

    assert response.usage is None


@pytest.mark.parametrize(
    ("raw", "failure"),
    [
        (None, ResponseTranslationFailure.RESPONSE_OBJECT),
        ({}, ResponseTranslationFailure.OUTPUT_OBJECT),
        ({"output": {}}, ResponseTranslationFailure.MESSAGE_OBJECT),
        (
            {"output": {"message": {"role": "user", "content": []}}},
            ResponseTranslationFailure.MESSAGE_ROLE,
        ),
        (
            {"output": {"message": {"role": "assistant", "content": None}}},
            ResponseTranslationFailure.CONTENT_ARRAY,
        ),
        (
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "ok", "toolUse": {}}],
                    }
                }
            },
            ResponseTranslationFailure.CONTENT_BLOCK,
        ),
        (
            {"output": {"message": {"role": "assistant", "content": [{"text": 1}]}}},
            ResponseTranslationFailure.TEXT_BLOCK,
        ),
        (
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call-1",
                                    "name": "submit",
                                    "input": {},
                                    "unexpected": True,
                                }
                            }
                        ],
                    }
                }
            },
            ResponseTranslationFailure.TOOL_USE_SHAPE,
        ),
        (
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call-1",
                                    "name": "submit",
                                    "input": [],
                                }
                            }
                        ],
                    }
                }
            },
            ResponseTranslationFailure.TOOL_USE_INPUT,
        ),
        (
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"reasoningContent": {}}],
                    }
                }
            },
            ResponseTranslationFailure.REASONING_CONTENT,
        ),
        (
            {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": 1,
            },
            ResponseTranslationFailure.STOP_REASON,
        ),
        (
            {
                "output": {"message": {"role": "assistant", "content": []}},
                "usage": [],
            },
            ResponseTranslationFailure.USAGE_OBJECT,
        ),
        (
            {
                "output": {"message": {"role": "assistant", "content": []}},
                "usage": {"inputTokens": 1},
            },
            ResponseTranslationFailure.USAGE_FIELDS,
        ),
        (
            {
                "output": {"message": {"role": "assistant", "content": []}},
                "usage": {"inputTokens": None, "outputTokens": 1},
            },
            ResponseTranslationFailure.USAGE_COUNTER,
        ),
    ],
)
def test_response_translation_failure_is_one_fixed_content_free_branch(
    raw: object,
    failure: ResponseTranslationFailure,
) -> None:
    with pytest.raises(BedrockResponseTranslationError) as caught:
        bedrock_converse_response(raw, _MODEL)

    assert caught.value.failure is failure
    assert caught.value.failure.value in {item.value for item in ResponseTranslationFailure}


def test_client_tool_type_is_preserved_in_bedrock_envelope_but_not_pi_projection() -> None:
    response = bedrock_converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-1",
                                "name": "submit",
                                "input": {"answer": "OK"},
                                "type": "tool_use",
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        },
        _MODEL,
        advertised_client_tools=frozenset({"submit"}),
    )

    [tool_call] = response.choices[0].message.tool_calls or []
    assert tool_call.type == "function"
    assert tool_call.function.name == "submit"
    assert json.loads(tool_call.function.arguments) == {"answer": "OK"}
    assert '"type":"tool_use"' not in json.dumps(
        response.wire_payload(),
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    "tool_type",
    ["server_tool_use", "client_tool_use", "function", "unknown", 1, None],
)
def test_non_client_tool_type_fails_closed_with_fixed_discriminator(
    tool_type: object,
) -> None:
    with pytest.raises(BedrockResponseTranslationError) as caught:
        bedrock_converse_response(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call-1",
                                    "name": "submit",
                                    "input": {"answer": "OK"},
                                    "type": tool_type,
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 1, "outputTokens": 1},
            },
            _MODEL,
        )

    assert caught.value.failure is ResponseTranslationFailure.TOOL_USE_TYPE


def test_typed_client_tool_requires_exact_request_advertisement() -> None:
    with pytest.raises(BedrockResponseTranslationError) as caught:
        bedrock_converse_response(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call-1",
                                    "name": "submit",
                                    "input": {"answer": "OK"},
                                    "type": "tool_use",
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 1, "outputTokens": 1},
            },
            _MODEL,
            advertised_client_tools=frozenset({"read_file"}),
        )

    assert caught.value.failure is ResponseTranslationFailure.TOOL_USE_ADVERTISEMENT


def test_usage_preserves_provider_dimensions_for_downstream_pricing() -> None:
    response = bedrock_converse_response(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 2,
                "totalTokens": 12,
                "cacheReadInputTokens": 4,
            },
        },
        _MODEL,
    )

    assert response.usage is not None
    assert response.usage.model_extra == {
        "total_tokens": 12,
        "cache_read_input_tokens": 4,
    }


def test_partial_usage_fails_closed() -> None:
    with pytest.raises(ValueError, match="must include inputTokens and outputTokens"):
        bedrock_converse_response(
            {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1},
            },
            _MODEL,
        )


def test_null_usage_counter_fails_closed() -> None:
    with pytest.raises(ValueError, match="usage counters"):
        bedrock_converse_response(
            {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": None, "outputTokens": 1},
            },
            _MODEL,
        )


@pytest.mark.parametrize(
    ("canonical_name", "provider_name"),
    [
        ("prompt_tokens", None),
        ("completion_tokens", None),
        ("total_tokens", "totalTokens"),
        ("cache_read_input_tokens", "cacheReadInputTokens"),
        ("cache_write_input_tokens", "cacheWriteInputTokens"),
    ],
)
def test_usage_alias_collision_fails_closed(
    canonical_name: str,
    provider_name: str | None,
) -> None:
    """Provider aliases cannot overwrite translated usage dimensions."""
    usage: dict[str, object] = {
        "inputTokens": 10,
        "outputTokens": 4,
        canonical_name: 0,
    }
    if provider_name is not None:
        usage[provider_name] = 1

    with pytest.raises(ValueError, match="duplicate ChatUsage field"):
        bedrock_converse_response(
            {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "end_turn",
                "usage": usage,
            },
            _MODEL,
        )


def test_signed_reasoning_replay_is_bound_to_originating_model() -> None:
    content = [
        {"reasoningContent": {"reasoningText": {"text": "inspect", "signature": "sig"}}},
        {"toolUse": {"toolUseId": "call-1", "name": "read_file", "input": {}}},
    ]
    response = bedrock_converse_response(
        {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 0, "outputTokens": 0},
        },
        _MODEL,
    )
    request = ChatRequest.model_validate(
        {"messages": [response.choices[0].message.model_dump(mode="json", exclude_none=True)]}
    )

    with pytest.raises(ValueError, match="belongs to a different model"):
        bedrock_converse_request(
            request,
            "us.anthropic.claude-sonnet-4-6",
            reasoning_effort="high",
        )


def test_signed_reasoning_model_binding_accepts_same_model_profile_prefix() -> None:
    content = [
        {"reasoningContent": {"reasoningText": {"text": "inspect", "signature": "sig"}}},
        {"toolUse": {"toolUseId": "call-1", "name": "read_file", "input": {}}},
    ]
    response = bedrock_converse_response(
        {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 0, "outputTokens": 0},
        },
        _MODEL,
    )
    request = ChatRequest.model_validate(
        {"messages": [response.choices[0].message.model_dump(mode="json", exclude_none=True)]}
    )

    wire = bedrock_converse_request(
        request,
        "global.anthropic.claude-opus-4-6-v1",
        reasoning_effort="max",
    )

    assert wire["messages"] == [{"role": "assistant", "content": content}]


def test_tampered_signed_turn_fails_closed_before_request() -> None:
    response = bedrock_converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "inspect", "signature": "sig"}
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "call-1",
                                "name": "read_file",
                                "input": {"path": "a"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 0, "outputTokens": 0},
        },
        _MODEL,
    )
    message = response.choices[0].message.model_dump(mode="json", exclude_none=True)
    message["tool_calls"][0]["function"]["arguments"] = json.dumps({"path": "changed"})
    request = ChatRequest.model_validate({"messages": [message]})

    with pytest.raises(ValueError, match="does not match its signed Bedrock snapshot"):
        bedrock_converse_request(request, _MODEL, reasoning_effort="max")


@pytest.mark.parametrize(
    "reasoning_content",
    [
        {},
        {"reasoningText": {"text": "x", "signature": ""}},
        {"reasoningText": {"text": "x", "signature": "s"}, "redactedContent": b"x"},
        {"redactedContent": "not-bytes"},
    ],
)
def test_malformed_reasoning_response_fails_closed(reasoning_content: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="reasoningContent"):
        bedrock_converse_response(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"reasoningContent": reasoning_content}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 0, "outputTokens": 0},
            },
            _MODEL,
        )


def test_unsigned_reasoning_text_remains_valid_and_replayable() -> None:
    content = [
        {"reasoningContent": {"reasoningText": {"text": "inspect"}}},
        {"toolUse": {"toolUseId": "call-1", "name": "read_file", "input": {}}},
    ]
    response = bedrock_converse_response(
        {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 0, "outputTokens": 0},
        },
        _MODEL,
    )
    request = ChatRequest.model_validate(
        {"messages": [response.choices[0].message.model_dump(mode="json", exclude_none=True)]}
    )

    wire = bedrock_converse_request(request, _MODEL, reasoning_effort="max")
    assert wire["messages"] == [{"role": "assistant", "content": content}]


def test_malformed_or_foreign_reasoning_envelope_fails_closed() -> None:
    request = ChatRequest.model_validate(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                    "reasoning_details": [
                        {
                            "type": "reasoning.encrypted",
                            "id": "call-1",
                            "data": '{"type":"reasoning","encrypted_content":"foreign"}',
                        }
                    ],
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="Bedrock reasoning envelope"):
        bedrock_converse_request(request, _MODEL, reasoning_effort="max")
