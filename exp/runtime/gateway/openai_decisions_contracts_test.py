"""Contract tests for OpenAI Decisions API request decoding and reservation."""

from __future__ import annotations

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.openai_decisions_contracts import decode_openai_decision_request


def _decode(body: JsonObject) -> JsonObject:
    """Decode one body and return the upstream body for wire id ``gpt-6-luna``."""
    return decode_openai_decision_request(json.dumps(body)).request.provider_body("gpt-6-luna")


def test_decoder_round_trips_string_input_and_provider_body() -> None:
    """A plain-text input forwards verbatim with the deployment's wire id."""
    decoded = decode_openai_decision_request(
        json.dumps(
            {
                "model": "luna-decisions",
                "input": "I want a refund.",
                "questions": [{"type": "predicate", "name": "refund", "instructions": "Refund?"}],
            }
        )
    )
    assert decoded.alias == "luna-decisions"
    assert decoded.request.provider_body("gpt-6-luna") == {
        "model": "gpt-6-luna",
        "input": "I want a refund.",
        "questions": [{"type": "predicate", "name": "refund", "instructions": "Refund?"}],
    }


def test_every_optional_schema_field_is_accepted_and_forwarded() -> None:
    """Unnamed questions, typed boolean values, absent descriptions, the message
    discriminator, original image detail, and safety_identifier all cross unchanged."""
    body: JsonObject = {
        "model": "luna-decisions",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Refund please."},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,iVBORw0KGgo=",
                        "detail": "original",
                    },
                ],
            }
        ],
        "questions": [
            {"type": "predicate", "instructions": "Angry?"},
            {
                "type": "choice",
                "instructions": "Refund?",
                "choices": [{"value": True}, {"value": "true", "description": "the text"}],
            },
            {
                "type": "score",
                "name": "urgency",
                "instructions": "How urgent?",
                "levels": [{"label": "low"}, {"label": "high", "description": "now"}],
            },
        ],
        "safety_identifier": "user-123",
    }
    assert _decode(body) == {**body, "model": "gpt-6-luna"}
    assert _decode({**body, "safety_identifier": None}) == {
        key: value for key, value in body.items() if key != "safety_identifier"
    } | {"model": "gpt-6-luna"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"questions": [{"type": "predicate", "name": "a", "instructions": "i"}] * 2}, "unique"),
        (
            {
                "questions": [
                    {
                        "type": "choice",
                        "instructions": "i",
                        "choices": [{"value": "a"}, {"value": "a"}],
                    }
                ]
            },
            "unique",
        ),
        ({"input": [{"role": "system", "content": "x"}]}, "role"),
        ({"input": [{"role": "user", "content": "x", "name": "n"}]}, "Extra"),
        ({"safety_identifier": "x" * 129}, "128"),
        ({"safety_identifier": 7}, "safety_identifier"),
        ({"stream": True}, "accepts only"),
    ],
)
def test_invalid_bodies_are_refused(change: JsonObject, message: str) -> None:
    """Duplicate names or values, non-user roles, unknown fields and bad bounds fail."""
    body: JsonObject = {
        "model": "m",
        "input": "x",
        "questions": [{"type": "predicate", "instructions": "i"}],
        **change,
    }
    with pytest.raises(ValueError, match=message):
        decode_openai_decision_request(json.dumps(body))


def test_unnamed_questions_never_collide() -> None:
    """Several unnamed questions are valid; only named ones must be unique."""
    body: JsonObject = {
        "model": "m",
        "input": "x",
        "questions": [{"type": "predicate", "instructions": "i"}] * 3,
    }
    assert len(decode_openai_decision_request(json.dumps(body)).request.questions) == 3


def test_image_parts_are_capped_across_messages() -> None:
    """At most 128 inline images cross one request, matching OpenAI's limit."""
    image = {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}

    def body(images: int) -> str:
        return json.dumps(
            {
                "model": "m",
                "input": [{"role": "user", "content": [image] * images}],
                "questions": [{"type": "predicate", "instructions": "i"}],
            }
        )

    assert decode_openai_decision_request(body(128)).request.input_token_reservation > 128 * 2048
    with pytest.raises(ValueError, match="128 image parts"):
        decode_openai_decision_request(body(129))


def test_message_arrays_are_capped_and_reserve_per_message_framing() -> None:
    """Many tiny messages cannot slip under the budget hold or past the cap."""
    question = [{"type": "predicate", "name": "a", "instructions": "i"}]

    def reservation(messages: int) -> int:
        decoded = decode_openai_decision_request(
            json.dumps(
                {
                    "model": "m",
                    "input": [{"role": "user", "content": "x"}] * messages,
                    "questions": question,
                }
            )
        )
        return decoded.request.input_token_reservation

    assert reservation(200) - reservation(100) == 100 * (1 + 16)
    with pytest.raises(ValueError, match="at most 256 messages"):
        reservation(257)
