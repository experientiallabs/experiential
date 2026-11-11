"""Contract tests for the OpenAI-to-Google decisions request translation."""

from __future__ import annotations

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.google_decisions_contracts import (
    SAFETY_IDENTIFIER_DROPPED,
    GoogleDecisionTranslationError,
    google_decision_body,
    google_level_criterion,
    google_option_key,
    google_question_key,
)
from exp.runtime.gateway.openai_decisions_contracts import (
    DecisionLevel,
    decode_openai_decision_request,
)


def _translate(body: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Decode one OpenAI-shaped body and translate it for ``gemini-decision-1``."""
    decoded = decode_openai_decision_request(json.dumps(body))
    translated, dropped = google_decision_body(decoded.request, "gemini-decision-1")
    return translated, dropped


def test_every_question_type_maps_onto_googles_shape() -> None:
    """Predicates become noul, choices keep their values as option keys, scores keep order."""
    translated, dropped = _translate(
        {
            "model": "gemini-decision-1",
            "input": "The blue mug arrived cracked.",
            "questions": [
                {"type": "predicate", "name": "damaged", "instructions": "Arrived damaged?"},
                {
                    "type": "choice",
                    "name": "queue",
                    "instructions": "Which queue?",
                    "choices": [
                        {"value": "billing", "description": "Payments and refunds."},
                        {"value": "replacements"},
                        {"value": True, "description": "Escalate to a person."},
                    ],
                },
                {
                    "type": "score",
                    "instructions": "How severe?",
                    "levels": [
                        {"label": "cosmetic", "description": "appearance only"},
                        {"label": "blocked"},
                    ],
                },
            ],
        }
    )
    assert dropped == ()
    assert translated == {
        "model": "gemini-decision-1",
        "input": "The blue mug arrived cracked.",
        "questions": {
            "q0": {"type": "noul", "instructions": "Arrived damaged?"},
            "q1": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {
                    "billing": "Payments and refunds.",
                    "replacements": "",
                    "true": "Escalate to a person.",
                },
            },
            "q2": {
                "type": "score",
                "instructions": "How severe?",
                "criteria": ["cosmetic: appearance only", "blocked"],
            },
        },
    }
    assert google_question_key(7) == "q7"
    assert google_level_criterion(DecisionLevel(label="low", description="")) == "low"


def test_messages_become_content_blocks_with_images_and_separators() -> None:
    """Text parts and inline images cross as Google blocks; messages are newline-separated.

    ``original`` detail is served at Google's ``high`` (1,120 tokens), never
    ``ultra_high`` (2,240), which would exceed the surface's 2,048-token
    per-image reservation.
    """
    translated, _dropped = _translate(
        {
            "model": "gemini-decision-1",
            "input": [
                {"role": "user", "content": "Order 8821, Gold tier."},
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Customer photo:"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64,/9j/4AAQ",
                            "detail": "low",
                        },
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "original",
                        },
                        {"type": "input_image", "image_url": "data:image/webp;base64,UklGR"},
                    ],
                },
            ],
            "questions": [{"type": "predicate", "instructions": "Damaged?"}],
        }
    )
    assert translated["input"] == [
        {"type": "text", "text": "Order 8821, Gold tier."},
        {"type": "text", "text": "\n"},
        {"type": "text", "text": "Customer photo:"},
        {"type": "image", "data": "/9j/4AAQ", "mime_type": "image/jpeg", "resolution": "low"},
        {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png", "resolution": "high"},
        {"type": "image", "data": "UklGR", "mime_type": "image/webp"},
    ]


def test_a_models_prefixed_wire_id_is_sent_bare() -> None:
    """Google's Decisions API reference names the bare id; a resource prefix is stripped."""
    decoded = decode_openai_decision_request(
        json.dumps(
            {"model": "x", "input": "x", "questions": [{"type": "predicate", "instructions": "i"}]}
        )
    )
    body, _dropped = google_decision_body(decoded.request, "models/gemini-decision-1")
    assert body["model"] == "gemini-decision-1"


def test_safety_identifier_is_dropped_with_a_disclosure() -> None:
    """Google has no end-user identifier field, so the answer discloses the drop."""
    translated, dropped = _translate(
        {
            "model": "gemini-decision-1",
            "input": "x",
            "questions": [{"type": "predicate", "instructions": "i"}],
            "safety_identifier": "user-123",
        }
    )
    assert "safety_identifier" not in json.dumps(translated)
    assert dropped == (SAFETY_IDENTIFIER_DROPPED,)


@pytest.mark.parametrize(
    "value",
    [" padded", "padded ", "x" * 129, "tab\there", "line\nbreak", "no\u00a0break", "\u3000wide"],
)
def test_unexpressible_choice_values_are_refused_without_being_quoted(value: str) -> None:
    """A value outside Google's option-key rules is a typed refusal naming the rule."""
    with pytest.raises(GoogleDecisionTranslationError) as refused:
        google_option_key(value)
    assert refused.value.param == "questions"
    assert value.strip() not in refused.value.message or value.strip() in {"x" * 129}
    assert "128 characters" in refused.value.message


@pytest.mark.parametrize(
    "value", ["caf\u00e9", "\u65e5\u672c\u8a9e", "Needs review", "\U0001f600" * 128]
)
def test_unicode_choice_values_are_google_option_keys(value: str) -> None:
    """Google keys are Unicode text; only control characters and non-ASCII spaces are out."""
    assert google_option_key(value) == value


def test_a_boolean_beside_its_spelled_string_collides() -> None:
    """OpenAI keeps true and "true" distinct; Google's text keys cannot, so refuse."""
    body: JsonObject = {
        "model": "gemini-decision-1",
        "input": "x",
        "questions": [
            {
                "type": "choice",
                "instructions": "i",
                "choices": [{"value": True}, {"value": "true"}],
            }
        ],
    }
    with pytest.raises(GoogleDecisionTranslationError, match="distinct"):
        _translate(body)
    assert google_option_key(False) == "false"
    assert google_option_key("Needs review") == "Needs review"
