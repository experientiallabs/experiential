"""Translate OpenAI-shaped decisions to Google's Decisions API body.

Google's Decisions API (``POST {GEMINI_BASE_URL}/decisions``, served only by
``gemini-decision-1``) answers typed ``noul``, ``choice`` and ``score``
questions over shared text, image, audio, video and PDF input, billing input
tokens only. The gateway keeps ONE customer-facing decisions request shape per
wire, so a Google decision lane is reached through the OpenAI Decisions API
body on ``POST /v1/decisions``: this module is the request half of that
translation (the answer half lives in the native data plane,
``route_decisions/google.rs``), and both halves agree on the keys below.

What maps:

- ``input`` text forwards verbatim; user messages become Google content
  blocks (``input_text`` to ``text``, an inline base64 ``input_image`` data
  URL to an ``image`` block with its MIME type; ``detail`` low/high become
  the ``resolution`` low/high, and ``original`` is served at ``high``, Google's
  largest level inside the surface's per-image token reservation; ``auto``
  leaves Google's default), with a newline text block between messages
  because Google concatenates adjacent text with no separator.
- ``predicate`` becomes ``noul``; ``choice`` becomes ``choice`` whose
  ``criteria`` map each requested value to its description (empty when none);
  ``score`` becomes ``score`` whose ordered ``criteria`` carry each level's
  label and description.
- Questions are keyed positionally (``q0``, ``q1``, ...) since Google never
  shows keys to the model and OpenAI names may exceed Google's key rules.

What cannot map is refused with a typed 400 before dispatch: a choice value
that is not a valid Google option key (1 through 128 characters, no leading
or trailing whitespace, no control characters, no non-ASCII spaces) or that
collides with another value's key (the boolean ``true`` beside the string
``"true"``).
``safety_identifier`` has no Google counterpart and is dropped with a
disclosure on the answer (``x-experiential-ignored-parameters``).
"""

from __future__ import annotations

import unicodedata
from typing import Literal, assert_never

from exp.common.core.artifacts import JsonValue
from exp.runtime.gateway.openai_decisions_contracts import (
    DecisionLevel,
    InputImagePart,
    OpenAIChoiceQuestion,
    OpenAIDecisionQuestion,
    OpenAIDecisionRequest,
    OpenAIScoreQuestion,
    PredicateQuestion,
)
from exp.runtime.models.providers.gemini_requests import gemini_model_path

MAX_GOOGLE_OPTION_KEY_CHARACTERS = 128
"""Google's documented option-key length (at most 512 UTF-8 bytes, which 128
characters never exceed)."""

SAFETY_IDENTIFIER_DROPPED = "safety_identifier->dropped(google_decisions)"
"""Disclosure recorded when the Google wire cannot carry ``safety_identifier``."""

# Google's ``ultra_high`` (2,240 tokens per image) exceeds the surface's
# per-image reservation (2,048), so ``original`` is served at ``high`` (1,120).
_RESOLUTIONS: dict[str, str] = {"low": "low", "high": "high", "original": "high"}
_MESSAGE_SEPARATOR: dict[str, JsonValue] = {"type": "text", "text": "\n"}


class GoogleDecisionTranslationError(ValueError):
    """A request the Google Decisions API cannot express; safe to show the caller.

    Messages name fields and rules only; they never quote input, instructions,
    question names, or choice values.
    """

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param
        self.message = message


def google_question_key(index: int) -> str:
    """The positional Google question key of the admitted question at ``index``."""
    return f"q{index}"


def google_option_key(value: str | bool) -> str:
    """The Google option key of one requested choice value.

    Args:
        value: The typed OpenAI choice value.

    Returns:
        The key, the boolean spelled ``true`` or ``false`` and a string itself.

    Raises:
        GoogleDecisionTranslationError: The string is not a valid Google option
            key.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if (
        len(value) > MAX_GOOGLE_OPTION_KEY_CHARACTERS
        or value != value.strip()
        or any(_disallowed_in_option_key(character) for character in value)
    ):
        raise GoogleDecisionTranslationError(
            "questions",
            "choice values on this model must be 1 through 128 characters with no leading "
            "or trailing spaces, no control characters and no non-ASCII spaces",
        )
    return value


def _disallowed_in_option_key(character: str) -> bool:
    """Google refuses control characters and every space other than ASCII space."""
    category = unicodedata.category(character)
    return category == "Cc" or (category == "Zs" and character != " ")


def google_level_criterion(level: DecisionLevel) -> str:
    """The criterion text Google shows for one ordered score level."""
    return f"{level.label}: {level.description}" if level.description else level.label


def _question(question: OpenAIDecisionQuestion) -> dict[str, JsonValue]:
    """Translate one typed question into Google's question object."""
    match question:
        case PredicateQuestion():
            return {"type": "noul", "instructions": question.instructions}
        case OpenAIChoiceQuestion():
            criteria: dict[str, JsonValue] = {}
            for choice in question.choices:
                key = google_option_key(choice.value)
                if key in criteria:
                    raise GoogleDecisionTranslationError(
                        "questions",
                        "choice values on this model must stay distinct when spelled as "
                        'text (a boolean true beside the string "true" cannot be told apart)',
                    )
                criteria[key] = choice.description or ""
            return {"type": "choice", "instructions": question.instructions, "criteria": criteria}
        case OpenAIScoreQuestion():
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": [google_level_criterion(level) for level in question.levels],
            }
        case _:
            assert_never(question)


def _image_block(part: InputImagePart) -> dict[str, JsonValue]:
    """Translate one inline base64 data URL into a Google image block."""
    header, _separator, data = part.image_url.partition(";base64,")
    block: dict[str, JsonValue] = {
        "type": "image",
        "data": data,
        "mime_type": header.removeprefix("data:"),
    }
    resolution = None if part.detail is None else _RESOLUTIONS.get(part.detail)
    if resolution is not None:
        block["resolution"] = resolution
    return block


def _input(request: OpenAIDecisionRequest) -> JsonValue:
    """Translate the shared input into Google's string or content-block list."""
    if isinstance(request.input, str):
        return request.input
    blocks: list[JsonValue] = []
    for index, message in enumerate(request.input):
        if index:
            blocks.append(dict(_MESSAGE_SEPARATOR))
        if isinstance(message.content, str):
            blocks.append({"type": "text", "text": message.content})
            continue
        for part in message.content:
            if isinstance(part, InputImagePart):
                blocks.append(_image_block(part))
            else:
                blocks.append({"type": "text", "text": part.text})
    return blocks


def google_decision_body(
    request: OpenAIDecisionRequest, model: str
) -> tuple[
    dict[str, JsonValue], tuple[Literal["safety_identifier->dropped(google_decisions)"], ...]
]:
    """Build the Google Decisions API body for an admitted OpenAI-shaped request.

    Args:
        request: The typed, already validated customer request.
        model: The deployment's Google wire id (``gemini-decision-1``, with or
            without the ``models/`` resource prefix; the body carries the bare
            id Google's Decisions API reference documents).

    Returns:
        The upstream body and the disclosures the translation dropped.

    Raises:
        GoogleDecisionTranslationError: A question cannot be expressed on this wire.
    """
    body: dict[str, JsonValue] = {
        "model": gemini_model_path(model),
        "input": _input(request),
        "questions": {
            google_question_key(index): _question(question)
            for index, question in enumerate(request.questions)
        },
    }
    dropped: tuple[Literal["safety_identifier->dropped(google_decisions)"], ...] = (
        (SAFETY_IDENTIFIER_DROPPED,) if request.safety_identifier is not None else ()
    )
    return body, dropped
