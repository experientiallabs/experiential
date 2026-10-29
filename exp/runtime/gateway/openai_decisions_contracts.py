"""Typed OpenAI Decisions API requests, a second wire of the decisions surface.

OpenAI's ``POST /v1/decisions`` (public beta) asks one model to answer a fixed
list of typed questions (``predicate``, ``choice``, ``score``) about text or
inline images, and bills input tokens only. It shares the decisions surface,
ledger value and pricing rule with the TypeSafe SystemOne wire, but its body is
a different shape, so it is its own request type. It subclasses
:class:`DecisionRequest` so every surface-wide consumer (reservations, cost
ceilings, replay digests, the ledger) treats it as a decision without a new
branch.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictStr, field_validator, model_validator

from exp.common.core.artifacts import ContractModel, JsonValue
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_JSON_DEPTH,
    DecisionRequest,
    _native_json,
)

MAX_OPENAI_DECISION_QUESTIONS = 32
MAX_OPENAI_DECISION_CHOICES = 64
MAX_OPENAI_DECISION_LEVELS = 10
MAX_OPENAI_DECISION_INPUT_BYTES = 32 * 1024 * 1024
OPENAI_DECISION_IMAGE_TOKEN_RESERVATION = 2048
"""Planning tokens per inline image; settlement bills the provider's own count."""
OPENAI_DECISION_QUESTION_OVERHEAD_TOKENS = 256
MAX_OPENAI_DECISION_MESSAGES = 256
MAX_OPENAI_DECISION_IMAGES = 128
"""OpenAI's cap on image parts across all messages of one request."""
MAX_OPENAI_SAFETY_IDENTIFIER_CHARACTERS = 128
OPENAI_DECISION_MESSAGE_OVERHEAD_TOKENS = 16
"""Planning tokens per message for role and framing, so many tiny messages cannot
exceed the hold."""

_NAME_BYTES = 256


def _bounded_name(value: str, label: str) -> str:
    """Require 1 through 256 UTF-8 bytes for an identifier or answer value."""
    if not value or len(value.encode("utf-8")) > _NAME_BYTES:
        raise ValueError(f"{label} must contain 1 through 256 bytes")
    return value


class InputTextPart(ContractModel):
    """One text part of a decision input message.

    Attributes:
        type: Always ``input_text``.
        text: The text evidence.
    """

    type: Literal["input_text"] = "input_text"
    text: str


class InputImagePart(ContractModel):
    """One inline image part; OpenAI accepts only base64 data URLs here.

    Attributes:
        type: Always ``input_image``.
        image_url: A ``data:image/...;base64,`` URL; remote URLs and file ids
            are refused.
        detail: The image detail level; the provider defaults to ``auto``.
    """

    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: Literal["low", "high", "auto", "original"] | None = None

    @field_validator("image_url")
    @classmethod
    def _data_url(cls, value: str) -> str:
        """Refuse remote URLs and file ids, which the Decisions API does not accept."""
        if not value.startswith("data:image/") or ";base64," not in value:
            raise ValueError("decision images must be inline base64 data URLs")
        return value


InputPart = Annotated[InputTextPart | InputImagePart, Field(discriminator="type")]


class DecisionInputMessage(ContractModel):
    """One user message of a structured decision input.

    Attributes:
        type: The optional ``message`` discriminator SDKs may serialize.
        role: Always ``user``; OpenAI refuses every other role here.
        content: Text, or an ordered list of text and inline image parts.
    """

    type: Literal["message"] | None = None
    role: Literal["user"]
    content: str | tuple[InputPart, ...] = Field(min_length=1)


class DecisionChoice(ContractModel):
    """One allowed answer of a ``choice`` question.

    Attributes:
        value: The answer value. Values are typed: the string ``"true"`` and
            the boolean ``true`` are different values, never coerced.
        description: Optional guidance on when this value applies.
    """

    value: StrictStr | StrictBool
    description: str | None = None

    @field_validator("value")
    @classmethod
    def _value(cls, value: str | bool) -> str | bool:
        """Keep text answer values bounded identifiers."""
        if isinstance(value, bool):
            return value
        return _bounded_name(value, "choice values")


class DecisionLevel(ContractModel):
    """One ordered level of a ``score`` question.

    Attributes:
        label: The level's label, unique within its question.
        description: Optional guidance on when this level applies.
    """

    label: str
    description: str | None = None

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        """Keep level labels bounded identifiers."""
        return _bounded_name(value, "score level labels")


class PredicateQuestion(ContractModel):
    """Ask for the probability that a statement is true.

    Attributes:
        type: Always ``predicate``.
        name: Optional identifier, unique among named questions; answers come
            back in question order either way.
        instructions: The statement to judge.
    """

    type: Literal["predicate"] = "predicate"
    name: str | None = None
    instructions: str = Field(min_length=1)


class OpenAIChoiceQuestion(ContractModel):
    """Choose exactly one of the developer-defined values.

    Attributes:
        type: Always ``choice``.
        name: Optional identifier, unique among named questions.
        instructions: What to choose.
        choices: Two through 64 values, unique by typed value.
    """

    type: Literal["choice"] = "choice"
    name: str | None = None
    instructions: str = Field(min_length=1)
    choices: tuple[DecisionChoice, ...] = Field(
        min_length=2, max_length=MAX_OPENAI_DECISION_CHOICES
    )

    @field_validator("choices")
    @classmethod
    def _unique(cls, value: tuple[DecisionChoice, ...]) -> tuple[DecisionChoice, ...]:
        """Reject duplicate values, which would make the answer ambiguous."""
        # A (type, value) key keeps "true" and true distinct.
        if len({(type(choice.value), choice.value) for choice in value}) != len(value):
            raise ValueError("choice values must be unique within a question")
        return value


class OpenAIScoreQuestion(ContractModel):
    """Rate the input on an ordered list of levels.

    Attributes:
        type: Always ``score``.
        name: Optional identifier, unique among named questions.
        instructions: What to rate.
        levels: Two through 10 ordered levels, unique by label.
    """

    type: Literal["score"] = "score"
    name: str | None = None
    instructions: str = Field(min_length=1)
    levels: tuple[DecisionLevel, ...] = Field(min_length=2, max_length=MAX_OPENAI_DECISION_LEVELS)

    @field_validator("levels")
    @classmethod
    def _unique(cls, value: tuple[DecisionLevel, ...]) -> tuple[DecisionLevel, ...]:
        """Reject duplicate labels, which would make the answer ambiguous."""
        if len({level.label for level in value}) != len(value):
            raise ValueError("score level labels must be unique within a question")
        return value


OpenAIDecisionQuestion = Annotated[
    PredicateQuestion | OpenAIChoiceQuestion | OpenAIScoreQuestion,
    Field(discriminator="type"),
]


class OpenAIDecisionRequest(DecisionRequest):
    """Canonical OpenAI Decisions API request on the decisions surface.

    Input tokens are reserved from the UTF-8 bytes of every text part and
    question (bytes upper-bound tokens) plus a fixed allowance per inline image
    and per question. The wire bills no output tokens, so none are reserved;
    settlement always uses the provider's reported usage.

    Attributes:
        state: Always ``None``; the SystemOne field this wire does not carry.
        input: Text, or up to 256 user messages with at most 128 image parts.
        questions: One through 32 questions; named ones carry unique names.
        safety_identifier: Optional opaque end-user identifier forwarded to
            the provider, at most 128 characters.
    """

    state: JsonValue = None
    input: str | tuple[DecisionInputMessage, ...]
    questions: tuple[OpenAIDecisionQuestion, ...] = Field(  # type: ignore[assignment]
        min_length=1, max_length=MAX_OPENAI_DECISION_QUESTIONS
    )
    safety_identifier: str | None = Field(
        default=None, max_length=MAX_OPENAI_SAFETY_IDENTIFIER_CHARACTERS
    )

    @model_validator(mode="after")
    def _bounded_input(self) -> OpenAIDecisionRequest:
        """Validate identifiers, text, and the total serialized request size."""
        if self.state is not None:
            raise ValueError("OpenAI decisions carry input, not state")
        if isinstance(self.input, str):
            if not self.input:
                raise ValueError("decision input must not be empty")
        elif not self.input:
            raise ValueError("decision input must contain at least one message")
        elif len(self.input) > MAX_OPENAI_DECISION_MESSAGES:
            raise ValueError("decision input may contain at most 256 messages")
        elif self._image_count() > MAX_OPENAI_DECISION_IMAGES:
            raise ValueError("decision input may contain at most 128 image parts")
        names = [
            _bounded_name(question.name, "question names")
            for question in self.questions
            if question.name is not None
        ]
        if len(set(names)) != len(names):
            raise ValueError("question names must be unique")
        dumped = self.model_dump(mode="json", exclude_none=True)
        _native_json(dumped)
        encoded = json.dumps(dumped, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_OPENAI_DECISION_INPUT_BYTES:
            raise ValueError("decision request exceeds the 33554432-byte input limit")
        return self

    def _image_count(self) -> int:
        """Count inline image parts across every message."""
        if isinstance(self.input, str):
            return 0
        return sum(
            isinstance(part, InputImagePart)
            for message in self.input
            if not isinstance(message.content, str)
            for part in message.content
        )

    @property
    def wire(self) -> Literal["openai"]:
        """Name the decisions wire this request speaks."""
        return "openai"

    @property
    def input_token_reservation(self) -> int:
        """Reserve text and question bytes plus fixed image and question allowances."""
        text_bytes = 0
        images = 0
        messages = 0 if isinstance(self.input, str) else len(self.input)
        if isinstance(self.input, str):
            text_bytes += len(self.input.encode("utf-8"))
        else:
            for message in self.input:
                if isinstance(message.content, str):
                    text_bytes += len(message.content.encode("utf-8"))
                    continue
                for part in message.content:
                    if isinstance(part, InputTextPart):
                        text_bytes += len(part.text.encode("utf-8"))
                    else:
                        images += 1
        question_bytes = sum(
            len(
                json.dumps(
                    question.model_dump(mode="json", exclude_none=True), ensure_ascii=False
                ).encode("utf-8")
            )
            for question in self.questions
        )
        return (
            text_bytes
            + question_bytes
            + images * OPENAI_DECISION_IMAGE_TOKEN_RESERVATION
            + messages * OPENAI_DECISION_MESSAGE_OVERHEAD_TOKENS
            + len(self.questions) * OPENAI_DECISION_QUESTION_OVERHEAD_TOKENS
        )

    @property
    def output_token_reservation(self) -> int:
        """Reserve no output tokens: the OpenAI decisions wire bills input only."""
        return 0

    def provider_body(self, model: str) -> dict[str, JsonValue]:
        """Build the upstream ``/v1/decisions`` body with the deployment's wire id."""
        payload: dict[str, JsonValue] = self.model_dump(
            mode="json", exclude_none=True, include={"input", "questions", "safety_identifier"}
        )
        return {"model": model, **payload}

    def question_definitions(self) -> list[JsonValue]:
        """The admitted questions the data plane validates every answer against."""
        return [question.model_dump(mode="json", exclude_none=True) for question in self.questions]


class DecodedOpenAIDecisionRequest(ContractModel):
    """The public alias separated from the typed OpenAI decisions request.

    Attributes:
        alias: The public model alias the caller named.
        request: The typed request.
    """

    alias: str = Field(min_length=1, max_length=512)
    request: OpenAIDecisionRequest


def decode_openai_decision_request(body: str) -> DecodedOpenAIDecisionRequest:
    """Decode strict JSON for ``POST /v1/decisions`` before admission.

    Args:
        body: Raw request body text.

    Returns:
        The public alias and the typed request.

    Raises:
        ValueError: The body is not strict JSON, repeats a key, exceeds the
            size or nesting bounds, or carries fields other than ``model``,
            ``input``, ``questions``, and the optional ``safety_identifier``.
    """

    def unique(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        """Reject duplicate keys rather than silently losing a question."""
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in decision request")
            result[key] = value
        return result

    def finite(value: str) -> JsonValue:
        """Refuse JSON's nonstandard non-finite literals."""
        raise ValueError(f"non-finite JSON value {value} is not allowed")

    try:
        raw_bytes = len(body.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("decision JSON must contain valid UTF-8 text") from exc
    if raw_bytes > MAX_OPENAI_DECISION_INPUT_BYTES:
        raise ValueError("decision request exceeds the 33554432-byte input limit")
    try:
        raw = json.loads(body, object_pairs_hook=unique, parse_constant=finite)
    except RecursionError as exc:
        raise ValueError(
            f"decision JSON nesting exceeds the {MAX_DECISION_JSON_DEPTH}-level limit"
        ) from exc
    _native_json(raw)
    required = {"model", "input", "questions"}
    if not isinstance(raw, dict) or not required <= set(raw) <= required | {"safety_identifier"}:
        raise ValueError(
            "decision body requires model, input, and questions, and accepts only "
            "safety_identifier besides them"
        )
    model = raw["model"]
    if not isinstance(model, str):
        raise ValueError("decision model must be a string")
    safety_identifier = raw.get("safety_identifier")
    if safety_identifier is not None and not isinstance(safety_identifier, str):
        raise ValueError("decision safety_identifier must be a string")
    return DecodedOpenAIDecisionRequest(
        alias=model,
        request=OpenAIDecisionRequest(
            input=raw["input"], questions=raw["questions"], safety_identifier=safety_identifier
        ),
    )
