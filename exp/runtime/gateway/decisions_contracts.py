"""Typed SystemOne decisions, separate from conversational gateway requests."""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from exp.common.core.artifacts import ContractModel, JsonValue
from exp.runtime.gateway.contracts import GatewayApiSurface

MAX_DECISION_QUESTIONS = 32
MAX_DECISION_CRITERIA = 64
MAX_SCORE_CRITERIA = 10
MAX_DECISION_INPUT_BYTES = 262_144
MAX_DECISION_JSON_DEPTH = 64


def _native_json(value: JsonValue) -> None:
    """Reject values Rust cannot preserve exactly through its bounded JSON bridge.

    Args:
        value: JSON-native state, instructions, criteria, or a complete request.

    Raises:
        ValueError: Text is invalid UTF-8, numbers exceed native JSON representation,
            or nesting exceeds the gateway's bridge-safe depth.
    """
    pending: list[tuple[JsonValue, int]] = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_DECISION_JSON_DEPTH:
            raise ValueError("decision JSON nesting exceeds the 64-level limit")
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("decision JSON must contain valid UTF-8 text") from exc
        elif isinstance(item, bool) or item is None:
            continue
        elif isinstance(item, int):
            if not -(2**63) <= item <= 2**64 - 1:
                raise ValueError("decision JSON integers must fit signed or unsigned 64-bit values")
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("decision JSON numbers must be finite")
        elif isinstance(item, dict):
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)


def _json_bytes(value: JsonValue) -> int:
    """Validate native JSON and return its strict UTF-8 serialized byte count."""
    _native_json(value)
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


class NoulCriteria(ContractModel):
    """Optional descriptions of the two proposition outcomes."""

    true: str | None = None
    false: str | None = None


class NoulQuestion(ContractModel):
    """Ask for the probability of a proposition."""

    type: Literal["noul"] = "noul"
    instructions: JsonValue
    criteria: NoulCriteria | None = None


class ChoiceQuestion(ContractModel):
    """Choose among explicitly named, unordered categories."""

    type: Literal["choice"] = "choice"
    instructions: JsonValue
    criteria: dict[str, JsonValue] = Field(min_length=2, max_length=MAX_DECISION_CRITERIA)

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Keep category names bounded and their optional descriptions structured."""
        _native_json(value)
        if any(not name or len(name.encode()) > 256 for name in value):
            raise ValueError("decision category names must contain 1 through 256 bytes")
        if any(
            item is not None and not isinstance(item, (str, dict, list)) for item in value.values()
        ):
            raise ValueError("choice criteria must be text, JSON objects, arrays, or null")
        return value


class ScoreQuestion(ContractModel):
    """Rate state against an ordered list of criteria."""

    type: Literal["score"] = "score"
    instructions: JsonValue
    criteria: tuple[JsonValue, ...] = Field(min_length=2, max_length=MAX_SCORE_CRITERIA)

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, value: tuple[JsonValue, ...]) -> tuple[JsonValue, ...]:
        """Require text or structured descriptions for every ordered score level."""
        _native_json(list(value))
        if any(not isinstance(item, (str, dict, list)) for item in value):
            raise ValueError("score criteria must be text, JSON objects, or arrays")
        return value


DecisionQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class DecisionRequest(ContractModel):
    """Canonical non-streaming decision request with bounded provider work.

    State is repeated per question in the conservative token reservation. JSON
    bytes upper-bound visible text tokens and the per-question allowance covers
    protocol instructions. Providers exceeding that estimate still report their
    actual usage at settlement; no usage is invented from decision values.
    """

    surface: Literal[GatewayApiSurface.DECISIONS] = GatewayApiSurface.DECISIONS
    state: JsonValue
    questions: dict[str, DecisionQuestion] = Field(min_length=1, max_length=MAX_DECISION_QUESTIONS)

    @model_validator(mode="after")
    def _bounded_input(self) -> DecisionRequest:
        """Validate provider input shapes and total serialized request size."""
        _native_json(self.state)
        if not isinstance(self.state, (str, dict, list)):
            raise ValueError("state must be text, a JSON object, or an array")
        for name, question in self.questions.items():
            _native_json(name)
            _native_json(question.instructions)
            if question.criteria is not None:
                _native_json(question.model_dump(mode="python")["criteria"])
            if not name or len(name.encode()) > 256:
                raise ValueError("question identifiers must contain 1 through 256 bytes")
            if not isinstance(question.instructions, (str, dict, list)):
                raise ValueError("question instructions must be text, a JSON object, or an array")
        if _json_bytes(self.model_dump(mode="json")) > MAX_DECISION_INPUT_BYTES:
            raise ValueError("decision request exceeds the 262144-byte input limit")
        return self

    @property
    def attribution_label(self) -> None:
        """Return no end-user attribution; the native decision wire defines none."""
        return None

    @property
    def input_token_reservation(self) -> int:
        """Reserve every question's repeated state plus bounded protocol overhead."""
        state_bytes = _json_bytes(self.state)
        return sum(
            state_bytes + _json_bytes(question.model_dump(mode="json", exclude_none=True)) + 1024
            for question in self.questions.values()
        )

    @property
    def output_token_reservation(self) -> int:
        """Bound decision-answer accounting without manufacturing an upstream limit."""
        return sum(
            1024
            + 512
            * (
                len(question.criteria)
                if isinstance(question, (ChoiceQuestion, ScoreQuestion))
                else 2
            )
            for question in self.questions.values()
        )

    def provider_body(self, model: str) -> dict[str, JsonValue]:
        """Build the native upstream document with the selected deployment wire ID."""
        return {
            "model": model,
            "state": self.state,
            "questions": {
                key: question.model_dump(mode="json", exclude_none=True)
                for key, question in self.questions.items()
            },
        }


class DecisionUsage(ContractModel):
    """Provider-reported token counts required to settle a decision request."""

    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _nonempty_usage(self) -> DecisionUsage:
        """Require credible usage for a response containing at least one decision."""
        if self.input_tokens == 0 and self.output_tokens == 0:
            raise ValueError("decision usage must contain at least one token")
        return self


class DecodedDecisionRequest(ContractModel):
    """The public alias separated from the provider-neutral request."""

    alias: str = Field(min_length=1, max_length=512)
    request: DecisionRequest


def decode_decision_request(body: str) -> DecodedDecisionRequest:
    """Decode strict JSON and reject duplicate keys before request admission."""

    def unique(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        """Reject duplicate identifiers rather than silently losing a question."""
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
    if raw_bytes > MAX_DECISION_INPUT_BYTES:
        raise ValueError("decision request exceeds the 262144-byte input limit")
    try:
        raw = json.loads(body, object_pairs_hook=unique, parse_constant=finite)
    except RecursionError as exc:
        raise ValueError("decision JSON nesting exceeds the 64-level limit") from exc
    _native_json(raw)
    if not isinstance(raw, dict) or set(raw) != {"model", "state", "questions"}:
        raise ValueError("decision body requires only model, state, and questions")
    return DecodedDecisionRequest(
        alias=raw["model"], request=DecisionRequest(state=raw["state"], questions=raw["questions"])
    )
