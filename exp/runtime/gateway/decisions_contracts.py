"""Typed SystemOne decisions, separate from conversational gateway requests."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import Field, StrictInt, ValidationError, field_validator, model_validator

from exp.common.core.artifacts import ContractModel, JsonValue
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.lone_surrogates import (
    SURROGATE_ESCAPE,
    has_lone_surrogate,
    repaired_text,
)

MAX_DECISION_QUESTIONS = 32
MAX_DECISION_CRITERIA = 64
MAX_SCORE_CRITERIA = 10
MAX_DECISION_INPUT_BYTES = 4_194_304
"""Serialized request bytes excluding image data (asset entries)."""
MAX_DECISION_JSON_DEPTH = 64
MAX_DECISION_IMAGE_BYTES = 33_554_432
"""Decoded image bytes across every asset of one request."""
MAX_DECISION_ASSET_BYTES = 16_777_216
"""Decoded bytes of one image file, TypeSafe's documented per-file limit."""
MAX_DECISION_ASSETS = 256
MAX_DECISION_REFERENCE_BYTES = 1_024
MAX_DECISION_REFERENCES_BYTES = 65_536
DECISION_IMAGE_TOKEN_RESERVATION = 4_096
"""Planning tokens per image per question; settlement bills the provider's own count."""
MAX_DECISION_BODY_BYTES = (
    MAX_DECISION_INPUT_BYTES + 4 * math.ceil(MAX_DECISION_IMAGE_BYTES / 3) + 4 * 1024 * 1024
)
"""Raw body ceiling: the text budget, base64 image data, and bounded asset framing."""
DECISION_FIELDS = frozenset({"model", "state", "questions", "assets"})
_STANDARD_BASE64 = re.compile(r"[A-Za-z0-9+/]*={0,2}")
STATE_SURROGATES_REPLACED = "state->replaced(lone_surrogate)"
"""Disclosure recorded when lone UTF-16 surrogate escapes in ``state`` become U+FFFD."""
INSTRUCTIONS_SURROGATES_REPLACED = "questions.instructions->replaced(lone_surrogate)"
"""Disclosure recorded when lone surrogate escapes in question instructions become U+FFFD."""
CRITERIA_SURROGATES_REPLACED = "questions.criteria->replaced(lone_surrogate)"
"""Disclosure recorded when lone surrogate escapes in criteria descriptions become U+FFFD."""
_CHAT_CONTROLS = frozenset(
    {
        "messages",
        "input",
        "prompt",
        "stream",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_tokens",
        "max_output_tokens",
        "response_format",
        "user",
    }
)


class DecisionRequestError(ValueError):
    """A decision request refusal whose text is safe to return to the caller.

    Messages name fields and limits only; they never quote state, instructions,
    criteria, question identifiers, asset references, or image data.
    """

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param
        self.message = message


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
            raise DecisionRequestError("body", "decision JSON nesting exceeds the 64-level limit")
        if isinstance(item, str):
            if item.isascii():
                continue
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise DecisionRequestError(
                    "body", "decision JSON must contain valid UTF-8 text"
                ) from exc
        elif isinstance(item, bool) or item is None:
            continue
        elif isinstance(item, int):
            if not -(2**63) <= item <= 2**64 - 1:
                raise DecisionRequestError(
                    "body", "decision JSON integers must fit signed or unsigned 64-bit values"
                )
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise DecisionRequestError("body", "decision JSON numbers must be finite")
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
    """Optional descriptions of the two proposition outcomes.

    Each outcome is text or structured JSON, so it can carry example images by
    asset reference (for example ``{"description": ..., "examples": [...]}``).
    """

    true: JsonValue = None
    false: JsonValue = None

    @model_validator(mode="after")
    def _outcomes(self) -> NoulCriteria:
        """Require text or structured descriptions for each stated outcome."""
        for outcome in (self.true, self.false):
            _native_json(outcome)
            if outcome is not None and not isinstance(outcome, (str, dict, list)):
                raise DecisionRequestError(
                    "questions",
                    "noul criteria true and false must be text, JSON objects, or arrays",
                )
        return self


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
            raise DecisionRequestError(
                "questions", "decision category names must contain 1 through 256 bytes"
            )
        if any(
            item is not None and not isinstance(item, (str, dict, list)) for item in value.values()
        ):
            raise DecisionRequestError(
                "questions", "choice criteria must be text, JSON objects, arrays, or null"
            )
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
            raise DecisionRequestError(
                "questions", "score criteria must be text, JSON objects, or arrays"
            )
        return value


DecisionQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class ImageAsset(ContractModel):
    """One inline image, placed wherever its reference string appears.

    Images arrive as base64 JSON only. Multipart file parts are not accepted on
    this surface, and the provider enforces its own per-image format and pixel
    limits.
    """

    type: Literal["image"] = "image"
    format: Literal["png", "jpg", "jpeg"]
    data_base64: str = Field(min_length=1)

    def decoded_bytes(self) -> int:
        """Return the decoded size from the validated text, without decoding it."""
        data = self.data_base64
        if len(data) % 4 or _STANDARD_BASE64.fullmatch(data) is None:
            raise DecisionRequestError(
                "assets",
                "asset data_base64 must be standard base64 with = padding, no whitespace, "
                "and no data: prefix",
            )
        padding = 2 if data.endswith("==") else 1 if data.endswith("=") else 0
        size = len(data) // 4 * 3 - padding
        if size > MAX_DECISION_ASSET_BYTES:
            raise DecisionRequestError(
                "assets", "each image must be at most 16777216 bytes after base64 decoding"
            )
        return size


def _strings(value: JsonValue) -> Iterator[str]:
    """Yield every string value in nested JSON; object keys are never references."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


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
    assets: dict[str, ImageAsset] | None = None

    @model_validator(mode="after")
    def _bounded_input(self) -> DecisionRequest:
        """Validate provider input shapes and total serialized request size."""
        _native_json(self.state)
        if not isinstance(self.state, (str, dict, list)):
            raise DecisionRequestError("state", "state must be text, a JSON object, or an array")
        for name, question in self.questions.items():
            _native_json(name)
            _native_json(question.instructions)
            if question.criteria is not None:
                _native_json(question.model_dump(mode="python")["criteria"])
            if not name or len(name.encode()) > 256:
                raise DecisionRequestError(
                    "questions", "question identifiers must contain 1 through 256 bytes"
                )
            if not isinstance(question.instructions, (str, dict, list)):
                raise DecisionRequestError(
                    "questions", "question instructions must be text, a JSON object, or an array"
                )
        if _json_bytes(self.model_dump(mode="json", exclude={"assets"})) > MAX_DECISION_INPUT_BYTES:
            raise DecisionRequestError(
                "body",
                f"decision request exceeds the {MAX_DECISION_INPUT_BYTES}-byte input limit "
                "(image data not counted); shorten state or questions",
            )
        if self.assets is not None:
            self._bounded_assets(self.assets)
        return self

    def _bounded_assets(self, assets: dict[str, ImageAsset]) -> None:
        """Bound references and decoded image bytes, and require every image be used."""
        if not assets:
            raise DecisionRequestError("assets", "assets must contain at least one image when sent")
        if len(assets) > MAX_DECISION_ASSETS:
            raise DecisionRequestError("assets", "assets may contain at most 256 images")
        sizes = [len(reference.encode("utf-8")) for reference in assets]
        if any(not reference.strip() for reference in assets):
            raise DecisionRequestError("assets", "asset references must not be empty or whitespace")
        if max(sizes) > MAX_DECISION_REFERENCE_BYTES:
            raise DecisionRequestError("assets", "each asset reference must be at most 1024 bytes")
        if sum(sizes) > MAX_DECISION_REFERENCES_BYTES:
            raise DecisionRequestError("assets", "asset references must total at most 65536 bytes")
        if sum(asset.decoded_bytes() for asset in assets.values()) > MAX_DECISION_IMAGE_BYTES:
            raise DecisionRequestError(
                "assets", "decoded image data must total at most 33554432 bytes per request"
            )
        texts = self._reference_texts()
        if any(not any(reference in text for text in texts) for reference in assets):
            raise DecisionRequestError(
                "assets",
                "every asset reference must appear in state, a question's instructions, "
                "or its criteria",
            )

    def _reference_texts(self) -> list[str]:
        """Return every string where an asset reference may stand for its image."""
        texts = list(_strings(self.state))
        for question in self.questions.values():
            texts.extend(_strings(question.instructions))
            if question.criteria is not None:
                texts.extend(_strings(question.model_dump(mode="python")["criteria"]))
        return texts

    def _referenced_assets(self, value: JsonValue) -> frozenset[str]:
        """Return the asset references that appear anywhere in one value's strings."""
        if not self.assets:
            return frozenset()
        texts = list(_strings(value))
        return frozenset(
            reference for reference in self.assets if any(reference in text for text in texts)
        )

    @property
    def attribution_label(self) -> None:
        """Return no end-user attribution; the native decision wire defines none."""
        return None

    @property
    def input_token_reservation(self) -> int:
        """Reserve every question's repeated state, its images, and protocol overhead."""
        state_bytes = _json_bytes(self.state)
        state_images = self._referenced_assets(self.state)
        total = 0
        for question in self.questions.values():
            definition = question.model_dump(mode="json", exclude_none=True)
            # Each image counts once per question however often its reference
            # repeats, bounding the estimate by assets x questions.
            images = len(state_images | self._referenced_assets(definition))
            total += (
                state_bytes
                + _json_bytes(definition)
                + 1024
                + images * DECISION_IMAGE_TOKEN_RESERVATION
            )
        return total

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
        body: dict[str, JsonValue] = {
            "model": model,
            "state": self.state,
            "questions": {
                key: question.model_dump(mode="json", exclude_none=True)
                for key, question in self.questions.items()
            },
        }
        if self.assets:
            # Built by hand so the base64 text is shared, not copied, per rung.
            body["assets"] = {
                reference: {
                    "type": asset.type,
                    "format": asset.format,
                    "data_base64": asset.data_base64,
                }
                for reference, asset in self.assets.items()
            }
        return body

    @property
    def has_images(self) -> bool:
        """Whether the request carries image assets."""
        return bool(self.assets)


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
    """The public alias separated from the provider-neutral request.

    Attributes:
        alias: The public model alias the caller named.
        request: The provider-neutral decision request.
        ignored_parameters: Disclosed request rewrites, returned in
            ``x-experiential-ignored-parameters``.
    """

    alias: str = Field(min_length=1, max_length=512)
    request: DecisionRequest
    ignored_parameters: tuple[str, ...] = ()


def decode_decision_request(body: str, byte_length: int | None = None) -> DecodedDecisionRequest:
    """Decode strict JSON and reject duplicate keys before request admission.

    Args:
        body: The raw request body text.
        byte_length: Its UTF-8 length when the caller already measured it, so a
            large image body is not re-encoded only to be counted.
    """

    def unique(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        """Reject duplicate identifiers rather than silently losing a question."""
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise DecisionRequestError("body", "decision JSON must not repeat an object key")
            result[key] = value
        return result

    def finite(value: str) -> JsonValue:
        """Refuse JSON's nonstandard non-finite literals."""
        raise DecisionRequestError("body", f"non-finite JSON value {value} is not allowed")

    try:
        raw_bytes = byte_length if byte_length is not None else len(body.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise DecisionRequestError("body", "decision JSON must contain valid UTF-8 text") from exc
    if raw_bytes > MAX_DECISION_BODY_BYTES:
        raise DecisionRequestError(
            "body", "decision request body exceeds the gateway's request size limit"
        )
    try:
        raw = json.loads(body, object_pairs_hook=unique, parse_constant=finite)
    except RecursionError as exc:
        raise DecisionRequestError(
            "body", "decision JSON nesting exceeds the 64-level limit"
        ) from exc
    ignored_parameters: tuple[str, ...] = ()
    if isinstance(raw, dict) and SURROGATE_ESCAPE.search(body) is not None:
        raw, ignored_parameters = _replace_lone_surrogates(raw)
    _native_json(raw)
    if not isinstance(raw, dict):
        raise DecisionRequestError("body", "decision request body must be a JSON object")
    _require_fields(raw)
    request = DecisionRequest.model_validate(
        {"state": raw["state"], "questions": raw["questions"], "assets": raw.get("assets")}
    )
    return DecodedDecisionRequest.model_validate(
        {"alias": raw["model"], "request": request, "ignored_parameters": ignored_parameters}
    )


def _replace_lone_surrogates(
    raw: dict[str, JsonValue],
) -> tuple[dict[str, JsonValue], tuple[str, ...]]:
    """Replace lone UTF-16 surrogates in decision content with U+FFFD, disclosed.

    JavaScript's well-formed ``JSON.stringify`` writes a string cut inside a
    surrogate pair (an emoji sliced by length) as a lone ``\\udXXX`` escape:
    valid JSON that has no UTF-8 form, so the text cannot reach a provider.
    Content (state, instructions, criteria descriptions, and the keys inside
    them) is repaired the way a lenient UTF-8 decoder would. Identifiers the
    answer is keyed by (the model, question IDs, choice names, asset
    references) are refused instead: a rewritten key could not match the
    caller's own.

    Raises:
        DecisionRequestError: An identifier holds a lone surrogate, or two
            content keys become equal once repaired.
    """
    # The decoded body is owned here, so it is repaired in place: nothing is
    # copied per question or per entry before the input bounds are checked.
    disclosed: list[str] = []
    if "state" in raw:
        raw["state"], changed = _repaired_content(raw["state"], "state", 1)
        if changed:
            disclosed.append(STATE_SURROGATES_REPLACED)
    questions = raw.get("questions")
    if isinstance(questions, dict):
        instructions_changed = criteria_changed = False
        for question in questions.values():
            if not isinstance(question, dict):
                continue
            if "instructions" in question:
                question["instructions"], changed = _repaired_content(
                    question["instructions"], "questions", 3
                )
                instructions_changed |= changed
            criteria = question.get("criteria")
            # Choice names are answer keys, and noul's true/false are fixed
            # vocabulary: only their descriptions are content.
            if isinstance(criteria, dict):
                for key, item in criteria.items():
                    repaired, changed = _repaired_content(item, "questions", 4)
                    if changed:
                        criteria[key] = repaired
                        criteria_changed = True
            elif isinstance(criteria, list):
                for index, item in enumerate(criteria):
                    repaired, changed = _repaired_content(item, "questions", 4)
                    if changed:
                        criteria[index] = repaired
                        criteria_changed = True
        if instructions_changed:
            disclosed.append(INSTRUCTIONS_SURROGATES_REPLACED)
        if criteria_changed:
            disclosed.append(CRITERIA_SURROGATES_REPLACED)
    _refuse_surrogate_identifiers(raw)
    assets = raw.get("assets")
    if (
        disclosed
        and isinstance(assets, dict)
        and any("\N{REPLACEMENT CHARACTER}" in reference for reference in assets)
    ):
        # A repair writes U+FFFD into the text, so it could newly reference
        # this asset and send an image the caller's own text never named.
        raise DecisionRequestError(
            "assets",
            "an asset reference containing U+FFFD cannot be used while lone UTF-16 "
            "surrogates in the request text are being replaced; resend the text as valid Unicode",
        )
    return raw, tuple(disclosed)


def _repaired_content(value: JsonValue, param: str, depth: int) -> tuple[JsonValue, bool]:
    """Return ``value`` with every lone surrogate replaced, and whether any was.

    Past the nesting limit the value is returned untouched for
    ``_native_json`` to refuse, which also bounds this recursion.
    """
    if depth > MAX_DECISION_JSON_DEPTH:
        return value, False
    if isinstance(value, str):
        return repaired_text(value)
    if isinstance(value, list):
        # The decoded body is owned here, so a list is repaired in place:
        # nothing is allocated per entry of a large unchanged array.
        any_changed = False
        for index, item in enumerate(value):
            repaired, changed = _repaired_content(item, param, depth + 1)
            if changed:
                value[index] = repaired
                any_changed = True
        return value, any_changed
    if isinstance(value, dict):
        if not any(has_lone_surrogate(key) for key in value):
            # Keys stay as they are, so the owned mapping is repaired in place.
            unchanged = True
            for key, item in value.items():
                repaired, changed = _repaired_content(item, param, depth + 1)
                if changed:
                    value[key] = repaired
                    unchanged = False
            return value, not unchanged
        result: dict[str, JsonValue] = {}
        any_changed = False
        for key, item in value.items():
            new_key, key_changed = repaired_text(key)
            if new_key in result:
                raise DecisionRequestError(
                    param,
                    "decision JSON object keys must stay distinct after lone UTF-16 "
                    "surrogates are replaced with U+FFFD",
                )
            result[new_key], item_changed = _repaired_content(item, param, depth + 1)
            any_changed |= key_changed or item_changed
        return result, any_changed
    return value, False


def _refuse_surrogate_identifiers(raw: dict[str, JsonValue]) -> None:
    """Name the identifier field holding a lone surrogate, without echoing it."""

    def lone(text: JsonValue | None) -> bool:
        """Whether ``text`` is a string carrying a lone UTF-16 surrogate."""
        return isinstance(text, str) and has_lone_surrogate(text)

    questions = raw.get("questions")
    assets = raw.get("assets")
    if lone(raw.get("model")):
        param = "model"
    elif isinstance(questions, dict) and any(
        lone(name)
        or (
            isinstance(question, dict)
            and question.get("type") == "choice"
            and isinstance(question.get("criteria"), dict)
            and any(lone(key) for key in question["criteria"])
        )
        for name, question in questions.items()
    ):
        param = "questions"
    elif isinstance(assets, dict) and any(lone(reference) for reference in assets):
        param = "assets"
    else:
        return
    raise DecisionRequestError(
        param,
        "model, question IDs, choice category names, and asset references must not contain "
        "lone UTF-16 surrogates (a \\uD800-\\uDFFF escape without its pair); these keys "
        "identify answers, so they are not replaced",
    )


def _require_fields(raw: dict[str, JsonValue]) -> None:
    """Name the missing or unsupported top-level field without echoing caller keys."""
    for field in ("model", "state", "questions"):
        if field not in raw:
            raise DecisionRequestError(
                field, f"decision requests require model, state, and questions; {field} is missing"
            )
    unknown = sorted(set(raw) - DECISION_FIELDS)
    if unknown:
        named = next((field for field in unknown if field in _CHAT_CONTROLS), None)
        raise DecisionRequestError(
            named or "body",
            (f"{named} is not supported on /v1/systemone; " if named else "")
            + "decision requests accept only model, state, questions, and optional assets "
            "(chat, streaming, tools, and generation controls are not supported)",
        )
    assets = raw.get("assets")
    if isinstance(assets, dict) and any(
        isinstance(asset, dict) and "part" in asset for asset in assets.values()
    ):
        raise DecisionRequestError(
            "assets",
            "multipart image uploads (asset part) are not supported; send each image as "
            "data_base64 with format in a JSON request",
        )


_QUESTION_FIELD_MESSAGES = {
    "instructions": "every question needs instructions as text, a JSON object, or an array",
    "noul": "noul criteria must be an object with optional true and false outcomes",
    "choice": "choice criteria must map 2 through 64 category names to descriptions",
    "score": "score criteria must be a list of 2 through 10 levels",
}


def public_decision_error(exc: Exception) -> tuple[str, str]:
    """Return a caller-safe ``(param, message)`` for a decision decode failure.

    Pydantic and JSON error text can quote state, instructions, question ids, or
    asset references, so only error locations and this module's own refusals are
    translated; values and caller-chosen keys are never copied.
    """
    if isinstance(exc, DecisionRequestError):
        return exc.param, exc.message
    if isinstance(exc, RecursionError):
        return "body", "decision JSON nesting exceeds the 64-level limit"
    if isinstance(exc, ValidationError) and exc.errors():
        error = exc.errors()[0]
        cause = error.get("ctx", {}).get("error")
        if isinstance(cause, DecisionRequestError):
            return cause.param, cause.message
        return _located_error(tuple(error["loc"]), error["type"])
    if isinstance(exc, json.JSONDecodeError):
        return "body", "decision request body must be valid JSON"
    return (
        "body",
        "Invalid decision request. Send JSON with model, state, questions (noul, choice, or "
        "score), and optional image assets.",
    )


def _located_error(location: tuple[int | str, ...], kind: str) -> tuple[str, str]:
    """Map a pydantic error location and type to fixed text; caller keys stay positional."""
    head = location[0] if location else "body"
    if head == "alias":
        return "model", "model must be a model id of 1 through 512 characters"
    if head == "state":
        return "state", "state must be text, a JSON object, or an array"
    if head == "assets":
        field = location[2] if len(location) > 2 else None
        if field == "format":
            return "assets", "asset format must be png, jpg, or jpeg"
        if field == "type":
            return "assets", 'asset type must be "image" when sent'
        return "assets", "each asset must be an object with format and data_base64"
    if head == "questions":
        if kind in {"union_tag_invalid", "union_tag_not_found"} or len(location) == 3:
            return "questions", "each question type must be noul, choice, or score"
        if len(location) <= 2:
            return "questions", "questions must map 1 through 32 ids to typed questions"
        field = location[3]
        if field == "criteria" and location[2] in _QUESTION_FIELD_MESSAGES:
            return "questions", _QUESTION_FIELD_MESSAGES[str(location[2])]
        if field == "instructions":
            return "questions", _QUESTION_FIELD_MESSAGES["instructions"]
        return "questions", "questions accept only type, instructions, and criteria"
    return "body", "decision requests accept only model, state, questions, and optional assets"
