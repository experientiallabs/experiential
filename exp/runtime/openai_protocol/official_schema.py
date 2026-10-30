"""Validate request bodies against the installed official OpenAI SDK request types.

Split from ``requests`` for the module line budget. The official Chat and
Responses request types are unions of a non-streaming and a streaming branch
that differ only in ``stream``. Validating the one branch the caller's
``stream`` selects accepts exactly what the union accepts, and keeps the other
branch's "stream: missing" complaint from being reported as the request's
fault when the request is actually wrong elsewhere (a non-string ``model``, a
``null`` ``tool_choice``): union error selection otherwise named ``stream``
with "expected a value" for every such body.
"""

from __future__ import annotations

from collections.abc import Collection

from openai.types import EmbeddingCreateParams
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
    ResponseCreateParamsStreaming,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.validation_errors import validation_protocol_error

type OfficialBranches = tuple[TypeAdapter[object], TypeAdapter[object]]
"""The (non-streaming, streaming) official schema branches of one surface."""

CHAT_OFFICIAL: OfficialBranches = (
    TypeAdapter[object](CompletionCreateParamsNonStreaming),
    TypeAdapter[object](CompletionCreateParamsStreaming),
)
RESPONSES_OFFICIAL: OfficialBranches = (
    TypeAdapter[object](ResponseCreateParamsNonStreaming),
    TypeAdapter[object](ResponseCreateParamsStreaming),
)
EMBEDDINGS_OFFICIAL: TypeAdapter[object] = TypeAdapter[object](EmbeddingCreateParams)


def streaming_branch(branches: OfficialBranches, payload: JsonObject) -> TypeAdapter[object]:
    """Pick the official branch the caller's ``stream`` selects.

    Only ``stream: true`` selects the streaming branch; an absent, false or
    otherwise-valued ``stream`` can only satisfy the non-streaming one, whose
    own complaint about ``stream`` is then the accurate one.
    """
    non_streaming, streaming = branches
    return streaming if payload.get("stream") is True else non_streaming


def validate_official(
    adapter: TypeAdapter[object],
    payload: JsonObject,
    *,
    extension_fields: Collection[str] = frozenset(),
) -> None:
    """Run the installed official SDK request schema before gateway narrowing."""
    try:
        official_payload = {
            key: value for key, value in payload.items() if key not in extension_fields
        }
        adapter.validate_python(official_payload)
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc


def validate_wire[ModelT: BaseModel](model: type[ModelT], payload: JsonObject) -> ModelT:
    """Validate one strict private wire model with a field-specific public error."""
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
