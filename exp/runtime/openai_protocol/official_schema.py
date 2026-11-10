"""Validate request bodies against the installed official OpenAI SDK request types.

The official Chat and Responses request types are unions of a non-streaming
and a streaming branch that differ only in ``stream``. Validating the one
branch the caller's ``stream`` selects accepts exactly what the union accepts,
and keeps the other branch's "stream: missing" complaint from being reported
as the request's fault when the request is wrong elsewhere (a non-string
``model``, a ``null`` ``tool_choice``): union error selection would otherwise
name ``stream`` with "expected a value" for every such body.
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


def stream_requested(payload: JsonObject) -> bool:
    """Whether the body's ``stream`` asks for a streamed answer.

    The official streaming branch's ``Literal[True]`` validates in pydantic's
    lax mode, so it admits every non-string JSON value equal to true (``true``,
    ``1``, ``1.0``). Every decision that depends on whether a call streams
    reads this one predicate, so a numeric true streams everywhere.

    Args:
        payload: Parsed request body.

    Returns:
        True exactly when ``stream`` is a non-string number equal to 1.
    """
    stream = payload.get("stream")
    return isinstance(stream, int | float) and stream == 1


def streaming_branch(branches: OfficialBranches, payload: JsonObject) -> TypeAdapter[object]:
    """Pick the official branch the caller's ``stream`` selects.

    A ``stream`` that ``stream_requested`` reads as true selects the streaming
    branch. An absent, false or otherwise-valued ``stream`` can only satisfy
    the non-streaming branch, whose own complaint about ``stream`` is then the
    accurate one.

    Args:
        branches: The surface's (non-streaming, streaming) official schemas.
        payload: The body about to be validated.

    Returns:
        The one branch that can accept the caller's ``stream`` value.
    """
    non_streaming, streaming = branches
    return streaming if stream_requested(payload) else non_streaming


def validate_official(
    adapter: TypeAdapter[object],
    payload: JsonObject,
    *,
    extension_fields: Collection[str] = frozenset(),
) -> None:
    """Run the installed official SDK request schema before gateway narrowing.

    Args:
        adapter: The official schema (or the one branch of it) to validate against.
        payload: Parsed request body.
        extension_fields: Gateway-only top-level fields the official schema
            does not know, removed before validation.

    Raises:
        OpenAIProtocolError: The body fails the official schema, naming the field.
    """
    try:
        official_payload = {
            key: value for key, value in payload.items() if key not in extension_fields
        }
        adapter.validate_python(official_payload)
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc


def validate_wire[ModelT: BaseModel](model: type[ModelT], payload: JsonObject) -> ModelT:
    """Validate one strict private wire model with a field-specific public error.

    Args:
        model: The gateway's closed wire model for the surface.
        payload: Parsed request body.

    Returns:
        The validated wire model.

    Raises:
        OpenAIProtocolError: The body fails the wire model, naming the field.
    """
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
