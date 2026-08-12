"""Shared non-streaming OpenAI-compatible conversion and focused HTTP client."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    Embedding,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ToolCall,
    Usage,
)
from wmo.runtime.models.providers.errors import ProviderResponseError
from wmo.runtime.models.providers.request import post_json
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.transport import HttpxJsonTransport, JsonHttpTransport

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RETRY_POLICY = RetryPolicy()


class OpenAICompatibleResponseError(ProviderResponseError):
    """An OpenAI-compatible endpoint returned a response outside the typed contract."""


def openai_compatible_request(model_id: str, request: ModelRequest) -> JsonObject:
    """Convert a WMO request into one non-streaming Chat Completions payload.

    Args:
        model_id: Provider model identifier to place on the wire.
        request: Typed WMO request.

    Returns:
        A JSON object for ``/chat/completions``.

    Raises:
        ValueError: A request message cannot be represented without losing tool context.
    """
    payload: JsonObject = {
        "model": model_id,
        "messages": [_openai_message(message) for message in request.messages],
        "stream": False,
    }
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        payload["tool_choice"] = (
            {
                "type": "function",
                "function": {"name": request.tool_choice.name},
            }
            if not isinstance(request.tool_choice, str)
            else request.tool_choice
        )
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.maximum_output_tokens is not None:
        payload["max_tokens"] = request.maximum_output_tokens
    return payload


def openai_embedding_request(model_id: str, texts: Sequence[str]) -> JsonObject:
    """Convert ordered text into one OpenAI-compatible embedding request."""
    return {"model": model_id, "input": list(texts)}


def openai_compatible_response(
    payload: JsonObject,
    *,
    configured_model: ModelSnapshot,
    latency_seconds: float,
) -> ModelResponse:
    """Convert one complete Chat Completions response into WMO's shared contract.

    Args:
        payload: Decoded provider response.
        configured_model: Resolved identity before the request was sent.
        latency_seconds: Wall-clock duration for the successful request sequence.

    Returns:
        Typed output, actual-or-configured model identity, and observed usage and latency.

    Raises:
        OpenAICompatibleResponseError: The response has no usable first choice or invalid tools.
    """
    choices = _array(payload.get("choices"), "choices")
    if not choices:
        raise OpenAICompatibleResponseError("OpenAI-compatible response has no choices")
    choice = _object(choices[0], "choices[0]")
    message = _object(choice.get("message"), "choices[0].message")
    content_value = message.get("content")
    content = content_value if isinstance(content_value, str) else None
    tool_call_values = _array_or_empty(message)
    tool_calls = tuple(_tool_call(value, index) for index, value in enumerate(tool_call_values))
    try:
        output = AssistantAction(content=content, tool_calls=tool_calls)
    except ValueError as exc:
        raise OpenAICompatibleResponseError(
            "OpenAI-compatible response has neither text nor a complete tool call"
        ) from exc
    return ModelResponse(
        output=output,
        model=_resolved_model_snapshot(payload, configured_model),
        economics=OperationEconomics(
            usage=_usage(payload),
            latency_seconds=NumericMeasurement(value=latency_seconds, provenance="observed"),
        ),
    )


def openai_embedding_response(payload: JsonObject, *, expected_count: int) -> tuple[Embedding, ...]:
    """Convert and normalize an OpenAI-compatible embedding response.

    Args:
        payload: Decoded ``/embeddings`` response.
        expected_count: Number of requested input strings.

    Returns:
        One normalized embedding in input order for every input.

    Raises:
        OpenAICompatibleResponseError: The provider omitted, duplicated, or malformed vectors.
    """
    data = _array(payload.get("data"), "data")
    if len(data) != expected_count:
        raise OpenAICompatibleResponseError(
            f"embedding response count {len(data)} does not match request count {expected_count}"
        )
    ordered: list[Embedding | None] = [None] * expected_count
    for position, value in enumerate(data):
        item = _object(value, f"data[{position}]")
        index_value = item.get("index", position)
        if not isinstance(index_value, int) or isinstance(index_value, bool):
            raise OpenAICompatibleResponseError(f"data[{position}].index must be an integer")
        if index_value < 0 or index_value >= expected_count or ordered[index_value] is not None:
            raise OpenAICompatibleResponseError(
                "embedding response indexes must be unique input indexes"
            )
        vector = _array(item.get("embedding"), f"data[{position}].embedding")
        ordered[index_value] = Embedding(values=normalize_embedding_vector(vector))
    if any(item is None for item in ordered):
        raise OpenAICompatibleResponseError("embedding response omitted an input index")
    return tuple(cast("Embedding", item) for item in ordered)


class OpenAICompatibleClient:
    """Calls one explicit OpenAI-compatible connection without streaming or failover."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        base_url: str,
        api_key: str,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Create a client with a single explicit endpoint and credential.

        Args:
            model: Resolved configured model identity.
            base_url: Endpoint root that exposes OpenAI-compatible routes.
            api_key: Credential already read from the named environment variable.
            transport: Optional deterministic transport used by tests.
            retry_policy: Bounded same-endpoint retry policy.
            timeout_seconds: Timeout for every transport attempt.
            extra_headers: Provider-specific non-secret headers.
        """
        if not api_key:
            raise ValueError("OpenAI-compatible clients require a non-empty API key")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport or HttpxJsonTransport()
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds
        self._extra_headers = dict(extra_headers or {})

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one non-streaming request through Chat Completions."""
        started_at = time.monotonic()
        response = post_json(
            self._transport,
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            payload=openai_compatible_request(self._model.model_id, request),
            timeout_seconds=self._timeout_seconds,
            retry_policy=self._retry_policy,
        )
        return openai_compatible_response(
            response,
            configured_model=self._model,
            latency_seconds=time.monotonic() - started_at,
        )

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed ordered text through the configured model without making empty requests."""
        if not texts:
            return ()
        response = post_json(
            self._transport,
            f"{self._base_url}/embeddings",
            headers=self._headers(),
            payload=openai_embedding_request(self._model.model_id, texts),
            timeout_seconds=self._timeout_seconds,
            retry_policy=self._retry_policy,
        )
        return openai_embedding_response(response, expected_count=len(texts))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }


def _openai_message(message: ModelMessage) -> JsonObject:
    """Convert one WMO message while retaining assistant tool history."""
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content or "",
            "tool_call_id": message.tool_call_id or "",
        }
    if message.role != "assistant":
        if message.assistant_action is not None:
            raise ValueError(f"{message.role} messages cannot carry assistant actions")
        if message.content is None:
            raise ValueError(f"{message.role} messages need text content")
        return {"role": message.role, "content": message.content}
    action = message.assistant_action
    content = message.content if message.content is not None else action.content if action else None
    payload: JsonObject = {"role": "assistant", "content": content or ""}
    if action is not None and action.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in action.tool_calls
        ]
    return payload


def _tool_call(value: JsonValue, index: int) -> ToolCall:
    """Parse one OpenAI-compatible tool call without accepting malformed JSON arguments."""
    item = _object(value, f"tool_calls[{index}]")
    call_id = _string(item.get("id"), f"tool_calls[{index}].id")
    function = _object(item.get("function"), f"tool_calls[{index}].function")
    name = _string(function.get("name"), f"tool_calls[{index}].function.name")
    raw_arguments = _string(function.get("arguments"), f"tool_calls[{index}].function.arguments")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise OpenAICompatibleResponseError(
            f"tool_calls[{index}].function.arguments is not JSON"
        ) from exc
    if not isinstance(arguments, dict):
        raise OpenAICompatibleResponseError(
            f"tool_calls[{index}].function.arguments must decode to an object"
        )
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _array_or_empty(message: JsonObject) -> list[JsonValue]:
    """Return optional tool calls as an array, rejecting every other wire shape."""
    value = message.get("tool_calls")
    if value is None:
        return []
    return _array(value, "choices[0].message.tool_calls")


def _resolved_model_snapshot(payload: JsonObject, configured: ModelSnapshot) -> ModelSnapshot:
    """Prefer a provider-returned concrete model ID while retaining pinned metadata."""
    model_id = payload.get("model")
    if not isinstance(model_id, str) or not model_id:
        return configured
    return configured.model_copy(update={"model_id": model_id})


def _usage(payload: JsonObject) -> Usage | None:
    """Read optional OpenAI-compatible token usage without inventing absent measurements."""
    value = payload.get("usage")
    if value is None:
        return None
    usage = _object(value, "usage")
    prompt_tokens = _integer(usage.get("prompt_tokens"), "usage.prompt_tokens", default=0)
    completion_tokens = _integer(
        usage.get("completion_tokens"), "usage.completion_tokens", default=0
    )
    details_value = usage.get("prompt_tokens_details")
    cached_input_tokens = None
    if details_value is not None:
        details = _object(details_value, "usage.prompt_tokens_details")
        cached_input_tokens = _integer(
            details.get("cached_tokens"), "usage.prompt_tokens_details.cached_tokens", default=0
        )
    return Usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def normalize_embedding_vector(values: Sequence[JsonValue]) -> tuple[float, ...]:
    """Return one finite, non-zero unit vector from a provider response."""
    vector: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OpenAICompatibleResponseError(f"embedding values[{index}] must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise OpenAICompatibleResponseError(f"embedding values[{index}] must be finite")
        vector.append(numeric)
    if not vector:
        raise OpenAICompatibleResponseError("embedding vectors cannot be empty")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0:
        raise OpenAICompatibleResponseError("embedding vectors cannot have zero norm")
    return tuple(item / norm for item in vector)


def _array(value: JsonValue | None, label: str) -> list[JsonValue]:
    """Return a JSON array or raise a focused conversion error."""
    if not isinstance(value, list):
        raise OpenAICompatibleResponseError(f"{label} must be an array")
    return value


def _object(value: JsonValue | None, label: str) -> JsonObject:
    """Return a JSON object or raise a focused conversion error."""
    if not isinstance(value, dict):
        raise OpenAICompatibleResponseError(f"{label} must be an object")
    return value


def _string(value: JsonValue | None, label: str) -> str:
    """Return a non-empty JSON string or raise a focused conversion error."""
    if not isinstance(value, str) or not value:
        raise OpenAICompatibleResponseError(f"{label} must be a non-empty string")
    return value


def _integer(value: JsonValue | None, label: str, *, default: int) -> int:
    """Read a non-negative JSON integer or the documented omitted default."""
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OpenAICompatibleResponseError(f"{label} must be a non-negative integer")
    return value
