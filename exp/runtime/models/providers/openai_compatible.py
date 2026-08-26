"""Shared non-streaming OpenAI-compatible conversion with the compatible and OpenRouter clients."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import ClassVar, cast

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    ChatMaxTokensField,
    Embedding,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
    Usage,
)
from exp.runtime.models.providers.async_transport import (
    AsyncJsonHttpTransport,
)
from exp.runtime.models.providers.base import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TIMEOUT_SECONDS,
    GatewayWireProfile,
    ProviderHttpClient,
    ReasoningWireFormat,
)
from exp.runtime.models.providers.errors import (
    ProviderRefusalError,
    ProviderRefusalSignal,
    ProviderResponseError,
    require_array,
    require_integer,
    require_object,
    require_string,
)
from exp.runtime.models.providers.reasoning_compat import (
    openai_reasoning_effort,
    require_sampling_reasoning_compatibility,
)
from exp.runtime.models.providers.transport import JsonHttpTransport, RetryPolicy

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REFERER = "https://github.com/experientiallabs/experiential"
OPENROUTER_TITLE = "experiential"


class OpenAICompatibleResponseError(ProviderResponseError):
    """An OpenAI-compatible endpoint returned a response outside the typed contract."""


def openai_compatible_request(
    model_id: str,
    request: ModelRequest,
    *,
    token_limit_key: ChatMaxTokensField = "max_tokens",
    supports_temperature: bool = True,
    supports_top_p: bool | None = None,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
    reasoning_wire_format: ReasoningWireFormat = "reasoning_effort",
    sampling_requires_reasoning_none: bool = False,
) -> JsonObject:
    """Convert a EXP request into one non-streaming Chat Completions payload.

    Args:
        model_id: Provider model identifier to place on the wire.
        request: Typed EXP request.
        token_limit_key: Wire field carrying the output-token ceiling. Azure OpenAI
            reasoning deployments reject ``max_tokens`` and require
            ``max_completion_tokens``.
        supports_temperature: Whether this exact model accepts explicit sampling controls.
        supports_top_p: Whether this exact model accepts nucleus sampling. ``None`` follows
            ``supports_temperature`` for older catalog records.
        supports_top_k: Whether this exact route accepts top-k sampling.
        supports_logprobs: Reserved route capability retained for contract parity. Chat
            logprob controls are currently ignored because the normalized gateway response
            cannot return provider logprob details.
        supports_reasoning: Whether this exact model accepts a reasoning control.
        reasoning_effort: Optional catalog-pinned reasoning effort.
        reasoning_wire_format: Provider field used for normalized reasoning effort.

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
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    require_sampling_reasoning_compatibility(
        reasoning_effort=effective_reasoning_effort,
        sampling_requires_reasoning_none=sampling_requires_reasoning_none,
        temperature_requested=request.temperature is not None,
        top_p_requested=request.top_p is not None,
    )
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    top_p_supported = supports_temperature if supports_top_p is None else supports_top_p
    if request.top_p is not None and top_p_supported:
        payload["top_p"] = request.top_p
    if request.top_k is not None and supports_top_k:
        payload["top_k"] = request.top_k
    # The public compatibility manifest accepts logprob controls, but the
    # normalized response has no probability representation. Ignore them
    # consistently instead of forwarding a request whose result is discarded.
    del supports_logprobs
    if request.maximum_output_tokens is not None:
        payload[token_limit_key] = request.maximum_output_tokens
    if supports_reasoning and effective_reasoning_effort is not None:
        if reasoning_wire_format == "reasoning":
            payload["reasoning"] = {"effort": effective_reasoning_effort}
        elif reasoning_wire_format == "reasoning_effort":
            payload["reasoning_effort"] = openai_reasoning_effort(
                model_id, effective_reasoning_effort
            )
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
    """Convert one complete Chat Completions response into EXP's shared contract.

    Args:
        payload: Decoded provider response.
        configured_model: Resolved identity before the request was sent.
        latency_seconds: Wall-clock duration for the successful request sequence.

    Returns:
        Typed output, actual-or-configured model identity, and observed usage and latency.

    Raises:
        ProviderResponseError: The response has no usable first choice or invalid tools.
    """
    choices = require_array(payload.get("choices"), "choices")
    if not choices:
        raise OpenAICompatibleResponseError("OpenAI-compatible response has no choices")
    choice = require_object(choices[0], "choices[0]")
    message = require_object(choice.get("message"), "choices[0].message")
    if choice.get("finish_reason") in {"content_filter", "safety"} or isinstance(
        message.get("refusal"), str
    ):
        raise ProviderRefusalError(
            provider="openai-compatible",
            signal=ProviderRefusalSignal.CONTENT_POLICY,
        )
    content_value = message.get("content")
    content = content_value if isinstance(content_value, str) else None
    tool_call_values = _array_or_empty(message)
    tool_calls = tuple(
        parse_openai_wire_tool_call(value, index) for index, value in enumerate(tool_call_values)
    )
    try:
        output = AssistantAction(content=content, tool_calls=tool_calls)
    except ValueError as exc:
        raise OpenAICompatibleResponseError(
            "OpenAI-compatible response has neither text nor a complete tool call"
        ) from exc
    return ModelResponse.completed(
        output=output,
        configured_model=configured_model,
        served_model_id=payload.get("model"),
        usage=_usage(payload),
        latency_seconds=latency_seconds,
        hit_length_limit=choice.get("finish_reason") == "length",
    )


def openai_embedding_response(payload: JsonObject, *, expected_count: int) -> tuple[Embedding, ...]:
    """Convert and normalize an OpenAI-compatible embedding response.

    Args:
        payload: Decoded ``/embeddings`` response.
        expected_count: Number of requested input strings.

    Returns:
        One normalized embedding in input order for every input.

    Raises:
        ProviderResponseError: The provider omitted, duplicated, or malformed vectors.
    """
    data = require_array(payload.get("data"), "data")
    if len(data) != expected_count:
        raise OpenAICompatibleResponseError(
            f"embedding response count {len(data)} does not match request count {expected_count}"
        )
    ordered: list[Embedding | None] = [None] * expected_count
    for position, value in enumerate(data):
        item = require_object(value, f"data[{position}]")
        index_value = item.get("index", position)
        if not isinstance(index_value, int) or isinstance(index_value, bool):
            raise OpenAICompatibleResponseError(f"data[{position}].index must be an integer")
        if index_value < 0 or index_value >= expected_count or ordered[index_value] is not None:
            raise OpenAICompatibleResponseError(
                "embedding response indexes must be unique input indexes"
            )
        vector = require_array(item.get("embedding"), f"data[{position}].embedding")
        ordered[index_value] = Embedding(values=normalize_embedding_vector(vector))
    if any(item is None for item in ordered):
        raise OpenAICompatibleResponseError("embedding response omitted an input index")
    return tuple(cast("Embedding", item) for item in ordered)


class OpenAIEmbeddingMixin(ProviderHttpClient):
    """Adds the shared OpenAI-wire embeddings endpoint to one HTTP provider client."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed ordered text through the configured model without making empty requests.

        Args:
            texts: Ordered visible text values to embed.

        Returns:
            Unit-normalized embeddings in the input order, or an empty tuple for no texts.
        """
        if not texts:
            return ()
        response = self._post("embeddings", openai_embedding_request(self._model.model_id, texts))
        return openai_embedding_response(response, expected_count=len(texts))


class OpenAICompatibleClient(OpenAIEmbeddingMixin):
    """Calls one explicit OpenAI-compatible connection without cross-provider failover."""

    token_limit_key: ClassVar[ChatMaxTokensField] = "max_tokens"
    reasoning_wire_format: ClassVar[ReasoningWireFormat] = "reasoning_effort"

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        supports_temperature: bool = True,
        supports_top_p: bool | None = None,
        supports_top_k: bool = False,
        supports_logprobs: bool = False,
        supports_reasoning: bool = False,
        reasoning_effort: str | None = None,
        chat_max_tokens_field: ChatMaxTokensField | None = None,
        sampling_requires_reasoning_none: bool = False,
    ) -> None:
        """Create one compatible client with explicit model wire capabilities."""
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            transport=transport,
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
        )
        self._supports_temperature = supports_temperature
        self._supports_top_p = supports_temperature if supports_top_p is None else supports_top_p
        self._supports_top_k = supports_top_k
        self._supports_logprobs = supports_logprobs
        self._supports_reasoning = supports_reasoning
        self._reasoning_effort = reasoning_effort
        self._token_limit_key: ChatMaxTokensField = chat_max_tokens_field or self.token_limit_key
        self._sampling_requires_reasoning_none = sampling_requires_reasoning_none

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the Chat Completions wire profile for this connection."""
        return GatewayWireProfile(
            dialect="openai_compatible",
            url=f"{self._base_url}/{self._request_path(self._completion_path())}",
            headers=self._headers(),
            model_id=self._model.model_id,
            timeout_seconds=self._timeout_seconds,
            supports_temperature=self._supports_temperature,
            supports_top_p=self._supports_top_p,
            supports_top_k=self._supports_top_k,
            supports_logprobs=self._supports_logprobs,
            supports_reasoning=self._supports_reasoning,
            reasoning_wire_format=self.reasoning_wire_format,
            reasoning_effort=self._reasoning_effort,
            token_limit_key=self._token_limit_key,
            sampling_requires_reasoning_none=self._sampling_requires_reasoning_none,
        )

    def _completion_path(self) -> str:
        """Return the shared Chat Completions route."""
        return "chat/completions"

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into a Chat Completions payload."""
        return openai_compatible_request(
            self._model.model_id,
            request,
            token_limit_key=self._token_limit_key,
            supports_temperature=self._supports_temperature,
            supports_top_p=self._supports_top_p,
            supports_top_k=self._supports_top_k,
            supports_logprobs=self._supports_logprobs,
            supports_reasoning=self._supports_reasoning,
            reasoning_effort=self._reasoning_effort,
            reasoning_wire_format=self.reasoning_wire_format,
            sampling_requires_reasoning_none=self._sampling_requires_reasoning_none,
        )

    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Convert one Chat Completions payload into the shared response contract."""
        return openai_compatible_response(
            payload, configured_model=self._model, latency_seconds=latency_seconds
        )


class OpenRouterClient(OpenAICompatibleClient):
    """Calls one OpenRouter model with attribution headers and no failover chain."""

    default_headers: ClassVar[Mapping[str, str]] = {
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }
    reasoning_wire_format: ClassVar[ReasoningWireFormat] = "reasoning"


def _openai_message(message: ModelMessage) -> JsonObject:
    """Convert one EXP message while retaining assistant tool history."""
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
                "function": {"name": call.name, "arguments": call.arguments_json()},
            }
            for call in action.tool_calls
        ]
    return payload


def parse_openai_wire_tool_call(value: object, index: int) -> ToolCall:
    """Parse one OpenAI-wire tool call without accepting malformed JSON arguments.

    Args:
        value: One decoded ``tool_calls`` array element.
        index: Zero-based array position used in error messages.

    Returns:
        The typed tool call with its arguments decoded as a JSON object.

    Raises:
        ProviderResponseError: The call lacks identity fields or its arguments do not decode
            to a JSON object.
    """
    item = require_object(cast("JsonValue", value), f"tool_calls[{index}]")
    call_id = require_string(item.get("id"), f"tool_calls[{index}].id")
    function = require_object(item.get("function"), f"tool_calls[{index}].function")
    name = require_string(function.get("name"), f"tool_calls[{index}].function.name")
    raw_arguments = require_string(
        function.get("arguments"), f"tool_calls[{index}].function.arguments"
    )
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
    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=raw_arguments,
    )


def _array_or_empty(message: JsonObject) -> list[JsonValue]:
    """Return optional tool calls as an array, rejecting every other wire shape."""
    value = message.get("tool_calls")
    if value is None:
        return []
    return require_array(value, "choices[0].message.tool_calls")


def _usage(payload: JsonObject) -> Usage | None:
    """Read optional OpenAI-compatible token usage without inventing absent measurements."""
    value = payload.get("usage")
    if value is None:
        return None
    usage = require_object(value, "usage")
    prompt_tokens = require_integer(usage.get("prompt_tokens"), "usage.prompt_tokens")
    completion_tokens = require_integer(usage.get("completion_tokens"), "usage.completion_tokens")
    details_value = usage.get("prompt_tokens_details")
    cached_input_tokens = None
    if details_value is not None:
        details = require_object(details_value, "usage.prompt_tokens_details")
        cached_input_tokens = require_integer(
            details.get("cached_tokens"), "usage.prompt_tokens_details.cached_tokens"
        )
    return Usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def normalize_embedding_vector(values: Sequence[JsonValue]) -> tuple[float, ...]:
    """Return one finite, non-zero unit vector from a provider response.

    Args:
        values: Numeric values in one provider-returned embedding vector.

    Returns:
        The same vector normalized to unit length.

    Raises:
        OpenAICompatibleResponseError: A value is nonnumeric, nonfinite, or the vector is zero.
    """
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
