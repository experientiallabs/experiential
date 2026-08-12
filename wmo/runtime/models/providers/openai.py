"""Native non-streaming OpenAI Responses conversion and model client."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    Embedding,
    ModelFinishReason,
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
from wmo.runtime.models.providers.openai_compatible import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TIMEOUT_SECONDS,
    openai_embedding_request,
    openai_embedding_response,
)
from wmo.runtime.models.providers.request import post_json
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.transport import HttpxJsonTransport, JsonHttpTransport

OPENAI_BASE_URL = "https://api.openai.com/v1"


def openai_responses_request(model_id: str, request: ModelRequest) -> JsonObject:
    """Convert one WMO request into OpenAI's native Responses API shape.

    Args:
        model_id: OpenAI model identifier.
        request: Typed WMO request.

    Returns:
        Non-streaming Responses API JSON with provider-side storage disabled.

    Raises:
        ValueError: A message cannot be represented without losing tool linkage.
    """
    instructions: list[str] = []
    input_items: list[JsonObject] = []
    for message in request.messages:
        if message.role == "system":
            if message.content is None:
                raise ValueError("system messages need text content")
            instructions.append(message.content)
            continue
        input_items.extend(_responses_items_for_message(message))
    payload: JsonObject = {
        "model": model_id,
        "input": input_items,
        "store": False,
        "stream": False,
    }
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        payload["tool_choice"] = (
            {"type": "function", "name": request.tool_choice.name}
            if not isinstance(request.tool_choice, str)
            else request.tool_choice
        )
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.maximum_output_tokens is not None:
        payload["max_output_tokens"] = request.maximum_output_tokens
    return payload


def openai_responses_response(
    payload: JsonObject,
    *,
    configured_model: ModelSnapshot,
    latency_seconds: float,
) -> ModelResponse:
    """Convert one completed native Responses response into a shared WMO response.

    Args:
        payload: Decoded completed OpenAI Responses payload.
        configured_model: Resolved catalog identity used for the request.
        latency_seconds: Observed duration of the successful request sequence.

    Returns:
        The typed assistant action, served model identity, and observed economics.

    Raises:
        ProviderResponseError: The response status, output, tools, or usage is malformed.
    """
    status = payload.get("status")
    if status not in {None, "completed", "incomplete"}:
        raise ProviderResponseError(f"OpenAI response ended with status {status!r}")
    incomplete_details = payload.get("incomplete_details")
    incomplete_reason = (
        incomplete_details.get("reason") if isinstance(incomplete_details, dict) else None
    )
    if status == "incomplete" and incomplete_reason not in {"max_output_tokens", "max_tokens"}:
        raise ProviderResponseError(
            f"OpenAI response ended incompletely for unsupported reason {incomplete_reason!r}"
        )
    output = _array(payload.get("output"), "output")
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, item_value in enumerate(output):
        item = _object(item_value, f"output[{index}]")
        item_type = item.get("type")
        if item_type == "message":
            text_parts.extend(_response_text_parts(item, index))
        elif item_type == "function_call":
            tool_calls.append(_response_tool_call(item, index))
        elif item_type == "reasoning":
            continue
        else:
            raise ProviderResponseError(
                f"OpenAI Responses output[{index}] has unsupported type {item_type!r}"
            )
    content = "".join(text_parts) if text_parts else None
    try:
        action = AssistantAction(content=content, tool_calls=tuple(tool_calls))
    except ValueError as exc:
        raise ProviderResponseError("OpenAI Responses output has no text or tool call") from exc
    model_id = payload.get("model")
    model = (
        configured_model.model_copy(update={"model_id": model_id})
        if isinstance(model_id, str) and model_id
        else configured_model
    )
    return ModelResponse(
        output=action,
        model=model,
        economics=OperationEconomics(
            usage=_responses_usage(payload),
            latency_seconds=NumericMeasurement(value=latency_seconds, provenance="observed"),
        ),
        finish_reason=(
            ModelFinishReason.LENGTH if status == "incomplete" else ModelFinishReason.COMPLETED
        ),
    )


class OpenAIClient:
    """Calls direct OpenAI through its native Responses and embeddings endpoints."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        base_url: str = OPENAI_BASE_URL,
    ) -> None:
        """Create a direct OpenAI client with one explicitly resolved credential."""
        if not api_key:
            raise ValueError("OpenAI clients require a non-empty API key")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or HttpxJsonTransport()
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one request through the native non-streaming Responses endpoint.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.

        Returns:
            The typed non-streaming model response with observed request economics.
        """
        started_at = time.monotonic()
        response = post_json(
            self._transport,
            f"{self._base_url}/responses",
            headers=self._headers(),
            payload=openai_responses_request(self._model.model_id, request),
            timeout_seconds=self._timeout_seconds,
            retry_policy=self._retry_policy,
        )
        return openai_responses_response(
            response,
            configured_model=self._model,
            latency_seconds=time.monotonic() - started_at,
        )

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed texts through OpenAI's native embeddings endpoint.

        Args:
            texts: Ordered visible text values to embed.

        Returns:
            Unit-normalized embeddings in the input order.
        """
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
        }


def _responses_items_for_message(message: ModelMessage) -> list[JsonObject]:
    """Convert one typed history message into its native Responses input items."""
    if message.role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id or "",
                "output": message.content or "",
            }
        ]
    if message.role == "user":
        if message.content is None:
            raise ValueError("user messages need text content")
        return [{"role": "user", "content": message.content}]
    if message.role != "assistant":
        raise ValueError(f"unsupported Responses message role {message.role!r}")
    action = message.assistant_action
    text = message.content if message.content is not None else action.content if action else None
    items: list[JsonObject] = []
    if text is not None:
        items.append({"role": "assistant", "content": text})
    if action is not None:
        items.extend(
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": json.dumps(call.arguments),
            }
            for call in action.tool_calls
        )
    if not items:
        raise ValueError("assistant messages need text or a tool call")
    return items


def _response_text_parts(item: JsonObject, index: int) -> list[str]:
    """Read text and refusal content blocks from one native response message."""
    parts: list[str] = []
    for content_index, value in enumerate(_array(item.get("content"), f"output[{index}].content")):
        block = _object(value, f"output[{index}].content[{content_index}]")
        block_type = block.get("type")
        text = block.get("text")
        refusal = block.get("refusal")
        if block_type == "output_text" and isinstance(text, str):
            parts.append(text)
        elif block_type == "refusal" and isinstance(refusal, str):
            parts.append(refusal)
        else:
            raise ProviderResponseError(
                f"OpenAI Responses content block has unsupported type {block_type!r}"
            )
    return parts


def _response_tool_call(item: JsonObject, index: int) -> ToolCall:
    """Map one native function call while validating object arguments."""
    call_id = _string(item.get("call_id"), f"output[{index}].call_id")
    name = _string(item.get("name"), f"output[{index}].name")
    raw_arguments = _string(item.get("arguments"), f"output[{index}].arguments")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(
            f"OpenAI Responses output[{index}].arguments is not JSON"
        ) from exc
    if not isinstance(arguments, dict):
        raise ProviderResponseError(
            f"OpenAI Responses output[{index}].arguments must decode to an object"
        )
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _responses_usage(payload: JsonObject) -> Usage | None:
    """Read optional Responses usage fields without substituting a different wire schema."""
    value = payload.get("usage")
    if value is None:
        return None
    usage = _object(value, "usage")
    details = usage.get("input_tokens_details")
    cached = 0
    if details is not None:
        cached = _integer(_object(details, "usage.input_tokens_details").get("cached_tokens"), 0)
    return Usage(
        input_tokens=_integer(usage.get("input_tokens"), 0),
        output_tokens=_integer(usage.get("output_tokens"), 0),
        cached_input_tokens=cached,
    )


def _array(value: JsonValue | None, label: str) -> list[JsonValue]:
    """Return a response JSON array or raise a native conversion error."""
    if not isinstance(value, list):
        raise ProviderResponseError(f"OpenAI Responses {label} must be an array")
    return value


def _object(value: JsonValue | None, label: str) -> JsonObject:
    """Return a response JSON object or raise a native conversion error."""
    if not isinstance(value, dict):
        raise ProviderResponseError(f"OpenAI Responses {label} must be an object")
    return value


def _string(value: JsonValue | None, label: str) -> str:
    """Return a required non-empty response string."""
    if not isinstance(value, str) or not value:
        raise ProviderResponseError(f"OpenAI Responses {label} must be a non-empty string")
    return value


def _integer(value: JsonValue | None, default: int) -> int:
    """Read an optional non-negative usage integer."""
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderResponseError("OpenAI Responses usage values must be non-negative")
    return value
