"""Native Anthropic Messages conversion and focused non-streaming client."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ToolCall,
    ToolChoice,
    Usage,
)
from wmo.runtime.models.providers.errors import ProviderResponseError
from wmo.runtime.models.providers.openai_compatible import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TIMEOUT_SECONDS,
)
from wmo.runtime.models.providers.request import post_json
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.sampling import include_temperature
from wmo.runtime.models.providers.transport import HttpxJsonTransport, JsonHttpTransport

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAXIMUM_OUTPUT_TOKENS = 4096


def anthropic_messages_request(
    model_id: str,
    request: ModelRequest,
    capabilities: ModelCapabilities | None = None,
) -> JsonObject:
    """Convert one WMO request into native Anthropic Messages JSON.

    Args:
        model_id: Anthropic model identifier.
        request: Typed WMO request.
        capabilities: Catalog sampling capabilities for this model, when known.

    Returns:
        Native Messages payload preserving tool-use and tool-result blocks.

    Raises:
        ValueError: A visible request message cannot be represented by the native protocol.
    """
    system_parts: list[str] = []
    messages: list[JsonObject] = []
    for message in request.messages:
        if message.role == "system":
            if message.content is None:
                raise ValueError("system messages need text content")
            system_parts.append(message.content)
            continue
        role, blocks = _anthropic_blocks(message)
        _append_anthropic_message(messages, role, blocks)
    payload: JsonObject = {
        "model": model_id,
        "messages": messages,
        "max_tokens": request.maximum_output_tokens or DEFAULT_MAXIMUM_OUTPUT_TOKENS,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if request.tools:
        payload["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        payload["tool_choice"] = _anthropic_tool_choice(request.tool_choice)
    if include_temperature(request, capabilities):
        payload["temperature"] = request.temperature
    return payload


def anthropic_messages_response(
    payload: JsonObject,
    *,
    configured_model: ModelSnapshot,
    latency_seconds: float,
) -> ModelResponse:
    """Convert native Anthropic content blocks without OpenAI-wire intermediates.

    Args:
        payload: Decoded completed Anthropic response.
        configured_model: Resolved catalog identity used for the request.
        latency_seconds: Observed duration of the successful request sequence.

    Returns:
        The typed assistant action, served model identity, and observed economics.

    Raises:
        ProviderResponseError: The completed response has malformed or unsupported content.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, value in enumerate(_array(payload.get("content"), "content")):
        block = _object(value, f"content[{index}]")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ProviderResponseError(f"Anthropic content[{index}].text must be text")
            text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(_anthropic_tool_call(block, index))
        else:
            raise ProviderResponseError(
                f"Anthropic content[{index}] has unsupported type {block_type!r}"
            )
    content = "".join(text_parts) if text_parts else None
    try:
        output = AssistantAction(content=content, tool_calls=tuple(tool_calls))
    except ValueError as exc:
        raise ProviderResponseError("Anthropic response has no text or tool call") from exc
    model_id = payload.get("model")
    model = (
        configured_model.model_copy(update={"model_id": model_id})
        if isinstance(model_id, str) and model_id
        else configured_model
    )
    return ModelResponse(
        output=output,
        model=model,
        economics=OperationEconomics(
            usage=_anthropic_usage(payload),
            latency_seconds=NumericMeasurement(value=latency_seconds, provenance="observed"),
        ),
        finish_reason=(
            ModelFinishReason.LENGTH
            if payload.get("stop_reason") == "max_tokens"
            else ModelFinishReason.COMPLETED
        ),
    )


class AnthropicClient:
    """Calls one Anthropic Messages model, which intentionally has no embedding method."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        base_url: str = ANTHROPIC_BASE_URL,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        """Build one explicit Anthropic Messages connection."""
        if not api_key:
            raise ValueError("Anthropic clients require a non-empty API key")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or HttpxJsonTransport()
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds
        self._capabilities = capabilities

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one native Messages request without OpenAI conversion.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.

        Returns:
            The typed non-streaming model response with observed request economics.
        """
        started_at = time.monotonic()
        response = post_json(
            self._transport,
            f"{self._base_url}/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            payload=anthropic_messages_request(self._model.model_id, request, self._capabilities),
            timeout_seconds=self._timeout_seconds,
            retry_policy=self._retry_policy,
            provider="anthropic",
            endpoint_class="messages",
        )
        return anthropic_messages_response(
            response,
            configured_model=self._model,
            latency_seconds=time.monotonic() - started_at,
        )


def _anthropic_blocks(message: ModelMessage) -> tuple[str, list[JsonObject]]:
    """Map one WMO message to native role and content blocks."""
    if message.role == "tool":
        return (
            "user",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content or "",
                }
            ],
        )
    if message.role == "user":
        if message.content is None:
            raise ValueError("user messages need text content")
        return "user", [{"type": "text", "text": message.content}]
    if message.role != "assistant":
        raise ValueError(f"unsupported Anthropic message role {message.role!r}")
    action = message.assistant_action
    text = message.content if message.content is not None else action.content if action else None
    blocks: list[JsonObject] = []
    if text is not None:
        blocks.append({"type": "text", "text": text})
    if action is not None:
        blocks.extend(
            {
                "type": "tool_use",
                "id": call.call_id,
                "name": call.name,
                "input": call.arguments,
            }
            for call in action.tool_calls
        )
    if not blocks:
        raise ValueError("assistant messages need text or tool calls")
    return "assistant", blocks


def _append_anthropic_message(
    messages: list[JsonObject], role: str, blocks: list[JsonObject]
) -> None:
    """Append or merge consecutive native roles, which Anthropic forbids on the wire."""
    if messages and messages[-1].get("role") == role:
        existing = messages[-1].get("content")
        if isinstance(existing, list):
            existing.extend(blocks)
            return
        raise ValueError("Anthropic message content must remain an array")
    messages.append({"role": role, "content": blocks})


def _anthropic_tool_choice(
    choice: Literal["auto", "none", "required"] | ToolChoice,
) -> JsonObject:
    """Map the closed WMO tool-choice shape to native Anthropic semantics."""
    if choice == "auto":
        return {"type": "auto"}
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, ToolChoice):
        return {"type": "tool", "name": choice.name}
    raise ValueError("unsupported Anthropic tool choice")


def _anthropic_tool_call(block: JsonObject, index: int) -> ToolCall:
    """Validate one native Anthropic tool-use block."""
    call_id = _string(block.get("id"), f"content[{index}].id")
    name = _string(block.get("name"), f"content[{index}].name")
    arguments = block.get("input")
    if not isinstance(arguments, dict):
        raise ProviderResponseError(f"Anthropic content[{index}].input must be an object")
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _anthropic_usage(payload: JsonObject) -> Usage | None:
    """Normalize Anthropic's separate cache counters into shared Usage semantics."""
    raw = payload.get("usage")
    if raw is None:
        return None
    usage = _object(raw, "usage")
    input_tokens = _integer(usage.get("input_tokens"), "usage.input_tokens")
    cache_read = _integer(usage.get("cache_read_input_tokens"), "usage.cache_read_input_tokens")
    cache_write = _integer(
        usage.get("cache_creation_input_tokens"), "usage.cache_creation_input_tokens"
    )
    return Usage(
        input_tokens=input_tokens + cache_read + cache_write,
        output_tokens=_integer(usage.get("output_tokens"), "usage.output_tokens"),
        cached_input_tokens=cache_read,
    )


def _array(value: JsonValue | None, label: str) -> list[JsonValue]:
    """Return one native array or raise a focused response error."""
    if not isinstance(value, list):
        raise ProviderResponseError(f"Anthropic {label} must be an array")
    return value


def _object(value: JsonValue | None, label: str) -> JsonObject:
    """Return one native object or raise a focused response error."""
    if not isinstance(value, dict):
        raise ProviderResponseError(f"Anthropic {label} must be an object")
    return value


def _string(value: JsonValue | None, label: str) -> str:
    """Return one required non-empty native string."""
    if not isinstance(value, str) or not value:
        raise ProviderResponseError(f"Anthropic {label} must be a non-empty string")
    return value


def _integer(value: JsonValue | None, label: str) -> int:
    """Read an optional non-negative native usage integer."""
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderResponseError(f"Anthropic {label} must be a non-negative integer")
    return value
