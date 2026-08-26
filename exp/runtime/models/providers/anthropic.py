"""Native Anthropic Messages conversion and focused non-streaming client."""

from __future__ import annotations

from typing import Literal

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
    ToolChoice,
    Usage,
)
from exp.runtime.models.providers.async_transport import AsyncJsonHttpTransport
from exp.runtime.models.providers.base import (
    DEFAULT_MAXIMUM_OUTPUT_TOKENS,
    DEFAULT_RETRY_POLICY,
    DEFAULT_TIMEOUT_SECONDS,
    GatewayWireProfile,
    ProviderHttpClient,
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
from exp.runtime.models.providers.reasoning_compat import anthropic_reasoning_effort
from exp.runtime.models.providers.transport import JsonHttpTransport, RetryPolicy

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


def anthropic_messages_request(
    model_id: str,
    request: ModelRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
) -> JsonObject:
    """Convert one EXP request into native Anthropic Messages JSON.

    Args:
        model_id: Anthropic model identifier.
        request: Typed EXP request.

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
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    if request.top_p is not None and supports_top_p:
        payload["top_p"] = request.top_p
    if request.top_k is not None and supports_top_k:
        payload["top_k"] = request.top_k
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    if supports_reasoning and effective_reasoning_effort is not None:
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {
            "effort": anthropic_reasoning_effort(model_id, effective_reasoning_effort)
        }
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
    if payload.get("stop_reason") == "refusal":
        raise ProviderRefusalError(
            provider="anthropic",
            signal=ProviderRefusalSignal.PROVIDER_REFUSAL,
        )
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, value in enumerate(require_array(payload.get("content"), "Anthropic content")):
        block = require_object(value, f"Anthropic content[{index}]")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ProviderResponseError(f"Anthropic content[{index}].text must be text")
            text_parts.append(text)
        elif block_type == "refusal":
            raise ProviderRefusalError(
                provider="anthropic",
                signal=ProviderRefusalSignal.PROVIDER_REFUSAL,
            )
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
    return ModelResponse.completed(
        output=output,
        configured_model=configured_model,
        served_model_id=payload.get("model"),
        usage=_anthropic_usage(payload),
        latency_seconds=latency_seconds,
        hit_length_limit=payload.get("stop_reason") == "max_tokens",
    )


class AnthropicClient(ProviderHttpClient):
    """Calls one Anthropic Messages model, which intentionally has no embedding method."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str = ANTHROPIC_BASE_URL,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        supports_temperature: bool = True,
        supports_top_p: bool = True,
        supports_top_k: bool = False,
        supports_logprobs: bool = False,
        supports_reasoning: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        """Create an Anthropic client with explicit generation capability gates."""
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            transport=transport,
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
        )
        self._supports_temperature = supports_temperature
        self._supports_top_p = supports_top_p
        self._supports_top_k = supports_top_k
        self._supports_logprobs = supports_logprobs
        self._supports_reasoning = supports_reasoning
        self._reasoning_effort = reasoning_effort

    def _headers(self) -> dict[str, str]:
        """Build native Anthropic Messages headers with the versioned API key scheme."""
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the native Messages wire profile for this connection."""
        return GatewayWireProfile(
            dialect="anthropic_messages",
            url=f"{self._base_url}/{self._request_path(self._completion_path())}",
            headers=self._headers(),
            model_id=self._model.model_id,
            timeout_seconds=self._timeout_seconds,
            supports_temperature=self._supports_temperature,
            maximum_temperature=1.0,
            supports_top_p=self._supports_top_p,
            supports_top_k=self._supports_top_k,
            supports_logprobs=self._supports_logprobs,
            supports_reasoning=self._supports_reasoning,
            reasoning_wire_format="anthropic_adaptive",
            reasoning_effort=self._reasoning_effort,
        )

    def _completion_path(self) -> str:
        """Return the native Messages route."""
        return "messages"

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into a native Messages payload."""
        return anthropic_messages_request(
            self._model.model_id,
            request,
            supports_temperature=self._supports_temperature,
            supports_top_p=self._supports_top_p,
            supports_top_k=self._supports_top_k,
            supports_reasoning=self._supports_reasoning,
            reasoning_effort=self._reasoning_effort,
        )

    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Convert one completed Messages payload into the shared response contract."""
        return anthropic_messages_response(
            payload, configured_model=self._model, latency_seconds=latency_seconds
        )


def _anthropic_blocks(message: ModelMessage) -> tuple[str, list[JsonObject]]:
    """Map one EXP message to native role and content blocks."""
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
    """Map the closed EXP tool-choice shape to native Anthropic semantics."""
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
    call_id = require_string(block.get("id"), f"Anthropic content[{index}].id")
    name = require_string(block.get("name"), f"Anthropic content[{index}].name")
    arguments = block.get("input")
    if not isinstance(arguments, dict):
        raise ProviderResponseError(f"Anthropic content[{index}].input must be an object")
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _anthropic_usage(payload: JsonObject) -> Usage | None:
    """Normalize Anthropic's separate cache counters into shared Usage semantics."""
    raw = payload.get("usage")
    if raw is None:
        return None
    usage = require_object(raw, "Anthropic usage")
    input_tokens = require_integer(usage.get("input_tokens"), "Anthropic usage.input_tokens")
    cache_read = require_integer(
        usage.get("cache_read_input_tokens"), "Anthropic usage.cache_read_input_tokens"
    )
    cache_write = require_integer(
        usage.get("cache_creation_input_tokens"), "Anthropic usage.cache_creation_input_tokens"
    )
    return Usage(
        input_tokens=input_tokens + cache_read + cache_write,
        output_tokens=require_integer(usage.get("output_tokens"), "Anthropic usage.output_tokens"),
        cached_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )
