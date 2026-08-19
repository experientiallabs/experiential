"""Native non-streaming OpenAI Responses conversion and model client."""

from __future__ import annotations

import json

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from pydantic import ValidationError

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
    Usage,
)
from wmo.runtime.models.providers.base import DEFAULT_RETRY_POLICY, DEFAULT_TIMEOUT_SECONDS
from wmo.runtime.models.providers.errors import (
    ProviderResponseError,
    ProviderRetryableResponseError,
)
from wmo.runtime.models.providers.openai_compatible import OpenAIEmbeddingMixin
from wmo.runtime.models.providers.transport import JsonHttpTransport, RetryPolicy

OPENAI_BASE_URL = "https://api.openai.com/v1"


def openai_responses_request(
    model_id: str,
    request: ModelRequest,
    *,
    supports_temperature: bool = True,
    reasoning_effort: str | None = None,
) -> JsonObject:
    """Convert one WMO request into OpenAI's native Responses API shape.

    Args:
        model_id: OpenAI model identifier.
        request: Typed WMO request.
        supports_temperature: Catalog declaration that the model accepts an explicit sampling
            temperature. Reasoning models pin their sampling and reject the parameter, so a
            ``False`` declaration omits any requested temperature from the wire payload.
        reasoning_effort: Optional catalog-pinned reasoning-effort level sent verbatim.

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
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
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
        ProviderRetryableResponseError: The completed output carries no text or tool call,
            typically because reasoning consumed the entire output budget.
        ProviderResponseError: The response status, output, tools, or usage is malformed.
    """
    try:
        parsed = Response.model_validate(payload)
    except ValidationError as exc:
        raise ProviderResponseError("OpenAI Responses payload is malformed") from exc
    status = parsed.status
    if status not in {None, "completed", "incomplete"}:
        raise ProviderResponseError(f"OpenAI response ended with status {status!r}")
    reason = parsed.incomplete_details.reason if parsed.incomplete_details else None
    if status == "incomplete" and reason != "max_output_tokens":
        raise ProviderResponseError(
            f"OpenAI response ended incompletely for unsupported reason {reason!r}"
        )
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, item in enumerate(parsed.output):
        if isinstance(item, ResponseOutputMessage):
            for part in item.content:
                if isinstance(part, ResponseOutputText):
                    text_parts.append(part.text)
                elif isinstance(part, ResponseOutputRefusal):
                    text_parts.append(part.refusal)
                else:
                    raise ProviderResponseError(
                        f"OpenAI Responses output[{index}] has unsupported content type "
                        f"{type(part).__name__!r}"
                    )
        elif isinstance(item, ResponseFunctionToolCall):
            tool_calls.append(_tool_call(item, index))
        elif isinstance(item, ResponseReasoningItem):
            continue
        else:
            raise ProviderResponseError(
                f"OpenAI Responses output[{index}] has unsupported type {item.type!r}"
            )
    content = "".join(text_parts) if text_parts else None
    try:
        action = AssistantAction(content=content, tool_calls=tuple(tool_calls))
    except ValueError as exc:
        raise ProviderRetryableResponseError(
            "OpenAI Responses output has no text or tool call"
        ) from exc
    return ModelResponse.completed(
        output=action,
        configured_model=configured_model,
        served_model_id=parsed.model,
        usage=_usage(parsed.usage),
        latency_seconds=latency_seconds,
        hit_length_limit=status == "incomplete",
    )


class OpenAIClient(OpenAIEmbeddingMixin):
    """Calls direct OpenAI through its native Responses and embeddings endpoints."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str = OPENAI_BASE_URL,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        supports_temperature: bool = True,
        reasoning_effort: str | None = None,
    ) -> None:
        """Create a direct OpenAI client with one explicitly resolved credential.

        Args:
            model: Resolved catalog identity for every request.
            api_key: Non-empty provider credential.
            base_url: Provider endpoint root.
            transport: Optional injected JSON transport for deterministic tests.
            retry_policy: Bounded retry behavior for transient transport failures.
            timeout_seconds: Positive per-attempt timeout floor; completion attempts scale
                above it with the requested maximum output tokens.
            supports_temperature: Catalog declaration that the model accepts an explicit
                sampling temperature; ``False`` omits requested temperatures from payloads.
            reasoning_effort: Optional catalog-pinned reasoning-effort level.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            transport=transport,
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
        )
        self._supports_temperature = supports_temperature
        self._reasoning_effort = reasoning_effort

    def _completion_path(self) -> str:
        """Return the native non-streaming Responses route."""
        return "responses"

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into a native Responses payload."""
        return openai_responses_request(
            self._model.model_id,
            request,
            supports_temperature=self._supports_temperature,
            reasoning_effort=self._reasoning_effort,
        )

    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Convert one completed Responses payload into the shared response contract."""
        return openai_responses_response(
            payload, configured_model=self._model, latency_seconds=latency_seconds
        )


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


def _tool_call(item: ResponseFunctionToolCall, index: int) -> ToolCall:
    """Map one typed native function call while validating object arguments."""
    try:
        arguments = json.loads(item.arguments)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(
            f"OpenAI Responses output[{index}].arguments is not JSON"
        ) from exc
    if not isinstance(arguments, dict):
        raise ProviderResponseError(
            f"OpenAI Responses output[{index}].arguments must decode to an object"
        )
    try:
        return ToolCall(call_id=item.call_id, name=item.name, arguments=arguments)
    except ValidationError as exc:
        raise ProviderResponseError(
            f"OpenAI Responses output[{index}] tool call is incomplete"
        ) from exc


def _usage(usage: ResponseUsage | None) -> Usage | None:
    """Map optional typed Responses usage without accepting negative token counts."""
    if usage is None:
        return None
    try:
        return Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.input_tokens_details.cached_tokens,
        )
    except ValidationError as exc:
        raise ProviderResponseError("OpenAI Responses usage values must be non-negative") from exc
