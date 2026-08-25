"""Canonical gateway request translation for launch-provider streaming protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.models.providers.bedrock_requests import converse_body
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderResponseError
from exp.runtime.models.providers.gemini_requests import gemini_generate_request
from exp.runtime.openai_protocol.model_adapter import model_request as gateway_model_request

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile


def dialect_stream_payload(
    profile: GatewayWireProfile,
    provider_request: GatewayRequest,
) -> JsonObject:
    """Build the provider wire payload for one resolved wire profile.

    Args:
        profile: The resolved connection's wire profile.
        provider_request: Canonical request forced into streaming mode.

    Returns:
        The exact JSON payload the gateway sends upstream for this dialect.

    Raises:
        ProviderCapabilityError: The request uses a capability this dialect
            cannot preserve.
    """
    if profile.dialect == "openai_responses":
        return openai_responses_stream_payload(
            profile.model_id,
            provider_request,
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_logprobs=profile.supports_logprobs,
            supports_reasoning=profile.supports_reasoning,
            reasoning_effort=profile.reasoning_effort,
        )
    if profile.dialect == "anthropic_messages":
        return anthropic_messages_stream_payload(
            profile.model_id,
            provider_request,
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_logprobs=profile.supports_logprobs,
        )
    if profile.dialect == "gemini_generate_content":
        return gemini_generate_content_stream_payload(
            profile.model_id,
            provider_request,
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_logprobs=profile.supports_logprobs,
        )
    if profile.dialect == "bedrock_converse_stream":
        return bedrock_converse_stream_payload(
            profile.model_id,
            provider_request,
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_logprobs=profile.supports_logprobs,
        )
    return openai_compatible_stream_payload(
        profile.model_id,
        provider_request,
        token_limit_key=profile.token_limit_key,
        supports_temperature=profile.supports_temperature,
        supports_top_p=(
            profile.supports_temperature
            if profile.supports_top_p is None
            else profile.supports_top_p
        ),
        supports_top_k=profile.supports_top_k,
        supports_logprobs=profile.supports_logprobs,
        supports_reasoning=profile.supports_reasoning,
        reasoning_effort=profile.reasoning_effort,
    )


def openai_responses_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool,
    supports_top_p: bool | None = None,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
) -> JsonObject:
    """Translate one canonical request to native streaming Responses JSON.

    Args:
        model_id: Exact OpenAI model identifier.
        request: Canonical gateway request.
        supports_temperature: Whether this exact model accepts explicit temperature.
        supports_reasoning: Whether this exact model accepts the reasoning parameter.
        reasoning_effort: Optional catalog-pinned reasoning effort.

    Returns:
        Native Responses request with storage disabled and streaming enabled.

    Raises:
        ProviderCapabilityError: The request uses unsupported stop sequences.
        ProviderResponseError: An instruction message has no text.
    """
    if request.stop:
        raise ProviderCapabilityError(capability="stop_sequences")
    instructions: list[str] = []
    items: list[JsonObject] = []
    for message in request.messages:
        if message.role in {"system", "developer"}:
            if message.content is None:
                raise ProviderResponseError("instruction messages require text")
            instructions.append(message.content)
        else:
            items.extend(_responses_items(message))
    payload: JsonObject = {
        "model": model_id,
        "input": items,
        "store": False,
        "stream": True,
    }
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    _add_openai_tools(payload, request, responses=True)
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.structured_text is not None:
        format_payload: JsonObject = {
            "type": "json_schema",
            "name": request.structured_text.name,
            "schema": request.structured_text.json_schema,
            "strict": request.structured_text.strict,
        }
        if request.structured_text.description is not None:
            format_payload["description"] = request.structured_text.description
        payload["text"] = {"format": format_payload}
    if request.maximum_output_tokens is not None:
        payload["max_output_tokens"] = request.maximum_output_tokens
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    top_p_supported = supports_temperature if supports_top_p is None else supports_top_p
    if request.top_p is not None and top_p_supported:
        payload["top_p"] = request.top_p
    # Native OpenAI Responses has no top-k request field. Never trust a
    # mistaken route declaration to send this extension to the API.
    del supports_top_k
    if request.top_logprobs is not None and supports_logprobs:
        payload["top_logprobs"] = request.top_logprobs
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    if supports_reasoning and effective_reasoning_effort is not None:
        payload["reasoning"] = {"effort": effective_reasoning_effort}
    return payload


def anthropic_messages_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
) -> JsonObject:
    """Translate one canonical request to native streaming Messages JSON.

    Args:
        model_id: Exact Anthropic model identifier.
        request: Canonical gateway request.

    Returns:
        Native Messages request with streaming enabled.

    Raises:
        ProviderCapabilityError: Structured text is requested on this adapter.
        ProviderResponseError: Instruction or message content is malformed.
    """
    # Anthropic Messages has no compatible logprob request/response surface in
    # this adapter. Keep the shared route signature for capability plumbing,
    # but never put an OpenAI-shaped field on the Anthropic wire.
    del supports_logprobs
    if request.structured_text is not None:
        raise ProviderCapabilityError(capability="structured_text")
    system_parts: list[str] = []
    messages: list[JsonObject] = []
    for message in request.messages:
        if message.role in {"system", "developer"}:
            if message.content is None:
                raise ProviderResponseError("instruction messages require text")
            system_parts.append(message.content)
            continue
        role, blocks = _anthropic_blocks(message)
        if messages and messages[-1].get("role") == role:
            existing = messages[-1].get("content")
            if not isinstance(existing, list):
                raise ProviderResponseError("Anthropic message content is malformed")
            existing.extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})
    payload: JsonObject = {
        "model": model_id,
        "messages": messages,
        "max_tokens": request.maximum_output_tokens or 4096,
        "stream": True,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if request.tools:
        payload["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        if isinstance(request.tool_choice, GatewayNamedToolChoice):
            payload["tool_choice"] = {"type": "tool", "name": request.tool_choice.name}
        else:
            mapping = {"auto": "auto", "none": "none", "required": "any"}
            payload["tool_choice"] = {"type": mapping[request.tool_choice]}
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    if request.top_p is not None and supports_top_p:
        payload["top_p"] = request.top_p
    if request.top_k is not None and supports_top_k:
        payload["top_k"] = request.top_k
    if request.stop:
        payload["stop_sequences"] = list(request.stop)
    return payload


def gemini_generate_content_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
) -> JsonObject:
    """Translate one canonical request to the native streamGenerateContent JSON.

    The payload is built by the exact converter the Gemini provider client
    uses (canonical request through the shared model adapter, then the native
    generateContent builder), so both engines send one identical body. Gemini
    streaming needs no body-level stream flag: streaming is selected by the
    ``streamGenerateContent`` route in the wire profile URL.

    Args:
        model_id: Exact Gemini model identifier; travels in the route path.
        request: Canonical gateway request.

    Returns:
        Native generation request for the SSE streaming route.

    Raises:
        ProviderResponseError: A message cannot preserve its tool linkage on
            Gemini's wire.
    """
    try:
        return gemini_generate_request(
            model_id,
            gateway_model_request(request),
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_logprobs=supports_logprobs,
        )
    except ValueError as exc:
        raise ProviderResponseError(str(exc)) from exc


def bedrock_converse_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
) -> JsonObject:
    """Translate one canonical request to the native ConverseStream REST body.

    The body is built by the exact converter the Bedrock provider client
    uses (canonical request through the shared model adapter, then the shared
    Converse body builder), so both engines send one identical document. On
    the REST route the model travels in the URL path, never the body, and
    streaming is selected by the ``converse-stream`` route itself.

    Args:
        model_id: Exact Bedrock model or inference-profile identifier; it
            travels in the wire profile URL and keeps the dispatch signature.
        request: Canonical gateway request.

    Returns:
        Native Converse request body for the streaming REST route.

    Raises:
        ProviderResponseError: A message cannot be represented without
            dropping tool context.
    """
    del model_id
    try:
        return converse_body(
            gateway_model_request(request),
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_logprobs=supports_logprobs,
        )
    except ValueError as exc:
        raise ProviderResponseError(str(exc)) from exc


def openai_compatible_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    token_limit_key: str = "max_tokens",
    supports_temperature: bool = True,
    supports_top_p: bool | None = None,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
) -> JsonObject:
    """Translate one canonical request to streaming Chat Completions JSON.

    Args:
        model_id: Exact provider model identifier.
        request: Canonical gateway request.
        token_limit_key: Wire field carrying the output-token ceiling. Azure OpenAI
            reasoning deployments reject ``max_tokens`` and require
            ``max_completion_tokens``.
        supports_temperature: Whether this exact model accepts explicit sampling controls.
        supports_reasoning: Whether this exact model accepts ``reasoning_effort``.
        reasoning_effort: Optional catalog-pinned reasoning effort.

    Returns:
        Chat Completions request that always asks the provider for terminal usage.
    """
    payload: JsonObject = {
        "model": model_id,
        "messages": [_openai_message(message) for message in request.messages],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    _add_openai_tools(payload, request, responses=False)
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.structured_text is not None:
        schema: JsonObject = {
            "name": request.structured_text.name,
            "schema": request.structured_text.json_schema,
            "strict": request.structured_text.strict,
        }
        if request.structured_text.description is not None:
            schema["description"] = request.structured_text.description
        payload["response_format"] = {"type": "json_schema", "json_schema": schema}
    if request.maximum_output_tokens is not None:
        payload[token_limit_key] = request.maximum_output_tokens
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    top_p_supported = supports_temperature if supports_top_p is None else supports_top_p
    if request.top_p is not None and top_p_supported:
        payload["top_p"] = request.top_p
    if request.top_k is not None and supports_top_k:
        payload["top_k"] = request.top_k
    if request.logprobs is not None and supports_logprobs:
        payload["logprobs"] = request.logprobs
    if request.top_logprobs is not None and supports_logprobs:
        payload["top_logprobs"] = request.top_logprobs
    if request.stop:
        payload["stop"] = list(request.stop)
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    if supports_reasoning and effective_reasoning_effort is not None:
        payload["reasoning_effort"] = effective_reasoning_effort
    return payload


def _responses_items(message: GatewayMessage) -> list[JsonObject]:
    """Translate one non-instruction gateway message to Responses input items."""
    if message.role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id or "",
                "output": message.content or "",
            }
        ]
    if message.role == "user":
        return [{"role": "user", "content": message.content or ""}]
    if message.role != "assistant":
        raise ProviderResponseError("unsupported Responses message role")
    items: list[JsonObject] = []
    if message.content is not None:
        items.append({"role": "assistant", "content": message.content})
    items.extend(
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments_json(),
        }
        for call in message.tool_calls
    )
    return items


def _anthropic_blocks(message: GatewayMessage) -> tuple[str, list[JsonObject]]:
    """Translate one non-instruction gateway message to Anthropic content blocks."""
    if message.role == "tool":
        result: JsonObject = {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id or "",
            "content": message.content or "",
        }
        # Only the Anthropic wire can express a failed tool invocation; the
        # marker is emitted solely when set so existing payloads are unchanged.
        if message.tool_is_error:
            result["is_error"] = True
        return ("user", [result])
    if message.role == "user":
        return "user", [{"type": "text", "text": message.content or ""}]
    if message.role != "assistant":
        raise ProviderResponseError("unsupported Anthropic message role")
    blocks: list[JsonObject] = []
    if message.content is not None:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {
            "type": "tool_use",
            "id": call.call_id,
            "name": call.name,
            "input": call.arguments,
        }
        for call in message.tool_calls
    )
    return "assistant", blocks


def _openai_message(message: GatewayMessage) -> JsonObject:
    """Translate one gateway message to OpenAI Chat wire JSON."""
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content or "",
            "tool_call_id": message.tool_call_id or "",
        }
    payload: JsonObject = {"role": message.role, "content": message.content or ""}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json()},
            }
            for call in message.tool_calls
        ]
    return payload


def _add_openai_tools(
    payload: JsonObject,
    request: GatewayRequest,
    *,
    responses: bool,
) -> None:
    """Add Responses-native or Chat-native tools and tool choice in place."""
    if request.tools:
        if responses:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
                for tool in request.tools
            ]
        else:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    },
                }
                for tool in request.tools
            ]
    if request.tool_choice is not None:
        if isinstance(request.tool_choice, GatewayNamedToolChoice):
            payload["tool_choice"] = (
                {"type": "function", "name": request.tool_choice.name}
                if responses
                else {
                    "type": "function",
                    "function": {"name": request.tool_choice.name},
                }
            )
        else:
            payload["tool_choice"] = request.tool_choice
