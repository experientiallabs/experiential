"""Native streaming payload builders for the Messages-family dialects.

Split from ``streaming_requests`` for the module line budget: the Anthropic,
Gemini, and Bedrock builders live here; ``dialect_stream_payload`` in
``streaming_requests`` remains the single dispatch seam.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.models.providers.bedrock_requests import converse_body
from exp.runtime.models.providers.errors import (
    ProviderParameterError,
    ProviderResponseError,
)
from exp.runtime.models.providers.gemini_requests import gemini_generate_request
from exp.runtime.models.providers.reasoning_compat import anthropic_reasoning_effort
from exp.runtime.models.providers.wire_messages import anthropic_blocks
from exp.runtime.openai_protocol.model_adapter import model_request as gateway_model_request


def anthropic_messages_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
) -> JsonObject:
    """Translate one canonical request to native streaming Messages JSON.

    Args:
        model_id: Exact Anthropic model identifier.
        request: Canonical gateway request.

    Returns:
        Native Messages request with streaming enabled.

    Raises:
        ProviderResponseError: Instruction or message content is malformed.
    """
    # Anthropic Messages has no compatible logprob request/response surface in
    # this adapter. Keep the shared route signature for capability plumbing,
    # but never put an OpenAI-shaped field on the Anthropic wire.
    del supports_logprobs
    system_parts: list[str] = []
    messages: list[JsonObject] = []
    for message in request.messages:
        if message.role in {"system", "developer"}:
            if message.content is None:
                raise ProviderResponseError("instruction messages require text")
            # Leading instructions ride the top-level system field; a system
            # turn after conversation began is a first-class mid-conversation
            # message on this wire (the provider enforces its own placement
            # rules), so its position is preserved verbatim.
            if messages:
                messages.append(
                    {"role": "system", "content": [{"type": "text", "text": message.content}]}
                )
            else:
                system_parts.append(message.content)
            continue
        role, blocks = anthropic_blocks(message)
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
        tools: list[JsonObject] = []
        for tool in request.tools:
            translated: JsonObject = {
                "name": tool.name,
                "input_schema": tool.parameters,
            }
            # Anthropic rejects an explicit null description ("Input should
            # be a valid string"), so an absent description stays absent.
            if tool.description is not None:
                translated["description"] = tool.description
            if tool.strict:
                translated["strict"] = True
            # Anthropic-native tool annotations forward verbatim on this
            # wire only; the provider owns their validity rules. An absent
            # value stays absent so provider defaults keep applying.
            if tool.cache_control is not None:
                translated["cache_control"] = tool.cache_control
            if tool.eager_input_streaming is not None:
                translated["eager_input_streaming"] = tool.eager_input_streaming
            if tool.defer_loading is not None:
                translated["defer_loading"] = tool.defer_loading
            if tool.allowed_callers is not None:
                translated["allowed_callers"] = list(tool.allowed_callers)
            if tool.input_examples is not None:
                translated["input_examples"] = list(tool.input_examples)
            tools.append(translated)
        payload["tools"] = tools
    if request.provider_server_tools:
        # Server tools re-emit verbatim after the converted custom tools (an
        # accepted ordering deviation); route admission guarantees this
        # dispatch is an Anthropic rung, which owns their validity rules.
        server_entries = [dict(entry) for entry in request.provider_server_tools]
        existing_tools = payload.get("tools")
        if isinstance(existing_tools, list):
            existing_tools.extend(server_entries)
        else:
            payload["tools"] = server_entries
    tool_choice: JsonObject | None = None
    if request.tool_choice is not None:
        if isinstance(request.tool_choice, GatewayNamedToolChoice):
            tool_choice = {"type": "tool", "name": request.tool_choice.name}
        else:
            mapping = {"auto": "auto", "none": "none", "required": "any"}
            tool_choice = {"type": mapping[request.tool_choice]}
    if request.parallel_tool_calls is not None:
        tool_choice = tool_choice or {"type": "auto"}
        tool_choice["disable_parallel_tool_use"] = not request.parallel_tool_calls
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    if request.top_p is not None and supports_top_p:
        payload["top_p"] = request.top_p
    if request.top_k is not None and supports_top_k:
        payload["top_k"] = request.top_k
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    # The caller's verbatim output_config seeds the object; engine-derived
    # keys only fill gaps, so the caller always wins byte-for-byte. A
    # canonical caller effort also rides request.reasoning_effort (decode
    # maps it), which keeps the engine's effective effort equal to the
    # caller's and structurally removes the two-sources fight.
    output_config: JsonObject = (
        dict(request.provider_output_config) if request.provider_output_config is not None else {}
    )
    if request.context_management is not None:
        # Anthropic-native context editing forwards byte-for-byte; the
        # required beta header joins the dispatch via
        # anthropic_request_headers.
        payload["context_management"] = request.context_management
    if request.diagnostics is not None:
        # Same treatment: verbatim object, beta header via
        # anthropic_request_headers.
        payload["diagnostics"] = request.diagnostics
    if request.speed is not None:
        payload["speed"] = request.speed
    if request.provider_cache_control is not None:
        # The top-level automatic caching marker forwards byte-for-byte; the
        # provider accepts it bare (verified live 2026-08-30).
        payload["cache_control"] = request.provider_cache_control
    if request.inference_geo is not None:
        payload["inference_geo"] = request.inference_geo
    if request.provider_thinking_config is not None:
        # The caller's exact thinking configuration wins over the catalog's
        # adaptive default and travels verbatim, so budget semantics are
        # never reinterpreted by the gateway. An adaptive config (caller-sent
        # or route-translated) still composes with the route's pinned effort,
        # exactly like a request that carried no thinking config.
        payload["thinking"] = request.provider_thinking_config
        if (
            request.provider_thinking_config.get("type") == "adaptive"
            and supports_reasoning
            and effective_reasoning_effort is not None
            and "effort" not in output_config
        ):
            output_config["effort"] = anthropic_reasoning_effort(
                model_id, effective_reasoning_effort
            )
    elif supports_reasoning and effective_reasoning_effort is not None:
        payload["thinking"] = {"type": "adaptive"}
        if "effort" not in output_config:
            output_config["effort"] = anthropic_reasoning_effort(
                model_id, effective_reasoning_effort
            )
    if request.structured_text is not None and "format" not in output_config:
        output_config["format"] = {
            "type": "json_schema",
            "schema": request.structured_text.json_schema,
        }
    if output_config:
        payload["output_config"] = output_config
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
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
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
            supports_reasoning=supports_reasoning,
            reasoning_effort=reasoning_effort,
            stop_sequences=request.stop,
            response_json_schema=(
                request.structured_text.json_schema if request.structured_text is not None else None
            ),
        )
    except ProviderParameterError:
        raise
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
            stop_sequences=request.stop,
            structured_output_name=(
                request.structured_text.name if request.structured_text is not None else None
            ),
            structured_output_description=(
                request.structured_text.description if request.structured_text is not None else None
            ),
            structured_output_schema=(
                request.structured_text.json_schema if request.structured_text is not None else None
            ),
            strict_tool_names=tuple(tool.name for tool in request.tools if tool.strict),
        )
    except ValueError as exc:
        raise ProviderResponseError(str(exc)) from exc
