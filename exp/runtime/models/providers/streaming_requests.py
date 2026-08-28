"""Canonical gateway request translation for launch-provider streaming protocols."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.common.models import ChatMaxTokensField
from exp.runtime.gateway.contracts import (
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.models.providers.bedrock_requests import converse_body
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
    ProviderResponseError,
    UnsupportedReasoningEffortError,
)
from exp.runtime.models.providers.gemini_requests import gemini_generate_request
from exp.runtime.models.providers.reasoning_compat import (
    REASONING_EFFORTS,
    anthropic_reasoning_effort,
    openai_reasoning_effort,
    require_sampling_reasoning_compatibility,
    supported_reasoning_efforts,
)
from exp.runtime.openai_protocol.model_adapter import model_request as gateway_model_request

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

_ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT = 4096
GATEWAY_GENERATION_PARAMETER_CONTRACT_VERSION = 2
"""Version of the route admission and provider wire-translation contract."""

_STRICT_STRUCTURED_OUTPUT_DIALECTS = frozenset(
    {"anthropic_messages", "gemini_generate_content", "bedrock_converse_stream"}
)
_NO_PARALLEL_TOOL_CONTROL_DIALECTS = frozenset(
    {"gemini_generate_content", "bedrock_converse_stream"}
)


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
    required_reasoning_effort = (
        profile.reasoning_effort if profile.reasoning_effort_required else None
    )
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
            reasoning_effort=required_reasoning_effort,
            sampling_requires_reasoning_none=profile.sampling_requires_reasoning_none,
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
            supports_reasoning=profile.supports_reasoning,
            reasoning_effort=required_reasoning_effort,
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
            supports_reasoning=profile.supports_reasoning,
            reasoning_effort=required_reasoning_effort,
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
    if profile.dialect == "openai_compatible":
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
            reasoning_wire_format=profile.reasoning_wire_format,
            reasoning_effort=required_reasoning_effort,
            sampling_requires_reasoning_none=profile.sampling_requires_reasoning_none,
        )
    raise ProviderCapabilityError(capability=f"wire_dialect:{profile.dialect}")


def route_generation_parameter_requests(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> tuple[GatewayRequest, GatewayRequest]:
    """Apply one stable generation-control policy across a provider waterfall.

    A caller-visible semantic parameter is forwarded only when every deployment
    in the certified route supports its exact value. Unsupported or out-of-range
    values fail locally before dispatch. Controls that are provable no-ops, such
    as tool selection without any tool definitions, are removed and disclosed
    through ``ignored_parameters`` on the public request.

    Args:
        profiles: Ordered wire profiles for every deployment in the route.
        request: Decoded public request before provider streaming is forced.

    Returns:
        A pair of ``(public_request, provider_request)``. The public copy keeps
        caller values for response reflection and adds no-op-field disclosure;
        the provider copy removes only controls that cannot change semantics.

    Raises:
        ProviderParameterError: A semantic control is unsupported or invalid
            anywhere in the route.
        ValueError: The route has no wire profiles.
    """
    if not profiles:
        raise ValueError("generation parameter shaping requires at least one wire profile")

    ignored = list(request.ignored_parameters)
    provider_updates: dict[str, object] = {}

    def ignore(field: str, public_path: str | None = None) -> None:
        provider_updates[field] = None
        path = public_path or field
        if path not in ignored:
            ignored.append(path)

    if request.maximum_output_tokens is not None:
        route_limits = tuple(
            profile.maximum_output_tokens
            for profile in profiles
            if profile.maximum_output_tokens is not None
        )
        if route_limits and request.maximum_output_tokens > min(route_limits):
            maximum = min(route_limits)
            param = request.maximum_output_tokens_parameter or "max_tokens"
            raise ProviderParameterError(
                message=(
                    f"The value {request.maximum_output_tokens!r} for {param!r} exceeds this "
                    f"model route's maximum of {maximum}."
                ),
                param=param,
                code="invalid_parameter",
            )
    elif any(profile.dialect == "anthropic_messages" for profile in profiles):
        # Anthropic requires max_tokens even when the public surface does not.
        # Pin one route-wide default so every waterfall rung sees the same
        # output budget, bounded by the smallest known model ceiling.
        route_limits = tuple(
            profile.maximum_output_tokens
            for profile in profiles
            if profile.maximum_output_tokens is not None
        )
        provider_updates["maximum_output_tokens"] = min(
            (_ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT, *route_limits)
        )
    effort_path = "reasoning.effort" if request.surface.value == "responses" else "reasoning_effort"
    effective_reasoning_effort = _effective_route_reasoning_effort(
        profiles,
        request.reasoning_effort,
        param=effort_path,
    )
    if request.reasoning_effort is None and effective_reasoning_effort is not None:
        provider_updates["reasoning_effort"] = effective_reasoning_effort

    def sampling_supported(profile: GatewayWireProfile, *, top_p: bool = False) -> bool:
        """Return whether one rung accepts this request's sampling mode."""
        declared = profile.supports_top_p is True if top_p else profile.supports_temperature
        return declared and (
            not profile.sampling_requires_reasoning_none or effective_reasoning_effort == "none"
        )

    if request.temperature is not None:
        _require_route_numeric_parameter(
            profiles,
            param="temperature",
            value=request.temperature,
            supported=sampling_supported,
            minimum=lambda profile: profile.minimum_temperature,
            maximum=lambda profile: profile.maximum_temperature,
        )
    if request.top_p is not None:
        _require_route_numeric_parameter(
            profiles,
            param="top_p",
            value=request.top_p,
            supported=lambda profile: sampling_supported(profile, top_p=True),
            minimum=lambda profile: profile.minimum_top_p,
            maximum=lambda profile: profile.maximum_top_p,
        )
    if request.top_k is not None:
        _require_route_numeric_parameter(
            profiles,
            param="top_k",
            value=request.top_k,
            supported=lambda profile: (
                profile.supports_top_k and profile.dialect != "openai_responses"
            ),
            minimum=lambda profile: profile.minimum_top_k,
            maximum=lambda profile: profile.maximum_top_k,
        )
    if effective_reasoning_effort is not None:
        portable_efforts = set(REASONING_EFFORTS)
        for profile in profiles:
            if not profile.supports_reasoning or profile.reasoning_wire_format == "none":
                portable_efforts.clear()
                break
            portable_efforts.intersection_update(
                supported_reasoning_efforts(
                    profile.model_id,
                    profile.reasoning_wire_format,
                    configured_effort=profile.reasoning_effort,
                    explicit_efforts=profile.supported_reasoning_efforts or None,
                )
            )
        if effective_reasoning_effort not in portable_efforts:
            raise UnsupportedReasoningEffortError(
                effort=effective_reasoning_effort,
                supported_efforts=tuple(
                    effort for effort in REASONING_EFFORTS if effort in portable_efforts
                ),
                param=effort_path,
            )
    if request.stop and any(profile.dialect == "openai_responses" for profile in profiles):
        raise ProviderParameterError(
            message=(
                "The parameter 'stop' is not supported by every deployment in this model "
                "route. Remove the field or choose a Chat-compatible model."
            ),
            param="stop",
            code="unsupported_parameter",
        )
    if request.reasoning_summary is not None and not all(
        profile.dialect == "openai_responses" and profile.supports_reasoning for profile in profiles
    ):
        path = next(
            iter(request.reasoning_summary_parameters),
            "reasoning.summary",
        )
        raise ProviderParameterError(
            message=(
                f"The parameter {path!r} is not supported by this model route. "
                "Remove the field or choose a different model."
            ),
            param=path,
            code="unsupported_parameter",
        )

    if (
        request.structured_text is not None
        and not request.structured_text.strict
        and any(profile.dialect in _STRICT_STRUCTURED_OUTPUT_DIALECTS for profile in profiles)
    ):
        path = (
            "text.format.strict"
            if request.surface.value == "responses"
            else "response_format.json_schema.strict"
        )
        raise ProviderParameterError(
            message=(
                f"The parameter {path!r} cannot be false on this model route. "
                "Every non-OpenAI structured-output deployment enforces the schema. "
                "Set it to true or choose a different model."
            ),
            param=path,
            code="unsupported_parameter",
        )

    if any(message.tool_is_error for message in request.messages) and not all(
        profile.dialect == "anthropic_messages" for profile in profiles
    ):
        raise ProviderParameterError(
            message=(
                "The parameter 'messages.content.is_error' is not supported by this model "
                "route. Remove the field or choose a native Anthropic-only route."
            ),
            param="messages.content.is_error",
            code="unsupported_parameter",
        )

    # Tool-selection controls have no semantics without tool definitions and
    # several provider APIs reject the otherwise harmless combination.
    if not request.tools:
        if request.tool_choice == "required" or isinstance(
            request.tool_choice, GatewayNamedToolChoice
        ):
            raise ProviderParameterError(
                message=(
                    "The parameter 'tool_choice' requires at least one matching tool "
                    "definition. Add the tool or remove the selector."
                ),
                param="tool_choice",
                code="invalid_parameter",
            )
        if request.tool_choice is not None:
            ignore("tool_choice")
        if request.parallel_tool_calls is not None:
            ignore("parallel_tool_calls")
    elif isinstance(request.tool_choice, GatewayNamedToolChoice) and not any(
        tool.name == request.tool_choice.name for tool in request.tools
    ):
        raise ProviderParameterError(
            message=(
                f"The tool named by 'tool_choice' ({request.tool_choice.name!r}) is not "
                "present in this request's tool definitions."
            ),
            param="tool_choice",
            code="invalid_parameter",
        )
    elif request.tool_choice == "none" and request.parallel_tool_calls is not None:
        ignore("parallel_tool_calls")
    elif request.parallel_tool_calls is not None and any(
        profile.dialect in _NO_PARALLEL_TOOL_CONTROL_DIALECTS for profile in profiles
    ):
        raise ProviderParameterError(
            message=(
                "The parameter 'parallel_tool_calls' is not supported by this model route. "
                "Remove the field or choose a provider route with an explicit parallel-tool "
                "control."
            ),
            param="parallel_tool_calls",
            code="unsupported_parameter",
        )

    # A true logprob request changes the requested result. Until the normalized
    # response can return those arrays, reject it rather than pretending it ran.
    if request.logprobs is True:
        path = (
            "top_logprobs"
            if request.surface.value == "responses" and request.top_logprobs is not None
            else "logprobs"
        )
        raise ProviderParameterError(
            message=(
                f"The parameter {path!r} is not supported by this gateway response contract. "
                "Remove the field and resend the request."
            ),
            param=path,
            code="unsupported_parameter",
        )
    if request.logprobs is False:
        ignore("logprobs")
    if request.top_logprobs is not None:
        raise ProviderParameterError(
            message=(
                "The parameter 'top_logprobs' is not supported by this gateway response "
                "contract. Remove the field and resend the request."
            ),
            param="top_logprobs",
            code="unsupported_parameter",
        )

    ignored_parameters = tuple(ignored)
    public_request = request.model_copy(update={"ignored_parameters": ignored_parameters})
    provider_request = public_request.model_copy(update=provider_updates)
    return public_request, provider_request


def _effective_route_reasoning_effort(
    profiles: Sequence[GatewayWireProfile],
    requested_effort: str | None,
    *,
    param: str,
) -> str | None:
    """Return an explicit effort or one consistent required provider default."""
    if requested_effort is not None:
        return requested_effort
    configured = {
        profile.reasoning_effort for profile in profiles if profile.reasoning_effort_required
    }
    if not configured:
        return None
    if None in configured or len(configured) != 1:
        raise ProviderParameterError(
            message=(
                "The model route has inconsistent required reasoning efforts. "
                "The gateway operator must align every waterfall deployment before retrying."
            ),
            param=param,
            code="invalid_parameter",
        )
    return next(iter(configured))


def _require_route_numeric_parameter(
    profiles: Sequence[GatewayWireProfile],
    *,
    param: str,
    value: float | int,
    supported: Callable[[GatewayWireProfile], bool],
    minimum: Callable[[GatewayWireProfile], float | int],
    maximum: Callable[[GatewayWireProfile], float | int | None],
) -> None:
    """Require every waterfall rung to accept one exact numeric control."""
    if not all(supported(profile) for profile in profiles):
        raise ProviderParameterError(
            message=(
                f"The parameter {param!r} is not supported by this model route. "
                "Remove the field or choose a different model."
            ),
            param=param,
            code="unsupported_parameter",
        )
    route_minimum = max(minimum(profile) for profile in profiles)
    maxima = tuple(bound for profile in profiles if (bound := maximum(profile)) is not None)
    route_maximum = min(maxima) if maxima else None
    if value >= route_minimum and (route_maximum is None or value <= route_maximum):
        return
    range_text = (
        f"{route_minimum} or greater"
        if route_maximum is None
        else f"between {route_minimum} and {route_maximum}"
    )
    raise ProviderParameterError(
        message=(
            f"The value {value!r} for {param!r} is not supported by this model route. "
            f"Supported values are {range_text}."
        ),
        param=param,
        code="invalid_parameter",
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
    sampling_requires_reasoning_none: bool = False,
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
    # Native OpenAI Responses has no top-k request field. Never trust a
    # mistaken route declaration to send this extension to the API.
    del supports_top_k
    # Responses output normalization has no probability representation. Keep
    # the shared capability argument, but ignore logprob controls before send.
    del supports_logprobs
    reasoning: JsonObject = {}
    if supports_reasoning and effective_reasoning_effort is not None:
        reasoning["effort"] = openai_reasoning_effort(model_id, effective_reasoning_effort)
    if supports_reasoning and request.reasoning_summary is not None:
        reasoning["summary"] = request.reasoning_summary
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


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
        tools: list[JsonObject] = []
        for tool in request.tools:
            translated: JsonObject = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            if tool.strict:
                translated["strict"] = True
            tools.append(translated)
        payload["tools"] = tools
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
    output_config: JsonObject = {}
    if supports_reasoning and effective_reasoning_effort is not None:
        payload["thinking"] = {"type": "adaptive"}
        output_config["effort"] = anthropic_reasoning_effort(model_id, effective_reasoning_effort)
    if request.structured_text is not None:
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


def openai_compatible_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    token_limit_key: ChatMaxTokensField = "max_tokens",
    supports_temperature: bool = True,
    supports_top_p: bool | None = None,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_wire_format: str = "reasoning_effort",
    reasoning_effort: str | None = None,
    sampling_requires_reasoning_none: bool = False,
) -> JsonObject:
    """Translate one canonical request to streaming Chat Completions JSON.

    Args:
        model_id: Exact provider model identifier.
        request: Canonical gateway request.
        token_limit_key: Wire field carrying the output-token ceiling. Azure OpenAI
            reasoning deployments reject ``max_tokens`` and require
            ``max_completion_tokens``.
        supports_temperature: Whether this exact model accepts explicit sampling controls.
        supports_reasoning: Whether this exact model accepts a reasoning control.
        reasoning_wire_format: Provider field used for normalized reasoning effort.
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
    # Compatible streaming responses also normalize logprobs to null, so an
    # accepted public control is intentionally ignored until projection exists.
    del supports_logprobs
    if request.stop:
        payload["stop"] = list(request.stop)
    if supports_reasoning and effective_reasoning_effort is not None:
        if reasoning_wire_format == "reasoning":
            payload["reasoning"] = {"effort": effective_reasoning_effort}
        elif reasoning_wire_format == "reasoning_effort":
            payload["reasoning_effort"] = openai_reasoning_effort(
                model_id, effective_reasoning_effort
            )
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
