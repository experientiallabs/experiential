"""Native streaming payload builders for the OpenAI-family dialects.

Split from ``streaming_requests`` for the module line budget, mirroring the
Messages-family split in ``messages_payloads``: the native Responses and
OpenAI-compatible Chat builders live here; ``dialect_stream_payload`` in
``dialect_dispatch`` remains the single dispatch seam.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ChatMaxTokensField
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.gateway.json_object import JSON_OBJECT_SYSTEM_INSTRUCTION
from exp.runtime.models.providers.deepseek import is_deepseek_model_id
from exp.runtime.models.providers.errors import (
    ProviderResponseError,
)
from exp.runtime.models.providers.fireworks import prepare_gateway_reasoning_history
from exp.runtime.models.providers.instruction_turns import (
    fold_instruction_turns_after_the_first,
    fold_trailing_instruction_turns,
)
from exp.runtime.models.providers.openai_item_ids import openai_issued_item_id
from exp.runtime.models.providers.prompt_cache_breakpoints import (
    carries_breakpoints,
    chat_breakpoints,
    responses_breakpoints,
)
from exp.runtime.models.providers.prompt_cache_options import prompt_cache_options_payload
from exp.runtime.models.providers.reasoning_compat import (
    anthropic_reasoning_effort,
    openai_reasoning_effort,
    reasoning_display_enabled,
    require_sampling_reasoning_compatibility,
)
from exp.runtime.models.providers.wire_messages import (
    add_openai_tools,
    fold_tool_result_attachments,
    openai_chat_message,
    responses_items,
)

_INPUT_MESSAGE_ROLES = frozenset({"user", "system", "developer"})
# Item ids another gateway's Responses emulation mints; no OpenAI wire issues
# them, and OpenAI refuses a replayed item carrying one ("Expected an ID that
# begins with 'rs'" / 'msg'; 164 requests across six orgs in the 48h to
# 2026-09-07). Only this observed shape is treated as foreign.
_FOREIGN_ITEM_ID_PREFIX = "item_"


def _without_probability_metadata(item: JsonObject) -> JsonObject:
    """Drop only output-text probability fields from a replayed message."""
    content_value = item.get("content")
    if item.get("type") != "message" or not isinstance(content_value, list):
        return item
    content: list[JsonValue] = []
    for part in content_value:
        if isinstance(part, dict) and part.get("type") == "output_text":
            content.append({key: value for key, value in part.items() if key != "logprobs"})
        else:
            content.append(part)
    return {**item, "content": content}


def _replayable_native_item(item: JsonObject) -> JsonObject | None:
    """Shape one replayed Responses item for the OpenAI wire; ``None`` drops it.

    Two client habits are repaired, everything else re-emits verbatim: an
    input MESSAGE or REASONING item loses the output-only ``status`` OpenAI
    rejects on it, and an item carrying a foreign ``id`` loses the id (a
    reasoning item with a foreign id is dropped whole: without its encrypted
    content the provider has nothing to resume from, and the id alone is
    refused).
    """
    shaped = _without_probability_metadata(item)
    item_id = shaped.get("id")
    item_type = shaped.get("type")
    if isinstance(item_id, str) and (
        item_id.startswith(_FOREIGN_ITEM_ID_PREFIX)
        or not openai_issued_item_id(
            item_id, item_type if isinstance(item_type, str) else "message"
        )
    ):
        if shaped.get("type") == "reasoning" and "encrypted_content" not in shaped:
            return None
        shaped = {key: value for key, value in shaped.items() if key != "id"}
    if "status" in shaped and (
        (shaped.get("type") in (None, "message") and shaped.get("role") in _INPUT_MESSAGE_ROLES)
        # OpenAI's own id-only reasoning replay serves, but not with the
        # status our response carried: 400 "Unknown parameter:
        # 'input[N].status'" (probed live 2026-10-09, api.openai.com).
        or shaped.get("type") == "reasoning"
    ):
        shaped = {key: value for key, value in shaped.items() if key != "status"}
    return shaped


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
    forwards_service_tier: bool = False,
    forwards_prompt_cache_key: bool = False,
    forwards_prompt_cache_options: bool = False,
    omits_output_token_limit: bool = False,
    requests_reasoning_summary: bool = False,
    forwards_web_search_sources: bool = True,
) -> JsonObject:
    """Translate one canonical request to native streaming Responses JSON.

    Args:
        model_id: Exact OpenAI model identifier.
        request: Canonical gateway request.
        supports_temperature: Whether this exact model accepts explicit temperature.
        supports_reasoning: Whether this exact model accepts the reasoning parameter.
        reasoning_effort: Optional catalog-pinned reasoning effort.
        omits_output_token_limit: Whether the wire rejects ``max_output_tokens`` (the
            ChatGPT plan backend), so the caller's ceiling is dropped structurally.
        requests_reasoning_summary: Whether this rung asks for an ``auto``
            reasoning summary when the caller chose none. Only host-managed
            rungs do: OpenAI rejects summaries for unverified organizations,
            so a customer's own key never receives one it did not ask for.
        forwards_web_search_sources: Whether this origin accepts the
            ``web_search_call.action.sources`` include selector. OpenRouter's
            Responses endpoint relays OpenAI's web search but refuses the
            selector ("Invalid Responses API request"), so it is left off there.
        forwards_prompt_cache_options: Whether this origin carries
            ``prompt_cache_options.mode`` and ``prompt_cache_breakpoint``. A
            marked instruction then rides an input item, since top-level
            ``instructions`` cannot hold a breakpoint.

    Returns:
        Native Responses request with storage disabled and streaming enabled.

    Raises:
        ProviderResponseError: An instruction message has no text.
    """
    # The Responses API has no stop field. Caller stop sequences never reach
    # this wire; the native data plane emulates them from the wire entry's
    # ``stop_sequences`` (see ``deployment_wire_entry``), cutting the stream
    # at the first match and reporting a stop-sequence terminal.
    instructions: list[str] = []
    instruction_roles: list[str] = []
    items: list[JsonObject] = []
    # A tool-returned document rides a following user turn: this wire's
    # function_call_output list carries text and image parts only, while a
    # user turn carries the document natively (see the fold's docstring).
    request = request.model_copy(
        update={"messages": fold_tool_result_attachments(request.messages, kinds=("document",))}
    )
    breakpoints = forwards_prompt_cache_options and request.prompt_cache_breakpoints
    for message in request.messages:
        if message.provider_native_item is not None:
            # Codex-native input items (tool namespaces, freeform tool
            # history, hosted tool echoes) re-emit byte-for-byte at their
            # position; route admission already required every rung to speak
            # this wire. The one exception is an input MESSAGE carrying the
            # output-only ``status`` a client copied from a prior response:
            # the input-message schema has no such field and OpenAI answers
            # 400 "Unknown parameter: 'input[N].status'". Hosted tool items
            # keep theirs (their schema defines it).
            replayable = _replayable_native_item(message.provider_native_item)
            if replayable is not None:
                items.append(replayable)
        elif message.role in {"system", "developer"}:
            if message.content is None:
                raise ProviderResponseError("instruction messages require text")
            # Leading instructions ride the instructions field; one arriving
            # after conversation began, or carrying a cache breakpoint, keeps its
            # position as an input item.
            if items or (breakpoints and carries_breakpoints(message)):
                item: JsonObject = {"role": message.role, "content": message.content}
                items.append(responses_breakpoints(message, item) if breakpoints else item)
            else:
                instructions.append(message.content)
                instruction_roles.append(message.role)
        elif breakpoints and message.role in {"user", "tool"}:
            # One input item each; assistant history carries no input breakpoints.
            items.extend(responses_breakpoints(message, item) for item in responses_items(message))
        else:
            items.extend(responses_items(message))
    if not items and instructions:
        # A request that is ONLY instructions (a system-prompt-only Chat call,
        # a Responses body whose input is a lone system item) has nothing for
        # the ``input`` field, and the provider refuses an empty one ("One of
        # 'input' or 'previous_response_id' ... must be provided") while it
        # serves the same instructions as input items (probed live
        # 2026-09-15, api.openai.com). Emit them as items instead.
        items = [
            {"role": role, "content": content}
            for role, content in zip(instruction_roles, instructions, strict=True)
        ]
        instructions = []
    if request.json_object_output:
        # JSON mode requires an instruction in input itself; the top-level
        # instructions field alone does not satisfy the provider's check.
        items.insert(0, {"role": "system", "content": JSON_OBJECT_SYSTEM_INSTRUCTION})
    # Upstream storage stays disabled regardless of the caller's `store`
    # selector: continuation state is gateway-owned, the gateway never
    # references a provider-stored response, and disabled storage is what
    # makes the provider return encrypted reasoning content.
    payload: JsonObject = {
        "model": model_id,
        "input": items,
        "store": False,
        "stream": True,
    }
    response_store = request.response_store
    include_paths: list[str] = []
    if request.include_encrypted_reasoning or supports_reasoning and response_store is not False:
        include_paths.append("reasoning.encrypted_content")
    if request.include_output_text_logprobs:
        if not supports_logprobs:
            raise ProviderResponseError(
                "This Responses route cannot preserve output text log probabilities."
            )
        include_paths.append("message.output_text.logprobs")
    if request.include_web_search_sources and forwards_web_search_sources:
        # The provider runs web search on this wire and its web_search_call
        # items relay verbatim, so the selector reaches the caller intact.
        include_paths.append("web_search_call.action.sources")
    if include_paths:
        payload["include"] = include_paths
    if request.top_logprobs is not None:
        if not supports_logprobs:
            raise ProviderResponseError(
                "This Responses route cannot preserve output text log probabilities."
            )
        payload["top_logprobs"] = request.top_logprobs
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    add_openai_tools(payload, request, responses=True)
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.client_metadata is not None:
        # Opaque client telemetry, forwarded verbatim (standard Responses
        # surface: accepted live with a plain API key, 2026-08-29).
        payload["client_metadata"] = request.client_metadata
    text_payload: JsonObject = {}
    if request.text_verbosity is not None:
        text_payload["verbosity"] = request.text_verbosity
    if request.structured_text is not None:
        format_payload: JsonObject = {
            "type": "json_schema",
            "name": request.structured_text.name,
            "schema": request.structured_text.json_schema,
            "strict": request.structured_text.strict,
        }
        if request.structured_text.description is not None:
            format_payload["description"] = request.structured_text.description
        text_payload["format"] = format_payload
    elif request.json_object_output:
        text_payload["format"] = {"type": "json_object"}
    if text_payload:
        payload["text"] = text_payload
    if request.maximum_output_tokens is not None and not omits_output_token_limit:
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
    if request.service_tier is not None and forwards_service_tier:
        # BYOK-only: the caller pays this provider directly, so their tier
        # selection (and its pricing) is between them and the provider.
        payload["service_tier"] = request.service_tier
    if request.provider_prompt_cache_key is not None and forwards_prompt_cache_key:
        # Tenant-namespaced cache-affinity key (derived at admission): it
        # changes cache-hit cost, never semantics, so rungs that do not route
        # by it omit it structurally with no decline and no disclosure.
        payload["prompt_cache_key"] = request.provider_prompt_cache_key
    # Admission reaches this wire with a lifetime only on a rung declaring
    # ``supports_prompt_cache_ttl_12h``; the defaults ("implicit", "30m") are never sent.
    cache_options = prompt_cache_options_payload(
        mode=request.prompt_cache_mode,
        ttl=request.prompt_cache_ttl,
        forwards_options=forwards_prompt_cache_options,
    )
    if cache_options is not None:
        payload["prompt_cache_options"] = cache_options
    # Native OpenAI Responses has no top-k request field. Never trust a
    # mistaken route declaration to send this extension to the API.
    del supports_top_k
    reasoning: JsonObject = {}
    if supports_reasoning and effective_reasoning_effort is not None:
        reasoning["effort"] = openai_reasoning_effort(model_id, effective_reasoning_effort)
    if supports_reasoning and request.reasoning_summary is not None:
        reasoning["summary"] = request.reasoning_summary
    elif (
        supports_reasoning
        and requests_reasoning_summary
        and effective_reasoning_effort != "none"
        and reasoning_display_enabled()
    ):
        # OpenAI returns readable reasoning only as summaries, and only when
        # asked; the caller's own selector above always wins. Host-managed
        # rungs only: an unverified customer key would 400 on a summary.
        reasoning["summary"] = "auto"
    if supports_reasoning and request.reasoning_context is not None:
        # Forwarded verbatim: the value controls provider-side re-rendering
        # of prior turns' reasoning and has no gateway semantics.
        reasoning["context"] = request.reasoning_context
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def openai_compatible_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    token_limit_key: ChatMaxTokensField = "max_tokens",
    supports_temperature: bool = True,
    supports_top_p: bool | None = None,
    supports_top_k: bool = False,
    supports_frequency_penalty: bool = False,
    supports_presence_penalty: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_wire_format: str = "reasoning_effort",
    reasoning_effort: str | None = None,
    sampling_requires_reasoning_none: bool = False,
    fireworks_reasoning_route_sha256: str | None = None,
    hunyuan_reasoning_route_sha256: str | None = None,
    reasoning_output_exposed: bool = False,
    deepseek_reasoning_history: bool = False,
    system_messages_leading_only: bool = False,
    forwards_service_tier: bool = False,
    forwards_prompt_cache_key: bool = False,
    forwards_cache_control: bool = False,
    forwards_prompt_cache_options: bool = False,
    reasoning_off_via_chat_template: bool = False,
    reasoning_via_thinking_toggle: bool = False,
    reasoning_off_via_thinking_disabled: bool = False,
    anthropic_thinking_model: str | None = None,
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
        reasoning_off_via_chat_template: Whether this origin turns thinking off only
            through ``chat_template_kwargs`` (Vertex MaaS GLM rejects
            ``reasoning_effort: "none"``), so effort ``none`` is sent as
            ``{"enable_thinking": false}`` and no ``reasoning_effort`` field.
        reasoning_via_thinking_toggle: Whether this origin controls reasoning with
            Z.ai's ``thinking: {"type": "enabled" | "disabled"}`` and no effort
            levels, so effort ``none`` is sent as ``disabled``, any other effort as
            ``enabled``, and no ``reasoning_effort`` field.
        reasoning_off_via_thinking_disabled: Whether this origin turns thinking
            off with ``thinking: {"type": "disabled"}`` while its other efforts
            stay ``reasoning_effort`` (Huawei MaaS DeepSeek-V4), so effort
            ``none`` is sent as that switch and no ``reasoning_effort`` field.
        anthropic_thinking_model: The Anthropic model id when this origin is a
            Claude passthrough that takes Anthropic's own thinking fields
            (Novita ``pa/claude-*-hs``): an effort other than ``none`` is sent as
            ``thinking: {type: adaptive}`` (with ``display: summarized`` unless
            the display kill switch is off) and ``output_config.effort``, and
            no ``reasoning_effort`` field. See ``anthropic_thinking_fields`` for
            the requests that keep thinking off.
        reasoning_output_exposed: Whether this rung replays the caller's plaintext
            ``reasoning_content`` history verbatim (exposure-gated Tencent/DeepSeek rung).
        deepseek_reasoning_history: Whether this rung is DeepSeek's own origin,
            which replays caller plaintext regardless of exposure and requires
            ``reasoning_content`` on every assistant message of the current turn
            (an absent one is backfilled empty on every assistant message); see
            ``openai_chat_message``.
        system_messages_leading_only: Whether this rung's chat template accepts a
            system message only as the very first message (the official Qwen3.6+
            template raises otherwise), so every other instruction turn is folded
            into user text; see ``fold_instruction_turns_after_the_first``.
        forwards_prompt_cache_options: Whether this origin carries
            ``prompt_cache_options.mode`` and ``prompt_cache_breakpoint``; caller
            breakpoints then re-emit in that spelling instead of ``cache_control``.

    Returns:
        Chat Completions request that always asks the provider for terminal usage.
    """
    # A rung is a preserved-thinking route under exactly one provider scheme;
    # its block must name that route to forward. Fireworks additionally toggles
    # the wire-native `reasoning_history` field, which Hunyuan does not use.
    reasoning_route_sha256 = fireworks_reasoning_route_sha256 or hunyuan_reasoning_route_sha256
    messages, active_reasoning = prepare_gateway_reasoning_history(
        request.messages,
        route_sha256=reasoning_route_sha256,
    )
    if deepseek_reasoning_history or is_deepseek_model_id(model_id):
        # DeepSeek ends a tools+reasoning turn empty when the conversation
        # ends on an instruction; see fold_trailing_instruction_turns.
        messages = fold_trailing_instruction_turns(messages)
    # Chat tool messages are text-only on every server behind this wire, so a
    # tool screenshot or tool-returned document rides a following user turn
    # (see the fold's docstring).
    messages = fold_tool_result_attachments(messages, kinds=("image", "document"))
    if system_messages_leading_only:
        # The rung's template 400s on any system turn past the first; the
        # text stays where the caller put it, as user text.
        messages = fold_instruction_turns_after_the_first(messages)
    breakpoints = forwards_prompt_cache_options and request.prompt_cache_breakpoints
    wire_messages = [
        openai_chat_message(
            message,
            reasoning_route_sha256=reasoning_route_sha256,
            reasoning_output_exposed=reasoning_output_exposed,
            deepseek_reasoning_history=deepseek_reasoning_history,
            # The caller's breakpoints re-emit in their own spelling below.
            forwards_cache_control=forwards_cache_control and not breakpoints,
        )
        for message in messages
    ]
    if breakpoints:
        wire_messages = [
            chat_breakpoints(message, wire)
            for message, wire in zip(messages, wire_messages, strict=True)
        ]
    if request.json_object_output:
        _instruct_json_object(wire_messages)
    payload: JsonObject = {
        "model": model_id,
        "messages": wire_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if active_reasoning and fireworks_reasoning_route_sha256 is not None:
        payload["reasoning_history"] = "interleaved"
    add_openai_tools(
        payload, request, responses=False, forwards_cache_control=forwards_cache_control
    )
    if forwards_cache_control and request.provider_cache_control is not None:
        payload["cache_control"] = request.provider_cache_control
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
    elif request.json_object_output:
        payload["response_format"] = {"type": "json_object"}
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
    if request.frequency_penalty is not None and supports_frequency_penalty:
        payload["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None and supports_presence_penalty:
        payload["presence_penalty"] = request.presence_penalty
    if request.logprobs is True:
        if not supports_logprobs:
            raise ProviderResponseError("logprobs was admitted without provider support")
        payload["logprobs"] = True
        if request.top_logprobs is not None:
            payload["top_logprobs"] = request.top_logprobs
    if request.stop:
        payload["stop"] = list(request.stop)
    if request.service_tier is not None and forwards_service_tier:
        # BYOK-only, matching the native Responses lane.
        payload["service_tier"] = request.service_tier
    if request.provider_prompt_cache_key is not None and forwards_prompt_cache_key:
        # Cache-affinity key, matching the native Responses lane.
        payload["prompt_cache_key"] = request.provider_prompt_cache_key
    # Chat's only lifetime is the "30m" default (decode refuses "12h").
    cache_options = prompt_cache_options_payload(
        mode=request.prompt_cache_mode, ttl=None, forwards_options=forwards_prompt_cache_options
    )
    if cache_options is not None:
        payload["prompt_cache_options"] = cache_options
    if supports_reasoning and effective_reasoning_effort is not None:
        if reasoning_via_thinking_toggle:
            payload["thinking"] = {
                "type": "disabled" if effective_reasoning_effort == "none" else "enabled"
            }
        elif effective_reasoning_effort == "none" and reasoning_off_via_thinking_disabled:
            payload["thinking"] = {"type": "disabled"}
        elif effective_reasoning_effort == "none" and reasoning_off_via_chat_template:
            template_kwargs = payload.get("chat_template_kwargs")
            payload["chat_template_kwargs"] = {
                **(template_kwargs if isinstance(template_kwargs, dict) else {}),
                "enable_thinking": False,
            }
        elif anthropic_thinking_model is not None:
            payload.update(
                anthropic_thinking_fields(
                    anthropic_thinking_model, effective_reasoning_effort, request
                )
            )
        elif reasoning_wire_format == "reasoning":
            payload["reasoning"] = {"effort": effective_reasoning_effort}
        elif reasoning_wire_format == "reasoning_effort":
            payload["reasoning_effort"] = openai_reasoning_effort(
                model_id, effective_reasoning_effort
            )
    return payload


def anthropic_thinking_fields(model_id: str, effort: str, request: GatewayRequest) -> JsonObject:
    """Return the Anthropic thinking fields one Claude-passthrough chat request carries.

    ``none`` sends nothing: the adaptive generations served this way run
    without thinking when no config is sent. Any other effort turns adaptive
    thinking on at that effort, validated against the model's own ladder.

    Two requests keep today's shape (no thinking) rather than fail, because
    Anthropic refuses them with thinking on and the provider error is not one
    a failover recovers from:

    * an open tool loop: Anthropic requires the signed thinking block that
      preceded a tool call on the last assistant turn, and a chat history has
      no place to carry it back to this host, so the turns after the first
      tool call of a user turn send no thinking;
    * sampling Anthropic forbids beside thinking: ``temperature`` other than
      1, any ``top_k``, or ``top_p`` below 0.95.

    Args:
        model_id: Exact Anthropic model identifier behind the lane.
        effort: The effective canonical effort.
        request: The canonical request.

    Returns:
        The top-level fields to merge into the chat payload, possibly empty.
    """
    if (
        effort == "none"
        or _in_open_tool_loop(request.messages)
        or _sampling_forbids_thinking(request)
    ):
        return {}
    display = {"display": "summarized"} if reasoning_display_enabled() else {}
    return {
        "thinking": {"type": "adaptive", **display},
        "output_config": {"effort": anthropic_reasoning_effort(model_id, effort)},
    }


def _sampling_forbids_thinking(request: GatewayRequest) -> bool:
    """Whether the request samples in a way Anthropic refuses beside thinking."""
    return (
        (request.temperature is not None and request.temperature != 1)
        or request.top_k is not None
        or (request.top_p is not None and request.top_p < 0.95)
    )


def _in_open_tool_loop(messages: Sequence[GatewayMessage]) -> bool:
    """Whether an assistant tool call follows the conversation's last user message."""
    for message in reversed(messages):
        if message.role == "user":
            return False
        if message.role == "assistant" and message.tool_calls:
            return True
    return False


def _instruct_json_object(messages: list[JsonObject]) -> None:
    """Add the JSON-mode instruction without a second system turn on strict templates."""
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, str):
            messages[0]["content"] = f"{content}\n\n{JSON_OBJECT_SYSTEM_INSTRUCTION}"
            return
        if isinstance(content, list):
            messages[0]["content"] = [
                *content,
                {"type": "text", "text": JSON_OBJECT_SYSTEM_INSTRUCTION},
            ]
            return
    messages.insert(0, {"role": "system", "content": JSON_OBJECT_SYSTEM_INSTRUCTION})
