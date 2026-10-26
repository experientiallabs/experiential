"""Provider wire dispatch: one canonical request to one dialect payload.

Split from ``streaming_requests`` for the module line budget: the single
dispatch seam (:func:`dialect_stream_payload`) and the pre-dispatch
tool-result image degrade live here; ``streaming_requests`` re-exports both
so import paths are unchanged, and route admission shaping stays there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import TextContentPart
from exp.common.models.known_models import known_model_metadata
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice, GatewayCustomToolChoice
from exp.runtime.models.providers.anthropic import safeguards_for_upstream
from exp.runtime.models.providers.anthropic_tool_compat import anthropic_rejects_forced_tool_choice
from exp.runtime.models.providers.base import SERVICE_TIER_DIALECTS as SERVICE_TIER_DIALECTS
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.fireworks import (
    require_responses_continuation_channel,
)
from exp.runtime.models.providers.messages_payloads import (
    anthropic_messages_stream_payload,
    bedrock_converse_stream_payload,
    gemini_generate_content_stream_payload,
)
from exp.runtime.models.providers.openai_payloads import (
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
)
from exp.runtime.models.providers.thinking_budget import (
    budgeted_provider_request,
    qwen_budget_payload,
    qwen_uses_total_budget_cap,
    require_thinking_budget_support,
    thinking_budget_value,
)
from exp.runtime.models.providers.tool_references import (
    NATIVE_TOOL_REFERENCE_DIALECT,
    resolve_client_tool_search,
)

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

TOOL_RESULT_IMAGE_DROP_DISCLOSURE = "messages.content.tool_result.image->placeholder"
TOOL_RESULT_IMAGE_FOLD_DISCLOSURE = "messages.content.tool_result.image->following_user_message"
"""Disclosed when a rung carries tool-result images in a user turn that follows
the tool run (Chat Completions and Gemini define no image carrier inside a tool
result; see ``wire_messages.fold_tool_result_attachments``)."""
TOOL_RESULT_IMAGE_FOLD_DIALECTS = frozenset({"openai_compatible", "gemini_generate_content"})
"""Dialects whose payload builders fold tool-result images into a user turn."""
TOOL_RESULT_DOCUMENT_FOLD_DISCLOSURE = (
    "messages.content.tool_result.document->following_user_message"
)
"""Disclosed when a rung carries tool-result documents in a user turn that
follows the tool run (only native Anthropic Messages serves a document inside a
tool result; see ``wire_messages.fold_tool_result_attachments``)."""
TOOL_RESULT_DOCUMENT_FOLD_DIALECTS = frozenset(
    {"openai_compatible", "openai_responses", "gemini_generate_content", "bedrock_converse_stream"}
)
"""Dialects whose payload builders fold tool-result documents into a user turn."""
TOOL_RESULT_DOCUMENT_DROP_DISCLOSURE = "messages.content.tool_result.document->placeholder"
"""Disclosed when a route with no PDF input serves a tool-result document as
placeholder text (see ``capability_policy.coerce_capability``)."""
THINKING_HISTORY_DROP_DISCLOSURE = "messages.thinking->dropped(unsupported_by_provider)"
GEMINI_MISSING_TOOL_RESULT_DISCLOSURE = "messages.tool_results.missing->placeholder"
"""Disclosed when a Gemini rung answers an unanswered function call with a placeholder.

Gemini refuses a turn whose following content does not answer every function
call of the model turn before it; a history whose call never got a result
(an interrupted client) is served with ``GEMINI_MISSING_TOOL_RESULT`` in that
call's slot instead (see ``gemini_requests._CallTurn``)."""


def disclose_tool_result_attachment_folds(
    profiles: Sequence[GatewayWireProfile],
    messages: tuple[GatewayMessage, ...],
    ignored: list[str],
) -> None:
    """Disclose, once per kind, the fold a rung applies to tool-result attachments.

    An attachment a dialect cannot carry inside a tool result rides the user
    turn after the tool run instead (see ``wire_messages.fold_tool_result_attachments``).

    Args:
        profiles: Ordered wire profiles of the route.
        messages: The request's canonical messages.
        ignored: The route's disclosure list, appended in place.
    """
    for dialects, disclosure, carries in (
        (
            TOOL_RESULT_IMAGE_FOLD_DIALECTS,
            TOOL_RESULT_IMAGE_FOLD_DISCLOSURE,
            lambda message: bool(message.images),
        ),
        (
            TOOL_RESULT_DOCUMENT_FOLD_DIALECTS,
            TOOL_RESULT_DOCUMENT_FOLD_DISCLOSURE,
            lambda message: bool(message.documents),
        ),
    ):
        if (
            any(message.role == "tool" and carries(message) for message in messages)
            and any(profile.dialect in dialects for profile in profiles)
            and disclosure not in ignored
        ):
            ignored.append(disclosure)


def disclose_gemini_missing_tool_results(
    profiles: Sequence[GatewayWireProfile],
    messages: tuple[GatewayMessage, ...],
    ignored: list[str],
) -> None:
    """Disclose, once, the placeholder a Gemini rung sends for an unanswered call.

    Args:
        profiles: Ordered wire profiles of the route.
        messages: The request's canonical messages.
        ignored: The route's disclosure list, appended in place.
    """
    if (
        any(profile.dialect == "gemini_generate_content" for profile in profiles)
        and gemini_unanswered_tool_calls(messages)
        and GEMINI_MISSING_TOOL_RESULT_DISCLOSURE not in ignored
    ):
        ignored.append(GEMINI_MISSING_TOOL_RESULT_DISCLOSURE)


def gemini_unanswered_tool_calls(messages: tuple[GatewayMessage, ...]) -> bool:
    """Whether a history leaves a function call unanswered before a later turn.

    Mirrors ``gemini_requests._CallTurn`` over the canonical messages: a call
    no tool message of its own turn answers is filled with the placeholder
    once any message follows the model turn. A leading system or developer
    turn never follows anything, and a trailing model turn awaiting results is
    not a gap.

    Args:
        messages: The request's canonical messages.

    Returns:
        True when the Gemini payload will carry a placeholder function response.
    """
    pending: set[str] = set()
    followed = False
    for message in messages:
        if message.role == "assistant":
            if pending and followed:
                return True
            pending = {call.call_id for call in message.tool_calls}
            followed = False
            continue
        if not pending:
            continue
        followed = True
        if message.role == "tool" and message.tool_call_id is not None:
            pending.discard(message.tool_call_id)
    return bool(pending) and followed


CACHE_CONTROL_NOT_FORWARDED_SUFFIX = (
    "->not_forwarded(provider_decides_caching;"
    " cache reads reported in usage.cache_read_input_tokens)"
)
"""Suffix for cache-marker disclosures on routes with no Anthropic rung.

The marker has no wire field there, so it is not forwarded, and whether the
prefix is cached is the provider's own decision: OpenAI-family and most
OpenAI-compatible servers cache implicitly, without breakpoints, while a
generic endpoint may never cache. Whatever the provider does shows up on the
Anthropic usage leg named here (billed at the cached rate when nonzero, `0`
when the provider caches nothing). The disclosure travels in
``x-experiential-ignored-parameters``, so it has to say where the caller's
caching is reported: a bare "not forwarded" next to a billed
``cache_read_input_tokens`` read as "caching is ignored" (Harbor, 2026-09-11).
The wording never claims caching is off or on."""
"""Disclosed when Anthropic-signed thinking history is omitted for a foreign wire."""
"""Disclosure recorded when tool-result images degrade to placeholder text.

A tool screenshot is baked into the caller's conversation history: rejecting
it wedges every later turn of a multi-turn session, which is strictly worse
than a disclosed degrade. Every wire now carries the image itself (natively
inside the tool result on Anthropic, Responses and Bedrock; folded into a
following user turn on Chat Completions and Gemini), so the degrade remains
only for a rung with no image input at all (``capability_policy``). Top-level
user images keep the fail-closed contract because the caller can re-send those
differently.
"""

TOOL_RESULT_IMAGE_PLACEHOLDER = "[image omitted: this model route cannot carry tool-result images]"
"""Text substituted for each dropped tool-result image, in block position."""


def strip_tool_result_images(
    messages: tuple[GatewayMessage, ...],
) -> tuple[GatewayMessage, ...] | None:
    """Replace tool-message image parts with positional placeholder text.

    Args:
        messages: The request's canonical messages.

    Returns:
        The degraded messages, or ``None`` when no tool message carries an
        image (nothing to strip).
    """
    if not any(message.role == "tool" and message.images for message in messages):
        return None
    out: list[GatewayMessage] = []
    for message in messages:
        if message.role != "tool" or not message.images:
            out.append(message)
            continue
        content = "".join(
            part.text if part.kind == "text" else TOOL_RESULT_IMAGE_PLACEHOLDER
            for part in message.content_parts
        )
        out.append(message.model_copy(update={"content": content, "content_parts": ()}))
    return tuple(out)


def strip_tool_result_documents(
    messages: tuple[GatewayMessage, ...],
) -> tuple[GatewayMessage, ...] | None:
    """Replace tool-message document parts with positional placeholder text.

    Every other part (text, a screenshot beside the document) keeps its place;
    a tool message left with only text collapses back to plain content.

    Args:
        messages: The request's canonical messages.

    Returns:
        The degraded messages, or ``None`` when no tool message carries a
        document (nothing to strip).
    """
    if not any(message.role == "tool" and message.documents for message in messages):
        return None
    out: list[GatewayMessage] = []
    for message in messages:
        if message.role != "tool" or not message.documents:
            out.append(message)
            continue
        parts = tuple(
            TextContentPart(text=_tool_result_document_placeholder(part.name))
            if part.kind == "document"
            else part
            for part in message.content_parts
        )
        content = "".join(part.text for part in parts if part.kind == "text")
        if all(part.kind == "text" for part in parts):
            parts = ()
        out.append(message.model_copy(update={"content": content, "content_parts": parts}))
    return tuple(out)


def _tool_result_document_placeholder(name: str | None) -> str:
    """Return the text left where a tool-result document cannot ride."""
    label = f" {name!r}" if name else ""
    return f"[document{label} omitted: this model route cannot carry documents]"


def fireworks_continuation_required(profile: GatewayWireProfile, request: GatewayRequest) -> bool:
    """Return whether a Fireworks Responses turn can emit an unretained tool call."""
    return (
        request.surface == GatewayApiSurface.RESPONSES
        and (urlsplit(profile.url).hostname or "").lower() == "api.fireworks.ai"
        and bool(request.tools)
        and request.tool_choice != "none"
    )


WEB_SEARCH_SOURCES_DROPPED = (
    "include.web_search_call.action.sources->dropped(unsupported_by_provider)"
)
"""Disclosure when the lead rung relays provider search but refuses the sources selector."""

WEB_SEARCH_SOURCES_DROPPED_ON_FALLBACK = (
    "include.web_search_call.action.sources->dropped(unsupported_by_fallback_provider)"
)
"""Disclosure when only a fallback rung refuses the selector: a failover may omit sources.

Disclosures are fixed at admission for the whole route, so a route whose
fallback could serve without sources says so up front rather than never."""


def responses_origin_refuses_web_search_sources(profile: GatewayWireProfile) -> bool:
    """Return whether a Responses origin refuses the web search sources selector.

    OpenRouter's Responses endpoint (a ``responses`` serving wire) relays
    OpenAI's own ``web_search_call`` items but answers 400 "Invalid Responses
    API request" to ``include: ["web_search_call.action.sources"]`` (probed
    2026-10-09, gpt-6.1-sol). Every other native Responses origin forwards it.
    """
    return profile.dialect == "openai_responses" and (
        (urlsplit(profile.url).hostname or "").lower() == "openrouter.ai"
    )


def reasoning_off_via_chat_template(profile: GatewayWireProfile) -> bool:
    """Return whether this origin turns thinking off only through ``chat_template_kwargs``.

    Vertex MaaS GLM rejects ``reasoning_effort: "none"`` ("Expected
    'reasoning_effort' to be one of: 'high', 'low', 'max', 'medium',
    'minimal'") and ignores ``thinking: {type: disabled}``, but
    ``chat_template_kwargs: {enable_thinking: false}`` answers with no
    reasoning at all (zai-org/glm-5-maas, probed 2026-10-18). Scoped to that
    host and publisher so no other Vertex model changes its wire.
    """
    host = (urlsplit(profile.url).hostname or "").lower()
    return (
        profile.dialect == "openai_compatible"
        and host.endswith("aiplatform.googleapis.com")
        and profile.model_id.lower().startswith("zai-org/glm")
    )


# Z.ai GLM models whose only reasoning control is thinking enabled/disabled.
_ZAI_THINKING_SWITCH_ONLY_MODELS = frozenset({"glm-5", "glm-5.1"})

# Huawei Cloud ModelArts Studio (MaaS) public inference hosts, e.g.
# api-ap-southeast-1.modelarts-maas.com.
_HUAWEI_MAAS_HOST_SUFFIX = ".modelarts-maas.com"


def _huawei_maas(profile: GatewayWireProfile) -> bool:
    """Whether the origin is Huawei Cloud MaaS's OpenAI-compatible API."""
    host = (urlsplit(profile.url).hostname or "").lower()
    return profile.dialect == "openai_compatible" and host.endswith(_HUAWEI_MAAS_HOST_SUFFIX)


def reasoning_via_zai_thinking(profile: GatewayWireProfile) -> bool:
    """Return whether this origin takes Z.ai's own ``thinking`` switch instead of an effort.

    Z.ai's chat completions reference (docs.z.ai, read 2026-10-18) documents
    ``reasoning_effort`` only for GLM-5.2 and above (5.2 skips thinking on none
    or minimal; 5.3 and 5.3-Flash take low, high and max), so those keep the
    effort field. GLM-5 and GLM-5.1 have no effort parameter: their only control
    is ``thinking: {"type": "enabled" | "disabled"}`` ("Thinking Mode"), on by
    default. Huawei Cloud MaaS serves GLM-5.1 with the same contract: its chat
    completions reference (support.huaweicloud.com model-call-019 / -022, read
    2026-10-19) lists ``thinking.type`` enabled | disabled for GLM-5.1, default
    enabled, and names GLM-5.1 in no ``reasoning_effort`` row. Before this the
    Huawei rung declared no effort ladder, so reasoning off was refused there.
    Scoped to the Z.ai and Huawei MaaS hosts and those two models so no other
    origin or model changes its wire.
    """
    host = (urlsplit(profile.url).hostname or "").lower()
    return (
        profile.dialect == "openai_compatible"
        and (host in {"api.z.ai", "open.bigmodel.cn"} or _huawei_maas(profile))
        and profile.model_id.lower() in _ZAI_THINKING_SWITCH_ONLY_MODELS
    )


def reasoning_off_via_thinking_disabled(profile: GatewayWireProfile) -> bool:
    """Return whether effort ``none`` is sent as ``thinking: {"type": "disabled"}`` here.

    Huawei Cloud MaaS DeepSeek-V4 models think by default and document two
    controls (support.huaweicloud.com model-call-019 / -022, read 2026-10-19):
    ``thinking.type`` enabled | disabled turns thinking off, while
    ``reasoning_effort`` takes only high and max ("takes effect only in thinking
    mode"). So none becomes the thinking switch and every other effort stays a
    ``reasoning_effort``. Scoped to the Huawei MaaS host and its DeepSeek-V4
    models so no other origin or model changes its wire.
    """
    return _huawei_maas(profile) and profile.model_id.lower().startswith("deepseek-v4")


_NOVITA_HOST = "api.novita.ai"
# Novita's Anthropic passthrough ids: ``pa/<anthropic model id>-hs``.
_NOVITA_CLAUDE_PREFIX = "pa/claude-"
_NOVITA_CLAUDE_SUFFIX = "-hs"


def novita_claude_thinking_model(profile: GatewayWireProfile) -> str | None:
    """Return the Anthropic model behind a Novita Claude lane that takes Anthropic thinking fields.

    Novita serves Claude over its OpenAI-compatible chat API under
    ``pa/<anthropic id>-hs`` ids. It ignores ``reasoning_effort``; with
    ``reasoning: {effort}`` the model thinks but the stream carries only
    signatures, never text; with Anthropic's own fields, ``thinking: {type:
    adaptive, display: summarized}`` and ``output_config.effort``, it streams
    the summarized thinking as ``reasoning_content`` (pa/claude-opus-4-8-hs,
    probed 2026-10-19). Only the adaptive generation qualifies (Anthropic
    metadata ``supports_reasoning_effort``: opus-4-6, opus-4-7, opus-4-8,
    sonnet-4-6): the 4.5 releases take a token budget instead, unprobed on
    this host. Scoped to the Novita host and those ids so no other origin or
    model changes its wire.

    Args:
        profile: The resolved connection's wire profile.

    Returns:
        The Anthropic model id to validate efforts against, or ``None``.
    """
    host = (urlsplit(profile.url).hostname or "").lower()
    model = profile.model_id.strip().lower()
    if (
        profile.dialect != "openai_compatible"
        or host != _NOVITA_HOST
        or not model.startswith(_NOVITA_CLAUDE_PREFIX)
        or not model.endswith(_NOVITA_CLAUDE_SUFFIX)
    ):
        return None
    anthropic_model = model.removeprefix("pa/").removesuffix(_NOVITA_CLAUDE_SUFFIX)
    known = known_model_metadata("anthropic", anthropic_model)
    if known is None or not known.supports_reasoning_effort:
        return None
    return anthropic_model


def _forces_tool_choice(request: GatewayRequest) -> bool:
    """Whether the request forces a tool call: required, a named tool, or required allowed tools."""
    choice = request.tool_choice
    if isinstance(choice, GatewayAllowedToolsChoice):
        return choice.mode == "required"
    return choice == "required" or isinstance(
        choice, (GatewayNamedToolChoice, GatewayCustomToolChoice)
    )


def _is_anthropic_files_block(block: object) -> bool:
    """Whether one Anthropic content block is a Files-API reference.

    A Files reference is a ``file`` block or an image/document block whose own
    ``source`` is ``{"type": "file", ...}`` (a ``file_id`` handle). This inspects
    ONLY the block's own type and source — NEVER a ``tool_use`` block's ``input``,
    a ``tool_result`` block's ``content``, or a tool schema, which carry ARBITRARY
    caller JSON that may legitimately contain keys like ``file_id`` or
    ``{"type": "file"}`` and must not be mistaken for a Files handle.

    Args:
        block: One entry from a message's ``content`` array.

    Returns:
        True only for a genuine Anthropic Files reference block.
    """
    if not isinstance(block, dict):
        return False
    if block.get("type") == "file":
        return True
    source = block.get("source")
    return isinstance(source, dict) and source.get("type") == "file"


def _refuse_vertex_anthropic_file_handle(messages: object) -> None:
    """Raise if any message's content carries an Anthropic Files reference.

    Vertex AI serves the Anthropic Messages wire but has NO Files API, so a
    request with a ``file_id`` handle (image or document source) cannot be served
    there. Refusing clearly lets route admission prefer a direct-Anthropic /
    Bedrock Claude rung that does support Files, rather than the Vertex endpoint
    400ing upstream. Inline content (base64 ``source``) and tool use are
    unaffected: only the direct content blocks of each message are inspected, not
    the arbitrary caller JSON inside ``tool_use``/``tool_result`` blocks.

    Args:
        messages: The built body's ``messages`` array.

    Raises:
        ProviderCapabilityError: A direct content block is a Files reference.
    """
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if _is_anthropic_files_block(block):
                raise ProviderCapabilityError(capability="anthropic_files")


def _apply_vertex_anthropic_envelope(payload: JsonObject, anthropic_version: str) -> None:
    """Rewrite a native Messages body in place for Claude on Vertex AI.

    Vertex names the model in the ``:rawPredict`` URL and reads the version from
    the body, the inverse of direct Anthropic (model in body, version in header).
    Files handles are refused first (Vertex has no Files API).

    Args:
        payload: The native streaming Messages body built for the anthropic wire.
        anthropic_version: The ``anthropic_version`` value Vertex requires.

    Raises:
        ProviderCapabilityError: The request references an Anthropic Files handle.
    """
    _refuse_vertex_anthropic_file_handle(payload.get("messages"))
    payload.pop("model", None)
    payload["anthropic_version"] = anthropic_version


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
    if profile.dialect != NATIVE_TOOL_REFERENCE_DIALECT:
        # Only Anthropic expands tool_reference blocks and honors
        # defer_loading without a tool-search tool; every other wire gets the
        # references as text and every deferred tool loaded (OpenAI refuses
        # the marker without its hosted tool_search).
        provider_request = resolve_client_tool_search(provider_request)
    if (
        provider_request.tools
        and (
            provider_request.tool_choice == "required"
            or isinstance(provider_request.tool_choice, GatewayNamedToolChoice)
        )
        and anthropic_rejects_forced_tool_choice(profile.model_id)
    ):
        # The release restriction also applies through relays and Bedrock. Raising
        # here lets admission select a capable rung or disclose its auto coercion.
        raise ProviderCapabilityError(capability="forced_tool_choice")
    if (
        provider_request.tools
        and profile.deepseek_reasoning_history
        and _forces_tool_choice(provider_request)
        and (provider_request.reasoning_effort or profile.reasoning_effort) != "none"
    ):
        # DeepSeek's own origin thinks by default and refuses a forced choice
        # while thinking ("Thinking mode does not support this tool_choice",
        # production 2026-10-18). Like Anthropic's budgeted thinking, admission
        # then prefers a rung that can force the tool or discloses ``auto``.
        raise ProviderCapabilityError(capability="forced_tool_choice")
    require_thinking_budget_support(profile, provider_request)
    budget = thinking_budget_value(provider_request)
    provider_request = budgeted_provider_request(profile, provider_request)
    if profile.inference_geo is not None and profile.dialect != "anthropic_messages":
        raise ProviderCapabilityError(capability="inference_geo")
    if fireworks_continuation_required(profile, provider_request):
        require_responses_continuation_channel(provider_request)
    if provider_request.service_tier is not None and profile.dialect not in SERVICE_TIER_DIALECTS:
        # A processing-tier hint changes pricing and latency semantics, so a
        # dialect with no wire field for it declines instead of dropping the
        # field silently: admission then prefers a tier-preserving rung and
        # otherwise retries with the disclosed drop in capability_policy.
        raise ProviderCapabilityError(capability="service_tier")
    if provider_request.prompt_cache_ttl is not None and not (
        profile.dialect == "openai_responses" and profile.forwards_prompt_cache_ttl
    ):
        # Only OpenAI's own Responses wire carries a cache lifetime (its Chat
        # answers 400 for "12h"). Any other rung declares it through inherited
        # capabilities at most, so it declines here instead of silently writing
        # at the default lifetime or sending a field its backend refuses.
        raise ProviderCapabilityError(capability="prompt_cache_ttl_12h")
    required_reasoning_effort = (
        profile.reasoning_effort if profile.reasoning_effort_required and budget is None else None
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
            supports_logprobs=profile.supports_responses_logprobs,
            supports_reasoning=profile.supports_reasoning,
            reasoning_effort=required_reasoning_effort,
            sampling_requires_reasoning_none=profile.sampling_requires_reasoning_none,
            forwards_service_tier=profile.forwards_tier(provider_request.service_tier),
            forwards_prompt_cache_key=profile.forwards_prompt_cache_key,
            forwards_prompt_cache_options=profile.forwards_prompt_cache_options,
            omits_output_token_limit=profile.omits_output_token_limit,
            requests_reasoning_summary=not profile.billing_customer_managed,
            forwards_web_search_sources=not responses_origin_refuses_web_search_sources(profile),
        )
    if profile.dialect == "anthropic_messages":
        payload = anthropic_messages_stream_payload(
            profile.model_id,
            safeguards_for_upstream(profile.url, provider_request),
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
            maximum_output_tokens=profile.maximum_output_tokens,
        )
        if profile.inference_geo is not None:
            payload["inference_geo"] = profile.inference_geo
        if profile.anthropic_vertex_version is not None:
            _apply_vertex_anthropic_envelope(payload, profile.anthropic_vertex_version)
        return payload
    if profile.dialect == "gemini_generate_content":
        payload = gemini_generate_content_stream_payload(
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
        if budget is not None:
            generation = payload["generationConfig"]
            assert isinstance(generation, dict)
            generation["thinkingConfig"] = {"thinkingBudget": budget}
        return payload
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
            supports_reasoning=profile.supports_reasoning,
            reasoning_effort=required_reasoning_effort,
        )
    if profile.dialect == "openai_compatible":
        if profile.fireworks_reasoning_route_sha256 is not None:
            require_responses_continuation_channel(provider_request)
        payload = openai_compatible_stream_payload(
            profile.model_id,
            provider_request,
            token_limit_key=(
                "max_completion_tokens"
                if budget is not None and qwen_uses_total_budget_cap(profile)
                else profile.token_limit_key
            ),
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_frequency_penalty=profile.supports_frequency_penalty,
            supports_presence_penalty=profile.supports_presence_penalty,
            supports_logprobs=profile.supports_logprobs,
            supports_reasoning=profile.supports_reasoning,
            reasoning_wire_format=profile.reasoning_wire_format,
            reasoning_effort=required_reasoning_effort,
            sampling_requires_reasoning_none=profile.sampling_requires_reasoning_none,
            fireworks_reasoning_route_sha256=profile.fireworks_reasoning_route_sha256,
            hunyuan_reasoning_route_sha256=profile.hunyuan_reasoning_route_sha256,
            reasoning_output_exposed=profile.reasoning_output_exposed,
            deepseek_reasoning_history=profile.deepseek_reasoning_history,
            system_messages_leading_only=profile.system_messages_leading_only,
            forwards_service_tier=profile.forwards_tier(provider_request.service_tier),
            forwards_prompt_cache_key=profile.forwards_prompt_cache_key,
            forwards_cache_control=profile.forwards_cache_control,
            forwards_prompt_cache_options=profile.forwards_prompt_cache_options,
            reasoning_off_via_chat_template=reasoning_off_via_chat_template(profile),
            reasoning_via_thinking_toggle=reasoning_via_zai_thinking(profile),
            reasoning_off_via_thinking_disabled=reasoning_off_via_thinking_disabled(profile),
            anthropic_thinking_model=novita_claude_thinking_model(profile),
        )
        qwen_budget_payload(profile, provider_request, payload)
        return payload
    raise ProviderCapabilityError(capability=f"wire_dialect:{profile.dialect}")
