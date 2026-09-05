"""Capability-preservation policy for one certified gateway route.

Admission prefers a rung that preserves every caller semantic verbatim; that
preference already exists in three layers (operational deadness skipping in
``native_execution.dispatchable_route_profiles``, per-rung generation-control
narrowing in ``generation_route_compat``, and the per-deployment capability
preflight plus payload build in the control plane's admit loop). This module
owns the step AFTER all three fail: the minimal COERCE-WITH-DISCLOSURE that
keeps a request servable when semantics allow; when they do not, the rung's
own field-scoped rejection stays the answer. A coercion is never silent:
every substitution is disclosed through ``ignored_parameters`` in
``path->effective`` form, logged, and counted by the control plane's
admission metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
)
from exp.runtime.models.providers.generation_parameter_validation import (
    profile_reasoning_efforts,
)
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.reasoning_compat import (
    MINIMUM_THINKING_BUDGET_TOKENS,
    anthropic_adaptive_only_thinking,
    anthropic_budgeted_enabled_only,
    anthropic_thinking_budget_tokens,
    efforts_by_nearness,
    thinking_config_reasoning_effort,
)
from exp.runtime.models.providers.streaming_requests import (
    TOOL_RESULT_IMAGE_DROP_DISCLOSURE,
    route_generation_parameter_requests,
    strip_tool_result_images,
)

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

STRICT_TOOLS_DISCLOSURE = "tools.strict->false"
"""Disclosure recorded when strict tools degrade to best-effort schemas."""

FORCED_TOOL_CHOICE_DISCLOSURE = "tool_choice->auto"
"""Disclosure recorded when a forced tool choice relaxes to ``auto`` because no
rung can force a tool (the model rejects it by name, or a budgeted thinking
config forbids it)."""

STRICT_TOOL_SCHEMA_CLOSED_DISCLOSURE = "tools.parameters.additionalProperties->false"
"""Disclosure recorded when strict tool schemas have their objects closed for a
rung whose strict validator requires it."""

EFFORT_DROP_DISCLOSURE = "reasoning_effort"
"""Disclosure recorded when a zero-reasoning route drops the caller effort."""

THINKING_DROP_DISCLOSURE = "thinking"
"""Disclosure recorded when a zero-reasoning route drops adaptive thinking."""

THINKING_DISABLED_DISCLOSURE = "thinking.type->adaptive"
"""Disclosure recorded when an adaptive-only route overrides a disabled thinking config."""

THINKING_TRANSLATED_DISCLOSURE = "thinking.type->enabled"
"""Disclosure recorded when an adaptive thinking config is translated to a
budgeted ``enabled`` config for a budgeted-enabled Anthropic route."""

THINKING_EFFORT_DISCLOSURE_PREFIX = "thinking->reasoning_effort:"
"""Disclosure prefix recorded when a thinking config translates to the effort
a non-Anthropic reasoning route speaks; the effective tier follows the colon."""

THINKING_BUDGET_IGNORED_DISCLOSURE = "thinking.budget_tokens"
"""Disclosure recorded when a caller budget was illegal and a derived budget
replaced it in the translated ``enabled`` config."""

CLOSED_SCHEMA_DISCLOSURE = "json_schema.additionalProperties->false"
"""Disclosure recorded when an open structured-output schema is closed."""

_SCHEMA_DIALECTS_REQUIRING_CLOSED_OBJECTS = frozenset({"anthropic_messages"})
"""Wire dialects whose structured-output validator rejects open objects."""

SERVICE_TIER_DROP_DISCLOSURE = "service_tier"
"""Disclosure recorded when no route rung can carry a processing-tier hint."""


@dataclass(frozen=True)
class RequestCoercion:
    """One disclosed request substitution admission may retry with."""

    request: GatewayRequest
    disclosures: tuple[str, ...]


def _all_budgeted_enabled_anthropic(profiles: Sequence[GatewayWireProfile]) -> bool:
    """Whether every rung is an Anthropic budgeted-enabled-only reasoning route.

    A budgeted-enabled-only rung reasons and honors a ``thinking: {type:
    enabled, budget_tokens}`` config but rejects ``adaptive`` by name (haiku-4-5).
    A mixed, non-reasoning, or adaptive-accepting route (the effort generation)
    is not one, so an adaptive config there is dropped or left verbatim rather
    than translated.
    """
    if not profiles or not all(profile.dialect == "anthropic_messages" for profile in profiles):
        return False
    return all(
        profile.supports_reasoning and anthropic_budgeted_enabled_only(profile.model_id)
        for profile in profiles
    )


def _caller_or_derived_budget(
    config: Mapping[str, object],
    maximum_output_tokens: int | None,
) -> int | None:
    """Return the budget for a translated enabled config, or None to drop.

    A caller-supplied ``budget_tokens`` names the depth and is honored when it
    is legal (``1024 <= budget < max_tokens``); an illegal or absent one falls
    back to the derived budget so the translated config never carries a value
    the provider would reject.
    """
    caller_budget = config.get("budget_tokens")
    if isinstance(caller_budget, int) and not isinstance(caller_budget, bool):
        ceiling = maximum_output_tokens if maximum_output_tokens is not None else caller_budget + 1
        if MINIMUM_THINKING_BUDGET_TOKENS <= caller_budget < ceiling:
            return caller_budget
    return anthropic_thinking_budget_tokens(maximum_output_tokens)


def _coerce_adaptive_budget(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> RequestCoercion | None:
    """Translate an adaptive thinking config for a budgeted-enabled route.

    A budgeted-enabled Anthropic model (haiku-4-5) rejects ``thinking.type:
    adaptive`` by name — that is the effort-ladder generation's channel — but
    accepts a budgeted ``enabled`` config. Claude Code, configured for an
    adaptive model, pins ``adaptive`` on every model, so the serviceable
    reading here is to translate it to ``enabled`` with a budget — the
    caller's own when they carried a legal one, a derived one otherwise
    (the model rejects ``adaptive`` by NAME, so leaving a budget-carrying
    adaptive config verbatim would still fail at the provider). When no
    legal budget fits the output ceiling the translation is impossible, so the
    config and any effort channel drop, all disclosed. History thinking blocks
    are left untouched: Anthropic accepts replayed thinking blocks with no live
    thinking config, so a coercion never strips them.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.

    Returns:
        The disclosed translation or drop, or ``None`` when the request is not
        an adaptive config on a budgeted-enabled Anthropic route.
    """
    config = request.provider_thinking_config
    if config is None or config.get("type") != "adaptive":
        return None
    if not _all_budgeted_enabled_anthropic(profiles):
        return None
    budget = _caller_or_derived_budget(config, request.maximum_output_tokens)
    if budget is not None:
        disclosures: tuple[str, ...] = (THINKING_TRANSLATED_DISCLOSURE,)
        if config.get("budget_tokens") is not None and config.get("budget_tokens") != budget:
            # The caller named an illegal depth; the substitution changes the
            # requested reasoning depth and cost, so it is disclosed by itself
            # rather than hiding behind the type translation.
            disclosures = (*disclosures, THINKING_BUDGET_IGNORED_DISCLOSURE)
        return RequestCoercion(
            request=request.model_copy(
                update={"provider_thinking_config": {"type": "enabled", "budget_tokens": budget}}
            ),
            disclosures=disclosures,
        )
    # No legal budget fits: drop every reasoning signal so the surviving effort
    # cannot re-emit adaptive thinking through output_config.
    dropped, disclosures = _drop_thinking_and_effort(request)
    return RequestCoercion(request=dropped, disclosures=disclosures)


def _coerce_thinking_to_effort(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
    *,
    admits: Callable[[GatewayRequest], bool] | None = None,
) -> RequestCoercion | None:
    """Translate a thinking config for an all-non-Anthropic route.

    Claude Code pins a ``thinking`` config on every model, so the named
    rejection at route shaping would make whole sessions unusable against
    OpenAI-family reasoning models the provider itself serves fine. On a
    route with NO Anthropic rung the config translates to the nearest effort
    the route actually serves (the tier table in
    :func:`thinking_config_reasoning_effort`, then nearest-first over the
    route's ladder), disclosed as ``thinking->reasoning_effort:<tier>``.
    The combined ladder is a union of per-rung ladders, so the naive nearest
    tier may be served only by rungs that reject some other control;
    candidates are therefore tried in nearness order (ties prefer the lower
    tier) and the translation is the closest tier that survives full route
    construction and the caller's admission probe, mirroring the
    explicit-effort snap in :func:`coerce_generation_parameters`. A route
    with no reasoning rung drops every reasoning signal with disclosure
    instead. An explicit caller effort is the same channel already stated in
    the route's own vocabulary, so it wins verbatim and the config drops
    with one disclosure. Routes with an Anthropic rung are left alone:
    narrowing already prefers the rung that honors the config verbatim, and
    stealing that preference here would trade real thinking for a
    translation. Replayed thinking blocks are signed provider state no
    translation can carry, so their presence declines the coercion (and the
    gateway never fabricates unsigned blocks on the response side: our own
    decode routes replayed blocks to Anthropic-only routes, so a fabricated
    block would wedge the caller's next turn).

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.
        admits: Optional caller probe that must accept a candidate before it
            is offered, threaded through so a nearer tier that dies one
            layer later never blocks a farther tier that serves.

    Returns:
        The disclosed translation or drop, or ``None`` when the request
        carries no thinking config, the route has an Anthropic rung,
        replayed thinking blocks make the route unservable regardless, or no
        translatable tier serves the request end to end.
    """
    config = request.provider_thinking_config
    if config is None or not profiles:
        return None
    if any(profile.dialect == "anthropic_messages" for profile in profiles):
        return None
    if any(
        block.kind in {"thinking", "redacted_thinking"}
        for message in request.messages
        for block in message.provider_reasoning
    ):
        return None

    def admitted(coercion: RequestCoercion) -> RequestCoercion | None:
        """Offer one coercion only when the caller's probe accepts it."""
        if admits is not None and not admits(coercion.request):
            return None
        return coercion

    ladder: set[str] = set()
    for profile in profiles:
        ladder.update(profile_reasoning_efforts(profile))
    if not ladder:
        dropped, disclosures = _drop_thinking_and_effort(request)
        return admitted(RequestCoercion(request=dropped, disclosures=disclosures))
    if request.reasoning_effort is not None:
        return admitted(
            RequestCoercion(
                request=request.model_copy(update={"provider_thinking_config": None}),
                disclosures=(THINKING_DROP_DISCLOSURE,),
            )
        )
    requested_tier = thinking_config_reasoning_effort(config)
    candidates = set(ladder)
    if requested_tier == "none":
        # A disabled config asked for NO reasoning; snapping it to an active
        # level would enable reasoning the caller explicitly turned off, so
        # only an exact 'none' translates and anything else takes the
        # disclosed drop (the model's own default behavior, stated openly).
        candidates.intersection_update({"none"})
    else:
        # An active config asked for reasoning; snapping it to 'none' would
        # silently disable reasoning while calling it a translation, so a
        # route whose only level is 'none' takes the disclosed drop instead.
        candidates.discard("none")
    for candidate in efforts_by_nearness(requested_tier, candidates):
        translated_request = request.model_copy(
            update={"provider_thinking_config": None, "reasoning_effort": candidate}
        )
        try:
            indexes = compatible_generation_parameter_profile_indexes(profiles, translated_request)
            # Per-rung admission is not enough here either: only a candidate
            # whose narrowed rung set survives full route construction is a
            # real translation (see the explicit-effort snap below).
            route_generation_parameter_requests(
                tuple(profiles[index] for index in indexes),
                translated_request,
            )
        except (ProviderParameterError, ProviderCapabilityError):
            continue
        if admits is not None and not admits(translated_request):
            continue
        return RequestCoercion(
            request=translated_request,
            disclosures=(f"{THINKING_EFFORT_DISCLOSURE_PREFIX}{candidate}",),
        )
    if candidates:
        # Active tiers exist but none serves this request end to end, so the
        # original rejection stands: dropping the config here would silently
        # disable reasoning the caller asked for.
        return None
    dropped, disclosures = _drop_thinking_and_effort(request)
    return admitted(RequestCoercion(request=dropped, disclosures=disclosures))


def _drop_thinking_and_effort(
    request: GatewayRequest,
) -> tuple[GatewayRequest, tuple[str, ...]]:
    """Null the thinking config and effort channels for a route that cannot reason.

    Shared drop for the routes that cannot honor a reasoning signal: the
    thinking config, the caller effort, and the Messages ``output_config.effort``
    channel all go, and a ``clear_thinking`` context edit rides on the thinking
    config and is stripped with it. Every removal is disclosed. History thinking
    blocks are NOT touched — Anthropic accepts replayed blocks without a live
    thinking config.
    """
    updates: dict[str, object] = {"provider_thinking_config": None}
    disclosures: list[str] = []
    if request.reasoning_effort is not None:
        updates["reasoning_effort"] = None
        disclosures.append(EFFORT_DROP_DISCLOSURE)
    disclosures.append(THINKING_DROP_DISCLOSURE)
    if request.provider_output_config is not None and "effort" in request.provider_output_config:
        remaining = {
            key: value for key, value in request.provider_output_config.items() if key != "effort"
        }
        updates["provider_output_config"] = remaining or None
    if request.context_management is not None:
        updates["context_management"] = _without_clear_thinking_edits(request.context_management)
    return request.model_copy(update=updates), tuple(disclosures)


def coerce_generation_parameters(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
    *,
    admits: Callable[[GatewayRequest], bool] | None = None,
) -> RequestCoercion | None:
    """Build the minimal disclosed coercion after verbatim narrowing failed.

    Only substitutions whose semantics survive are offered: a reasoning
    effort snaps to the nearest level any rung supports on the canonical
    ladder (ties prefer the lower level, so a coercion never spends more
    than requested), and ANY effort on a route with no reasoning support at
    all is dropped with disclosure. A zero-reasoning route cannot honor any
    depth, so the only serviceable semantic is the model's sole behavior,
    and first-party clients pin effort globally (Claude Code sends its
    configured effortLevel to every model), so a named rejection here makes
    whole sessions unusable against non-reasoning models the provider
    itself serves fine without the parameter (owner decision, 2026-09-01;
    previously only an explicit ``none`` dropped).

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.
        admits: Optional caller probe that must accept a candidate before it
            is offered. Admission passes its full downstream pipeline here
            (deployment capability preflight included), because this module
            sees only wire profiles and a candidate that dies one layer
            later would block a farther candidate that serves.

    Returns:
        The disclosed substitution to retry with, or ``None`` when nothing
        coercible applies.
    """
    disabled_thinking = _coerce_disabled_thinking(profiles, request)
    if disabled_thinking is not None:
        if admits is not None and not admits(disabled_thinking.request):
            return None
        return disabled_thinking
    adaptive_budget = _coerce_adaptive_budget(profiles, request)
    if adaptive_budget is not None:
        if admits is not None and not admits(adaptive_budget.request):
            return None
        return adaptive_budget
    thinking_effort = _coerce_thinking_to_effort(profiles, request, admits=admits)
    if thinking_effort is not None:
        return thinking_effort
    ladder: set[str] = set()
    for profile in profiles:
        ladder.update(profile_reasoning_efforts(profile))
    adaptive_thinking = (
        request.provider_thinking_config is not None
        and request.provider_thinking_config.get("type") == "adaptive"
    )
    # An adaptive config with no effort beside it still cannot survive a route
    # with no reasoning rung (the provider rejects it by name), so it reaches
    # the drop path below instead of returning here. A budgeted-enabled route
    # already translated it in ``_coerce_adaptive_budget``; a route that
    # accepts adaptive verbatim keeps a non-empty ladder and returns here.
    if request.reasoning_effort is None and not (adaptive_thinking and not ladder):
        return None
    if not ladder:
        updates: dict[str, object] = {"reasoning_effort": None}
        disclosures: tuple[str, ...] = (
            (EFFORT_DROP_DISCLOSURE,) if request.reasoning_effort is not None else ()
        )
        if request.provider_output_config is not None:
            # The Messages surface carries the same effort verbatim inside
            # output_config; a dropped effort must not reach the provider
            # through that channel (the provider rejects it by name).
            remaining = {
                key: value
                for key, value in request.provider_output_config.items()
                if key != "effort"
            }
            updates["provider_output_config"] = remaining or None
        if adaptive_thinking:
            # Adaptive thinking is the effort's own channel on the Messages
            # surface (the model picks its depth from output_config.effort),
            # so a route with no reasoning rung cannot honor it either and
            # the provider rejects it by name, with or without an effort beside
            # it. A budgeted config is left verbatim: its semantics do not
            # depend on an effort level. A clear_thinking context edit rides on
            # the thinking config and is stripped with it.
            updates["provider_thinking_config"] = None
            disclosures = (*disclosures, THINKING_DROP_DISCLOSURE)
            if request.context_management is not None:
                updates["context_management"] = _without_clear_thinking_edits(
                    request.context_management
                )
        dropped_request = request.model_copy(update=updates)
        if admits is not None and not admits(dropped_request):
            return None
        return RequestCoercion(request=dropped_request, disclosures=disclosures)
    if request.reasoning_effort is None or request.reasoning_effort in ladder:
        # The effort itself is portable; the verbatim failure lies elsewhere
        # and a snap would change semantics for nothing.
        return None
    # A heterogeneous waterfall can carry a nearby effort only on rungs that
    # reject some other control, so candidates are tried in nearness order
    # and the snap is the closest level that actually admits a rung.
    for candidate in efforts_by_nearness(request.reasoning_effort, ladder):
        snapped_request = request.model_copy(update={"reasoning_effort": candidate})
        try:
            indexes = compatible_generation_parameter_profile_indexes(profiles, snapped_request)
            # Per-rung admission is not enough: the narrowed rung set changes
            # with the candidate, and a route-wide gate (for example the
            # homogeneous encrypted-reasoning channel) can reject a mixed set
            # that a farther candidate would narrow past. Only a candidate
            # that survives full route construction is a real snap.
            route_generation_parameter_requests(
                tuple(profiles[index] for index in indexes),
                snapped_request,
            )
        except (ProviderParameterError, ProviderCapabilityError):
            continue
        if admits is not None and not admits(snapped_request):
            continue
        return RequestCoercion(
            request=snapped_request,
            disclosures=(f"reasoning_effort->{candidate}",),
        )
    return None


def _coerce_disabled_thinking(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> RequestCoercion | None:
    """Drop a ``thinking.type: disabled`` config the route's model cannot honor.

    The adaptive-thinking Anthropic generation always reasons and rejects an
    explicit ``disabled`` by name, so on a route whose Anthropic rungs are all
    adaptive-only the caller's only alternative to a rejection is removing the
    field. First-party clients pin the thinking mode globally (Claude Code
    sends its configured mode to every model), so the config is dropped with
    disclosure and the rung emits its sole supported mode, mirroring how a
    budgeted ``enabled`` config is translated to adaptive. Routes with a rung
    that honors ``disabled`` verbatim are left alone: narrowing already picks
    that rung.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.

    Returns:
        The disclosed drop, or ``None`` when the config is not a rejected
        ``disabled`` on an adaptive-only Anthropic route.
    """
    config = request.provider_thinking_config
    if config is None or config.get("type") != "disabled":
        return None
    anthropic_profiles = [
        profile for profile in profiles if profile.dialect == "anthropic_messages"
    ]
    if not anthropic_profiles or not all(
        anthropic_adaptive_only_thinking(profile.model_id) for profile in anthropic_profiles
    ):
        return None
    return RequestCoercion(
        request=request.model_copy(update={"provider_thinking_config": None}),
        disclosures=(THINKING_DISABLED_DISCLOSURE,),
    )


def _without_clear_thinking_edits(context_management: JsonObject) -> JsonObject | None:
    """Return the context-management object without ``clear_thinking_*`` edits.

    A ``clear_thinking`` context edit requires an active thinking config, so it
    is stripped alongside a dropped thinking config; the provider rejects it by
    name once the config is gone. Other edits are preserved verbatim.

    Args:
        context_management: Verbatim caller ``context_management`` object.

    Returns:
        The same object minus clear-thinking edits, or ``None`` when no edit
        survives so the field is omitted entirely.
    """
    edits = context_management.get("edits")
    if not isinstance(edits, list):
        return context_management
    retained = [
        edit
        for edit in edits
        if not (isinstance(edit, dict) and str(edit.get("type", "")).startswith("clear_thinking"))
    ]
    if len(retained) == len(edits):
        return context_management
    if not retained:
        remaining = {key: value for key, value in context_management.items() if key != "edits"}
        return remaining or None
    return {**context_management, "edits": retained}


def coerce_capability(capability: str, request: GatewayRequest) -> RequestCoercion | None:
    """Build the disclosed coercion for one preflight capability rejection.

    Four coercions exist, all only here — after every rung declined the
    verbatim request — and all only as a disclosed substitution. Degrading
    ``strict: true`` tools to best-effort schemas weakens a correctness
    guarantee. Relaxing a forced ``tool_choice`` to ``auto`` weakens a
    structural guarantee the same way (the model still sees the tools and the
    prompt, so it usually calls one, but nothing forces it); a named
    rejection instead would fail every forced-choice request against a model
    the provider serves fine under ``auto``. Dropping ``service_tier``
    changes pricing and latency
    semantics, which the caller can act on only when told, so the drop is
    disclosed rather than silent. Images inside TOOL results degrade to
    placeholder text on an image-incapable route because the block is baked
    into the caller's history and a rejection wedges the whole session; a
    top-level user image keeps the fail-closed contract (the caller can
    re-send it), so ``image_input`` coerces only when every image in the
    request lives in a tool message. Every other capability names a feature
    with no approximation and stays fail-closed.

    Args:
        capability: Stable capability literal from the preflight rejection.
        request: Decoded request no rung could preserve.

    Returns:
        The disclosed substitution to retry with, or ``None`` when the
        capability cannot be coerced.
    """
    if capability == "image_input":
        if any(message.role != "tool" and message.images for message in request.messages):
            return None
        stripped = strip_tool_result_images(request.messages)
        if stripped is None:
            return None
        return RequestCoercion(
            request=request.model_copy(update={"messages": stripped}),
            disclosures=(TOOL_RESULT_IMAGE_DROP_DISCLOSURE,),
        )
    if capability == "service_tier":
        if request.service_tier is None:
            return None
        return RequestCoercion(
            request=request.model_copy(update={"service_tier": None}),
            disclosures=(SERVICE_TIER_DROP_DISCLOSURE,),
        )
    if capability == "forced_tool_choice":
        if request.tool_choice != "required" and not isinstance(
            request.tool_choice, GatewayNamedToolChoice
        ):
            return None
        return RequestCoercion(
            request=request.model_copy(update={"tool_choice": "auto"}),
            disclosures=(FORCED_TOOL_CHOICE_DISCLOSURE,),
        )
    if capability != "strict_tools" or not any(tool.strict for tool in request.tools):
        return None
    return RequestCoercion(
        request=request.model_copy(
            update={
                "tools": tuple(
                    tool.model_copy(update={"strict": False}) if tool.strict else tool
                    for tool in request.tools
                )
            }
        ),
        disclosures=(STRICT_TOOLS_DISCLOSURE,),
    )


def coerce_route_rejections(
    errors: Sequence[ProviderParameterError | ProviderCapabilityError],
    deployment_count: int,
    request: GatewayRequest,
) -> RequestCoercion | None:
    """Pick the one disclosed coercion a set of per-rung rejections allows.

    A unanimous capability rejection may coerce any coercible capability.
    Mixed rejections may drop only the service tier: rungs declining for
    different reasons mean some rung offered to preserve any given guarantee,
    so degrading one (strict tools) would weaken semantics a rung could have
    kept — but the tier is a routing hint whose only alternative is a
    rejection the caller cannot act on, so the disclosed drop is offered
    whenever any rung named it and the per-rung probe decides whether the
    dropped request actually serves.

    Args:
        errors: One rejection per declined deployment, in route order.
        deployment_count: Number of deployments the route offered.
        request: Decoded request no rung could preserve.

    Returns:
        The disclosed substitution to retry with, or ``None`` when nothing
        coercible applies.
    """
    capability = route_wide_capability(errors, deployment_count)
    if capability is not None:
        return coerce_capability(capability, request)
    if any(
        isinstance(error, ProviderCapabilityError) and error.capability == "service_tier"
        for error in errors
    ):
        return coerce_capability("service_tier", request)
    return None


def route_wide_capability(
    errors: Sequence[ProviderParameterError | ProviderCapabilityError],
    deployment_count: int,
) -> str | None:
    """Return the one capability EVERY route deployment rejected, if any.

    Deployments can decline different requirements; a capability coercion is
    honest only when a single capability was rejected by every rung, so mixed
    rejections keep the first rung's own field-specific error instead of
    degrading a field some rung could have preserved.

    Args:
        errors: One rejection per declined deployment, in route order.
        deployment_count: Number of deployments the route offered.

    Returns:
        The universally rejected capability literal, or ``None``.
    """
    if len(errors) != deployment_count:
        return None
    capabilities = {
        error.capability for error in errors if isinstance(error, ProviderCapabilityError)
    }
    if len(capabilities) == 1 and all(
        isinstance(error, ProviderCapabilityError) for error in errors
    ):
        return next(iter(capabilities))
    return None


def coerce_structured_text_schema(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> RequestCoercion | None:
    """Close every object in a structured-output schema for a rung that needs it.

    The Anthropic Messages validator rejects a structured-output schema whose
    objects leave ``additionalProperties`` open, while the OpenAI-family
    validators accept the same schema, so a caller who tested against one
    provider gets a post-dispatch 400 from the other. Closing the objects is
    the only serviceable reading of the request (the provider has no open
    mode), and it tightens the output contract rather than loosening it, so
    it happens here as a disclosed coercion instead of a rejection. Schemas
    already closed everywhere, and routes with no rung on such a dialect,
    pass through untouched.

    Args:
        profiles: Ordered wire profiles for the rungs the request will reach.
        request: Admitted request, after generation-parameter narrowing.

    Returns:
        The disclosed substitution to dispatch, or ``None`` when nothing
        needs closing.
    """
    if request.structured_text is None:
        return None
    if not request.structured_text.strict:
        # A non-strict schema is permissive by the caller's own declaration
        # (notably a translated ``json_object`` = "any JSON object"). Closing it
        # would over-constrain the very intent the caller marked loose — a bare
        # open object would become "no properties allowed" — so it is left as-is
        # rather than silently tightened.
        return None
    if not any(
        profile.dialect in _SCHEMA_DIALECTS_REQUIRING_CLOSED_OBJECTS for profile in profiles
    ):
        return None
    closed, changed = _close_schema_objects(request.structured_text.json_schema)
    if not changed:
        return None
    return RequestCoercion(
        request=request.model_copy(
            update={
                "structured_text": request.structured_text.model_copy(
                    update={"json_schema": closed}
                )
            }
        ),
        disclosures=(CLOSED_SCHEMA_DISCLOSURE,),
    )


def coerce_strict_tool_schemas(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> RequestCoercion | None:
    """Close every object in each ``strict`` tool schema for a rung that needs it.

    The Anthropic strict validator requires ``additionalProperties: false``
    on every object of a strict tool's ``input_schema`` (verified live
    2026-09-05: an absent or ``true`` value is a 400 by name), exactly as it
    does for structured-output schemas. Closing the objects tightens the
    input contract the caller already asked to have enforced, so it is a
    disclosed coercion rather than a reason to drop ``strict``. Non-strict
    tools, schemas already closed everywhere, and routes with no rung on such
    a dialect pass through untouched.

    Args:
        profiles: Ordered wire profiles for the rungs the request will reach.
        request: Admitted request, after generation-parameter narrowing.

    Returns:
        The disclosed substitution to dispatch, or ``None`` when nothing
        needs closing.
    """
    if not any(tool.strict for tool in request.tools):
        return None
    if not any(
        profile.dialect in _SCHEMA_DIALECTS_REQUIRING_CLOSED_OBJECTS for profile in profiles
    ):
        return None
    changed_any = False
    tools: list[GatewayToolDefinition] = []
    for tool in request.tools:
        if not tool.strict:
            tools.append(tool)
            continue
        closed, changed = _close_schema_objects(tool.parameters)
        changed_any = changed_any or changed
        tools.append(tool.model_copy(update={"parameters": closed}) if changed else tool)
    if not changed_any:
        return None
    return RequestCoercion(
        request=request.model_copy(update={"tools": tuple(tools)}),
        disclosures=(STRICT_TOOL_SCHEMA_CLOSED_DISCLOSURE,),
    )


_SCHEMA_CHILD_KEYS = ("properties", "$defs", "definitions", "patternProperties")
"""Schema keys whose values map names to subschemas."""

_SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
"""Schema keys whose values list subschemas."""

_SCHEMA_SINGLE_KEYS = ("items", "not", "if", "then", "else")
"""Schema keys whose values are one subschema."""


def _close_schema_objects(schema: JsonObject) -> tuple[JsonObject, bool]:
    """Return ``schema`` with ``additionalProperties: false`` on every object.

    An object is any node typed ``object`` or carrying ``properties``. The
    walk descends through the standard composition and container keywords
    and copies only the nodes it changes.

    Args:
        schema: One JSON Schema node.

    Returns:
        The closed node and whether any node changed.
    """
    changed = False
    closed: JsonObject = dict(schema)
    is_object = schema.get("type") == "object" or "properties" in schema
    if is_object and schema.get("additionalProperties") is not False:
        closed["additionalProperties"] = False
        changed = True
    for key in _SCHEMA_CHILD_KEYS:
        children = schema.get(key)
        if isinstance(children, dict):
            closed_children: dict[str, JsonValue] = {}
            for name, child in children.items():
                if isinstance(child, dict):
                    closed_child, child_changed = _close_schema_objects(child)
                    changed = changed or child_changed
                    closed_children[name] = closed_child
                else:
                    closed_children[name] = child
            closed[key] = closed_children
    for key in _SCHEMA_LIST_KEYS:
        members = schema.get(key)
        if isinstance(members, list):
            closed_members: list[JsonValue] = []
            for member in members:
                if isinstance(member, dict):
                    closed_member, member_changed = _close_schema_objects(member)
                    changed = changed or member_changed
                    closed_members.append(closed_member)
                else:
                    closed_members.append(member)
            closed[key] = closed_members
    for key in _SCHEMA_SINGLE_KEYS:
        single = schema.get(key)
        if isinstance(single, dict):
            closed_single, single_changed = _close_schema_objects(single)
            changed = changed or single_changed
            closed[key] = closed_single
    return (closed if changed else schema), changed
