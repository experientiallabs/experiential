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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
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
from exp.runtime.models.providers.reasoning_compat import efforts_by_nearness
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

STRICT_TOOLS_DISCLOSURE = "tools.strict->false"
"""Disclosure recorded when strict tools degrade to best-effort schemas."""

EFFORT_DROP_DISCLOSURE = "reasoning_effort"
"""Disclosure recorded when a zero-reasoning route drops the caller effort."""

THINKING_DROP_DISCLOSURE = "thinking"
"""Disclosure recorded when a zero-reasoning route drops adaptive thinking."""

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
    if request.reasoning_effort is None:
        return None
    ladder: set[str] = set()
    for profile in profiles:
        ladder.update(profile_reasoning_efforts(profile))
    if not ladder:
        updates: dict[str, object] = {"reasoning_effort": None}
        disclosures: tuple[str, ...] = (EFFORT_DROP_DISCLOSURE,)
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
        if (
            request.provider_thinking_config is not None
            and request.provider_thinking_config.get("type") == "adaptive"
        ):
            # Adaptive thinking is the effort's own channel on the Messages
            # surface (the model picks its depth from output_config.effort),
            # so a route with no reasoning rung cannot honor it either and
            # the provider rejects it by name. A budgeted config is left
            # verbatim: its semantics do not depend on an effort level.
            updates["provider_thinking_config"] = None
            disclosures = (EFFORT_DROP_DISCLOSURE, THINKING_DROP_DISCLOSURE)
        dropped_request = request.model_copy(update=updates)
        if admits is not None and not admits(dropped_request):
            return None
        return RequestCoercion(request=dropped_request, disclosures=disclosures)
    if request.reasoning_effort in ladder:
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


def coerce_capability(capability: str, request: GatewayRequest) -> RequestCoercion | None:
    """Build the disclosed coercion for one preflight capability rejection.

    Two capabilities are coercible, both only here — after every rung declined
    the verbatim request — and both only as a disclosed drop. Degrading
    ``strict: true`` tools to best-effort schemas weakens a correctness
    guarantee. Dropping ``service_tier`` changes pricing and latency
    semantics, which the caller can act on only when told, so the drop is
    disclosed rather than silent. Every other capability names a feature with
    no approximation and stays fail-closed.

    Args:
        capability: Stable capability literal from the preflight rejection.
        request: Decoded request no rung could preserve.

    Returns:
        The disclosed substitution to retry with, or ``None`` when the
        capability cannot be coerced.
    """
    if capability == "service_tier":
        if request.service_tier is None:
            return None
        return RequestCoercion(
            request=request.model_copy(update={"service_tier": None}),
            disclosures=(SERVICE_TIER_DROP_DISCLOSURE,),
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
