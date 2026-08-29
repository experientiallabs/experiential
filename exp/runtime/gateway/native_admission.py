"""Route admission with disclosed coercion for the native control plane.

Admission prefers rungs that preserve every caller semantic verbatim: the
generation-control narrowing and the per-deployment capability preflight plus
payload build both run here. When no rung preserves the request verbatim,
this module applies the capability-preservation policy's minimal disclosed
coercion exactly once per layer and re-selects; when nothing coercible
remains, the first rung's own field-scoped rejection stays the answer. A
coercion is never
silent: every substitution is disclosed through ``ignored_parameters`` in
``path->effective`` form, warn-logged for operators, and counted in the
admission metrics.
"""

from __future__ import annotations

import logging

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_execution import select_route_deployments
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.models.providers import preflight_gateway_request
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import (
    coerce_capability,
    coerce_generation_parameters,
    route_wide_capability,
)
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
)
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.protocol import NativeWireClient
from exp.runtime.models.providers.streaming_requests import (
    dialect_stream_payload,
    route_generation_parameter_requests,
)

_logger = logging.getLogger(__name__)

_ResolvedWires = tuple[tuple[GatewayWireProfile, NativeWireClient], ...]


def admitted_route_requests(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    request: GatewayRequest,
    *,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
) -> tuple[GatewayRoute, _ResolvedWires, GatewayRequest, GatewayRequest]:
    """Narrow one certified route to rungs that serve the admitted request.

    Args:
        route: Frozen route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        request: Canonical request produced by the public protocol decoder.
        accounting: Shared accounting owning the coercion counter.
        authorization: Frozen authority for the accepted request.

    Returns:
        The narrowed route and wires plus the public request (carrying any
        coercion disclosures) and the streaming-forced provider request.

    Raises:
        ProviderParameterError: No rung preserves a generation control and no
            disclosed coercion applies, or rungs declined for different
            field-specific reasons.
        ProviderCapabilityError: The first rung's capability rejection when
            no rung is protocol-compatible; the shared admit handler scopes
            it to the exact public request field.
        GatewayRoutingError: No rung is protocol-compatible and none named a
            rejection.
    """
    admitted_request = request
    coercion_disclosures: tuple[str, ...] = ()
    try:
        compatible_indexes = compatible_generation_parameter_profile_indexes(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
        )
    except ProviderParameterError:
        # No rung preserves the request verbatim; retry once with the
        # minimal disclosed coercion when semantics allow, otherwise
        # keep the named rejection.
        coercion = coerce_generation_parameters(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
        )
        if coercion is None:
            raise
        compatible_indexes = compatible_generation_parameter_profile_indexes(
            tuple(profile for profile, _client in resolved_wires),
            coercion.request,
        )
        admitted_request = coercion.request
        coercion_disclosures = coercion.disclosures
    route = select_route_deployments(route, compatible_indexes)
    resolved_wires = tuple(resolved_wires[index] for index in compatible_indexes)
    public_request, provider_request = route_generation_parameter_requests(
        tuple(profile for profile, _client in resolved_wires),
        admitted_request,
    )
    provider_request = provider_request.model_copy(update={"stream": True, "include_usage": True})
    protocol_indexes, protocol_errors = protocol_compatible_indexes(
        route,
        resolved_wires,
        provider_request,
        public_stream=public_request.stream,
    )
    blocking_capability = route_wide_capability(protocol_errors, len(route.deployments))
    if not protocol_indexes and blocking_capability is not None:
        # Every rung declined the same capability verbatim; degrade
        # once with disclosure where semantics allow (strict tools
        # only).
        coercion = coerce_capability(blocking_capability, admitted_request)
        if coercion is not None:
            admitted_request = coercion.request
            coercion_disclosures = (*coercion_disclosures, *coercion.disclosures)
            public_request, provider_request = route_generation_parameter_requests(
                tuple(profile for profile, _client in resolved_wires),
                admitted_request,
            )
            provider_request = provider_request.model_copy(
                update={"stream": True, "include_usage": True}
            )
            protocol_indexes, protocol_errors = protocol_compatible_indexes(
                route,
                resolved_wires,
                provider_request,
                public_stream=public_request.stream,
            )
            blocking_capability = route_wide_capability(protocol_errors, len(route.deployments))
    if not protocol_indexes:
        if not protocol_errors:
            raise GatewayRoutingError("authorized route has no compatible deployment")
        # Nothing coercible remains; the first rung's own rejection stays the
        # accurate answer, and the shared admit handler scopes capability
        # rejections to their exact public request field.
        raise protocol_errors[0]
    if len(protocol_indexes) != len(route.deployments):
        selected_indexes = tuple(protocol_indexes)
        route = select_route_deployments(route, selected_indexes)
        resolved_wires = tuple(resolved_wires[index] for index in selected_indexes)
        public_request, provider_request = route_generation_parameter_requests(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
        )
        provider_request = provider_request.model_copy(
            update={"stream": True, "include_usage": True}
        )
    if coercion_disclosures:
        record_admission_coercions(accounting, authorization, coercion_disclosures)
        public_request = public_request.model_copy(
            update={
                "ignored_parameters": tuple(
                    dict.fromkeys((*public_request.ignored_parameters, *coercion_disclosures))
                )
            }
        )
    return route, resolved_wires, public_request, provider_request


def protocol_compatible_indexes(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    provider_request: GatewayRequest,
    *,
    public_stream: bool | None,
) -> tuple[tuple[int, ...], tuple[ProviderParameterError | ProviderCapabilityError, ...]]:
    """Select rungs that pass capability preflight and payload build.

    Args:
        route: Frozen route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        provider_request: Streaming-forced request to validate.
        public_stream: The caller's declared streaming intent.

    Returns:
        Ordered compatible indexes and every rung's rejection in route
        order, so the caller can distinguish a route-wide capability gap
        from rungs declining for different reasons.
    """
    indexes: list[int] = []
    errors: list[ProviderParameterError | ProviderCapabilityError] = []
    for index, (deployment, (profile, _client)) in enumerate(
        zip(route.deployments, resolved_wires, strict=True)
    ):
        try:
            preflight_gateway_request(
                provider_request,
                deployment.gateway.capabilities,
                model_capabilities=deployment.capabilities,
                public_stream=public_stream,
            )
            dialect_stream_payload(profile, provider_request)
        except (ProviderParameterError, ProviderCapabilityError) as exc:
            errors.append(exc)
            continue
        indexes.append(index)
    return tuple(indexes), tuple(errors)


def record_admission_coercions(
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    disclosures: tuple[str, ...],
) -> None:
    """Log and count one admission's disclosed request coercions.

    A coercion is never silent: the caller sees it in
    ``ignored_parameters``, the log names it for operators, and the
    metrics snapshot counts it so a persistently coerced alias reaches a
    human instead of quietly serving degraded semantics forever.

    Args:
        accounting: Shared accounting owning the coercion counter.
        authorization: Frozen authority for the accepted request.
        disclosures: Path->effective disclosure strings applied.
    """
    accounting.record_admission_coercions(len(disclosures))
    _logger.warning(
        "gateway admission coerced request semantics for alias %r: %s "
        "(disclosed through ignored_parameters)",
        authorization.alias,
        ", ".join(disclosures),
    )
