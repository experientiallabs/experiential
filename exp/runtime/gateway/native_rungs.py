"""Per-rung dispatch construction for the native bridge.

Admission hands the bridge an ordered route; this module turns one rung of
it into everything the data plane needs for that deployment: the frozen wire
entry, the dispatch signer, the frozen-body binding and the reasoning-carrier
authority. Rungs are shaped independently so a control (``parallel_tool_calls``
today) can be honored natively on one rung and emulated on the next without
the route-level request changing.
"""

from __future__ import annotations

from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject, sha256_bytes
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.native_admission import shape_parallel_tool_calls
from exp.runtime.gateway.native_dispatch import frozen_dispatch
from exp.runtime.gateway.native_execution import FrozenDispatchBinding, deployment_wire_entry
from exp.runtime.gateway.reasoning_carrier import (
    ReasoningCarrierAuthority,
    reasoning_carrier_authority,
    scheme_for_profile,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers import (
    emulated_gateway_capabilities,
    emulated_stop_sequences,
    preflight_gateway_request,
    require_gateway_provider,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.logprobs import require_chat_logprobs
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.models.providers.streaming_requests import dialect_stream_payload
from exp.runtime.models.providers.wire_messages import anthropic_request_headers


@dataclass(frozen=True, slots=True)
class RungDispatch:
    """One deployment's frozen dispatch material plus its shaping disclosure."""

    wire_entry: JsonObject
    signer: GatewayDispatchSigner | None
    binding: FrozenDispatchBinding | None
    carrier_authority: ReasoningCarrierAuthority | None
    parallel_disclosure: str | None


def build_rung_dispatch(
    route: GatewayRoute,
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
    client: NativeWireClient,
    *,
    provider_request: GatewayRequest,
    public_request: GatewayRequest,
    authorization: AuthorizationSnapshot,
    throttle_redial_budget: int = 0,
) -> RungDispatch:
    """Preflight, shape and freeze ``provider_request`` for one route rung.

    ``throttle_redial_budget`` rides onto the wire entry unchanged: it is
    the admission-time count of post-backoff redials a throttle on this rung
    is worth under the pool's ``throttle_redial`` schedule for this request.
    """
    require_gateway_provider(deployment.provider)
    preflight_gateway_request(
        provider_request,
        deployment.gateway.capabilities,
        model_capabilities=deployment.capabilities,
        public_stream=public_request.stream,
        route_provider=deployment.provider,
        emulated_capabilities=emulated_gateway_capabilities(
            profile.dialect, emulate_parallel_tool_calls=True
        ),
    )
    rung_request, parallel_disclosure = shape_parallel_tool_calls(
        provider_request, deployment.gateway.capabilities
    )
    require_chat_logprobs((profile,), rung_request)
    upstream_payload = dialect_stream_payload(profile, rung_request)
    upstream_body, signer = frozen_dispatch(profile, client, upstream_payload)
    request_headers = (
        anthropic_request_headers(dict(profile.headers), rung_request)
        if profile.dialect == "anthropic_messages"
        else None
    )
    wire_entry = deployment_wire_entry(
        route,
        deployment,
        profile,
        upstream_payload,
        upstream_body,
        headers=request_headers,
        stop_sequences=emulated_stop_sequences(profile.dialect, rung_request),
        serialize_tool_calls=rung_request.serialize_tool_calls,
        throttle_redial_budget=throttle_redial_budget,
    )
    binding = (
        None
        if signer is None or upstream_body is None
        else FrozenDispatchBinding(
            url=profile.url,
            body_sha256=sha256_bytes(upstream_body.encode("utf-8")),
        )
    )
    carrier_scheme = scheme_for_profile(profile)
    carrier_authority = (
        None
        if carrier_scheme is None
        else reasoning_carrier_authority(
            authorization=authorization,
            exact_model_id=route.snapshot.exact_model_id,
            pool_id=route.snapshot.pool_id,
            deployment=deployment,
            profile=profile,
            scheme=carrier_scheme,
        )
    )
    return RungDispatch(
        wire_entry=wire_entry,
        signer=signer,
        binding=binding,
        carrier_authority=carrier_authority,
        parallel_disclosure=parallel_disclosure,
    )
