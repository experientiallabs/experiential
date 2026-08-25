"""Resolve frozen gateway deployments into exact executable provider bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

from exp.common.core.artifacts import stable_id
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.health import DeploymentHealthKey
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models import ResolvedModel, RuntimeModelCatalog
from exp.runtime.models.providers import AsyncGatewayProvider
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.protocol import NativeWireClient


@dataclass(frozen=True)
class _ResolvedDeployment:
    """One verified provider binding for a frozen deployment."""

    deployment: ExactModelDeployment
    provider: AsyncGatewayProvider
    wire_profile: GatewayWireProfile
    health_key: DeploymentHealthKey
    idempotency_key: str


def resolve_route(
    route: GatewayRoute,
    catalogs: Mapping[tuple[str, str], RuntimeModelCatalog],
) -> tuple[_ResolvedDeployment, ...]:
    """Resolve and identity-check every deployment before billable dispatch."""
    authorization = route.snapshot.authorization
    catalog = catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
    if catalog is None:
        raise ValueError("runtime catalog is not loaded for the authorized revision")
    resolved: list[_ResolvedDeployment] = []
    for deployment in route.deployments:
        runtime_model = catalog.resolve(deployment.source_alias)
        _require_deployment_identity(deployment, runtime_model)
        if getattr(runtime_model.client, "stream", None) is None:
            raise TypeError("resolved gateway deployment has no async stream capability")
        wire_profile = _resolved_wire_profile(deployment, runtime_model)
        resolved.append(
            _ResolvedDeployment(
                deployment=deployment,
                provider=cast(AsyncGatewayProvider, runtime_model.client),
                wire_profile=wire_profile,
                health_key=(
                    authorization.catalog_sha256,
                    deployment.deployment_id,
                    deployment.connection_sha256,
                ),
                idempotency_key=_deployment_idempotency_key(route, deployment),
            )
        )
    return tuple(resolved)


def _resolved_wire_profile(
    deployment: ExactModelDeployment,
    runtime_model: ResolvedModel,
) -> GatewayWireProfile:
    """Return a native profile or a conservative Python-stream compatibility view."""
    capabilities = runtime_model.capabilities
    if isinstance(runtime_model.client, NativeWireClient):
        profile = runtime_model.client.gateway_wire_profile()
        output_limits = tuple(
            limit
            for limit in (
                profile.maximum_output_tokens,
                capabilities.maximum_output_tokens,
            )
            if limit is not None
        )
        return replace(
            profile,
            model_id=profile.model_id or runtime_model.snapshot.model_id,
            minimum_temperature=(
                max(profile.minimum_temperature, capabilities.minimum_temperature)
                if capabilities.minimum_temperature is not None
                else profile.minimum_temperature
            ),
            maximum_temperature=(
                min(profile.maximum_temperature, capabilities.maximum_temperature)
                if capabilities.maximum_temperature is not None
                else profile.maximum_temperature
            ),
            minimum_top_p=(
                max(profile.minimum_top_p, capabilities.minimum_top_p)
                if capabilities.minimum_top_p is not None
                else profile.minimum_top_p
            ),
            maximum_top_p=(
                min(profile.maximum_top_p, capabilities.maximum_top_p)
                if capabilities.maximum_top_p is not None
                else profile.maximum_top_p
            ),
            minimum_top_k=(
                capabilities.minimum_top_k
                if capabilities.minimum_top_k is not None
                else profile.minimum_top_k
            ),
            maximum_top_k=(
                min(profile.maximum_top_k, capabilities.maximum_top_k)
                if profile.maximum_top_k is not None and capabilities.maximum_top_k is not None
                else capabilities.maximum_top_k
                if capabilities.maximum_top_k is not None
                else profile.maximum_top_k
            ),
            sampling_requires_reasoning_none=capabilities.sampling_requires_reasoning_none,
            token_limit_key=capabilities.chat_max_tokens_field or profile.token_limit_key,
            maximum_output_tokens=min(output_limits) if output_limits else None,
        )
    # Python still supports injected/custom streaming clients that predate the
    # native protocol. Rust admission separately requires a real native profile.
    return GatewayWireProfile(
        dialect="python_stream",
        url="",
        model_id=deployment.provider_model,
        supports_temperature=capabilities.supports_temperature,
        minimum_temperature=(
            0.0 if capabilities.minimum_temperature is None else capabilities.minimum_temperature
        ),
        maximum_temperature=(
            2.0 if capabilities.maximum_temperature is None else capabilities.maximum_temperature
        ),
        supports_top_p=capabilities.supports_top_p,
        minimum_top_p=0.0 if capabilities.minimum_top_p is None else capabilities.minimum_top_p,
        maximum_top_p=1.0 if capabilities.maximum_top_p is None else capabilities.maximum_top_p,
        supports_top_k=capabilities.supports_top_k is True,
        minimum_top_k=1 if capabilities.minimum_top_k is None else capabilities.minimum_top_k,
        maximum_top_k=capabilities.maximum_top_k,
        supports_logprobs=capabilities.supports_logprobs is True,
        supports_reasoning=capabilities.supports_reasoning,
        reasoning_wire_format="reasoning_effort" if capabilities.supports_reasoning else "none",
        reasoning_effort=capabilities.reasoning_effort,
        sampling_requires_reasoning_none=capabilities.sampling_requires_reasoning_none,
        token_limit_key=capabilities.chat_max_tokens_field or "max_tokens",
        maximum_output_tokens=capabilities.maximum_output_tokens,
    )


def _require_deployment_identity(
    deployment: ExactModelDeployment,
    resolved: ResolvedModel,
) -> None:
    """Fail before accounting or network work when runtime identity drifts."""
    if (
        resolved.alias != deployment.source_alias
        or resolved.snapshot.provider != deployment.provider
        or deployment.provider_model not in {resolved.snapshot.model_id, resolved.served_model_id}
        or resolved.snapshot.revision != deployment.revision
        or resolved.snapshot.connection_sha256 != deployment.connection_sha256
        or resolved.snapshot.billing_source != deployment.billing_source
        or (
            deployment.capabilities is not None and resolved.capabilities != deployment.capabilities
        )
    ):
        raise ValueError("resolved runtime client differs from the frozen gateway deployment")


def _deployment_idempotency_key(
    route: GatewayRoute,
    deployment: ExactModelDeployment,
) -> str:
    """Derive one stable key reused only by retries of this deployment."""
    authorization = route.snapshot.authorization
    return stable_id(
        "gateway-provider-operation",
        {
            "request_id": authorization.request_id,
            "catalog_sha256": authorization.catalog_sha256,
            "deployment_id": deployment.deployment_id,
            "connection_sha256": deployment.connection_sha256,
        },
    )
