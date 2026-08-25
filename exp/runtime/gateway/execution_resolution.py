"""Resolve frozen gateway deployments into exact executable provider bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    if isinstance(runtime_model.client, NativeWireClient):
        return runtime_model.client.gateway_wire_profile()
    # Python still supports injected/custom streaming clients that predate the
    # native protocol. Rust admission separately requires a real native profile.
    capabilities = runtime_model.capabilities
    return GatewayWireProfile(
        dialect="python_stream",
        url="",
        model_id=deployment.provider_model,
        supports_temperature=capabilities.supports_temperature,
        supports_top_p=capabilities.supports_top_p,
        supports_top_k=capabilities.supports_top_k is True,
        supports_logprobs=capabilities.supports_logprobs is True,
        supports_reasoning=capabilities.supports_reasoning,
        reasoning_wire_format="reasoning_effort" if capabilities.supports_reasoning else "none",
        reasoning_effort=capabilities.reasoning_effort,
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
