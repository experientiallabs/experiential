"""Intersect resolved provider wire profiles with the frozen catalog capability contract."""

from __future__ import annotations

from dataclasses import replace

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.models import ResolvedModel
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.protocol import NativeWireClient


def _resolved_wire_profile(
    deployment: ExactModelDeployment,
    runtime_model: ResolvedModel,
) -> GatewayWireProfile:
    """Return the client's native wire profile bounded by the frozen catalog contract.

    Args:
        deployment: Frozen certified deployment from the authorized catalog.
        runtime_model: Runtime resolution of the deployment's source alias.

    Returns:
        The client's wire profile with every generation-parameter range
        intersected against the deployment's catalog capabilities.

    Raises:
        TypeError: The resolved client exposes no native wire profile.
    """
    capabilities = runtime_model.capabilities
    gateway_capabilities = deployment.gateway.capabilities
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
            supported_reasoning_efforts=(
                gateway_capabilities.supported_reasoning_efforts
                or profile.supported_reasoning_efforts
            ),
            reasoning_effort_required=(
                gateway_capabilities.reasoning_effort_required or profile.reasoning_effort_required
            ),
            token_limit_key=capabilities.chat_max_tokens_field or profile.token_limit_key,
            maximum_output_tokens=min(output_limits) if output_limits else None,
        )
    raise TypeError(
        f"provider {deployment.provider!r} resolved to a client without a native wire profile"
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
