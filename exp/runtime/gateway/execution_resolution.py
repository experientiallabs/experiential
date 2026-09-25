"""Intersect resolved provider wire profiles with the frozen catalog capability contract."""

from __future__ import annotations

from dataclasses import replace

from exp.common.models.gateway_catalog import ExactModelDeployment, NormalizedGatewayCatalog
from exp.common.models.model import BillingSource
from exp.runtime.models import ResolvedModel, RuntimeModelCatalog
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.protocol import NativeWireClient


class GatewayWireContractError(ValueError):
    """A resolved provider profile contradicts the frozen gateway contract."""


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
        GatewayWireContractError: Catalog reasoning metadata contradicts the
            resolved model or provider wire profile.
        TypeError: The resolved client exposes no native wire profile.
    """
    capabilities = runtime_model.capabilities
    gateway_capabilities = deployment.gateway.capabilities
    if isinstance(runtime_model.client, NativeWireClient):
        profile = runtime_model.client.gateway_wire_profile()
        if gateway_capabilities.declares_reasoning_contract and (
            not capabilities.supports_reasoning or not profile.supports_reasoning
        ):
            raise GatewayWireContractError(
                "gateway reasoning metadata conflicts with the resolved provider wire profile"
            )
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
            credential_receipt=runtime_model.credential_receipt
            if not profile.signs_request_body
            else None,
            model_id=profile.model_id or runtime_model.snapshot.model_id,
            supports_responses_logprobs=(
                gateway_capabilities.supports_responses_logprobs
                and profile.dialect == "openai_responses"
            ),
            billing_customer_managed=(deployment.billing_source == BillingSource.CUSTOMER_MANAGED),
            service_tier_pricing_enabled=capabilities.service_tier_pricing_enabled,
            service_tier_cards=frozenset(
                tier
                for tier in ("flex", "priority")
                if deployment.gateway.prices.service_tier(tier) is not None
            ),
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
            logprobs_reasoning_efforts=gateway_capabilities.logprobs_reasoning_efforts,
            supported_reasoning_efforts=(
                gateway_capabilities.supported_reasoning_efforts
                or profile.supported_reasoning_efforts
            ),
            reasoning_effort=(
                gateway_capabilities.reasoning_default_effort or profile.reasoning_effort
            ),
            reasoning_effort_required=(
                gateway_capabilities.reasoning_effort_required or profile.reasoning_effort_required
            ),
            token_limit_key=capabilities.chat_max_tokens_field or profile.token_limit_key,
            maximum_output_tokens=min(output_limits) if output_limits else None,
            # The provider's output-token floor is a catalog lane fact the
            # client profile cannot know (a relay serves floored and unfloored
            # models on one wire); the deployment declaration is the only
            # source, so it is carried, never intersected.
            minimum_output_tokens=gateway_capabilities.minimum_output_tokens,
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


def alias_native_blockers(
    alias: str,
    normalized: NormalizedGatewayCatalog,
    runtime_catalog: RuntimeModelCatalog,
) -> tuple[str, ...]:
    """Name why the native engine cannot serve one alias, or ``()`` if it can.

    Every deployment reachable from the alias's catalog snapshot (direct pools
    and project candidates alike) must resolve to a provider client with a
    native wire dialect and a valid wire contract, since no other engine exists
    to serve the request. This is the per-alias servability check the catalog
    build runs so a structurally unservable alias is excluded (marked
    UNAVAILABLE) rather than aborting the whole build; the same check names the
    fleet-level startup blockers.

    Args:
        alias: Public alias name, used only for the returned reason text.
        normalized: The alias's normalized catalog snapshot.
        runtime_catalog: The frozen runtime catalog for the alias's revision.

    Returns:
        Display-safe reasons the alias cannot be served natively, deduplicated,
        or an empty tuple when every deployment resolves to a native wire.
    """
    reasons: list[str] = []
    for deployment in normalized.deployments:
        try:
            resolved = runtime_catalog.resolve(deployment.source_alias)
        except Exception:  # noqa: BLE001 - name the deployment, not the internals.
            reasons.append(f"deployment {deployment.deployment_id!r} does not resolve")
            continue
        client = resolved.client
        if not isinstance(client, NativeWireClient):
            reasons.append(f"provider {deployment.provider!r} has no native wire profile")
            continue
        try:
            _resolved_wire_profile(deployment, resolved)
        except ProviderCapabilityError as exc:
            if exc.capability != "native_data_plane":
                raise
            reasons.append(f"provider {deployment.provider!r} has no native dialect implementation")
        except GatewayWireContractError:
            reasons.append(
                f"deployment {deployment.deployment_id!r} has an invalid reasoning wire contract"
            )
    return tuple(dict.fromkeys(reasons))
