"""Bind recovery evidence to the static authentication already resolved for a wire."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from urllib.parse import urlsplit, urlunsplit

from exp.common.core.artifacts import sha256_json
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.recovery import FrozenRecoveryBinding, RecoveryHost, RecoveryScope
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.protocol import NativeWireClient

_logger = logging.getLogger(__name__)
_GLOBAL_SERVICE_HOSTS = frozenset(
    {"api.openai.com", "api.anthropic.com", "openrouter.ai", "generativelanguage.googleapis.com"}
)


def bind_recovery_profiles(
    deployments: tuple[ExactModelDeployment, ...],
    wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...],
    organization_id: str,
    host: RecoveryHost | None,
    *,
    request_region: str | None = None,
) -> tuple[tuple[GatewayWireProfile, NativeWireClient], ...]:
    """Freeze scopes without I/O; unproven request-level region selectors disable recovery."""
    bound = []
    for deployment, (profile, client) in zip(deployments, wires, strict=True):
        receipt = profile.credential_receipt
        binding = None
        region = _resolved_region(profile)
        if (
            host is not None
            and receipt is not None
            and not profile.signs_request_body
            # Vertex receipts identify the OAuth source account for explicit
            # resources, not the rotating bearer used by static-auth recovery.
            and deployment.provider != "vertex"
            and request_region is None
            and region is not None
        ):
            scope = RecoveryScope(
                provider=deployment.provider,
                exact_model_id=deployment.exact_model_id,
                endpoint_scope=_endpoint_scope(profile),
                region_scope=region,
                credential_scope=str(receipt.binding_id),
                organization_id=organization_id,
            )
            binding = FrozenRecoveryBinding(
                deployment.deployment_id,
                deployment.connection_sha256,
                profile.url,
                profile.model_id,
                scope,
                profile.operational_region,
                profile.dialect,
            )
            try:
                host.observe_scope(scope.operational())
            except Exception:  # noqa: BLE001 - optional evidence never changes serving.
                _logger.warning("Recovery scope observation skipped")
        bound.append((replace(profile, recovery_binding=binding), client))
    return tuple(bound)


def _resolved_region(profile: GatewayWireProfile) -> str | None:
    """Use declared geography only when no unproven operator selector changes execution scope."""
    if profile.inference_geo is not None:
        return None
    if profile.operational_region:
        return profile.operational_region
    if (
        profile.operational_region is None
        and urlsplit(profile.url).hostname in _GLOBAL_SERVICE_HOSTS
    ):
        return "global-service"
    return None


def _endpoint_scope(profile: GatewayWireProfile) -> str:
    """Bind shared topology to the exact endpoint, provider model and wire dialect."""
    parsed = urlsplit(profile.url)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("resolved wire endpoint embeds credentials")
    endpoint = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", ""))
    return sha256_json(
        {"endpoint": endpoint, "model": profile.model_id, "dialect": profile.dialect}
    )


def validated_recovery_binding(
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
    organization_id: str,
) -> FrozenRecoveryBinding | None:
    """Reject accidental binding reassignment after route narrowing or reordering."""
    binding = profile.recovery_binding
    if binding is None or not binding.scope.region_scope or deployment.provider == "vertex":
        return None
    if (
        binding.deployment_id != deployment.deployment_id
        or binding.connection_sha256 != deployment.connection_sha256
        or binding.wire_url != profile.url
        or binding.wire_model != profile.model_id
        or profile.signs_request_body
        or binding.wire_dialect != profile.dialect
        or binding.wire_region != profile.operational_region
        or binding.scope.provider != deployment.provider
        or binding.scope.exact_model_id != deployment.exact_model_id
        or binding.scope.organization_id != organization_id
        or profile.credential_receipt is None
        or binding.scope.credential_scope != str(profile.credential_receipt.binding_id)
    ):
        raise ValueError("resolved recovery binding differs from its authorized wire")
    if _resolved_region(profile) is None:
        return None
    return binding


def frozen_scope(
    bindings: Mapping[str, FrozenRecoveryBinding],
    deployment: ExactModelDeployment,
    organization_id: str,
) -> RecoveryScope | None:
    """Read a retained scope without consulting mutable credential or catalog state."""
    binding = bindings.get(deployment.deployment_id)
    if binding is None or not binding.scope.region_scope:
        return None
    if (
        binding.deployment_id != deployment.deployment_id
        or binding.connection_sha256 != deployment.connection_sha256
        or binding.scope.organization_id != organization_id
        or binding.scope.provider != deployment.provider
        or binding.scope.exact_model_id != deployment.exact_model_id
    ):
        raise ValueError("retained recovery binding differs from its authorized deployment")
    return binding.scope
