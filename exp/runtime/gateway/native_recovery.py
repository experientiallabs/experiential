"""Integrate reason-aware session cache state into native admission and settlement."""

from __future__ import annotations

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.affinity import affinity_fingerprint, affinity_seed_material
from exp.runtime.gateway.contracts import (
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.recovery import (
    RecoveryCause,
    RecoveryHost,
    SessionCacheKey,
    SessionRecoveryRegistry,
)
from exp.runtime.gateway.replay_identity import canonical_request_sha256


def recovery_prefix_digest(request: GatewayRequest) -> str | None:
    """Hash actual stable prefix input, not a caller-controlled cache routing hint.

    Tool schemas and opaque tool declarations precede message input and join the
    digest. Without a leading system/developer run, only the first user turn is
    considered stable. Canonical replay identity includes structured/media and
    provider-significant carriers. Raw prompts never enter the history registry.
    """
    count = 0
    for message in request.messages:
        if message.role in ("system", "developer"):
            count += 1
        elif count:
            break
        elif message.role == "user":
            count = 1
            break
        else:
            return None
    if not count:
        return None
    prefix = request.model_copy(
        update={
            "messages": request.messages[:count],
            "prompt_cache_key": None,
            "provider_prompt_cache_key": None,
            "client_request_id": None,
            "idempotency_key": None,
            "previous_response_id": None,
            "metadata": {},
            "user": None,
            "safety_identifier": None,
        }
    )
    return canonical_request_sha256(prefix)


def session_cache_key(entry: InflightRequest) -> SessionCacheKey | None:
    """Derive tenant and prefix scope only for conversational requests."""
    request = entry.request
    if not isinstance(request, GatewayRequest) or not request.provider_prompt_cache_key:
        return None
    prefix = recovery_prefix_digest(request)
    if prefix is None:
        return None
    material = affinity_seed_material(
        request,
        continuation_episode_key=None
        if entry.continuation is None
        else entry.continuation.episode_key,
        request_id=entry.authorization.request_id,
    )
    fingerprint = affinity_fingerprint(
        organization_id=entry.authorization.organization_id,
        identity_id=entry.authorization.identity_id,
        material=material,
    )
    return SessionCacheKey(
        entry.authorization.organization_id,
        entry.authorization.identity_id,
        fingerprint,
        prefix,
    )


def record_departure(
    registry: SessionRecoveryRegistry,
    host: RecoveryHost | None,
    entry: InflightRequest,
    deployment: ExactModelDeployment,
    cause: RecoveryCause,
    retry_after_seconds: float = 0,
) -> None:
    """Record a departure reason without implying dispatch warmed a cache."""
    key = session_cache_key(entry)
    if key is not None:
        registry.depart(
            key,
            deployment.deployment_id,
            registry.scope(deployment, entry.authorization.organization_id, host),
            cause,
            retry_after_seconds=retry_after_seconds,
        )


def record_session_outcome(
    registry: SessionRecoveryRegistry,
    host: RecoveryHost | None,
    entry: InflightRequest,
    attempt_id: str,
    usage: GatewayUsage | None,
    failure: GatewayFailure | None,
) -> None:
    """Record each settled outcome once, separate from aggregate fairness samples."""
    depth = entry.attempt_depths.get(attempt_id)
    key = session_cache_key(entry)
    if depth is None or key is None or attempt_id in entry.recovery_recorded_attempts:
        return
    entry.recovery_recorded_attempts.add(attempt_id)
    deployment = entry.route.deployments[depth]
    if failure is not None:
        causes: dict[GatewayFailureClass, RecoveryCause] = {
            GatewayFailureClass.THROTTLED: "throttle",
            GatewayFailureClass.PROVIDER_AUTHENTICATION: "credential",
            GatewayFailureClass.PROVIDER_QUOTA: "credential",
            GatewayFailureClass.PROVIDER_NOT_FOUND: "credential",
            GatewayFailureClass.TRANSPORT: "transport",
            GatewayFailureClass.TIMEOUT: "transport",
            GatewayFailureClass.PROVIDER_INTERNAL: "transport",
        }
        cause = causes.get(failure.failure_class)
        if cause is not None:
            record_departure(
                registry,
                host,
                entry,
                deployment,
                cause,
                retry_after_seconds=float(failure.retry_after_seconds or 0),
            )
        return
    if usage is None:
        return
    dispatch = deployment.gateway.dispatch
    registry.record_success(
        key,
        deployment.deployment_id,
        registry.scope(deployment, entry.authorization.organization_id, host),
        cached_tokens=usage.cached_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        retention_seconds=deployment.gateway.cache_retention_seconds,
        sticky_seconds=None if dispatch is None else dispatch.sticky_spill_seconds,
    )
