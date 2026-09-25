"""Integrate reason-aware session cache state into native admission and settlement."""

from __future__ import annotations

import logging

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.affinity import affinity_fingerprint, affinity_seed_material
from exp.runtime.gateway.contracts import (
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.health import health_failure_cause
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.recovery import (
    RecoveryCause,
    RecoveryHost,
    SessionCacheKey,
    SessionRecoveryRegistry,
)
from exp.runtime.gateway.recovery_binding import frozen_scope
from exp.runtime.gateway.replay_identity import canonical_request_sha256

_logger = logging.getLogger(__name__)


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
    *,
    key: SessionCacheKey | None = None,
    observed_at: float | None = None,
) -> bool:
    """Record an optional departure without implying dispatch warmed a cache.

    Args:
        registry: Worker-local recovery history.
        host: Optional scope authority. None disables observation entirely.
        entry: Owning request with authorized tenant and stable prefix input.
        deployment: Actual deployment being left.
        cause: Health failure or local admission reason for leaving.
        retry_after_seconds: Provider backoff that bounds the next recovery trial.
        key: Already derived session key, avoiding repeat hashing at settlement.
        observed_at: First terminal receipt epoch, preserved through durable retries.

    Returns:
        Whether the departure was recorded. Invalid scopes and observer failures
        record nothing and emit only a content-free diagnostic.
    """
    if host is None:
        return False
    try:
        if key is None:
            key = session_cache_key(entry)
        if key is None:
            return False
        scope = frozen_scope(
            entry.recovery_bindings, deployment, entry.authorization.organization_id
        )
        if scope is None:
            return False
        registry.depart(
            key,
            deployment.deployment_id,
            scope,
            cause,
            retry_after_seconds=retry_after_seconds,
            observed_at=observed_at,
        )
    except Exception:  # noqa: BLE001 - optional observation cannot fail admission or settlement.
        _logger.warning("Session recovery departure observation skipped")
        return False
    return True


def observe_reserved_attempt(
    host: RecoveryHost | None,
    entry: InflightRequest,
    attempt_id: str,
    deployment: ExactModelDeployment,
) -> None:
    """Observe a successful reservation using only its retained resolved-wire binding."""
    if host is None:
        return
    try:
        scope = frozen_scope(
            entry.recovery_bindings, deployment, entry.authorization.organization_id
        )
        if scope is not None and scope.region_scope is not None:
            host.attempt_started(attempt_id, scope.operational())
    except Exception:  # noqa: BLE001 - optional observer cannot undo a reservation.
        _logger.warning("Recovery attempt observation skipped")


def recovery_failure(
    data: JsonObject | None, normalized: GatewayFailure | None
) -> GatewayFailure | None:
    """Retain credential-local auth/quota causes without changing ledger or house health."""
    if normalized is None or not normalized.customer_owned or data is None:
        return normalized
    failure = data.get("failure")
    if isinstance(failure, dict) and failure.get("failure_class") in (
        "provider_authentication",
        "provider_quota",
    ):
        return normalized.model_copy(
            update={"failure_class": GatewayFailureClass(str(failure["failure_class"]))}
        )
    return normalized


def retain_recovery_observation_time(
    registry: SessionRecoveryRegistry, entry: InflightRequest, attempt_id: str
) -> None:
    """Freeze first terminal arrival before ledger I/O without trusting a caller timestamp."""
    if attempt_id not in entry.attempt_depths:
        return
    with entry.recovery_observation_lock:
        entry.recovery_observed_at.setdefault(attempt_id, registry.observation_time())


def record_session_outcome(
    registry: SessionRecoveryRegistry,
    host: RecoveryHost | None,
    entry: InflightRequest,
    attempt_id: str,
    usage: GatewayUsage | None,
    failure: GatewayFailure | None,
) -> None:
    """Observe settled recovery evidence without affecting durable accounting.

    Both direct and swept settlements call this after the terminal write lands.
    Unconfigured hosts, repeated observations, unknown attempts and irrelevant
    outcomes return before hashing request input. Health-affecting failures record
    their shared circuit classification as a departure. Successful provider usage
    can establish bounded cache evidence; only the actual stage's affinity policy
    may also retain placement. Aggregate fairness sampling is independent.

    Scope validation must succeed before any history is written. Optional observer
    errors are logged without exception details and cannot prevent finalization.
    An attempt is marked observed only after the registry call succeeds, allowing
    a later delivery to retry failed observation without duplicating completed work.

    Args:
        registry: Worker-local, tenant-scoped recovery history.
        host: Optional authority for endpoint, model and credential scope.
        entry: Request and frozen route that own the settled attempt.
        attempt_id: Durable attempt identifier with a recorded route depth.
        usage: Provider-reported terminal usage, or None when unavailable.
        failure: Normalized terminal failure, or None for successful completion.
    """
    if host is None:
        return
    with entry.recovery_observation_lock:
        if attempt_id in entry.recovery_recorded_attempts:
            return
        depth = entry.attempt_depths.get(attempt_id)
        if depth is None:
            return
        cause = None if failure is None else health_failure_cause(failure.failure_class)
        if (failure is not None and cause is None) or (failure is None and usage is None):
            return
        observed_at = entry.recovery_observed_at.setdefault(attempt_id, registry.observation_time())
        try:
            key = session_cache_key(entry)
            if key is None:
                return
            deployment = entry.route.deployments[depth]
            if failure is not None and cause is not None:
                if not record_departure(
                    registry,
                    host,
                    entry,
                    deployment,
                    cause,
                    retry_after_seconds=float(failure.retry_after_seconds or 0),
                    key=key,
                    observed_at=observed_at,
                ):
                    return
            elif usage is not None:
                dispatch = deployment.gateway.dispatch
                stage = entry.route.snapshot.stage_for_depth(depth)
                scope = frozen_scope(
                    entry.recovery_bindings, deployment, entry.authorization.organization_id
                )
                if scope is None:
                    return
                registry.record_success(
                    key,
                    deployment.deployment_id,
                    scope,
                    cached_tokens=usage.cached_input_tokens or 0,
                    cache_write_tokens=usage.cache_creation_input_tokens or 0,
                    retention_seconds=deployment.gateway.cache_retention_seconds,
                    sticky_seconds=dispatch.sticky_spill_seconds
                    if dispatch is not None and stage.failover_mode == "maximize_cache_affinity"
                    else None,
                    observed_at=observed_at,
                )
        except Exception:  # noqa: BLE001 - optional observation cannot fail durable settlement.
            _logger.warning("Session recovery outcome observation skipped")
            return
        entry.recovery_recorded_attempts.add(attempt_id)
