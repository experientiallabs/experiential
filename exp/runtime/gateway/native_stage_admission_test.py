"""Native admission consumes session evidence while respecting stage and host gates."""

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest, GatewayUsage
from exp.runtime.gateway.model_plan import model_execution_snapshot
from exp.runtime.gateway.model_plan_test import catalog
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.native_recovery import record_session_outcome, session_cache_key
from exp.runtime.gateway.recovery import RecoveryScope, RecoverySnapshot
from exp.runtime.gateway.recovery_test import Clock
from exp.runtime.gateway.routing import GatewayRoute


class Host:
    """In-memory immutable observation host with explicit credential identity."""

    def scope_for(self, deployment: ExactModelDeployment, organization_id: str) -> RecoveryScope:
        """Freeze a known credential scope per destination model."""
        return RecoveryScope(
            provider=deployment.provider,
            exact_model_id=deployment.exact_model_id,
            endpoint_scope=deployment.connection_sha256,
            region_scope="region",
            credential_scope="credential",
            organization_id=organization_id,
        )

    def snapshot(self) -> RecoverySnapshot:
        """Return no fleet recovery evidence, so a healthy warm fallback stays retained."""
        return RecoverySnapshot(loaded_at=1000)


def test_settlement_records_successful_session_cache_once_and_never_dispatch_only() -> None:
    """The native settlement hook owns evidence, independent of aggregate fairness samples."""
    normalized = catalog()
    auth = _route().snapshot.authorization
    snapshot = model_execution_snapshot(normalized, auth, normalized.pools[0])
    by_id = {d.deployment_id: d for d in normalized.deployments}
    deployments = tuple(by_id[d] for d in snapshot.deployment_ids)
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )
    request = GatewayRequest(
        surface=auth.surface,
        messages=(
            GatewayMessage(role="system", content="prefix"),
            GatewayMessage(role="user", content="turn"),
        ),
        provider_prompt_cache_key="xpl-test",
    )
    entry = InflightRequest(authorization=auth, route=route, request=request, deadline_monotonic=10)
    key = session_cache_key(entry)
    assert key is not None
    from exp.runtime.gateway.recovery import SessionRecoveryRegistry

    registry = SessionRecoveryRegistry(clock=Clock())
    host = Host()
    scope = host.scope_for(route.deployment, auth.organization_id)
    assert (
        registry.choose(
            key, ((route.deployment.deployment_id, scope),), eligible=lambda _: True, snapshot=None
        ).deployment_id
        is None
    )
    entry.attempt_depths["attempt"] = 0
    usage = GatewayUsage(input_tokens=100, output_tokens=1, cached_input_tokens=80)
    record_session_outcome(registry, host, entry, "attempt", usage, None)
    # Unknown declared retention remains no evidence even with a cache read.
    assert (
        registry.choose(
            key, ((route.deployment.deployment_id, scope),), eligible=lambda _: True, snapshot=None
        ).deployment_id
        is None
    )
    assert entry.recovery_recorded_attempts == {"attempt"}
