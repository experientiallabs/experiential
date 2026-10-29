"""Recovery history hashes actual prefix input independently from cache routing hints."""

from dataclasses import dataclass, replace
from typing import Literal
from unittest.mock import patch

import pytest

from exp.common.models.catalog import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment, FailoverMode
from exp.common.models.gateway_chains import ModelExecutionStage
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
    GatewayUsage,
)
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.native_recovery import (
    record_departure,
    record_session_outcome,
    recovery_prefix_digest,
    session_cache_key,
)
from exp.runtime.gateway.recovery import (
    FrozenRecoveryBinding,
    OperationalScope,
    RecoveryLease,
    RecoveryObservation,
    RecoveryScope,
    RecoverySnapshot,
    SessionRecoveryRegistry,
)
from exp.runtime.gateway.recovery_test import Clock, eligible


def request(system: str = "Stable instructions", user: str = "First turn") -> GatewayRequest:
    """Build two requests deliberately sharing the same caller affinity hint."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="system", content=system),
            GatewayMessage(role="user", content=user),
        ),
        prompt_cache_key="same-public-hint",
        provider_prompt_cache_key="xpl-same-namespaced-hint",
    )


def test_actual_prefix_change_with_same_hint_loses_history() -> None:
    """Caller-chosen hints cannot establish that unrelated prompt prefixes match."""
    first, changed = request(), request("Completely different system")
    route = _route()
    entries = [
        InflightRequest(
            authorization=route.snapshot.authorization,
            route=route,
            request=r,
            deadline_monotonic=10,
        )
        for r in (first, changed)
    ]
    assert session_cache_key(entries[0]) != session_cache_key(entries[1])
    key = session_cache_key(entries[0])
    assert key is not None and key.prefix_key == recovery_prefix_digest(first)
    assert recovery_prefix_digest(first) == recovery_prefix_digest(request(user="Different suffix"))
    assert recovery_prefix_digest(first) == recovery_prefix_digest(
        first.model_copy(update={"prompt_cache_key": "changed"})
    )


def test_prefix_tools_and_order_are_part_of_cache_evidence_identity() -> None:
    """Tool changes and reordered system roles invalidate the actual cached prefix."""
    first = request()
    tool = GatewayToolDefinition(name="read", parameters={"type": "object"})
    with_tool = first.model_copy(update={"tools": (tool,)})
    assert recovery_prefix_digest(first) != recovery_prefix_digest(with_tool)
    changed_tool = with_tool.model_copy(
        update={"tools": (tool.model_copy(update={"parameters": {"type": "string"}}),)}
    )
    assert recovery_prefix_digest(with_tool) != recovery_prefix_digest(changed_tool)
    ordered = first.model_copy(
        update={
            "messages": (
                GatewayMessage(role="system", content="a"),
                GatewayMessage(role="developer", content="b"),
                first.messages[-1],
            )
        }
    )
    reordered = ordered.model_copy(
        update={"messages": (ordered.messages[1], ordered.messages[0], ordered.messages[2])}
    )
    assert recovery_prefix_digest(ordered) != recovery_prefix_digest(reordered)


@dataclass
class RecoveryHostFake:
    """Expose deterministic, synthetic scope metadata or a scripted host fault."""

    fault: Literal["raise", "provider", "exact_model_id", "organization_id"] | None = None
    calls: int = 0

    def scope_for(self, deployment: ExactModelDeployment, organization_id: str) -> RecoveryScope:
        """Resolve a valid scope unless the fixture requests a fault.

        Args:
            deployment: Actual authorized deployment under observation.
            organization_id: Tenant whose request reported the outcome.

        Returns:
            Matching metadata, or a mismatched field to exercise validation.

        Raises:
            RuntimeError: The fixture requests a host callback failure.
        """
        self.calls += 1
        if self.fault == "raise":
            raise RuntimeError("synthetic-private-detail")
        scope = RecoveryScope(
            provider=deployment.provider,
            exact_model_id=deployment.exact_model_id,
            organization_id=organization_id,
            endpoint_scope=deployment.deployment_id,
            region_scope="test-region",
            credential_scope="test-credential-scope",
        )
        if self.fault is not None:
            return scope.model_copy(update={self.fault: "synthetic-private-detail"})
        return scope

    def observe_scope(self, scope: OperationalScope) -> None:
        """Accept detached operational demand without resolving credentials."""
        self.calls += 1

    def attempt_started(self, attempt_id: str, scope: OperationalScope) -> None:
        """Accept an actual attempt's detached topology."""
        self.calls += 1

    def snapshot(self) -> RecoverySnapshot:
        """Return an empty host view without provider or credential access."""
        return RecoverySnapshot(loaded_at=1000)


def recovery_entry() -> InflightRequest:
    """Build a conversational entry with real bounded cache-retention metadata."""
    route = _route()
    deployments = tuple(
        deployment.model_copy(
            update={
                "gateway": deployment.gateway.model_copy(
                    update={
                        "cache_retention_seconds": 100,
                        "dispatch": GatewayRungDispatchPolicy(sticky_spill_seconds=100),
                    }
                )
            }
        )
        for deployment in route.deployments
    )
    route = route.model_copy(
        update={
            "deployment": deployments[0],
            "fallback_deployments": deployments[1:],
            "snapshot": route.snapshot.model_copy(
                update={"failover_mode": "maximize_cache_affinity"}
            ),
        }
    )
    return InflightRequest(
        authorization=route.snapshot.authorization,
        route=route,
        request=request(),
        deadline_monotonic=10,
        attempt_depths={"attempt": 0, "departure": 0, "fallback": 1},
        recovery_bindings={
            d.deployment_id: FrozenRecoveryBinding(
                d.deployment_id,
                d.connection_sha256,
                "https://test.invalid",
                d.provider_model,
                RecoveryHostFake().scope_for(d, route.snapshot.authorization.organization_id),
            )
            for d in route.deployments
        },
    )


@pytest.mark.parametrize("failure_class", [None, GatewayFailureClass.TRANSPORT])
def test_no_host_does_not_derive_recovery_keys(failure_class: GatewayFailureClass | None) -> None:
    """Unconfigured recovery hooks do no prompt hashing or history work."""
    registry, entry = SessionRecoveryRegistry(), recovery_entry()
    failure = (
        None
        if failure_class is None
        else GatewayFailure(failure_class=failure_class, safe_message="test failure")
    )
    with patch("exp.runtime.gateway.native_recovery.session_cache_key") as derive:
        record_session_outcome(
            registry,
            None,
            entry,
            "attempt",
            GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=50),
            failure,
        )
        record_departure(registry, None, entry, entry.route.deployment, "local_capacity")
    assert derive.call_count == 0
    assert not entry.recovery_recorded_attempts


@pytest.mark.parametrize("failure_class", [None, GatewayFailureClass.TRANSPORT])
def test_settled_outcome_hashes_once_and_duplicate_does_no_work(
    failure_class: GatewayFailureClass | None,
) -> None:
    """Success and failure each hash once; a repeated settlement never rehashes."""
    registry, host, entry = SessionRecoveryRegistry(), RecoveryHostFake(), recovery_entry()
    failure = (
        None
        if failure_class is None
        else GatewayFailure(failure_class=failure_class, safe_message="test failure")
    )
    with patch(
        "exp.runtime.gateway.native_recovery.session_cache_key", wraps=session_cache_key
    ) as derive:
        for _ in range(2):
            record_session_outcome(
                registry,
                host,
                entry,
                "attempt",
                GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=50),
                failure,
            )
    assert derive.call_count == 1
    assert host.calls == 0
    assert entry.recovery_recorded_attempts == {"attempt"}


@pytest.mark.parametrize("case", ["unknown_attempt", "no_usage", "caller_failure"])
def test_irrelevant_outcome_does_not_derive_recovery_keys(case: str) -> None:
    """Unknown attempts, absent usage and caller errors cannot affect recovery."""
    registry, host, entry = SessionRecoveryRegistry(), RecoveryHostFake(), recovery_entry()
    failure = (
        GatewayFailure(failure_class=GatewayFailureClass.INVALID_REQUEST, safe_message="bad input")
        if case == "caller_failure"
        else None
    )
    with patch("exp.runtime.gateway.native_recovery.session_cache_key") as derive:
        record_session_outcome(
            registry,
            host,
            entry,
            "missing" if case == "unknown_attempt" else "attempt",
            None
            if case == "no_usage"
            else GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=50),
            failure,
        )
    assert derive.call_count == 0
    assert host.calls == 0
    assert not entry.recovery_recorded_attempts


@pytest.mark.parametrize("fault", ["raise", "provider", "exact_model_id", "organization_id"])
@pytest.mark.parametrize("failure_class", [None, GatewayFailureClass.TRANSPORT])
def test_invalid_host_scope_records_nothing_and_remains_retryable(
    fault: Literal["raise", "provider", "exact_model_id", "organization_id"],
    failure_class: GatewayFailureClass | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Host failures never create cache evidence, departures, or completion markers."""
    registry, host, entry = SessionRecoveryRegistry(), RecoveryHostFake(fault), recovery_entry()
    deployment_id = entry.route.deployment.deployment_id
    valid = entry.recovery_bindings[deployment_id]
    entry.recovery_bindings[deployment_id] = replace(
        valid,
        scope=valid.scope.model_copy(
            update={"provider" if fault == "raise" else fault: "synthetic-private-detail"}
        ),
    )
    key = session_cache_key(entry)
    assert key is not None
    failure = (
        None
        if failure_class is None
        else GatewayFailure(failure_class=failure_class, safe_message="test failure")
    )
    usage = GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=50)
    record_session_outcome(registry, host, entry, "attempt", usage, failure)
    assert not registry._sessions  # noqa: SLF001 - invalid scopes must create no evidence.
    assert not registry._negative  # noqa: SLF001 - invalid scopes must create no negatives.
    assert not entry.recovery_recorded_attempts
    assert "synthetic-private-detail" not in caplog.text
    assert caplog.records and all(record.exc_info is None for record in caplog.records)
    host.fault = None
    entry.recovery_bindings[deployment_id] = valid
    record_session_outcome(registry, host, entry, "attempt", usage, failure)
    assert entry.recovery_recorded_attempts == {"attempt"}
    assert key in registry._sessions  # noqa: SLF001 - only the valid retry may write history.


@pytest.mark.parametrize("operation", ["record_success", "depart"])
def test_registry_write_failure_does_not_mark_the_attempt(operation: str) -> None:
    """A failed optional registry write can be retried by a later settlement."""
    registry, host, entry = SessionRecoveryRegistry(), RecoveryHostFake(), recovery_entry()
    failure = (
        GatewayFailure(failure_class=GatewayFailureClass.TRANSPORT, safe_message="test failure")
        if operation == "depart"
        else None
    )
    with patch.object(registry, operation, side_effect=RuntimeError("synthetic-private-detail")):
        record_session_outcome(
            registry,
            host,
            entry,
            "attempt",
            GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=50),
            failure,
        )
    assert not entry.recovery_recorded_attempts
    record_session_outcome(
        registry,
        host,
        entry,
        "attempt",
        GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=50),
        failure,
    )
    assert entry.recovery_recorded_attempts == {"attempt"}


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize(
    "mode", ["maximize_availability", "maximize_cache", "maximize_cache_affinity"]
)
def test_cache_evidence_retains_placement_only_for_actual_affinity_stage(
    mode: FailoverMode, staged: bool
) -> None:
    """Reported cache use never changes non-affinity placement, including child stages."""
    registry, host, entry = SessionRecoveryRegistry(), RecoveryHostFake(), recovery_entry()
    snapshot = entry.route.snapshot
    if staged:
        root = ModelExecutionStage(
            stage_index=0,
            exact_model_id=snapshot.exact_model_id,
            pool_id=snapshot.pool_id,
            deployment_ids=snapshot.deployment_ids[:1],
            failover_mode="maximize_cache_affinity",
        )
        child = ModelExecutionStage(
            stage_index=1,
            exact_model_id=snapshot.exact_model_id,
            pool_id="child-pool",
            deployment_ids=snapshot.deployment_ids[1:],
            failover_mode=mode,
        )
        snapshot = snapshot.model_copy(update={"model_stages": (root, child)})
    else:
        snapshot = snapshot.model_copy(update={"failover_mode": mode})
    entry.route = entry.route.model_copy(update={"snapshot": snapshot})
    record_session_outcome(
        registry,
        host,
        entry,
        "fallback",
        GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=50),
        None,
    )
    key = session_cache_key(entry)
    assert key is not None
    history = registry._sessions[key]  # noqa: SLF001 - separate evidence from elective placement.
    assert "two" in history.evidence
    candidates = tuple(
        (deployment.deployment_id, host.scope_for(deployment, entry.authorization.organization_id))
        for deployment in entry.route.deployments
    )
    decision = registry.choose(key, candidates, eligible=eligible, snapshot=None)
    if mode == "maximize_cache_affinity":
        assert (decision.deployment_id, decision.reason) == ("two", "retained_warm_fallback")
    else:
        assert decision.deployment_id is None
        assert decision.reason == "normal_selection"


def test_newer_malformed_outcome_blocks_older_shared_healthy_observation() -> None:
    """A different session's malformed response supersedes older transport recovery."""
    clock, host, entry = Clock(), RecoveryHostFake(), recovery_entry()
    registry = SessionRecoveryRegistry(clock=clock)
    key = session_cache_key(entry)
    assert key is not None
    usage = GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=50)
    record_session_outcome(registry, host, entry, "attempt", usage, None)
    record_session_outcome(
        registry,
        host,
        entry,
        "departure",
        None,
        GatewayFailure(failure_class=GatewayFailureClass.TRANSPORT, safe_message="test failure"),
    )
    record_session_outcome(registry, host, entry, "fallback", usage, None)
    candidates = tuple(
        (deployment.deployment_id, host.scope_for(deployment, entry.authorization.organization_id))
        for deployment in entry.route.deployments
    )
    clock.now += 10
    shared = RecoverySnapshot(
        loaded_at=clock.now,
        observations=(
            RecoveryObservation(
                scope=candidates[0][1], cause="transport", observed_at=clock.now, healthy=True
            ),
        ),
        leases=(
            RecoveryLease(lease_id="trial", scope=candidates[0][1], expires_at=clock.now + 30),
        ),
    )
    clock.now += 1
    other = recovery_entry()
    other.authorization = other.authorization.model_copy(update={"organization_id": "other-org"})
    other.recovery_bindings = {
        name: replace(
            binding, scope=binding.scope.model_copy(update={"organization_id": "other-org"})
        )
        for name, binding in other.recovery_bindings.items()
    }
    record_session_outcome(
        registry,
        host,
        other,
        "attempt",
        None,
        GatewayFailure(
            failure_class=GatewayFailureClass.MALFORMED_RESPONSE, safe_message="malformed output"
        ),
    )
    clock.now += 6
    decision = registry.choose(key, candidates, eligible=eligible, snapshot=shared)
    assert (decision.deployment_id, decision.reason, decision.trial) == (
        "two",
        "retained_warm_fallback",
        False,
    )
    fresh = shared.model_copy(
        update={
            "loaded_at": clock.now,
            "observations": (shared.observations[0].model_copy(update={"observed_at": clock.now}),),
        }
    )
    assert (
        registry.choose(key, candidates, eligible=eligible, snapshot=fresh).deployment_id == "one"
    )
