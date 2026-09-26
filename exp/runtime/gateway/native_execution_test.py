"""Tests for the native waterfall's candidate policy and wire building."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.models.catalog import GatewayDeploymentMetadata
from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment, FailoverMode
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.native_execution import (
    THROTTLE_BACKOFF,
    THROTTLE_FAILOVER_COLD,
    THROTTLE_SURFACED_CACHE_PRESERVING,
    claim_route_from,
    deployment_wire_entry,
    dispatch_disclosure,
    next_route_candidate,
    reorder_route_deployments,
    select_route_deployments,
    throttle_disposition,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile

_KEYS: tuple[DeploymentHealthKey, ...] = (
    ("catalog" + "0" * 57, "deployment-a", "connection-a"),
    ("catalog" + "0" * 57, "deployment-b", "connection-b"),
)


def _deployment(deployment_id: str) -> ExactModelDeployment:
    """Build one exact deployment for route-narrowing tests."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai-compatible",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(),
    )


def _route() -> GatewayRoute:
    """Build a three-rung exact-model route."""
    deployments = tuple(_deployment(name) for name in ("one", "two", "three"))
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _retryable() -> GatewayFailure:
    """Build one failure the executor may redial on the same deployment."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="provider service failed; retry after a short delay",
        retryable_same_deployment=True,
        failover_eligible=True,
    )


def _failover_only() -> GatewayFailure:
    """Build one failure that advances routes but never redials."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED,
        safe_message="provider throttled the request",
        failover_eligible=True,
    )


def test_route_narrowing_and_reordering_keep_the_reasoning_pin() -> None:
    """A narrowed or reordered pinned route still knows which rung sealed the reasoning.

    The pin survives dead-rung narrowing and affinity reordering so every
    surviving non-issuing rung still requires the strip and is still recorded
    as ``reasoning_continuation_failover``, even when narrowing removed the
    issuing rung itself.
    """
    pinned = _route().model_copy(
        update={
            "route_reason": "reasoning_continuation",
            "reasoning_pinned_deployment_id": "one",
        }
    )
    narrowed = select_route_deployments(pinned, (1, 2))
    assert narrowed.reasoning_pinned_deployment_id == "one"
    assert all(narrowed.requires_reasoning_strip(item) for item in narrowed.deployments)
    assert narrowed.attempt_route_reason(narrowed.deployment) == "reasoning_continuation_failover"
    reordered = reorder_route_deployments(pinned, (2, 0, 1))
    assert reordered.reasoning_pinned_deployment_id == "one"
    assert reordered.requires_reasoning_strip(reordered.deployment) is True
    assert reordered.requires_reasoning_strip(reordered.deployments[1]) is False
    assert reordered.attempt_route_reason(reordered.deployments[1]) == "reasoning_continuation"


def test_select_route_deployments_rebinds_the_execution_snapshot() -> None:
    """Accounting and wire order name exactly the compatible deployment subset."""
    selected = select_route_deployments(_route(), (1, 2))

    assert tuple(item.deployment_id for item in selected.deployments) == ("two", "three")
    assert selected.snapshot.deployment_ids == ("two", "three")


@pytest.mark.parametrize("indexes", ((), (1, 0), (0, 0), (3,)))
def test_select_route_deployments_rejects_invalid_indexes(indexes: tuple[int, ...]) -> None:
    """An invalid compatibility selection cannot corrupt waterfall accounting."""
    with pytest.raises(ValueError):
        select_route_deployments(_route(), indexes)


def test_admission_dead_failure_mirrors_runtime_circuit_classes() -> None:
    """A dead-at-admission rung feeds the circuit like the runtime failure it mirrors."""
    from exp.runtime.gateway.native_execution import _admission_dead_failure
    from exp.runtime.models import ModelConnectionError
    from exp.runtime.models.credentials import MissingModelCredentialError, ModelCredentialError
    from exp.runtime.models.providers.errors import ProviderCapabilityError

    # A missing credential opens the circuit at once, like a runtime 401.
    for credential_error in (
        ModelCredentialError("no key"),
        MissingModelCredentialError("TEST_KEY", connection_id="conn"),
    ):
        failure = _admission_dead_failure(credential_error)
        assert failure.failure_class == GatewayFailureClass.PROVIDER_AUTHENTICATION

    # A connection or capability drift is operational, like a runtime transport failure.
    assert (
        _admission_dead_failure(ModelConnectionError("drifted")).failure_class
        == GatewayFailureClass.TRANSPORT
    )
    assert (
        _admission_dead_failure(ProviderCapabilityError(capability="reasoning")).failure_class
        == GatewayFailureClass.TRANSPORT
    )


def test_claim_ladder_prefers_healthy_then_probe_then_forced() -> None:
    """A suppressed route still admits through bounded probe and forced claims."""
    health = DeploymentHealthRegistry(failure_threshold=1, open_seconds=60.0)
    assert claim_route_from(health, _KEYS, 0) == 0
    hard = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
        safe_message="provider authentication failed",
    )
    health.failed(_KEYS[0], hard)
    health.failed(_KEYS[1], hard)
    # Both circuits open: the first claim grants the bounded half-open probe.
    assert claim_route_from(health, _KEYS, 0) == 0
    # Probes taken on both routes: the forced claim still admits dispatch.
    assert claim_route_from(health, _KEYS, 1) == 1
    assert claim_route_from(health, _KEYS, 0) == 0


def test_retryable_failure_redials_within_the_per_deployment_cap() -> None:
    """A retryable failure redials the same deployment while its count allows."""
    health = DeploymentHealthRegistry()
    candidate = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=_retryable(),
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
    )
    assert candidate == 0
    # The per-deployment cap reached: the same failure fails over instead.
    candidate = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=_retryable(),
        current_depth=0,
        attempt_counts=[2, 0],
        total_attempts=2,
        refusal_failover=False,
    )
    assert candidate == 1


def test_total_attempt_cap_ends_the_ladder() -> None:
    """No candidate exists once the hard total dispatch cap is reached."""
    health = DeploymentHealthRegistry()
    candidate = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=_retryable(),
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=8,
        refusal_failover=False,
    )
    assert candidate is None


def test_failover_only_failure_skips_the_redial() -> None:
    """A failover-eligible, non-retryable failure advances immediately."""
    health = DeploymentHealthRegistry()
    candidate = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=_failover_only(),
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
    )
    assert candidate == 1
    exhausted = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=_failover_only(),
        current_depth=1,
        attempt_counts=[1, 1],
        total_attempts=2,
        refusal_failover=False,
    )
    assert exhausted is None


def test_refusal_advances_only_with_the_revision_opt_in() -> None:
    """A typed refusal fails over exactly when the alias revision enables it."""
    health = DeploymentHealthRegistry()
    refusal = GatewayFailure(
        failure_class=GatewayFailureClass.REFUSAL,
        safe_message="provider refused the request",
    )
    withheld = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=refusal,
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=True,
    )
    assert withheld == 1
    declined = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=refusal,
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
    )
    assert declined is None


def test_caller_invalid_request_never_advances() -> None:
    """A caller-owned rejection neither redials nor fails over."""
    health = DeploymentHealthRegistry()
    invalid = GatewayFailure(
        failure_class=GatewayFailureClass.INVALID_REQUEST,
        safe_message="provider rejected the request",
    )
    candidate = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=invalid,
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
    )
    assert candidate is None


def test_maximize_cache_returns_a_throttle_without_failing_over() -> None:
    """maximize_cache surfaces a throttle to the caller instead of failing over cold.

    A same-request redial is infeasible -- the 429 records the rung's throttle
    window before the next candidate is chosen -- so the cache-preserving move is
    to stop the ladder and let the caller retry the warm rung after backoff. The
    default policy still fails over on the same throttle.
    """
    health = DeploymentHealthRegistry()
    # Under maximize_cache a throttle ends the ladder (no cold failover)...
    assert (
        next_route_candidate(
            health=health,
            keys=_KEYS,
            failure=_failover_only(),
            current_depth=0,
            attempt_counts=[1, 0],
            total_attempts=1,
            refusal_failover=False,
            failover_mode="maximize_cache",
        )
        is None
    )
    # ...while the default maximize_availability policy fails over to the next rung.
    assert (
        next_route_candidate(
            health=health,
            keys=_KEYS,
            failure=_failover_only(),
            current_depth=0,
            attempt_counts=[1, 0],
            total_attempts=1,
            refusal_failover=False,
            failover_mode="maximize_availability",
        )
        == 1
    )


def test_maximize_cache_affinity_fails_over_on_a_throttle() -> None:
    """The affinity policy keeps availability-style failover on a throttle.

    Its cache story is the deterministic rendezvous ALTERNATE: spilling builds
    warm cache at the same alternate every time, so waiting out a backoff
    window (the maximize_cache move) would only add latency.
    """
    health = DeploymentHealthRegistry()
    assert (
        next_route_candidate(
            health=health,
            keys=_KEYS,
            failure=_failover_only(),
            current_depth=0,
            attempt_counts=[1, 0],
            total_attempts=1,
            refusal_failover=False,
            failover_mode="maximize_cache_affinity",
        )
        == 1
    )


def test_maximize_cache_does_not_redial_a_stalled_timeout_lane() -> None:
    """A stalled-lane timeout fails over even under maximize_cache.

    The engine marks first-byte and header-phase stalls as ``TIMEOUT`` with
    ``retryable_same_deployment=False`` precisely so a dead lane advances instead
    of burning another fail-fast window. maximize_cache must respect that signal:
    a lane that accepted the connection but never answered has no warm cache to
    preserve, so it is not redialed.
    """
    health = DeploymentHealthRegistry()
    stalled = GatewayFailure(
        failure_class=GatewayFailureClass.TIMEOUT,
        safe_message="provider did not send the first token in time",
        retryable_same_deployment=False,
        failover_eligible=True,
    )
    candidate = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=stalled,
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
        failover_mode="maximize_cache",
    )
    assert candidate == 1


def test_maximize_cache_still_fails_over_on_operational_deadness() -> None:
    """A dead rung (auth failure) fails over even under maximize_cache."""
    health = DeploymentHealthRegistry()
    dead = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
        safe_message="provider authentication failed",
        failover_eligible=True,
    )
    candidate = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=dead,
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
        failover_mode="maximize_cache",
    )
    # No cache to preserve on a rung that cannot authenticate -> advance.
    assert candidate == 1


def test_dialectless_provider_alias_is_excluded_not_a_startup_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dialectless alias is excluded (UNAVAILABLE); the worker serves the rest.

    Every currently supported provider implements a native dialect, so the
    "no native dialect implementation" branch is exercised by patching a real
    client's ``gateway_wire_profile`` back to the unimplemented base behavior.
    Under the per-alias fail-safe build such an alias no longer aborts worker
    startup: it is marked UNAVAILABLE and the build serves every other alias, so
    the fleet-level ``native_serving_blockers`` diagnostic comes back clean.
    """
    from exp.common.models import (
        GatewayDeploymentCapabilities,
        GatewayTokenPrices,
        ModelCapabilities,
    )
    from exp.runtime.gateway.catalog_authority import (
        ConnectionConfig,
        upsert_connection,
        upsert_singleton_deployment,
    )
    from exp.runtime.gateway.lifecycle import load_gateway_components
    from exp.runtime.gateway.lifecycle_test import _configured_gateway
    from exp.runtime.gateway.native_execution import native_serving_blockers
    from exp.runtime.models.providers.errors import ProviderCapabilityError
    from exp.runtime.models.providers.gemini import GeminiClient

    def _no_native_dialect(self: GeminiClient) -> object:
        del self
        raise ProviderCapabilityError(capability="native_data_plane")

    monkeypatch.setattr(GeminiClient, "gateway_wire_profile", _no_native_dialect)

    manager, _raw_key = _configured_gateway(tmp_path)
    upsert_connection(
        tmp_path,
        name="gemini-main",
        connection=ConnectionConfig(provider="gemini", api_key_env="TEST_GEMINI_KEY"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="escalated",
        connection_name="gemini-main",
        provider_model="gemini-model-exact",
        exact_model_id="gemini-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="escalated",
        alias_name="escalated",
        revision_id="revision-escalated",
        pool_id="escalated",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="escalated")
    components = load_gateway_components(
        tmp_path,
        environment={
            "TEST_PROVIDER_KEY": "provider-secret-canary",
            "TEST_GEMINI_KEY": "gemini-secret-canary",
        },
    )
    # The dialectless alias is excluded at build, so the served generation has no
    # blockers and the worker binds.
    assert native_serving_blockers(components) == ()
    unavailable = dict(components.unavailable_aliases)
    assert "escalated" in unavailable
    assert "gemini" in unavailable["escalated"]
    assert "native dialect" in unavailable["escalated"]
    # The other granted alias still loads and serves.
    served = {alias for alias, _revision, _digest in components.reloader.state.authorities}
    assert "coding" in served
    assert "escalated" not in served


def test_reasoning_wire_contract_conflict_excludes_the_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alias whose reasoning metadata the provider profile cannot carry is excluded.

    The build rejects the invalid reasoning wire contract per-alias (UNAVAILABLE)
    instead of aborting the worker, so a servable alias still serves.
    """
    from dataclasses import replace

    from exp.common.models import (
        GatewayDeploymentCapabilities,
        GatewayTokenPrices,
        ModelCapabilities,
    )
    from exp.runtime.gateway.catalog_authority import (
        ConnectionConfig,
        upsert_connection,
        upsert_singleton_deployment,
    )
    from exp.runtime.gateway.lifecycle import load_gateway_components
    from exp.runtime.gateway.lifecycle_test import _configured_gateway
    from exp.runtime.gateway.native_execution import native_serving_blockers
    from exp.runtime.models.providers.gemini import GeminiClient

    original_wire_profile = GeminiClient.gateway_wire_profile

    def _profile_without_reasoning(self: GeminiClient) -> GatewayWireProfile:
        """Return a valid profile that contradicts the authored reasoning contract."""
        return replace(
            original_wire_profile(self),
            supports_reasoning=False,
            reasoning_wire_format="none",
        )

    monkeypatch.setattr(GeminiClient, "gateway_wire_profile", _profile_without_reasoning)

    manager, _raw_key = _configured_gateway(tmp_path)
    upsert_connection(
        tmp_path,
        name="gemini-main",
        connection=ConnectionConfig(provider="gemini", api_key_env="TEST_GEMINI_KEY"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="reasoning",
        connection_name="gemini-main",
        provider_model="gemini-model-exact",
        exact_model_id="gemini-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(supports_reasoning=True),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supported_reasoning_efforts=("medium",),
            reasoning_default_effort="medium",
            reasoning_effort_required=True,
        ),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="reasoning",
        alias_name="reasoning",
        revision_id="revision-reasoning",
        pool_id="reasoning",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="reasoning")
    components = load_gateway_components(
        tmp_path,
        environment={
            "TEST_PROVIDER_KEY": "provider-secret-canary",
            "TEST_GEMINI_KEY": "gemini-secret-canary",
        },
    )

    # Excluded at build (UNAVAILABLE), not a fleet-wide startup blocker.
    assert native_serving_blockers(components) == ()
    unavailable = dict(components.unavailable_aliases)
    assert "reasoning" in unavailable
    assert "invalid reasoning wire contract" in unavailable["reasoning"]
    served = {alias for alias, _revision, _digest in components.reloader.state.authorities}
    assert "coding" in served
    assert "reasoning" not in served


def test_route_reorder_permutes_dispatch_order_and_rejects_non_permutations() -> None:
    """Reordering changes dispatch order over exactly the same deployments."""
    from exp.runtime.gateway.native_execution import reorder_route_deployments

    route = _route()
    unchanged = reorder_route_deployments(route, (0, 1, 2))
    assert unchanged is route
    rotated = reorder_route_deployments(route, (2, 0, 1))
    assert rotated.deployment.deployment_id == "three"
    assert tuple(item.deployment_id for item in rotated.fallback_deployments) == ("one", "two")
    assert rotated.snapshot.deployment_ids == ("three", "one", "two")
    assert rotated.route_reason == route.route_reason
    with pytest.raises(ValueError, match="permutation"):
        reorder_route_deployments(route, (0, 1))
    with pytest.raises(ValueError, match="permutation"):
        reorder_route_deployments(route, (0, 1, 1))


def test_cache_marker_predicate_sees_every_marker_carrier() -> None:
    """Each cache-marker position makes the request marker-carrying."""
    from exp.common.models.model import ToolCall
    from exp.runtime.gateway.contracts import GatewayMessage, GatewayToolDefinition
    from exp.runtime.gateway.native_execution import request_carries_cache_markers

    def request(**updates: object) -> GatewayRequest:
        base = GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=(GatewayMessage(role="user", content="hi"),),
        )
        return base.model_copy(update=updates)

    assert request_carries_cache_markers(request()) is False
    assert request_carries_cache_markers(request(provider_cache_control={"type": "ephemeral"}))
    assert request_carries_cache_markers(
        request(
            tools=(
                GatewayToolDefinition(
                    name="bash",
                    parameters={"type": "object"},
                    cache_control={"type": "ephemeral"},
                ),
            )
        )
    )
    marked_text = GatewayMessage(
        role="user",
        content="hi",
        provider_text_blocks=(
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
        ),
    )
    assert request_carries_cache_markers(request(messages=(marked_text,)))
    marked_tool_result = GatewayMessage(
        role="tool",
        content="ok",
        tool_call_id="call-1",
        cache_control={"type": "ephemeral"},
    )
    assert request_carries_cache_markers(request(messages=(marked_tool_result,)))
    marked_call = GatewayMessage(
        role="assistant",
        tool_calls=(
            ToolCall(
                call_id="call-1",
                name="bash",
                arguments={},
                cache_control={"type": "ephemeral"},
            ),
        ),
    )
    assert request_carries_cache_markers(request(messages=(marked_call,)))


def test_wire_entry_carries_emulated_stop_sequences_for_the_data_plane() -> None:
    """A Responses rung's entry names the caller's exact stop sequences; others carry none."""
    route = _route()
    profile = GatewayWireProfile(dialect="openai_responses", url="https://provider.test")

    entry = deployment_wire_entry(
        route,
        route.deployment,
        profile,
        {"model": "gpt-5.6-luna"},
        stop_sequences=("</severity>", "</block>"),
    )
    assert entry["stop_sequences"] == ["</severity>", "</block>"]
    assert entry["dialect"] == "openai_responses"

    default = deployment_wire_entry(route, route.deployment, profile, {"model": "gpt-5.6-luna"})
    assert default["stop_sequences"] == []


def test_wire_entry_carries_the_rungs_failover_only_on_tokens() -> None:
    """A failover-only rung's entry lists its tokens; an unrestricted rung carries null."""
    route = _route()
    profile = GatewayWireProfile(dialect="openai_responses", url="https://provider.test")
    assert deployment_wire_entry(route, route.deployment, profile, {})["failover_only_on"] is None
    restricted = route.deployment.model_copy(
        update={
            "gateway": route.deployment.gateway.model_copy(
                update={
                    "capabilities": route.deployment.gateway.capabilities.model_copy(
                        update={"failover_only_on": ("refusal:cyber_policy", "throttled")}
                    )
                }
            )
        }
    )
    assert deployment_wire_entry(route, restricted, profile, {})["failover_only_on"] == [
        "refusal:cyber_policy",
        "throttled",
    ]


def test_wire_entry_names_customer_managed_billing_for_the_data_plane() -> None:
    """A BYOK rung's entry says so, so the data plane re-owns credential failures."""
    route = _route()
    byok = GatewayWireProfile(
        dialect="openai_responses", url="https://provider.test", billing_customer_managed=True
    )
    house = GatewayWireProfile(dialect="openai_responses", url="https://provider.test")
    assert deployment_wire_entry(route, route.deployment, byok, {})["billing_customer_managed"]
    assert not deployment_wire_entry(route, route.deployment, house, {})["billing_customer_managed"]


@pytest.mark.parametrize(
    ("failover_mode", "expected"),
    (
        ("maximize_availability", 1),
        ("maximize_cache", None),
        ("maximize_cache_affinity", 1),
    ),
)
def test_no_threshold_keeps_each_modes_own_throttle_rule(
    failover_mode: FailoverMode,
    expected: int | None,
) -> None:
    """Without an authored threshold every mode decides a throttle as before.

    The observed cached fraction is deliberately high here: with no threshold
    it must be ignored, so an unauthored pool sees zero behavior change.
    """
    candidate = next_route_candidate(
        health=DeploymentHealthRegistry(),
        keys=_KEYS,
        failure=_failover_only(),
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
        failover_mode=failover_mode,
        throttle_cache_threshold=None,
        cached_fraction=0.95,
    )
    assert candidate == expected


@pytest.mark.parametrize(
    ("cached_fraction", "threshold", "expected"),
    (
        # Above the floor: the warm cache is worth waiting for.
        (0.8, 0.5, None),
        # Below it: fail over cold to the next claimable rung.
        (0.2, 0.5, 1),
        # Exactly at the floor surfaces (at-or-above).
        (0.5, 0.5, None),
        # 0.0 always surfaces, even with no cache evidence (maximize_cache).
        (0.0, 0.0, None),
        # A fresh organization (no evidence) fails over under any positive floor.
        (0.0, 0.5, 1),
        (0.0, 0.01, 1),
        # 1.0 surfaces only a fully cached prompt.
        (0.99, 1.0, 1),
        (1.0, 1.0, None),
    ),
)
def test_threshold_surfaces_a_throttle_only_when_the_cache_at_stake_meets_it(
    cached_fraction: float,
    threshold: float,
    expected: int | None,
) -> None:
    """An authored threshold decides a throttle by the observed cached fraction."""
    candidate = next_route_candidate(
        health=DeploymentHealthRegistry(),
        keys=_KEYS,
        failure=_failover_only(),
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
        throttle_cache_threshold=threshold,
        cached_fraction=cached_fraction,
    )
    assert candidate == expected


@pytest.mark.parametrize(
    ("failover_mode", "cached_fraction", "expected"),
    (
        # Availability would fail over; the warm cache says wait.
        ("maximize_availability", 0.9, None),
        # maximize_cache would surface; the cold cache says advance.
        ("maximize_cache", 0.1, 1),
        # Affinity would fail over; the warm cache says wait.
        ("maximize_cache_affinity", 0.9, None),
    ),
)
def test_threshold_is_the_authoritative_throttle_control_under_every_mode(
    failover_mode: FailoverMode,
    cached_fraction: float,
    expected: int | None,
) -> None:
    """When a threshold is authored the mode's fixed throttle rule no longer applies."""
    candidate = next_route_candidate(
        health=DeploymentHealthRegistry(),
        keys=_KEYS,
        failure=_failover_only(),
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
        failover_mode=failover_mode,
        throttle_cache_threshold=0.5,
        cached_fraction=cached_fraction,
    )
    assert candidate == expected


def test_threshold_failover_still_needs_a_claimable_later_rung() -> None:
    """A cold failover advances only to a rung the health registry will grant."""
    health = DeploymentHealthRegistry()
    health.failed(
        _KEYS[1],
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message="provider throttled the request",
            retry_after_seconds=30,
        ),
    )
    candidate = next_route_candidate(
        health=health,
        keys=_KEYS,
        failure=_failover_only(),
        current_depth=0,
        attempt_counts=[1, 0],
        total_attempts=1,
        refusal_failover=False,
        throttle_cache_threshold=0.5,
        cached_fraction=0.0,
    )
    # The only later rung sits inside its own throttle window: exhausted.
    assert candidate is None


def test_threshold_leaves_every_non_throttle_class_alone() -> None:
    """The threshold reads only against a throttle; other classes keep their rules.

    A high cached fraction that would surface a throttle must not strand a
    dead rung, and a low one must not fail over a caller-owned rejection.
    """
    health = DeploymentHealthRegistry()

    def decide(failure: GatewayFailure, cached_fraction: float) -> int | None:
        """Run one decision under maximize_cache with an authored threshold."""
        return next_route_candidate(
            health=health,
            keys=_KEYS,
            failure=failure,
            current_depth=0,
            attempt_counts=[1, 0],
            total_attempts=1,
            refusal_failover=False,
            failover_mode="maximize_cache",
            throttle_cache_threshold=0.5,
            cached_fraction=cached_fraction,
        )

    dead = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
        safe_message="provider authentication failed",
        failover_eligible=True,
    )
    assert decide(dead, 0.9) == 1
    stalled = GatewayFailure(
        failure_class=GatewayFailureClass.TIMEOUT,
        safe_message="provider did not send the first token in time",
        retryable_same_deployment=False,
        failover_eligible=True,
    )
    assert decide(stalled, 0.9) == 1
    invalid = GatewayFailure(
        failure_class=GatewayFailureClass.INVALID_REQUEST,
        safe_message="provider rejected the request",
    )
    assert decide(invalid, 0.0) is None
    # A retryable failure still redials the warm rung ahead of any failover.
    assert decide(_retryable(), 0.0) == 0


def test_throttle_disposition_names_each_branch_and_only_those() -> None:
    """The pure disposition is the single source for the decision and its disclosure."""
    throttle = _failover_only()
    assert (
        throttle_disposition(throttle, throttle_cache_threshold=None, cached_fraction=1.0) is None
    )
    assert (
        throttle_disposition(_retryable(), throttle_cache_threshold=0.5, cached_fraction=1.0)
        is None
    )
    assert (
        throttle_disposition(throttle, throttle_cache_threshold=0.5, cached_fraction=0.5)
        == THROTTLE_SURFACED_CACHE_PRESERVING
        == "throttle_surfaced_cache_preserving"
    )
    assert (
        throttle_disposition(throttle, throttle_cache_threshold=0.5, cached_fraction=0.49)
        == THROTTLE_FAILOVER_COLD
        == "throttle_failover_cold"
    )


def test_wire_entry_carries_the_tool_call_serialization_flag() -> None:
    """A rung emulating parallel_tool_calls=false tells the data plane to serialize."""
    route = _route()
    profile = GatewayWireProfile(dialect="gemini_generate_content", url="https://provider.test")
    assert deployment_wire_entry(route, route.deployment, profile, {}, serialize_tool_calls=True)[
        "serialize_tool_calls"
    ]
    assert not deployment_wire_entry(route, route.deployment, profile, {})["serialize_tool_calls"]


_REDIAL = GatewayThrottleRedialPolicy(max_attempts=2, base_delay_ms=100, max_delay_ms=2_000)


def test_throttle_redial_redials_the_warm_rung_through_its_window_until_the_cap() -> None:
    """A post-backoff redial claims the throttled rung; the spent cap advances cold.

    The settle already recorded the rung's throttle window, so a plain claim
    refuses it; the data plane's ``throttle_backoff`` reservation is the one
    request deliberately probing the rung back. Once the per-rung redials are
    spent the throttle advances like any failover-eligible failure, even under
    ``maximize_cache``, whose surfacing rule the schedule replaces.
    """
    health = DeploymentHealthRegistry(throttle_seconds=30.0)
    health.failed(_KEYS[0], _failover_only())
    assert not health.claim(_KEYS[0])

    def candidate(
        counts: list[int],
        redials: int,
        *,
        backoff: bool,
        keys: tuple[DeploymentHealthKey, ...] = _KEYS,
    ) -> int | None:
        """Ask the frozen policy after ``counts`` dispatches and ``redials`` backoffs."""
        return next_route_candidate(
            health=health,
            keys=keys,
            failure=_failover_only(),
            current_depth=0,
            attempt_counts=counts,
            total_attempts=sum(counts),
            refusal_failover=False,
            failover_mode="maximize_cache",
            throttle_redial=_REDIAL,
            throttle_backoff=backoff,
            throttle_redial_budget=_REDIAL.max_attempts - redials,
        )

    # Two redials of the warm rung after its throttled dispatch...
    assert candidate([1, 0], 0, backoff=True) == 0
    assert candidate([2, 0], 1, backoff=True) == 0
    # ...then the budget is spent and the ladder advances cold (no surfacing).
    assert candidate([3, 0], 2, backoff=True) == 1
    # A retryable-class redial of the same rung spends no throttle budget:
    # three dispatches there but only one post-backoff redial still redials.
    assert candidate([3, 0], 1, backoff=True) == 0
    # A throttle the data plane did not wait out advances at once.
    assert candidate([1, 0], 0, backoff=False) == 1
    # A single-rung ladder past its redials is exhausted, and only then.
    assert candidate([3], 2, backoff=True, keys=_KEYS[:1]) is None
    # The hard total cap still bounds the whole story.
    assert (
        next_route_candidate(
            health=health,
            keys=_KEYS,
            failure=_failover_only(),
            current_depth=0,
            attempt_counts=[1, 0],
            total_attempts=8,
            refusal_failover=False,
            throttle_redial=_REDIAL,
            throttle_backoff=True,
        )
        is None
    )


def test_throttle_redial_replaces_the_threshold_surfacing_rule_and_leaves_other_classes() -> None:
    """Warm cache above the threshold now backs off or advances; it never surfaces."""
    health = DeploymentHealthRegistry()

    def candidate(
        failure: GatewayFailure,
        *,
        throttle_redial: GatewayThrottleRedialPolicy | None = None,
        throttle_backoff: bool = False,
    ) -> int | None:
        """Ask the frozen policy with warm cache (0.9) above an authored 0.5 threshold."""
        return next_route_candidate(
            health=health,
            keys=_KEYS,
            failure=failure,
            current_depth=0,
            attempt_counts=[1, 0],
            total_attempts=1,
            refusal_failover=False,
            throttle_cache_threshold=0.5,
            cached_fraction=0.9,
            throttle_redial=throttle_redial,
            throttle_backoff=throttle_backoff,
            throttle_redial_budget=_REDIAL.max_attempts,
        )

    # Without a schedule the met threshold surfaces the throttle.
    assert candidate(_failover_only()) is None
    # With one, the same cache means "wait here" (redial) or advance cold.
    assert candidate(_failover_only(), throttle_redial=_REDIAL, throttle_backoff=True) == 0
    assert candidate(_failover_only(), throttle_redial=_REDIAL) == 1
    # Other classes keep their own rules: a retryable failure redials on its
    # own flag, a client error never advances, backoff flag or not.
    assert candidate(_retryable(), throttle_redial=_REDIAL, throttle_backoff=True) == 0
    invalid = GatewayFailure(
        failure_class=GatewayFailureClass.INVALID_REQUEST, safe_message="bad request"
    )
    assert candidate(invalid, throttle_redial=_REDIAL, throttle_backoff=True) is None


def test_dispatch_disclosure_names_a_backoff_redial_on_every_pool() -> None:
    """The redial is ``throttle_backoff`` with no counterfactual, affinity pools included."""
    route = _route()
    assert dispatch_disclosure(
        route, 0, policy_sheds=[], forced_overflow=False, throttle_backoff=True
    ) == (THROTTLE_BACKOFF, None)
    affinity = route.model_copy(
        update={
            "snapshot": route.snapshot.model_copy(
                update={"failover_mode": "maximize_cache_affinity"}
            )
        }
    )
    assert dispatch_disclosure(
        affinity, 0, policy_sheds=[], forced_overflow=False, throttle_backoff=True
    ) == (THROTTLE_BACKOFF, None)
    assert dispatch_disclosure(affinity, 0, policy_sheds=[], forced_overflow=False) == (
        "affinity",
        None,
    )
    # A redial the rung's own policy shed and the accounting force-admitted
    # stays throttle_backoff: neither the shed reason nor saturated_overflow
    # may relabel the attempt the caller waited the backoff for.
    for pool in (route, affinity):
        assert dispatch_disclosure(
            pool,
            0,
            policy_sheds=[(0, "rate_limit")],
            forced_overflow=True,
            throttle_backoff=True,
        ) == (THROTTLE_BACKOFF, None)


@pytest.mark.parametrize("root_affinity", [False, True])
def test_dispatch_disclosure_uses_actual_stage_policy_and_preference(root_affinity: bool) -> None:
    """A child affinity mode and its preferred provider never inherit root identity."""
    route = _route()
    lead, child = route.deployments[:2]
    sibling = child.model_copy(update={"deployment_id": "child-sibling"})
    root = route.snapshot.stage_for_depth(0).model_copy(
        update={
            "deployment_ids": (lead.deployment_id,),
            "failover_mode": "maximize_cache_affinity"
            if root_affinity
            else "maximize_availability",
        }
    )
    descendant = root.model_copy(
        update={
            "stage_index": 1,
            "deployment_ids": (child.deployment_id, sibling.deployment_id),
            "failover_mode": "maximize_availability"
            if root_affinity
            else "maximize_cache_affinity",
        }
    )
    route = route.model_copy(
        update={
            "fallback_deployments": (child, sibling),
            "snapshot": route.snapshot.model_copy(
                update={
                    "deployment_ids": (
                        lead.deployment_id,
                        child.deployment_id,
                        sibling.deployment_id,
                    ),
                    "model_stages": (root, descendant),
                    "failover_mode": root.failover_mode,
                }
            ),
        }
    )
    assert dispatch_disclosure(route, 1, policy_sheds=[], forced_overflow=False) == (
        None if root_affinity else "affinity",
        None,
    )
    reason, preferred = dispatch_disclosure(route, 2, policy_sheds=[], forced_overflow=False)
    assert reason == (None if root_affinity else "rung_dead")
    assert preferred == (None if root_affinity else child)


def test_wire_entry_carries_the_throttle_redial_budget() -> None:
    """The admission-time redial budget rides the wire entry; unauthored rungs carry zero."""
    route = _route()
    profile = GatewayWireProfile(dialect="openai_responses", url="https://provider.test")
    assert (
        deployment_wire_entry(route, route.deployment, profile, {})["throttle_redial_budget"] == 0
    )
    assert (
        deployment_wire_entry(route, route.deployment, profile, {}, throttle_redial_budget=3)[
            "throttle_redial_budget"
        ]
        == 3
    )


def test_route_narrowing_and_reordering_keep_the_zdr_constraint_flags() -> None:
    """The snapshot's constrained ids survive every route rebuild the admission does."""
    route = _route()
    flagged = route.model_copy(
        update={
            "snapshot": route.snapshot.model_copy(
                update={"zdr_constrained_deployment_ids": ("two",)}
            )
        }
    )

    narrowed = select_route_deployments(flagged, (1, 2))
    reordered = reorder_route_deployments(flagged, (2, 0, 1))

    assert narrowed.snapshot.zdr_constrained_deployment_ids == ("two",)
    assert reordered.snapshot.zdr_constrained_deployment_ids == ("two",)
    assert route.snapshot.zdr_constrained_deployment_ids == ()


def test_wire_entry_carries_the_codex_native_tool_translation_map() -> None:
    """A translated Codex request carries the inversion map to the data plane."""
    route = _route()
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://provider.test")
    mapping = {"multi_agent_v1__close_agent": ("close_agent", "multi_agent_v1", False)}
    entry = deployment_wire_entry(
        route, route.deployment, profile, {}, native_tool_translation=mapping
    )
    # Tuples serialize to JSON arrays for the Rust `(String, Option<String>, bool)`.
    assert entry["native_tool_translation"] == {
        "multi_agent_v1__close_agent": ["close_agent", "multi_agent_v1", False]
    }
    assert (
        deployment_wire_entry(route, route.deployment, profile, {})["native_tool_translation"] == {}
    )
