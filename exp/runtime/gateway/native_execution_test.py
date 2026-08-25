"""Tests for the native waterfall's candidate policy and wire building."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.native_execution import (
    claim_route_from,
    next_route_candidate,
)

_KEYS: tuple[DeploymentHealthKey, ...] = (
    ("catalog" + "0" * 57, "deployment-a", "connection-a"),
    ("catalog" + "0" * 57, "deployment-b", "connection-b"),
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


def test_native_serving_blockers_name_dialectless_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust-only startup validation names every alias the engine cannot serve.

    Every currently supported provider implements a native dialect, so the
    "no native dialect implementation" branch is exercised by patching a real
    client's ``gateway_wire_profile`` back to the unimplemented base
    behavior, rather than by a provider connection string this registry can
    still resolve.
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
    blockers = native_serving_blockers(components)
    assert len(blockers) == 1
    assert blockers[0].startswith("escalated: ")
    assert "gemini" in blockers[0]
    assert "native dialect" in blockers[0]
