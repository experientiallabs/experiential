"""Tests for the control-plane half of per-rung conditional failover."""

from __future__ import annotations

import pytest

from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.common.models.failover_tokens import FailoverToken
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRefusalReason,
)
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.native_execution import (
    claim_route_from,
    deployment_health_key,
    next_route_candidate,
)
from exp.runtime.gateway.native_fallback_rules import (
    FALLBACK_REASON_PREFIX,
    FailoverRulesError,
    eligible_depths,
    eligible_ladder,
    failure_token,
    matched_token,
    require_unrestricted_rung,
    route_fallback_rules,
    rule_fallback_reason,
)
from exp.runtime.gateway.routing import GatewayRoute


def _deployment(name: str, tokens: tuple[FailoverToken, ...] | None = None) -> ExactModelDeployment:
    return ExactModelDeployment(
        deployment_id=name,
        source_alias=name,
        exact_model_id="exact-one",
        connection=f"connection-{name}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="d" * 64,
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(
                supports_streaming=True, failover_only_on=tokens
            ),
        ),
    )


def _route(*deployments: ExactModelDeployment, fallback_reason: str | None = None) -> GatewayRoute:
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
        fallback_reason=fallback_reason,
    )


_CYBER = GatewayFailure(
    failure_class=GatewayFailureClass.REFUSAL,
    safe_message="provider refused the request: cybersecurity policy",
    refusal_reason=GatewayRefusalReason.CYBER_POLICY,
)
_UNNAMED_REFUSAL = GatewayFailure(
    failure_class=GatewayFailureClass.REFUSAL, safe_message="provider refused the request"
)
_THROTTLE = GatewayFailure(
    failure_class=GatewayFailureClass.THROTTLED,
    safe_message="provider throttled the request",
    failover_eligible=True,
)
_INVALID = GatewayFailure(
    failure_class=GatewayFailureClass.INVALID_REQUEST, safe_message="bad request"
)


def test_failure_tokens_spell_the_class_or_the_refusal_reason() -> None:
    assert failure_token(_THROTTLE) == "throttled"
    assert failure_token(_CYBER) == "refusal:cyber_policy"
    assert failure_token(_UNNAMED_REFUSAL) == "refusal:unspecified"


def test_matched_token_honors_exact_tokens_and_the_bare_refusal() -> None:
    exact = frozenset({"refusal:cyber_policy"})
    assert matched_token(exact, _CYBER) == "refusal:cyber_policy"
    assert matched_token(exact, _UNNAMED_REFUSAL) is None
    assert matched_token(exact, _THROTTLE) is None
    any_refusal = frozenset({"refusal", "throttled"})
    # The bare token matches any refusal and still names the precise one.
    assert matched_token(any_refusal, _CYBER) == "refusal:cyber_policy"
    assert matched_token(any_refusal, _UNNAMED_REFUSAL) == "refusal:unspecified"
    assert matched_token(any_refusal, _THROTTLE) == "throttled"
    # Unrestricted rungs and first dials never match.
    assert matched_token(None, _CYBER) is None
    assert matched_token(exact, None) is None


def test_route_rules_and_eligible_depths_follow_the_authored_sets() -> None:
    route = _route(
        _deployment("house"),
        _deployment("byok", ("refusal:cyber_policy",)),
        _deployment("spill"),
    )
    rules = route_fallback_rules(route)
    assert rules == (None, frozenset({"refusal:cyber_policy"}), None)
    # A first dial (no failure) reaches unrestricted rungs only.
    assert eligible_ladder(route, None) == (0, 2)
    # A cyber refusal reaches the rule rung even when the policy would not
    # advance an unrestricted rung.
    assert eligible_depths(rules, 1, _CYBER, unrestricted=False) == (1,)
    assert eligible_depths(rules, 1, _CYBER, unrestricted=True) == (1, 2)
    # A throttle skips the rule rung.
    assert eligible_depths(rules, 1, _THROTTLE, unrestricted=True) == (2,)
    assert eligible_depths(rules, 1, _THROTTLE, unrestricted=False) == ()
    # With no sets authored the ladder is the historical one.
    plain = route_fallback_rules(_route(_deployment("a"), _deployment("b")))
    assert eligible_depths(plain, 0, _THROTTLE) == (0, 1)
    assert eligible_depths(plain, 1, None) == (1,)


def test_next_route_candidate_claims_a_rule_rung_only_on_a_matching_failure() -> None:
    route = _route(_deployment("house"), _deployment("byok", ("refusal:cyber_policy",)))
    rules = route_fallback_rules(route)
    keys = tuple(deployment_health_key(route.snapshot.authorization, d) for d in route.deployments)

    def candidate(failure: GatewayFailure, *, refusal_failover: bool = False) -> int | None:
        return next_route_candidate(
            health=DeploymentHealthRegistry(),
            keys=keys,
            failure=failure,
            current_depth=0,
            attempt_counts=[1, 0],
            total_attempts=1,
            refusal_failover=refusal_failover,
            fallback_rules=rules,
        )

    # The refusal reaches the opted-in rung without the alias revision's opt-in.
    assert candidate(_CYBER) == 1
    assert candidate(_CYBER, refusal_failover=True) == 1
    # Other refusals and other classes never reach it.
    assert candidate(_UNNAMED_REFUSAL, refusal_failover=True) is None
    assert candidate(_THROTTLE) is None
    assert candidate(_INVALID) is None
    # Without rules the historical answer holds: the throttle advances.
    assert (
        next_route_candidate(
            health=DeploymentHealthRegistry(),
            keys=keys,
            failure=_THROTTLE,
            current_depth=0,
            attempt_counts=[1, 0],
            total_attempts=1,
            refusal_failover=False,
        )
        == 1
    )


def test_claim_route_from_stays_inside_the_given_depths() -> None:
    route = _route(_deployment("a"), _deployment("b"), _deployment("c"))
    keys = tuple(deployment_health_key(route.snapshot.authorization, d) for d in route.deployments)
    health = DeploymentHealthRegistry()
    assert claim_route_from(health, keys, 0, (2,)) == 2
    assert claim_route_from(health, keys, 1, (0, 2)) == 2
    assert claim_route_from(health, keys, 0, ()) is None
    assert claim_route_from(health, keys, 0) == 0


def test_rule_fallback_reason_names_the_token_only_for_a_matched_successor() -> None:
    route = _route(
        _deployment("house"),
        _deployment("byok", ("refusal",)),
        _deployment("spill"),
        fallback_reason="route_default",
    )
    assert (
        rule_fallback_reason(route, 1, 0, _CYBER) == f"{FALLBACK_REASON_PREFIX}refusal:cyber_policy"
    )
    # First dial, same-rung redial, unrestricted successor: the route's own reason.
    assert rule_fallback_reason(route, 0, None, None) == "route_default"
    assert rule_fallback_reason(route, 1, 1, _THROTTLE) == "route_default"
    assert rule_fallback_reason(route, 2, 0, _THROTTLE) == "route_default"
    # A non-depth `current_depth` from the wire is a first dial.
    assert rule_fallback_reason(route, 1, "0", _CYBER) == "route_default"


def test_every_rung_restricted_fails_closed_at_admission() -> None:
    require_unrestricted_rung(_route(_deployment("house"), _deployment("byok", ("refusal",))))
    with pytest.raises(FailoverRulesError, match="failover_only_on"):
        require_unrestricted_rung(
            _route(_deployment("a", ("refusal",)), _deployment("b", ("throttled",)))
        )
