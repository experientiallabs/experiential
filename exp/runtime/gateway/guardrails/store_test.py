"""Tests for organization-and-identity guardrail policy lookup."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    ProjectTarget,
)
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore


def _authorization(
    organization_id: str = "organization-one",
    identity_id: str = "identity-one",
    *,
    pool_id: str = "pool-one",
) -> AuthorizationSnapshot:
    """Return one identity-scoped authority snapshot for an alias that targets one pool."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id=organization_id,
        identity_id=identity_id,
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id=pool_id),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="b" * 64,
        deadline_monotonic=200.0,
    )


def _policy(*, organization_id: str, identity_id: str) -> GuardrailPolicy:
    """Build one empty policy for the requested scope."""
    return GuardrailPolicy(
        policy_id=f"{organization_id}-{identity_id}",
        organization_id=organization_id,
        identity_id=identity_id,
    )


def test_missing_pair_returns_none() -> None:
    """Unguarded organization and identity pairs leave the hot path unchanged."""
    store = MappingGuardrailStore(
        (_policy(organization_id="organization-one", identity_id="identity-one"),)
    )

    assert store.policies_for(_authorization()) != ()
    assert store.policies_for(_authorization("organization-one", "identity-two")) == ()
    assert store.policies_for(_authorization("organization-two", "identity-one")) == ()


def test_identical_identity_ids_do_not_share_policies_across_organizations() -> None:
    """The same identity ID in two organizations keeps distinct assignments."""
    first = _policy(organization_id="organization-one", identity_id="shared")
    second = _policy(organization_id="organization-two", identity_id="shared")
    store = MappingGuardrailStore((first, second))

    assert store.policies_for(_authorization("organization-one", "shared"))[0] is first
    assert store.policies_for(_authorization("organization-two", "shared"))[0] is second
    assert store.policies_for(_authorization("organization-one", "shared"))[0] is not second


def test_store_rejects_duplicate_organization_identity_pairs() -> None:
    """Two policies for the same organization and identity fail closed."""
    with pytest.raises(ValueError, match="must be unique"):
        MappingGuardrailStore(
            (
                _policy(organization_id="organization-one", identity_id="identity-one"),
                _policy(organization_id="organization-one", identity_id="identity-one"),
            )
        )


def test_pool_scoped_policies_skip_requests_to_other_models() -> None:
    """A scoped policy covers listed pools and router aliases; other pools get no policy."""
    scoped_global = GuardrailPolicy(
        policy_id="classify-pool-one", protected=True, target_pool_ids=("pool-one",)
    )
    scoped_identity = GuardrailPolicy(
        policy_id="classify-pool-two",
        organization_id="organization-one",
        identity_id="identity-one",
        target_pool_ids=("pool-two",),
    )
    store = MappingGuardrailStore((scoped_global, scoped_identity))
    routed = _authorization().model_copy(
        update={
            "target": ProjectTarget(
                project_ref="router-one", activation_ref="activation-one", catalog_sha256="c" * 64
            )
        }
    )

    def ids(authorization: AuthorizationSnapshot) -> list[str]:
        """Return resolved policy identities in order."""
        return [policy.policy_id for policy in store.policies_for(authorization)]

    assert ids(_authorization(pool_id="pool-one")) == ["classify-pool-one"]
    assert ids(_authorization(pool_id="pool-two")) == ["classify-pool-two"]
    assert ids(_authorization(pool_id="pool-three")) == []
    assert ids(routed) == ["classify-pool-one", "classify-pool-two"]
