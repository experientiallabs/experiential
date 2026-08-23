"""Tests for organization-and-identity guardrail policy lookup."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore


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

    assert store.policy_for("organization-one", "identity-one") is not None
    assert store.policy_for("organization-one", "identity-two") is None
    assert store.policy_for("organization-two", "identity-one") is None


def test_identical_identity_ids_do_not_share_policies_across_organizations() -> None:
    """The same identity ID in two organizations keeps distinct assignments."""
    first = _policy(organization_id="organization-one", identity_id="shared")
    second = _policy(organization_id="organization-two", identity_id="shared")
    store = MappingGuardrailStore((first, second))

    assert store.policy_for("organization-one", "shared") is first
    assert store.policy_for("organization-two", "shared") is second
    assert store.policy_for("organization-one", "shared") is not second


def test_store_rejects_duplicate_organization_identity_pairs() -> None:
    """Two policies for the same organization and identity fail closed."""
    with pytest.raises(ValueError, match="unique per organization"):
        MappingGuardrailStore(
            (
                _policy(organization_id="organization-one", identity_id="identity-one"),
                _policy(organization_id="organization-one", identity_id="identity-one"),
            )
        )
