"""Tests for identity-keyed guardrail policy lookup."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore


def test_missing_identity_returns_none() -> None:
    """Unguarded identities leave the existing hot path unchanged."""
    store = MappingGuardrailStore(
        {
            "identity-one": GuardrailPolicy(
                policy_id="member-policy",
                identity_id="identity-one",
            )
        }
    )

    assert store.policy_for("identity-one") is not None
    assert store.policy_for("identity-two") is None


def test_store_rejects_mismatched_identity_keys() -> None:
    """A map key that does not match the policy identity is a configuration error."""
    with pytest.raises(ValueError, match="identity must match"):
        MappingGuardrailStore(
            {
                "other": GuardrailPolicy(
                    policy_id="member-policy",
                    identity_id="identity-one",
                )
            }
        )
