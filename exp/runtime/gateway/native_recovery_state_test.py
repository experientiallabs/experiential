"""Private bindings remain confined to the native admission export."""

from exp.runtime.gateway.native_recovery_state import key_registration, scope_registration
from exp.runtime.gateway.recovery_test import key, scope


def test_native_recovery_scope_keeps_private_authority_out_of_shared_topology() -> None:
    """Only the native owner receives exact credential and tenant identity."""
    binding = scope("generation", "tenant")
    assert "credential_scope" not in binding.model_dump(mode="json")
    assert scope_registration(binding) == {
        **binding.operational().model_dump(mode="json"),
        "credential_scope": "generation",
        "organization_id": "tenant",
    }


def test_native_session_key_preserves_both_tenant_and_stable_prefix() -> None:
    """Similar sessions remain distinct across tenant and prefix boundaries."""
    first, second = key(), key(org="another")
    assert key_registration(first) != key_registration(second)
    assert key_registration(first)[2] == first.fingerprint.hex()
