"""Tests for setup-owned guardrail identifiers and document mutation."""

from __future__ import annotations

from exp.cli.gateway.guardrail_setup_store import (
    setup_adapter_id,
    setup_policy_id,
)
from exp.common.core.artifacts import stable_id, validate_artifact_id


def test_hyphenated_organization_and_identity_pairs_do_not_collide() -> None:
    """Structured subjects keep (a-b, c) distinct from (a, b-c)."""
    first = setup_adapter_id("a-b", "c")
    second = setup_adapter_id("a", "b-c")
    assert first != second
    assert first == stable_id(
        "setup-http-json",
        {"organization_id": "a-b", "identity_id": "c"},
    )
    assert second == stable_id(
        "setup-http-json",
        {"organization_id": "a", "identity_id": "b-c"},
    )
    assert setup_policy_id("a-b", "c") != setup_policy_id("a", "b-c")
    assert setup_policy_id("a-b", "c") != first


def test_long_valid_identifiers_still_produce_bounded_artifact_ids() -> None:
    """Long organization and identity values hash into short valid ArtifactIds."""
    organization_id = "organization-" + ("a" * 80)
    identity_id = "identity-" + ("b" * 80)
    adapter_id = setup_adapter_id(organization_id, identity_id)
    policy_id = setup_policy_id(organization_id, identity_id)
    assert validate_artifact_id(adapter_id) == adapter_id
    assert validate_artifact_id(policy_id) == policy_id
    assert len(adapter_id) <= 128
    assert len(policy_id) <= 128
    assert organization_id not in adapter_id
    assert identity_id not in policy_id
    assert adapter_id != policy_id
    assert adapter_id.startswith("setup-http-json-")
    assert policy_id.startswith("setup-standard-")
