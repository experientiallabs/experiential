"""Tests for the grounded world-model artifact's content-addressed identity."""

from __future__ import annotations

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.models import BillingSource, ModelCapabilities, ModelSnapshot
from exp.simulation.world_model.artifact import OriginalEvidence, grounded_world_model_content

# The identity fields of a grounded world model persisted before memory and knowledge existed.
_ORIGINAL_FIELDS = {
    "serving_rag",
    "model_alias",
    "model",
    "prompt_version",
    "prompt_sha256",
    "top_k",
    "code_revision",
}


def _content(
    *,
    selection_policy_sha256: str | None = None,
    original_evidence: OriginalEvidence = "retrieved",
) -> dict[str, object]:
    """One world model's identity content, with any optional inputs given."""
    return grounded_world_model_content(
        serving_rag=ArtifactInput(artifact_id="trace-rag-fixture", sha256="0" * 64),
        model_alias="world",
        model=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="fixture",
            model_id="world-id",
            capabilities_sha256=ModelCapabilities().identity_sha256(),
            connection_sha256=sha256_json({"connection": "world-id"}),
        ),
        prompt_version="v3",
        prompt_sha256="1" * 64,
        top_k=5,
        code_revision="fixture",
        selection_policy_sha256=selection_policy_sha256,
        original_evidence=original_evidence,
    )


def test_defaults_keep_the_identity_a_world_model_was_saved_under() -> None:
    """Optional inputs at their defaults add no field, so earlier artifacts still verify."""
    assert set(_content()) == _ORIGINAL_FIELDS


def test_each_optional_input_given_joins_the_identity() -> None:
    """A selection policy or knowledge-free memory profile changes the identity only when set."""
    assert set(_content(selection_policy_sha256="2" * 64)) == {
        *_ORIGINAL_FIELDS,
        "selection_policy_sha256",
    }
    assert set(_content(original_evidence="none")) == {*_ORIGINAL_FIELDS, "original_evidence"}
