"""Tests for the grounded world-model artifact's content-addressed identity."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from exp.common.core.artifacts import ArtifactInput, canonical_json_bytes, sha256_json, stable_id
from exp.common.models import BillingSource, ModelCapabilities, ModelSnapshot
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.retrieval import CorpusScope
from exp.simulation.world_model.artifact import (
    GroundedWorldModelArtifact,
    OriginalEvidence,
    grounded_world_model_content,
)

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


_RAG = ArtifactInput(artifact_id="trace-rag-fixture", sha256="0" * 64)
_MODEL = ModelSnapshot(
    billing_source=BillingSource.CUSTOMER_MANAGED,
    provider="fixture",
    model_id="world-id",
    capabilities_sha256=ModelCapabilities().identity_sha256(),
    connection_sha256=sha256_json({"connection": "world-id"}),
)
_KNOWLEDGE = SimulationKnowledge(source_trace_ids=("trace-0", "trace-1"))

# Identity and envelope digest of the fixture world model, without and with knowledge, recorded
# before construction scopes existed. Unscoped artifacts must keep exactly these.
_UNSCOPED_GOLDEN = {
    False: (
        "grounded-world-model-a103bfb9426317ac774d",
        "753eff0db33b7e7ee89aafd1b95eebbc98a66f856c6c86cc204c8e19764af6b6",
    ),
    True: (
        "grounded-world-model-f2b659bf71cf50f95e35",
        "e05170acc6a82f205bfa73a9519363d8e576c109aa7ff492971e8cc875d260ee",
    ),
}


def _content(
    *,
    selection_policy_sha256: str | None = None,
    original_evidence: OriginalEvidence = "retrieved",
    knowledge: SimulationKnowledge | None = None,
    construction_scope: CorpusScope | None = None,
) -> dict[str, object]:
    """One world model's identity content, with any optional inputs given.

    Args:
        selection_policy_sha256: Optional reranker identity.
        original_evidence: Retrieval profile.
        knowledge: Optional construction knowledge.
        construction_scope: Optional recorded construction scope.

    Returns:
        The canonical identity content.
    """
    return grounded_world_model_content(
        serving_rag=_RAG,
        model_alias="world",
        model=_MODEL,
        prompt_version="v3",
        prompt_sha256="1" * 64,
        top_k=5,
        code_revision="fixture",
        knowledge=knowledge,
        selection_policy_sha256=selection_policy_sha256,
        original_evidence=original_evidence,
        construction_scope=construction_scope,
    )


def _artifact(
    knowledge: SimulationKnowledge | None, construction_scope: CorpusScope | None = None
) -> GroundedWorldModelArtifact:
    """Build the fixture world-model envelope under its content-derived identity.

    Args:
        knowledge: Optional construction knowledge.
        construction_scope: Optional recorded construction scope.

    Returns:
        The validated envelope.
    """
    content = _content(knowledge=knowledge, construction_scope=construction_scope)
    return GroundedWorldModelArtifact(
        schema_version=1,
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        inputs=(_RAG,),
        code_revision="fixture",
        source=None,
        world_model_id=stable_id("grounded-world-model", content),
        serving_rag=_RAG,
        model_alias="world",
        model=_MODEL,
        prompt_version="v3",
        prompt_sha256="1" * 64,
        top_k=5,
        knowledge=knowledge,
        construction_scope=construction_scope,
    )


def _trace_scope(*ids: str) -> CorpusScope:
    """Declare a construction trace scope bound to a fixed plan digest.

    Args:
        *ids: Admitted trace identities.

    Returns:
        The canonical scope.
    """
    return CorpusScope.declare(
        name="world-learning-construction", kind="trace", ids=ids, source_plan="d" * 64
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


@pytest.mark.parametrize("with_knowledge", [False, True])
def test_unscoped_artifacts_keep_their_bytes_and_identity(with_knowledge: bool) -> None:
    """An absent construction scope adds no field, so identity and envelope bytes are unchanged.

    Args:
        with_knowledge: Whether the fixture carries construction knowledge.
    """
    artifact = _artifact(_KNOWLEDGE if with_knowledge else None)
    encoded = canonical_json_bytes(artifact)
    assert b"construction_scope" not in encoded
    assert (artifact.world_model_id, hashlib.sha256(encoded).hexdigest()) == _UNSCOPED_GOLDEN[
        with_knowledge
    ]
    assert GroundedWorldModelArtifact.model_validate_json(encoded) == artifact


def test_a_recorded_construction_scope_joins_identity_and_envelope() -> None:
    """A recorded scope changes the identity, serializes, and round-trips exactly."""
    scope = _trace_scope("trace-0", "trace-1", "trace-2")
    assert set(_content(knowledge=_KNOWLEDGE, construction_scope=scope)) == {
        *_ORIGINAL_FIELDS,
        "knowledge",
        "construction_scope",
    }
    artifact = _artifact(_KNOWLEDGE, scope)
    assert artifact.world_model_id != _UNSCOPED_GOLDEN[True][0]
    encoded = canonical_json_bytes(artifact)
    assert b"construction_scope" in encoded
    assert GroundedWorldModelArtifact.model_validate_json(encoded).construction_scope == scope


def test_a_recorded_construction_scope_must_contain_the_construction_corpus() -> None:
    """A scope needs construction traces, and a trace scope must contain every one of them."""
    with pytest.raises(ValueError, match="needs recorded construction source traces"):
        _artifact(None, _trace_scope("trace-0"))
    with pytest.raises(ValueError, match="trace-1 lies outside construction scope"):
        _artifact(_KNOWLEDGE, _trace_scope("trace-0"))
