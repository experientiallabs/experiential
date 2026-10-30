"""Read existing same-scheme null annotations without accepting scope claims or rewriting data."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    canonical_json_bytes,
)
from exp.common.project import ArtifactStore, ProjectPaths
from exp.simulation.world_model.experience_memory import (
    KNOWLEDGE_IDENTITY_SCHEME,
    MemoryVersion,
    memory_head,
    memory_version_id,
    read_memory_version,
)

NOW = datetime(2026, 10, 23, 10, 46, 31, tzinfo=UTC)


def _record() -> JsonObject:
    """Same field shape as the78 saved Literal2 checkpoints, without benchmark data."""
    world: JsonObject = {"artifact_id": "grounded-world-model-fixture", "sha256": "a" * 64}
    library: JsonObject = {"artifact_id": "experience-library-fixture", "sha256": "b" * 64}
    return {
        "schema_version": 1,
        "created_at": "2026-10-23T10:46:31Z",
        "inputs": [library],
        "code_revision": "experiential==0.7.186",
        "source": None,
        "lineage": "fixture-memory",
        "version": 1,
        "world_model": world,
        "selection_policy_sha256": "c" * 64,
        "library": library,
        "consumed": ["d" * 64],
        "publisher": "fidelity-fixture",
        "knowledge_identity": "e" * 64,
        "knowledge_identity_scheme": 2,
        "serving_policy": {
            "memory_selector_sha256": "c" * 64,
            "model_id": "glm-5.3-flash",
            "prediction_guidance_sha256": "f" * 64,
            "prompt_sha256": "1" * 64,
            "prompt_version": "text-world-model-v3",
            "selection_policy_sha256": None,
            "top_k": 5,
            "world_model": world,
        },
        "log_length": 1,
        "appended": [
            {
                "kind": "observed",
                "text": "[observed] completed native fixture trajectory",
                "origin": "fixture-trajectory",
                "derived_from": {
                    "artifact_id": "consumed-trajectory-fixture",
                    "sha256": "2" * 64,
                },
                "depends_on": [],
            }
        ],
    }


def _store_record(tmp_path: Path, record: JsonObject) -> tuple[ArtifactStore, str, bytes]:
    """Persist the original raw shape, not a newly normalized MemoryVersion."""
    store = ArtifactStore(ProjectPaths(tmp_path, "memory"))
    identity = memory_version_id("fixture-memory", 1)
    raw = canonical_json_bytes(record)
    store.write(
        artifact_id=identity,
        artifact_type="experience-memory-version",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=NOW,
            code_revision="experiential==0.7.186",
            inputs=(ArtifactInput.model_validate(record["library"]),),
        ),
        files={"memory.json": raw},
    )
    return store, identity, raw


def test_saved_null_scope_checkpoint_replays_without_writes_or_byte_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit null is metadata absence; replay preserves raw bytes and source bindings."""
    record = _record()
    record["knowledge_scope"] = None
    store, identity, raw = _store_record(tmp_path, record)
    before_ids = store.list_ids()
    before_manifest = store.read(identity).manifest

    def forbidden_write(*args: object, **kwargs: object) -> None:
        """A format read has no authority to publish or migrate an artifact."""
        raise AssertionError("memory format read attempted a write")

    monkeypatch.setattr(ArtifactStore, "write", forbidden_write)
    monkeypatch.setattr(ArtifactStore, "write_json", forbidden_write)
    first = read_memory_version(store, "fixture-memory", 1)
    assert first == memory_head(store, "fixture-memory")
    assert first == read_memory_version(store, "fixture-memory", 1)
    assert first.knowledge_scope is None
    assert first.knowledge_identity_scheme == KNOWLEDGE_IDENTITY_SCHEME == 2
    assert first.inputs == before_manifest.inputs
    assert first.world_model.model_dump(mode="json") == record["world_model"]
    assert first.knowledge_identity == record["knowledge_identity"]
    assert first.serving_policy == record["serving_policy"]
    assert store.read_bytes(identity, "memory.json") == raw
    assert store.read(identity).manifest == before_manifest
    assert store.list_ids() == before_ids


@pytest.mark.parametrize("claim", [{}, {"scope": "fit"}, [], "fit", False, 0])
def test_nonnull_scope_claims_are_rejected(tmp_path: Path, claim: JsonValue) -> None:
    """Even empty, false or zero claims cannot introduce scope authority through this field."""
    record = _record()
    record["knowledge_scope"] = claim
    with pytest.raises(ValidationError, match="knowledge_scope"):
        MemoryVersion.model_validate(record)
    store, identity, raw = _store_record(tmp_path, record)
    with pytest.raises(ValueError, match="memory checkpoint does not match the current format"):
        read_memory_version(store, "fixture-memory", 1)
    assert store.read_bytes(identity, "memory.json") == raw


def test_unknown_fields_and_different_knowledge_schemes_remain_rejected() -> None:
    """The null-only field does not enable extra-ignore behavior or another memory format."""
    unknown = _record()
    unknown["knowledge_scope"] = None
    unknown["unrecognized_scope_claim"] = None
    with pytest.raises(ValidationError, match="unrecognized_scope_claim"):
        MemoryVersion.model_validate(unknown)
    different_scheme = _record()
    different_scheme["knowledge_identity_scheme"] = 1
    with pytest.raises(ValidationError, match="knowledge_identity_scheme"):
        MemoryVersion.model_validate(different_scheme)


def test_current_omitted_field_serialization_is_canonically_unchanged() -> None:
    """Both accepted absence forms emit the exact current canonical record, without null."""
    current = _record()
    expected = canonical_json_bytes(current)
    omitted = MemoryVersion.model_validate(current)
    explicit = MemoryVersion.model_validate({**current, "knowledge_scope": None})
    assert omitted.knowledge_scope is None and explicit.knowledge_scope is None
    assert canonical_json_bytes(omitted) == canonical_json_bytes(explicit) == expected
    assert "knowledge_scope" not in omitted.model_dump(mode="json", exclude_none=False)
    assert "knowledge_scope" not in explicit.model_dump(mode="json", exclude_none=False)
