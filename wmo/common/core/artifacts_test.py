"""Tests for canonical artifact provenance, stable IDs, and secret boundaries."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    SecretBoundaryError,
    SourceIdentity,
    assert_secret_free,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def test_deterministic_json_hash_and_stable_id_ignore_mapping_order() -> None:
    """Canonical JSON makes equivalent mappings hash and identify identically."""
    first = {"nested": {"b": 2, "a": 1}, "items": ["x", "y"]}
    second = {"items": ["x", "y"], "nested": {"a": 1, "b": 2}}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert sha256_json(first) == sha256_json(second)
    assert stable_id("task-set", first) == stable_id("task-set", second)


def test_envelope_requires_timezone_and_sorted_unique_inputs() -> None:
    """Envelope provenance rejects ambiguous clocks and duplicate input identities."""
    with pytest.raises(ValidationError, match="timezone"):
        ArtifactEnvelope(
            schema_version=1,
            created_at=datetime(2026, 8, 11),
            code_revision="e7aad17",
        )

    with pytest.raises(ValidationError, match="sorted"):
        ArtifactEnvelope(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            inputs=(
                ArtifactInput(artifact_id="trace-b", sha256=_DIGEST_B),
                ArtifactInput(artifact_id="trace-a", sha256=_DIGEST_A),
            ),
        )

    with pytest.raises(ValidationError, match="repeat"):
        ArtifactEnvelope(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            inputs=(
                ArtifactInput(artifact_id="trace-a", sha256=_DIGEST_A),
                ArtifactInput(artifact_id="trace-a", sha256=_DIGEST_B),
            ),
        )


def test_source_identity_and_secret_boundary_round_trip() -> None:
    """Source provenance serializes, while credential values and references do not."""
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="e7aad17",
        source=SourceIdentity(kind="file", source_id="traces.jsonl", sha256=_DIGEST_A),
    )

    restored = ArtifactEnvelope.model_validate_json(envelope.model_dump_json())

    assert restored == envelope
    with pytest.raises(SecretBoundaryError, match="api_key_env"):
        assert_secret_free({"api_key_env": "OPENAI_API_KEY"})
    with pytest.raises(SecretBoundaryError, match="secret-like"):
        assert_secret_free({"note": "sk-abcdefghijklmnopqrstuvwxyz123456"})
