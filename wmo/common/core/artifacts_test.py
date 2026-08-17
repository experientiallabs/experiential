"""Tests for canonical artifact provenance, stable IDs, and secret boundaries."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import (
    SECRET_REDACTION_PLACEHOLDER,
    ArtifactEnvelope,
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    SecretBoundaryError,
    SourceIdentity,
    StructuredFailure,
    assert_secret_free,
    assert_text_secret_free,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    envelope_matches_manifest,
    redact_secret_json,
    redact_secret_text,
    sha256_bytes,
    sha256_json,
    sorted_unique_inputs,
    stable_id,
    validate_artifact_file_path,
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


def test_canonical_jsonl_frames_records_and_digests_exact_payload_bytes() -> None:
    """JSONL serialization newline-terminates records and hashes the framed payload."""
    records = ({"b": 2, "a": 1}, {"c": 3})

    payload = canonical_jsonl_bytes(records)
    expected = b"".join(canonical_json_bytes(record) + b"\n" for record in records)

    assert canonical_jsonl_bytes(()) == b""
    assert payload == expected
    assert sha256_bytes(payload) != sha256_bytes(payload[:-1])
    assert sha256_bytes(canonical_json_bytes(records[0])) == sha256_json(records[0])


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


def test_sorted_unique_inputs_dedupes_sorts_and_raises_the_domain_error() -> None:
    """The canonical input normalizer dedupes exact repeats and rejects digest drift."""
    first = ArtifactInput(artifact_id="trace-a", sha256=_DIGEST_A)
    second = ArtifactInput(artifact_id="trace-b", sha256=_DIGEST_B)

    assert sorted_unique_inputs(second, first, second) == (first, second)
    assert sorted_unique_inputs() == ()

    conflicting = ArtifactInput(artifact_id="trace-a", sha256=_DIGEST_B)
    with pytest.raises(KeyError, match="conflicting manifest digests"):
        sorted_unique_inputs(first, conflicting, error_type=KeyError)


def test_envelope_matches_manifest_compares_the_five_shared_provenance_fields() -> None:
    """The shared predicate accepts identical provenance and rejects any field drift."""
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="e7aad17",
        inputs=(ArtifactInput(artifact_id="trace-a", sha256=_DIGEST_A),),
        source=SourceIdentity(kind="file", source_id="traces.jsonl", sha256=_DIGEST_A),
    )
    manifest = envelope.model_copy()

    assert envelope_matches_manifest(envelope, manifest)
    assert not envelope_matches_manifest(
        envelope, manifest.model_copy(update={"code_revision": "f00dcafe"})
    )
    assert not envelope_matches_manifest(
        envelope,
        manifest.model_copy(update={"created_at": datetime(2026, 8, 12, tzinfo=UTC)}),
    )


def test_envelope_matches_manifest_covers_every_base_envelope_field() -> None:
    """Growing `ArtifactEnvelope` must grow the predicate, or replay verification silently thins.

    `envelope_matches_manifest` hand-enumerates the shared provenance fields, so a new base field
    would be persisted and mirrored into manifests yet never compared. This pin turns that silent
    gap into a failing test naming the function to extend.
    """
    assert set(ArtifactEnvelope.model_fields) == {
        "schema_version",
        "created_at",
        "inputs",
        "code_revision",
        "source",
    }, "ArtifactEnvelope grew a field: add it to envelope_matches_manifest and this pin"


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
    with pytest.raises(SecretBoundaryError, match="environment name"):
        assert_secret_free({"connection_hint": "OPENAI_API_KEY"})


def test_artifact_file_paths_reject_nonportable_components() -> None:
    """Portable artifact paths cannot escape or change meaning on Windows."""
    assert validate_artifact_file_path("nested/data.json").as_posix() == "nested/data.json"
    for path in ("../outside.json", "nested\\outside.json", "C:/outside.json"):
        with pytest.raises(ValueError, match="relative POSIX"):
            validate_artifact_file_path(path)


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "AKIAABCDEFGHIJKLMNOP",
        "xoxb-1234567890-abcdefghij",
        "Bearer abcdefghijklmnop",
        "OPENAI_API_KEY",
    ],
)
def test_redact_secret_text_replaces_every_secret_pattern(secret: str) -> None:
    """Each secret-like value pattern and credential environment name is replaced once.

    Args:
        secret: One representative match for a rejected secret pattern.
    """
    redacted, count = redact_secret_text(f"model output includes {secret} inline")

    assert count == 1
    assert secret not in redacted
    assert redacted == f"model output includes {SECRET_REDACTION_PLACEHOLDER} inline"
    assert_secret_free({"content": redacted})


def test_redaction_placeholder_passes_every_secret_boundary() -> None:
    """The fixed placeholder itself never trips the immutable-artifact secret boundary."""
    assert_secret_free({"content": SECRET_REDACTION_PLACEHOLDER})
    assert_text_secret_free(SECRET_REDACTION_PLACEHOLDER)
    assert redact_secret_text(SECRET_REDACTION_PLACEHOLDER) == (SECRET_REDACTION_PLACEHOLDER, 0)


def test_redact_secret_json_counts_nested_replacements() -> None:
    """Nested JSON content is redacted structurally with an aggregate replacement count."""
    payload = {
        "response": {
            "output": [
                {"content": "export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456"},
                {"content": "no secrets here"},
            ]
        },
        "steps": 3,
    }

    redacted, count = redact_secret_json(payload)

    assert count == 2
    assert redacted == {
        "response": {
            "output": [
                {
                    "content": "export "
                    f"{SECRET_REDACTION_PLACEHOLDER}={SECRET_REDACTION_PLACEHOLDER}"
                },
                {"content": "no secrets here"},
            ]
        },
        "steps": 3,
    }
    assert_secret_free(redacted)


def test_structured_failures_preserve_runtime_attribution() -> None:
    """A failure distinguishes model and environment lifecycle ownership."""
    failure = StructuredFailure(
        code=FailureCode.INTERNAL,
        message="environment cleanup failed",
        attribution=FailureAttribution.CLEANUP,
    )

    assert StructuredFailure.model_validate_json(failure.model_dump_json()) == failure
