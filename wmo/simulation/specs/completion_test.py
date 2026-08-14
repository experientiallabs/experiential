"""Tests for immutable simulation completion reservations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from wmo.common.core.artifacts import canonical_json_bytes
from wmo.common.models import (
    CompletionCostReservation,
    ModelSnapshot,
    completion_cost_reservation,
)
from wmo.common.project import ArtifactStore, ProjectPaths
from wmo.common.project.manifests import file_digest
from wmo.simulation.specs.completion import (
    CandidateCompletionReservation,
    load_simulation_completion_contract,
    persist_simulation_completion_contract,
)

_TIME = datetime(2026, 8, 14, tzinfo=UTC)


def _model(model_id: str) -> ModelSnapshot:
    """Return one exact provider model identity.

    Args:
        model_id: Provider model identifier.

    Returns:
        Stable model snapshot for a completion reservation.
    """
    return ModelSnapshot(
        provider="openai",
        model_id=model_id,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def _request(model: ModelSnapshot) -> CompletionCostReservation:
    """Return one conservative completion reservation for a model.

    Args:
        model: Exact provider model identity.

    Returns:
        Validated retry-bound request ceiling.
    """
    return completion_cost_reservation(
        model=model,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=1.5,
        maximum_attempts=2,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )


@pytest.mark.parametrize("schema_version", [0, 2])
def test_completion_loader_rejects_unsupported_canonical_schema(
    tmp_path: Path,
    schema_version: int,
) -> None:
    """Reject unsupported envelopes even when their v1 semantic ID is valid.

    Args:
        tmp_path: Isolated artifact root.
        schema_version: Unsupported version written into payload and manifest.
    """
    candidate_a = _model("candidate-a")
    candidate_b = _model("candidate-b")
    world = _model("world")
    paths = ProjectPaths(root=tmp_path / "source", project_id="project-a")
    source = ArtifactStore(paths)
    contract, _contract_input = persist_simulation_completion_contract(
        source,
        inputs=(),
        candidate_requests=(
            CandidateCompletionReservation(
                candidate_alias="candidate-a", request=_request(candidate_a)
            ),
            CandidateCompletionReservation(
                candidate_alias="candidate-b", request=_request(candidate_b)
            ),
        ),
        world_model_alias="world",
        world_model_request=_request(world),
        maximum_attempts=2,
        created_at=_TIME,
        code_revision="test-revision",
    )
    unsupported = contract.model_copy(update={"schema_version": cast(Any, schema_version)})
    stored = source.read(contract.completion_contract_id)
    payload = canonical_json_bytes(unsupported)
    manifest = stored.manifest.model_copy(
        update={
            "schema_version": cast(Any, schema_version),
            "files": (file_digest("completion-contract.json", payload),),
        }
    )
    artifact_directory = paths.artifact_directory(contract.completion_contract_id)
    (artifact_directory / "completion-contract.json").write_bytes(payload)
    (artifact_directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError):
        load_simulation_completion_contract(source, contract.completion_contract_id)
