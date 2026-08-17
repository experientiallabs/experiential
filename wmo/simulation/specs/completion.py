"""Immutable completion-call reservations consumed by text simulation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    canonical_json_bytes,
    envelope_matches_manifest,
    stable_id,
)
from wmo.common.models import CompletionCostReservation
from wmo.common.project import (
    ArtifactStore,
    ArtifactStoreError,
    artifact_input,
)


class CandidateCompletionReservation(ContractModel):
    """One candidate alias and its exact conservative completion-call ceiling."""

    candidate_alias: ArtifactId
    request: CompletionCostReservation


class SimulationCompletionContract(ArtifactEnvelope):
    """Exact candidate and world-model reservations for one automatic simulation."""

    schema_version: Literal[1] = 1
    completion_contract_id: ArtifactId
    candidate_requests: tuple[CandidateCompletionReservation, ...]
    world_model_alias: ArtifactId
    world_model_request: CompletionCostReservation
    maximum_attempts: int

    @field_validator("candidate_requests")
    @classmethod
    def _require_unique_candidates(
        cls, values: tuple[CandidateCompletionReservation, ...]
    ) -> tuple[CandidateCompletionReservation, ...]:
        """Require at least two unique candidate aliases.

        Args:
            values: Candidate request reservations.

        Returns:
            The unchanged complete reservation tuple.

        Raises:
            ValueError: Fewer than two unique candidates are supplied.
        """
        aliases = tuple(value.candidate_alias for value in values)
        if len(aliases) < 2 or len(set(aliases)) != len(aliases):
            raise ValueError("completion contract needs at least two unique candidates")
        return values

    @model_validator(mode="after")
    def _require_consistent_attempts(self) -> SimulationCompletionContract:
        """Require every request reservation to use the frozen retry bound.

        Returns:
            The unchanged validated contract.

        Raises:
            ValueError: A candidate or world-model retry ceiling differs.
        """
        if self.maximum_attempts <= 0:
            raise ValueError("completion contract maximum_attempts must be positive")
        requests = tuple(item.request for item in self.candidate_requests) + (
            self.world_model_request,
        )
        if any(request.maximum_attempts != self.maximum_attempts for request in requests):
            raise ValueError("completion reservation retry bounds differ from the contract")
        return self


def _completion_contract_semantic(
    *,
    inputs: tuple[ArtifactInput, ...],
    candidate_requests: tuple[CandidateCompletionReservation, ...],
    world_model_alias: ArtifactId,
    world_model_request: CompletionCostReservation,
    maximum_attempts: int,
) -> dict[str, object]:
    """Build the canonical semantic identity payload.

    Args:
        inputs: Sorted unique immutable dependencies.
        candidate_requests: Exact candidate reservations.
        world_model_alias: Selected world-model alias.
        world_model_request: Exact world-model reservation.
        maximum_attempts: Runtime retry ceiling.

    Returns:
        JSON-compatible content-addressed identity payload.
    """
    return {
        "version": "simulation-completion-contract-v1",
        "inputs": [item.model_dump(mode="json") for item in inputs],
        "candidate_requests": [item.model_dump(mode="json") for item in candidate_requests],
        "world_model_alias": world_model_alias,
        "world_model_request": world_model_request.model_dump(mode="json"),
        "maximum_attempts": maximum_attempts,
    }


def persist_simulation_completion_contract(
    store: ArtifactStore,
    *,
    inputs: tuple[ArtifactInput, ...],
    candidate_requests: tuple[CandidateCompletionReservation, ...],
    world_model_alias: ArtifactId,
    world_model_request: CompletionCostReservation,
    maximum_attempts: int,
    created_at: datetime,
    code_revision: str,
) -> tuple[SimulationCompletionContract, ArtifactInput]:
    """Persist or replay one exact completion reservation contract.

    Args:
        store: Project-local immutable artifact store.
        inputs: Sorted unique immutable dependencies.
        candidate_requests: Exact candidate request reservations.
        world_model_alias: Selected world-model alias.
        world_model_request: Exact world-model request reservation.
        maximum_attempts: Runtime retry ceiling.
        created_at: Artifact materialization time.
        code_revision: Exact producer revision.

    Returns:
        Verified contract and its immutable manifest input.

    Raises:
        ValueError: Inputs are noncanonical or replay differs.
    """
    if inputs != tuple(sorted(inputs, key=lambda item: item.artifact_id)) or len(
        {item.artifact_id for item in inputs}
    ) != len(inputs):
        raise ValueError("completion contract inputs must be sorted and unique")
    semantic = _completion_contract_semantic(
        inputs=inputs,
        candidate_requests=candidate_requests,
        world_model_alias=world_model_alias,
        world_model_request=world_model_request,
        maximum_attempts=maximum_attempts,
    )
    contract_id = stable_id("simulation-completion", semantic)
    value = SimulationCompletionContract(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        completion_contract_id=contract_id,
        candidate_requests=candidate_requests,
        world_model_alias=world_model_alias,
        world_model_request=world_model_request,
        maximum_attempts=maximum_attempts,
    )
    try:
        stored, manifest = store.write_or_replay(
            artifact_id=contract_id,
            artifact_type="simulation-completion-contract",
            envelope=value,
            envelope_path="completion-contract.json",
            envelope_type=SimulationCompletionContract,
            files={"completion-contract.json": canonical_json_bytes(value)},
        )
    except ValueError as exc:
        raise ValueError("existing completion contract differs from replay") from exc
    return stored, artifact_input(manifest)


def load_simulation_completion_contract(
    store: ArtifactStore, artifact_id: ArtifactId
) -> tuple[SimulationCompletionContract, ArtifactInput]:
    """Load and recursively verify one completion reservation contract.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Expected completion contract identity.

    Returns:
        Verified contract and exact manifest input.

    Raises:
        ValueError: Artifact type, manifest, or content identity differs.
    """
    try:
        stored = store.read(artifact_id)
        if stored.manifest.artifact_type != "simulation-completion-contract":
            raise ValueError(f"artifact {artifact_id} is not a completion contract")
        value = SimulationCompletionContract.model_validate_json(
            store.read_bytes(artifact_id, "completion-contract.json")
        )
        if value.completion_contract_id != artifact_id:
            raise ValueError("completion contract identity differs from its artifact")
        if not envelope_matches_manifest(value, stored.manifest):
            raise ValueError("completion contract differs from its manifest")
        expected_id = stable_id(
            "simulation-completion",
            _completion_contract_semantic(
                inputs=value.inputs,
                candidate_requests=value.candidate_requests,
                world_model_alias=value.world_model_alias,
                world_model_request=value.world_model_request,
                maximum_attempts=value.maximum_attempts,
            ),
        )
        if expected_id != artifact_id:
            raise ValueError("completion contract content identity is invalid")
        return value, artifact_input(stored.manifest)
    except ArtifactStoreError as exc:
        raise ValueError(str(exc)) from exc
