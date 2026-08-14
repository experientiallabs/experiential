"""Terminal artifact-set persistence for text-world-model simulation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import datetime

from wmo.common.core.artifacts import ArtifactInput, canonical_json_bytes, stable_id
from wmo.common.project import ArtifactAlreadyExistsError, ArtifactCorruptionError, ArtifactStore
from wmo.common.rollouts import RolloutArtifact, SimulationArtifactSet
from wmo.simulation.engines.text.bindings import sorted_artifact_inputs
from wmo.simulation.engines.text.errors import SimulationResumeError
from wmo.simulation.engines.text.rollout_support import jsonl_bytes, timestamp
from wmo.simulation.specs import SimulationSpec

_ARTIFACT_SET_FILE = "artifact-set.json"
_ARTIFACT_IDS_FILE = "artifact-ids.jsonl"


def persist_artifact_set(
    *,
    store: ArtifactStore,
    plan_input: ArtifactInput,
    task_set_input: ArtifactInput,
    fit_rag_input: ArtifactInput,
    spec: SimulationSpec,
    spec_input: ArtifactInput,
    resolution_input: ArtifactInput,
    rollouts: Sequence[RolloutArtifact],
    clock: Callable[[], datetime],
) -> SimulationArtifactSet:
    """Persist or verify the terminal rollout index for selected cells.

    Args:
        store: Immutable artifact store receiving the terminal index.
        plan_input: Exact persisted evaluation-plan pointer.
        task_set_input: Exact persisted task-set pointer.
        fit_rag_input: Exact persisted fit-only retrieval pointer.
        spec: Simulation specification owning the artifact set.
        spec_input: Exact persisted specification pointer.
        resolution_input: Exact persisted resolution pointer.
        rollouts: Ordered terminal evidence for every selected cell.
        clock: Time source for newly persisted evidence.

    Returns:
        Newly persisted or exactly verified simulation artifact set.

    Raises:
        SimulationResumeError: Existing terminal index content or inputs differ.
    """
    if spec.world_model is None:  # pragma: no cover - text simulator validates this first
        raise SimulationResumeError("text artifact sets require grounded world-model settings")
    artifact_ids = tuple(rollout.artifact_id for rollout in rollouts)
    index_payload = jsonl_bytes(tuple({"artifact_id": item} for item in artifact_ids))
    artifact_set_id = stable_id(
        "simulation-artifact-set",
        {"simulation_id": spec.simulation_id, "artifact_ids": artifact_ids},
    )
    artifact_set = SimulationArtifactSet(
        schema_version=1,
        created_at=timestamp(clock),
        inputs=sorted_artifact_inputs(
            plan_input,
            task_set_input,
            fit_rag_input,
            spec.world_model.grounded_world_model_input,
            spec_input,
            resolution_input,
        ),
        code_revision=spec.code_revision,
        artifact_set_id=artifact_set_id,
        simulation_id=spec.simulation_id,
        artifact_ids=artifact_ids,
        artifacts_path=_ARTIFACT_IDS_FILE,
        artifacts_sha256=hashlib.sha256(index_payload).hexdigest(),
    )
    try:
        store.write(
            artifact_id=artifact_set_id,
            artifact_type="simulation-artifact-set",
            envelope=artifact_set,
            files={
                _ARTIFACT_SET_FILE: canonical_json_bytes(artifact_set),
                _ARTIFACT_IDS_FILE: index_payload,
            },
        )
        return artifact_set
    except ArtifactAlreadyExistsError as exc:
        stored = store.read(artifact_set_id)
        if stored.manifest.artifact_type != "simulation-artifact-set":
            raise SimulationResumeError(
                f"artifact set ID {artifact_set_id!r} is already bound to another artifact type"
            ) from exc
        try:
            existing = SimulationArtifactSet.model_validate_json(
                store.read_bytes(artifact_set_id, _ARTIFACT_SET_FILE)
            )
        except (ArtifactCorruptionError, ValueError) as parse_error:
            raise SimulationResumeError(
                f"simulation artifact set {artifact_set_id!r} cannot be read safely"
            ) from parse_error
        if (
            existing.simulation_id != artifact_set.simulation_id
            or existing.artifact_ids != artifact_set.artifact_ids
            or existing.artifacts_path != artifact_set.artifacts_path
            or existing.artifacts_sha256 != artifact_set.artifacts_sha256
            or existing.inputs != artifact_set.inputs
            or existing.code_revision != artifact_set.code_revision
        ):
            raise SimulationResumeError(
                f"artifact set ID {artifact_set_id!r} already names different rollout evidence"
            ) from exc
        return existing
