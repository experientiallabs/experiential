"""Canonical persistence of one immutable simulation specification."""

from __future__ import annotations

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ArtifactStore,
    artifact_input,
)
from wmo.simulation.engines.text.errors import SimulationResumeError
from wmo.simulation.specs import SimulationSpec

SPEC_FILE = "simulation-spec.json"


def persist_canonical_specification(
    store: ArtifactStore,
    spec: SimulationSpec,
) -> tuple[SimulationSpec, ArtifactInput]:
    """Persist or adopt the canonical specification for timestamp-stable retries.

    Args:
        store: Project artifact store owning simulation evidence.
        spec: Requested sparse simulation specification.

    Returns:
        Canonical specification and its manifest input.

    Raises:
        SimulationResumeError: Existing specification differs semantically.
    """
    try:
        manifest = store.write_json(
            artifact_id=spec.simulation_id,
            artifact_type="simulation-spec",
            envelope=spec,
            files={SPEC_FILE: spec},
        )
        return spec, artifact_input(manifest)
    except ArtifactAlreadyExistsError as exc:
        stored = store.read(spec.simulation_id)
        if stored.manifest.artifact_type != "simulation-spec":
            raise SimulationResumeError(
                f"artifact {spec.simulation_id!r} exists but is not a simulation specification"
            ) from exc
        try:
            persisted = SimulationSpec.model_validate_json(
                store.read_bytes(spec.simulation_id, SPEC_FILE)
            )
        except (ArtifactCorruptionError, ValueError) as exc:
            raise SimulationResumeError(
                f"simulation specification {spec.simulation_id!r} cannot be read safely"
            ) from exc
        if persisted.model_copy(update={"created_at": spec.created_at}) != spec:
            raise SimulationResumeError(
                f"simulation ID {spec.simulation_id!r} already names a different immutable spec"
            ) from exc
        return persisted, artifact_input(stored.manifest)
