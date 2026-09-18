"""Replay-safe simulation execution shared by evaluations and router optimization."""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

from exp.common.evaluations import EvaluationPlan
from exp.common.evaluations.evidence import read_rollout
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore
from exp.common.rollouts import SimulationArtifactSet
from exp.optimize.router.errors import RouterCompositionError
from exp.simulation.engines.text.resume import MAXIMUM_CELL_ATTEMPTS, reexecutable_dispatch_failure
from exp.simulation.orchestration import Simulator
from exp.simulation.specs import SimulationSpec, simulation_spec_digest

logger = logging.getLogger(__name__)


class SimulatorFactory(Protocol):
    """Construct a simulator only after its exact evaluation plan is persisted."""

    def __call__(self, project: ProjectStore, plan: EvaluationPlan) -> Simulator:
        """Return a simulator bound to the exact persisted plan."""


def run_or_load_simulation(
    project: ProjectStore,
    plan: EvaluationPlan,
    spec: SimulationSpec,
    simulator_factory: SimulatorFactory,
    *,
    progress: ProgressHook | None = None,
    progress_detail: str | None = None,
) -> SimulationArtifactSet:
    """Load an exactly completed simulation set or run the simulator to a final one.

    A completed prior run replays without invoking its simulator, so no new provider calls
    are dispatched. When the simulator must run, a produced set that still contains a
    retryable dispatch failure below the attempt cap is superseded evidence, not a final
    result: the simulator is re-invoked so resume re-executes only those cells as fresh
    attempts under whatever ceiling remains. The loop is bounded by
    ``MAXIMUM_CELL_ATTEMPTS``; a cell that exhausts its generations keeps its terminal
    failure rollout and the set becomes final.

    Args:
        project: Project store holding completed simulation artifacts.
        plan: Frozen evaluation plan bound to the injected simulator.
        spec: Phase-scoped simulation specification to load or run.
        simulator_factory: Injected constructor invoked only when no final set exists.
        progress: Optional observer of exact replayed evaluation-cell counts.
        progress_detail: Phase qualifier attached to replayed evaluation-cell counts.

    Returns:
        Immutable index of one final rollout artifact for every selected cell.

    Raises:
        RouterCompositionError: A stored artifact set is ambiguous, drifted, or mismatched,
            or retries did not converge within the attempt cap.
    """
    matches = []
    for artifact_id in project.artifacts.list_ids():
        stored = project.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "simulation-artifact-set":
            continue
        artifact_set = SimulationArtifactSet.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "artifact-set.json")
        )
        if artifact_set.simulation_id != spec.simulation_id:
            continue
        index_payload = project.artifacts.read_bytes(artifact_id, artifact_set.artifacts_path)
        if hashlib.sha256(index_payload).hexdigest() != artifact_set.artifacts_sha256:
            raise RouterCompositionError("simulation artifact-set index digest has drifted")
        rollouts = tuple(
            read_rollout(project.artifacts, rollout_id)[0]
            for rollout_id in artifact_set.artifact_ids
        )
        expected_digest = simulation_spec_digest(spec)
        rollout_cell_ids = tuple(rollout.cell_id for rollout in rollouts)
        if (
            artifact_set.artifact_set_id != artifact_id
            or any(cell_id is None for cell_id in rollout_cell_ids)
            or tuple(sorted(cell_id for cell_id in rollout_cell_ids if cell_id is not None))
            != spec.cell_ids
            or any(
                rollout.source_run_id != spec.simulation_id
                or rollout.simulation_id != spec.simulation_id
                or rollout.simulation_spec_sha256 != expected_digest
                for rollout in rollouts
            )
        ):
            raise RouterCompositionError(
                "completed simulation artifact set differs from phase spec"
            )
        if any(reexecutable_dispatch_failure(rollout) for rollout in rollouts):
            continue
        matches.append(artifact_set)
    if len(matches) > 1:
        raise RouterCompositionError("multiple completed artifact sets name one simulation phase")
    if matches:
        cell_count = len(matches[0].artifact_ids)
        report(
            progress,
            "evaluation cells",
            completed=cell_count,
            total=cell_count,
            detail=progress_detail,
        )
        return matches[0]
    artifact_set = simulator_factory(project, plan).run(spec)
    for _ in range(MAXIMUM_CELL_ATTEMPTS - 1):
        superseded = _reexecutable_cell_count(project, artifact_set)
        if superseded == 0:
            return artifact_set
        logger.warning(
            "%d simulated cell(s) failed with a retryable dispatch failure; "
            "re-executing only those cells as fresh attempts",
            superseded,
        )
        artifact_set = simulator_factory(project, plan).run(spec)
    if _reexecutable_cell_count(project, artifact_set) > 0:
        raise RouterCompositionError(
            "simulation retries did not converge to final evidence within the attempt cap"
        )
    return artifact_set


def _reexecutable_cell_count(
    project: ProjectStore,
    artifact_set: SimulationArtifactSet,
) -> int:
    """Count rollouts in one set that resume would supersede with another attempt.

    Args:
        project: Project store holding the set's immutable rollout artifacts.
        artifact_set: Simulation artifact set produced for one phase.

    Returns:
        Number of retryable dispatch failures still below the attempt cap.
    """
    return sum(
        1
        for rollout_id in artifact_set.artifact_ids
        if reexecutable_dispatch_failure(read_rollout(project.artifacts, rollout_id)[0])
    )
