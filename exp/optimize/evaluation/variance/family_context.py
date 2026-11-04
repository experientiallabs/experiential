"""The saved inputs of one decision family: its point, rules stage, before-state and context.

Split from :mod:`~exp.optimize.evaluation.variance.family`; nothing here calls a model. The
before-state is probed once through the family job's runtime and then read back from the saved
``decision-context`` artifact.
"""

from __future__ import annotations

from datetime import UTC, datetime

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    load_continuation_runtime,
    preflight_continuation_job,
)
from exp.optimize.evaluation.continuations.study import restore_continuation_world
from exp.optimize.evaluation.rules.stage import REPORT_TYPE, RulesStageReport
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.authors import DecisionContext
from exp.optimize.evaluation.variance.contracts import EstimationPoint
from exp.optimize.evaluation.variance.records import bound_job
from exp.simulation.continuations import load_continuation_checkpoint


def estimation_point(report: ActorCriticReport, point: int) -> EstimationPoint:
    """Return the first judge's estimation point at the chosen step.

    Args:
        report: The actor-critic run.
        point: The chosen point.

    Returns:
        The judge's point.

    Raises:
        ValueError: No usable judge selection chose the point.
    """
    for draw in report.judges:
        if draw.selection is None:
            continue
        for chosen in draw.selection.points:
            if chosen.step == point:
                return chosen
    raise ValueError(f"no judge selection chose point {point}")


def rules_stage_report(project: ProjectStore, reference: ArtifactInput) -> RulesStageReport:
    """Read the pinned rules stage report.

    Args:
        project: Project holding it.
        reference: The ``rules-stage-report`` artifact.

    Returns:
        The report.

    Raises:
        ValueError: The reference is not the saved stage report it pins.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        REPORT_TYPE
    ):
        raise ValueError(
            f"artifact {reference.artifact_id} is not the pinned rules stage report; pass the "
            "reference of a saved run_rules_stage result"
        )
    return RulesStageReport.model_validate_json(stored.payloads["report.json"])


def saved_before(
    project: ProjectStore, draws: str, job: ContinuationJob, checkpoint: ArtifactInput
) -> JsonObject:
    """Return the evaluator state before the step, from the saved context or by probing once.

    Probing restores the point's world through the job's explicitly selected runtime and reads
    its private evaluator projection; no model is called.

    Args:
        project: Project holding the checkpoint.
        draws: Identity the saved context derives from.
        job: Frozen job selecting the runtime.
        checkpoint: The point's checkpoint.

    Returns:
        The before-state.
    """
    store = project.artifacts
    if store.exists(draws + "-context"):
        saved = DecisionContext.model_validate_json(
            store.read_bytes(draws + "-context", "context.json")
        )
        return dict(saved.before)
    bound = bound_job(job, checkpoint)
    preflight_continuation_job(project, bound)
    original = load_continuation_checkpoint(store, checkpoint)
    with (
        load_continuation_runtime(project, bound) as runtime,
        restore_continuation_world(runtime.world, original.world_task, original.world) as world,
    ):
        return dict(runtime.probe(world.checkpoint()))


def save_context(
    project: ProjectStore,
    draws: str,
    context: DecisionContext,
    inputs: tuple[ArtifactInput, ...],
    code_revision: str,
) -> ArtifactInput:
    """Save the decision context once, refusing a different one under the same identity.

    Args:
        project: Project receiving it.
        draws: Identity the context derives from.
        context: What the decision-state author is shown.
        inputs: Saved artifacts it was rendered from, besides the checkpoint.
        code_revision: Producing source identity.

    Returns:
        The context artifact.

    Raises:
        ValueError: A different context is saved under the same identity.
    """
    store, artifact_id = project.artifacts, draws + "-context"
    if store.exists(artifact_id):
        saved = DecisionContext.model_validate_json(store.read_bytes(artifact_id, "context.json"))
        if saved != context:
            raise ValueError("saved decision context differs from the one rendered now")
        return artifact_input(store.read(artifact_id).manifest)
    return artifact_input(
        store.write_json(
            artifact_id=artifact_id,
            artifact_type="decision-context",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(
                    sorted({context.checkpoint, *inputs}, key=lambda ref: ref.artifact_id)
                ),
            ),
            files={"context.json": context},
        )
    )
