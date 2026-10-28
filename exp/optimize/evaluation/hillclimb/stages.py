"""One climb stage as an actor panel: two prompt versions of one actor on one side's lineages.

A stage copies each case of its side with the stage's repeat count and the request deadline,
then compares the two prompt versions as panel actors sharing the climbed model. The baseline
version runs the cases' own checkpoints; any other version runs each case's initial checkpoint
re-instructed under that version's prompt. Panel labels name the climb, the stage occurrence and
the arm, so every stage draws fresh continuations even when it repeats an earlier comparison.
"""

from datetime import UTC, datetime

from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.actors import (
    ActorPanelSpec,
    PanelActor,
    instructed_checkpoint,
    run_actor_panel,
)
from exp.optimize.evaluation.continuations.job import ContinuationJob, preflight_continuation_job
from exp.optimize.evaluation.hillclimb.design import ClimbSpec, ClimbSplit, Stage
from exp.optimize.evaluation.hillclimb.measurement import (
    INCUMBENT,
    StageMeasurement,
    arm_outcomes,
    load_panel_report,
    measure_stage,
    require_headroom,
)
from exp.optimize.evaluation.hillclimb.prompts import PromptVersion
from exp.simulation.continuations import persist_continuation_checkpoint

Arms = tuple[PromptVersion, PromptVersion]


def stage_jobs(
    spec: ClimbSpec, split: ClimbSplit, stage: Stage, deadline: datetime
) -> tuple[ContinuationJob, ...]:
    """The side's cases with the stage's repeats and the request's deadline."""
    repeats = spec.stage_repeats(stage)
    return tuple(
        spec.cases[index].model_copy(
            update={
                "study": spec.cases[index].study.model_copy(update={"repeats": repeats}),
                "deadline": deadline,
            }
        )
        for index in split.side(stage)
    )


def stage_lineages(split: ClimbSplit, stage: Stage) -> tuple[str, ...]:
    """Lineage of each stage case, in case order."""
    return tuple(split.lineages[index] for index in split.side(stage))


def stage_attempts(spec: ClimbSpec, split: ClimbSplit, stage: Stage) -> int:
    """Maximum actor HTTP attempts of one stage, both arms."""
    return 2 * sum(spec.cases[index].maximum_physical_attempts for index in split.side(stage))


def validate_stage(
    project: ProjectStore, spec: ClimbSpec, split: ClimbSplit, stage: Stage, arms: Arms
) -> None:
    """Check a stage without saving anything, importing runtime code or calling a provider.

    Args:
        project: Project holding the cases.
        spec: Climb design.
        split: Lineage split.
        stage: Stage to check.
        arms: Incumbent and edited prompt versions.

    Raises:
        ValueError: A case fails continuation preflight, or a version's prompt cannot start
            from a case's checkpoint.
    """
    for job in stage_jobs(spec, split, stage, spec.cases[0].deadline):
        preflight_continuation_job(project, job)
        for version in arms:
            if version.system_prompt != spec.baseline_prompt:
                assert version.system_prompt is not None
                instructed_checkpoint(project, job, version.system_prompt)


def stage_panel(
    project: ProjectStore,
    spec: ClimbSpec,
    split: ClimbSplit,
    stage: Stage,
    occurrence: str,
    arms: Arms,
    deadline: datetime,
    *,
    code_revision: str,
) -> ActorPanelSpec:
    """Save each edited arm's re-instructed checkpoints and return the stage panel.

    Args:
        project: Project holding the cases.
        spec: Climb design.
        split: Lineage split.
        stage: Stage to build.
        occurrence: Words naming this stage occurrence, such as ``round 3 screen``.
        arms: Incumbent (actor 0) and edited (actor 1) prompt versions.
        deadline: Deadline of every stage continuation.
        code_revision: Producing implementation of saved checkpoints.

    Returns:
        The panel comparing the two versions on the side's lineages.
    """
    jobs = stage_jobs(spec, split, stage, deadline)
    climb, first = spec.identity(), spec.cases[0]
    actors = []
    for role, version in zip(("incumbent", "candidate"), arms, strict=True):
        label = f"hill climb {climb} {occurrence} {role}"
        actor = PanelActor(alias=first.actor_alias, model=first.study.actor_model, label=label)
        if version.system_prompt != spec.baseline_prompt:
            assert version.system_prompt is not None
            checkpoints = tuple(
                persist_continuation_checkpoint(
                    project.artifacts,
                    instructed_checkpoint(project, job, version.system_prompt),
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                )
                for job in jobs
            )
            actor = actor.model_copy(
                update={"system_prompt": version.system_prompt, "checkpoints": checkpoints}
            )
        actors.append(actor)
    return ActorPanelSpec(cases=jobs, actors=tuple(actors))


def run_stage(
    project: ProjectStore,
    spec: ClimbSpec,
    split: ClimbSplit,
    stage: Stage,
    occurrence: str,
    arms: Arms,
    deadline: datetime,
    *,
    code_revision: str,
) -> StageMeasurement:
    """Run or replay one stage panel and measure it.

    Args:
        project: Project holding the cases.
        spec: Climb design.
        split: Lineage split.
        stage: Stage to run.
        occurrence: Words naming this stage occurrence.
        arms: Incumbent and edited prompt versions.
        deadline: Deadline of every stage continuation.
        code_revision: Installed execution revision of the panel.

    Returns:
        The stage measurement and decision.

    Raises:
        ValueError: A screening panel shows a saturated train set.
    """
    panel = stage_panel(
        project, spec, split, stage, occurrence, arms, deadline, code_revision=code_revision
    )
    report = run_actor_panel(project, panel, code_revision=code_revision)
    reference = artifact_input(
        project.artifacts.read(panel.identity(code_revision) + "-result").manifest
    )
    lineages = stage_lineages(split, stage)
    if stage == "screen":
        require_headroom(arm_outcomes(project, report, INCUMBENT, lineages))
    return measure_stage(project, spec, stage, report, reference, lineages)


def remeasure(
    project: ProjectStore,
    spec: ClimbSpec,
    split: ClimbSplit,
    measurement: StageMeasurement,
    arms: Arms,
) -> None:
    """Refuse a saved measurement that no longer reproduces from its saved panel.

    Args:
        project: Project holding the panel.
        spec: Climb design.
        split: Lineage split.
        measurement: Saved stage measurement.
        arms: Prompt versions the stage compared, incumbent first.

    Raises:
        ValueError: The panel compared other prompts or lineages, or its measurement differs.
    """
    report = load_panel_report(project, measurement.panel)
    prompts = [actor.system_prompt or spec.baseline_prompt for actor in report.spec.actors]
    if prompts != [version.system_prompt for version in arms] or len(report.spec.cases) != len(
        split.side(measurement.stage)
    ):
        raise ValueError("saved climb panel compared other prompts or lineages")
    lineages = stage_lineages(split, measurement.stage)
    again = measure_stage(project, spec, measurement.stage, report, measurement.panel, lineages)
    if again != measurement:
        raise ValueError("saved climb measurement does not reproduce from its saved panel")
