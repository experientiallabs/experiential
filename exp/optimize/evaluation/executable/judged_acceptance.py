"""Acceptance of a judged executable task by its frozen rubric judge, with no verifier.

A judged task (one carrying an episode judge) is rewarded only by that judge, optionally with a
bounded dense signal from its tags. It is accepted for reward when:

- the caller's admission evidence holds: for a decision-variance export, the family's admission
  that the decision is real, its outcome differs and the better option is clear, measured by
  judged contrasts from the frozen decision state (:mod:`~exp.optimize.evaluation.variance`);
- a fresh feasibility run of the exact task, from its frozen simulator state, is judged by the
  task's own frozen judge, at least ``minimum_judged`` of its branches receive a verdict and at
  least one is judged a pass, so the task is solvable under the reward it ships with.

No predicate over the final state, no controls and no verifier take part. The acceptance is
saved, passing or not, bound to the task's exact profile.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel, stable_id
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.qualification import load_executable_qualification
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.variance.judged import judge_study, slot_rewards

JUDGED_ACCEPTANCE_TYPE = "judged-acceptance"
"""Artifact type of a saved :class:`JudgedAcceptance`."""

JUDGED_SCOPE = (
    "frozen rubric-judge acceptance: admission evidence plus a judged feasible run of the exact "
    "task; no verifier, no mechanical controls, no native-fidelity or containment certificate"
)
"""The explicit limits of a judged acceptance."""


class JudgedAcceptance(ContractModel):
    """Whether a judged executable task may emit its judge's reward.

    Attributes:
        task: The exact executable task.
        profile_sha256: Its frozen profile.
        admission: The caller's admission evidence, such as a family report.
        admitted: Whether that evidence admits the task.
        admission_reasons: Unmet admission conditions, when it does not.
        feasibility: The saved feasibility run of the exact task.
        judged: The feasibility run judged by the task's frozen judge.
        planned: Planned feasibility branches.
        assessed: Branches that received a pass or fail verdict.
        passes: Branches judged a pass.
        qualified: Whether every acceptance condition holds.
        refusals: Every unmet condition.
        scope: Explicit limits of this acceptance.
    """

    task: ArtifactInput
    profile_sha256: str
    admission: ArtifactInput | None = None
    admitted: bool
    admission_reasons: tuple[str, ...] = ()
    feasibility: ArtifactInput
    judged: ArtifactInput
    planned: int = Field(ge=0)
    assessed: int = Field(ge=0)
    passes: int = Field(ge=0)
    qualified: bool
    refusals: tuple[str, ...] = ()
    scope: str = JUDGED_SCOPE


def accept_judged_task(
    project: ProjectStore,
    task_ref: ArtifactInput,
    *,
    admission: ArtifactInput | None,
    admitted: bool,
    admission_reasons: tuple[str, ...] = (),
    created_at: datetime,
    code_revision: str,
    minimum_judged: float = 0.5,
    concurrency: int = 1,
) -> ArtifactInput:
    """Run, judge and save the acceptance of one judged executable task, passing or not.

    Args:
        project: Project holding the task and its evidence.
        task_ref: The saved executable task; it must carry an episode judge.
        admission: The caller's admission evidence.
        admitted: Whether that evidence admits the task.
        admission_reasons: Unmet admission conditions, when it does not.
        created_at: Record time.
        code_revision: Producing source identity.
        minimum_judged: Least share of feasibility branches that must receive a verdict.
        concurrency: Branches judged at once.

    Returns:
        The saved acceptance.

    Raises:
        ValueError: The task carries no judge or still registers a verifier.
    """
    task = load_executable_task(project, task_ref)
    if task.judge is None:
        raise ValueError("judged acceptance needs a task with an episode judge")
    if task.job.study.verifier is not None:
        raise ValueError("a judged task registers no verifier; its judge is the only grader")
    if saved_continuation_job(project, task.job, code_revision=code_revision) is None:
        execute_continuation_job(project, task.job, code_revision=code_revision)
    feasibility = continuation_result_reference(project, task.job, code_revision=code_revision)
    judged, labels = judge_study(
        project,
        feasibility,
        task.judge,
        configuration=task.job.configuration,
        code_revision=code_revision,
        concurrency=concurrency,
    )
    rewards = [
        reward
        for reward in slot_rewards(project, labels.branches, labels.judge)
        if reward is not None
    ]
    planned, assessed, passes = len(labels.branches), len(rewards), sum(r == 1 for r in rewards)
    refusals = list(admission_reasons if not admitted else ())
    if not admitted and not refusals:
        refusals.append("the admission evidence does not admit the task")
    if planned == 0 or assessed < minimum_judged * planned:
        refusals.append(
            f"the frozen judge assessed {assessed} of {planned} feasibility branches; at least "
            f"{minimum_judged:.0%} must receive a verdict"
        )
    if passes == 0:
        refusals.append("no feasibility branch was judged a pass; the task is not shown solvable")
    report = JudgedAcceptance(
        task=task_ref,
        profile_sha256=task.profile(),
        admission=admission,
        admitted=admitted,
        admission_reasons=admission_reasons,
        feasibility=feasibility,
        judged=judged,
        planned=planned,
        assessed=assessed,
        passes=passes,
        qualified=not refusals,
        refusals=tuple(refusals),
    )
    identity = stable_id(JUDGED_ACCEPTANCE_TYPE, report.model_dump(mode="json"))
    store = project.artifacts
    if not store.exists(identity):
        inputs = (task_ref, feasibility, judged, *(() if admission is None else (admission,)))
        store.write_json(
            artifact_id=identity,
            artifact_type=JUDGED_ACCEPTANCE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=code_revision,
                inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
            ),
            files={"acceptance.json": report},
        )
    return artifact_input(store.read(identity).manifest)


def load_judged_acceptance(project: ProjectStore, reference: ArtifactInput) -> JudgedAcceptance:
    """Read one saved acceptance, refusing another artifact type or changed bytes."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != JUDGED_ACCEPTANCE_TYPE:
        raise ValueError(f"{reference.artifact_id} is not the pinned judged acceptance")
    return JudgedAcceptance.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "acceptance.json")
    )


def reward_qualification(
    project: ProjectStore, reference: ArtifactInput, *, judged: bool
) -> tuple[ArtifactInput, str, bool, str]:
    """Return a reward gate's task, profile, verdict and scope.

    A judged task is gated only by a judged acceptance; an unjudged task only by its witnessed
    observable-contract qualification.

    Args:
        project: Project holding the gate.
        reference: The saved gate.
        judged: Whether the gated task carries an episode judge.

    Raises:
        ValueError: The gate is the wrong kind for the task.
    """
    kind = project.artifacts.read(reference.artifact_id).manifest.artifact_type
    if judged:
        if kind != JUDGED_ACCEPTANCE_TYPE:
            raise ValueError("a judged task is gated only by its judged acceptance")
        accepted = load_judged_acceptance(project, reference)
        return accepted.task, accepted.profile_sha256, accepted.qualified, accepted.scope
    if kind == JUDGED_ACCEPTANCE_TYPE:
        raise ValueError("a judged acceptance gates only a task with an episode judge")
    report = load_executable_qualification(project, reference)
    return report.task, report.profile_sha256, report.qualified, report.scope
