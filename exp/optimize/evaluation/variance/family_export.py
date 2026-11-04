"""Harbor export of one decision family's variants, and the job each variant runs under.

:func:`family_executable_task` is the exact task a variant's package starts from: its decision
checkpoint under the family's frozen job (rebound to a frozen variant's runtime definition by
:func:`variant_job`), rewarded only by the variant's frozen judge. :func:`export_family_harbor`
packages it, diagnostic unless the family is eligible, the variant's winner was confirmed on
fresh samples and an executable qualification for that exact task is supplied. Both are
re-exported by :mod:`~exp.optimize.evaluation.variance.family`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.executable.package import export_executable_harbor
from exp.optimize.evaluation.executable.task import ExecutableTask, persist_executable_task
from exp.optimize.evaluation.variance.pipeline_dense import DenseSettings, bind_dense_task
from exp.optimize.evaluation.variance.records import bound_job

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.family import FamilyReport


def export_family_harbor(
    project: ProjectStore,
    family_report: FamilyReport,
    scenario_id: str,
    destination: Path,
    *,
    name: str,
    intent: str,
    code_revision: str,
    qualification: ArtifactInput | None = None,
    dense: DenseSettings | None = None,
) -> Path:
    """Package one variant's checkpoint as a Harbor environment that starts there.

    The package starts at the variant's decision checkpoint, the replayed one when the variant
    forks before the decision. It is diagnostic, awarding no reward, unless the family is
    eligible, the variant's winner was confirmed on fresh samples, and an executable
    qualification for the exact task (:func:`family_executable_task`) is supplied.

    Args:
        project: Project holding the family.
        family_report: The saved family.
        scenario_id: The variant to package.
        destination: New package directory.
        name: Harbor ``organization/name``.
        intent: The capability the environment measures.
        dense: Frozen training signal for the exact task and world.
        code_revision: Producing source identity.
        qualification: Qualification report for the exact executable task, when it has one.

    Returns:
        The published package directory.

    Raises:
        ValueError: The scenario has no decision checkpoint in this family.
    """
    task = family_executable_task(family_report, scenario_id, intent=intent, dense=dense)
    confirmed = family_report.confirmation is not None and any(
        verdict.scenario_id == scenario_id and verdict.status == "separated_matches"
        for verdict in family_report.confirmation.variants
    )
    qualified = family_report.eligibility.eligible and confirmed and qualification is not None
    reference = persist_executable_task(
        project, task, created_at=datetime.now(UTC), code_revision=code_revision
    )
    return export_executable_harbor(
        project,
        reference,
        destination,
        name=name,
        qualification=qualification if qualified else None,
        diagnostic=not qualified,
    )


def family_executable_task(
    family_report: FamilyReport,
    scenario_id: str,
    *,
    intent: str,
    dense: DenseSettings | None = None,
) -> ExecutableTask:
    """Return the executable task that starts at one variant's decision checkpoint.

    It is the exact task :func:`export_family_harbor` packages, so a qualification for it can be
    built before export.

    Args:
        family_report: The saved family.
        scenario_id: The variant.
        intent: The capability the environment measures.
        dense: Frozen training signal for the exact task and world.

    Returns:
        The task under the family's frozen job, rewarded only by the variant's frozen judge.

    Raises:
        ValueError: The scenario has no decision checkpoint or judge binding in this family.
    """
    rows = {row.scenario_id: row for row in family_report.variants}
    row = rows.get(scenario_id)
    checkpoint = None if row is None else row.decision_checkpoint
    if checkpoint is None:
        built = sorted(key for key, item in rows.items() if item.decision_checkpoint is not None)
        raise ValueError(
            f"scenario {scenario_id} has no persisted decision checkpoint; choose from {built}"
        )
    judging = (family_report.judging or {}).get(scenario_id)
    if judging is None:
        raise ValueError(f"scenario {scenario_id} has no frozen judge binding in this family")
    job = family_report.job
    if row is not None and row.frozen is not None:
        if judging.runtime is None:
            raise ValueError(f"frozen scenario {scenario_id} has no frozen runtime definition")
        job = variant_job(job, (row.frozen, judging.runtime))
    task = ExecutableTask(intent=intent, job=bound_job(job, checkpoint), judge=judging.judge)
    return task if dense is None else bind_dense_task(task, dense)


def variant_job(
    job: ContinuationJob, start: tuple[ArtifactInput, JsonObject] | None
) -> ContinuationJob:
    """Return the job a variant's witness and contrasts run under.

    Args:
        job: The family's frozen job.
        start: The variant's frozen start checkpoint and runtime configuration, if frozen.

    Returns:
        The family job, or for a frozen variant the same job rebound to its frozen runtime
        definition; the study checkpoint is rebound per contrast.
    """
    if start is None:
        return job
    return job.model_copy(update={"configuration": start[1]})
