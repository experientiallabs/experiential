"""Bind qualified task and paired measurement evidence to exact continuation generation.

Qualification covers challenged observable predicates. A paired report supplies finite measurements
against its selected reference. Neither establishes population fidelity or training benefit.
"""

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.continuation_job import ContinuationJob
from exp.optimize.evaluation.continuation_study import ContinuationStudySpec, TrainingEvidence
from exp.optimize.evaluation.equivalence import EquivalenceReport, saved_equivalence
from exp.optimize.evaluation.equivalence_evidence import equivalence_metrics
from exp.optimize.evaluation.executable_qualification import load_executable_qualification
from exp.optimize.evaluation.executable_task import load_executable_task


class TrainingEvidenceAssessment(ContractModel):
    """Re-derived evidence scope, retained with training admission and its dependencies.

    Attributes:
        qualification: Exact independently challenged observable-contract receipt.
        task: Frozen executable task to which the receipt applies.
        qualification_scope: Original limited authority, never widened to native truth.
        comparison: Optional reproduced actor/environment report, including all failed cells.
        comparison_case: Candidate case matching this task's execution settings.
        comparison_scope: Original interpretation limits of the paired measurement.
        comparison_metrics: Measurements for the attached case, with all draws and both
            conditions retained. These are not an admission threshold or a population estimate.
    """

    qualification: ArtifactInput
    task: ArtifactInput
    qualification_scope: str
    comparison: ArtifactInput | None = None
    comparison_case: int | None = None
    comparison_scope: str | None = None
    comparison_metrics: JsonObject | None = None


def qualified_training_job(
    project: ProjectStore,
    *,
    qualification: ArtifactInput,
    comparison: ArtifactInput | None = None,
    comparison_case: int | None = None,
) -> ContinuationJob:
    """Prepare the exact qualified job with evidence pinned before any generation calls.

    This reads saved records only. It does not import environment code, execute native witnesses,
    dispatch providers or choose a favorable comparison case. Callers name the applicable case.
    """
    evidence = TrainingEvidence(
        qualification=qualification, comparison=comparison, comparison_case=comparison_case
    )
    report = load_executable_qualification(project, qualification, allow_native_replay=False)
    task = load_executable_task(project, report.task)
    if task.job.study.training_evidence is not None:
        raise ValueError("qualify a base task without inherited training evidence")
    job = task.job.model_copy(
        update={"study": task.job.study.model_copy(update={"training_evidence": evidence})}
    )
    revision = project.artifacts.read(report.task.artifact_id).manifest.code_revision
    verify_training_evidence(
        project.artifacts, job.study, execution_revision=job.implementation(revision)
    )
    return job


def verify_training_evidence(
    store: ArtifactStore, spec: ContinuationStudySpec, *, execution_revision: str
) -> TrainingEvidenceAssessment | None:
    """Reproduce authority and reject transfer to different worlds or execution policies.

    The execution revision binds the complete job as well as application code. Merely matching
    a checkpoint or verifier cannot admit runs made with another runtime file or configuration.
    """
    evidence = spec.training_evidence
    if evidence is None:
        return None
    project = ProjectStore(store.paths.root, store.paths.project_id)
    qualification = load_executable_qualification(
        project, evidence.qualification, allow_native_replay=False
    )
    if not qualification.qualified:
        raise ValueError("training task is unqualified: " + "; ".join(qualification.refusals))
    task = load_executable_task(project, qualification.task)
    if task.job.study.training_evidence is not None:
        raise ValueError("qualify a base task without inherited training evidence")
    if spec.training_specification is None:
        raise ValueError("qualified training requires an independent pre-registered specification")
    revision = store.read(qualification.task.artifact_id).manifest.code_revision
    expected = task.job.model_copy(
        update={"study": task.job.study.model_copy(update={"training_evidence": evidence})}
    )
    if spec != expected.study or execution_revision != expected.implementation(revision):
        raise ValueError(
            "generation differs from the qualified execution; qualify the changed task and "
            "prepare a fresh training job"
        )
    comparison = None
    if evidence.comparison is not None:
        comparison = _comparison(project, evidence.comparison)
        index = evidence.comparison_case
        if index is None or index >= len(comparison.spec.cases):
            raise ValueError("comparison_case is outside the saved paired report")
        candidate = comparison.spec.cases[index].candidate
        comparison_revision = store.read(evidence.comparison.artifact_id).manifest.code_revision
        if (
            _comparison_execution(candidate) != _comparison_execution(task.job)
            or comparison_revision != revision
        ):
            raise ValueError(
                "comparison candidate differs from the qualified execution; measure this "
                "exact world and actor policy before attaching comparison evidence"
            )
    return TrainingEvidenceAssessment(
        qualification=evidence.qualification,
        task=qualification.task,
        qualification_scope=qualification.scope,
        comparison=evidence.comparison,
        comparison_case=evidence.comparison_case,
        comparison_scope=None if comparison is None else comparison.scope,
        comparison_metrics=None
        if comparison is None
        else equivalence_metrics(
            tuple(row for row in comparison.measurements if row.case == evidence.comparison_case)
        ),
    )


def _comparison(project: ProjectStore, reference: ArtifactInput) -> EquivalenceReport:
    """Reconstruct a report from saved executions and reviews without fresh inference."""
    store = project.artifacts
    manifest = store.read(reference.artifact_id).manifest
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != "actor-equivalence-report"
    ):
        raise ValueError("comparison reference differs from the saved paired report")
    report = EquivalenceReport.model_validate_json(
        store.read_bytes(reference.artifact_id, "report.json")
    )
    if reference.artifact_id != report.spec.identity(manifest.code_revision) + "-result":
        raise ValueError("comparison report lost its frozen identity")
    rebuilt = saved_equivalence(project, report.spec, code_revision=manifest.code_revision)
    if rebuilt != report:
        raise ValueError("comparison report no longer reproduces from saved evidence")
    return report


def _comparison_execution(job: ContinuationJob) -> JsonObject:
    """Project exact execution settings, excluding study bookkeeping and draw count.

    Comparisons measure a separately sampled panel. Its number of draws, selector labels and
    prospective critic records do not change the actor or world. Training witnesses are not
    passed to actors. Checkpoints, tools, prompts, sampling, runtime configuration, probes,
    catalog and budgets must all match; no state or behavior field is discarded.
    """
    return job.model_dump(
        mode="json",
        exclude={
            "study": {
                "repeats",
                "selector",
                "predicted_success",
                "critic_prediction",
                "selection_plan",
                "training_specification",
                "training_evidence",
            }
        },
    )
