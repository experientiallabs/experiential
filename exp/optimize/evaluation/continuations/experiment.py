"""Execute matched checkpoint studies with prospective critics and complete branch accounting."""

from datetime import UTC, datetime

from filelock import FileLock, Timeout

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.critic import (
    CriticSpec,
    load_critic_prediction,
    predict_success,
    preflight_critic,
)
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.selection import load_selection_plan, selection_jobs
from exp.optimize.evaluation.executable.deferral import (
    deferred_execution,
    record_execution_deferral,
)


class CheckpointExperimentSpec(ContractModel):
    """One frozen matched population, with an optional prospective critic for every cell.

    Attributes:
        selection: Immutable plan including selector explanations, probabilities and controls.
        critic: Estimator template; only its checkpoint changes between cells. No outcome
            statistics, private state or completed continuation is supplied to it.
    """

    selection: ArtifactInput
    critic: CriticSpec | None = None

    def identity(self, code_revision: str) -> str:
        """Bind the complete experiment before drawing any predictor or continuation result."""
        return stable_id(
            "checkpoint-experiment", {"spec": sha256_json(self), "code": code_revision}
        )


class ExperimentCell(ContractModel):
    """One matched arm with its actual prediction, complete study or explicit failure.

    Attributes:
        arm: Selector-proposed checkpoint or matched random control.
        job: Exact continuation settings, including a saved critic when obtained.
        result: Complete immutable study reference, including unavailable branches.
        failure: Sanitized setup/estimator failure if execution could not produce a study.
    """

    arm: str
    job: ContinuationJob
    result: ArtifactInput | None = None
    failure: str | None = None


class CheckpointExperiment(ContractModel):
    """Every planned matched cell and explicit unmatched count, never a selector reward.

    Attributes:
        spec: Frozen plan and estimator settings.
        cells: All matched cells in their registered alternating order.
        unmatched: Proposals with no admissible random control.
        metrics: Resumed success, unknown counts and critic scores separated by arm.
        scope: Correlation and causal limits on the observations.
    """

    spec: CheckpointExperimentSpec
    cells: tuple[ExperimentCell, ...]
    unmatched: int
    metrics: JsonObject
    scope: str = (
        "development; related checkpoints of one task; recoverability is not action-level blame"
    )


def _jobs(
    project: ProjectStore, spec: CheckpointExperimentSpec
) -> tuple[tuple[str, ContinuationJob], ...]:
    """Read the immutable plan without consulting runtime files or credentials."""
    plan = load_selection_plan(project, spec.selection)
    revision = project.artifacts.read(spec.selection.artifact_id).manifest.code_revision
    return selection_jobs(plan, code_revision=revision, plan_reference=spec.selection)


def _critic(template: CriticSpec, job: ContinuationJob) -> CriticSpec:
    """Permit checkpoint variation only; reject changed actor or grading semantics."""
    spec = template.model_copy(update={"checkpoint": job.study.checkpoint})
    if (
        spec.actor_model != job.study.actor_model
        or spec.maximum_model_calls != job.maximum_model_calls
        or spec.system_prompt != job.system_prompt
        or spec.actor_output_tokens != job.study.maximum_output_tokens
        or spec.temperature != job.study.temperature
        or spec.verifier_sha256 != sha256_json(job.study.verifier)
        or spec.probe_sha256 != job.study.probe_sha256
    ):
        raise ValueError("checkpoint critic template differs from the actor or verifier contract")
    return spec


def preflight_checkpoint_experiment(project: ProjectStore, spec: CheckpointExperimentSpec) -> None:
    """Validate every planned model and context before any client or runtime is constructed."""
    jobs = _jobs(project, spec)
    attempts = 0
    for _, job in jobs:
        preflight_continuation_job(project, job)
        attempts += job.maximum_physical_attempts
        if spec.critic is not None:
            preflight_critic(project, _critic(spec.critic, job))
            attempts += spec.critic.maximum_physical_attempts
    if attempts > 100_000:
        raise ValueError("checkpoint experiment exceeds its combined actor/critic attempt ceiling")


def _metrics(project: ProjectStore, cells: tuple[ExperimentCell, ...]) -> JsonObject:
    """Keep missing outcomes visible and distinguish recoverability from causal attribution."""
    metrics: JsonObject = {"scope": "resumed-checkpoint measurements; correlated within one task"}
    for arm in ("proposed", "random"):
        group = [cell for cell in cells if cell.arm == arm]
        planned = sum(cell.job.study.repeats for cell in group)
        passed = known = mixed = actor_requests = 0
        briers = []
        for cell in group:
            if cell.result is None:
                continue
            result = load_study(project.artifacts, cell.result)
            labels = [
                int(row.assessment.status == "pass")
                for row in result.branches
                if row.assessment is not None and row.assessment.status != "unassessable"
            ]
            passed += sum(labels)
            known += len(labels)
            mixed += bool(labels and 0 < sum(labels) < len(labels))
            for branch in result.branches:
                calls = branch.outcome.get("model_calls", [])
                actor_requests += len(calls) if isinstance(calls, list) else 0
            brier = result.metrics()["brier"]
            if isinstance(brier, (float, int)):
                briers.append(float(brier))
        metrics[arm] = {
            "checkpoints": len(group),
            "planned_branches": planned,
            "successes": passed,
            "assessable": known,
            "unknown": planned - known,
            "success_bounds": None
            if not planned
            else [passed / planned, (passed + planned - known) / planned],
            "checkpoints_with_both_success_and_failure": mixed,
            "critic_assessable_checkpoints": len(briers),
            "mean_checkpoint_brier": sum(briers) / len(briers) if briers else None,
            "actor_logical_requests": actor_requests,
        }
    metrics["resource_scope"] = (
        "logical actor requests only; provider retries, simulator, selector and critic costs "
        "remain in their original receipts; no compute-efficiency conclusion"
    )
    return metrics


def saved_checkpoint_experiment(
    project: ProjectStore, spec: CheckpointExperimentSpec, *, code_revision: str
) -> CheckpointExperiment | None:
    """Reproduce completed measurements and every dependency without rerunning any cell."""
    store = project.artifacts
    identity = spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    report = CheckpointExperiment.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    plan = load_selection_plan(project, spec.selection)
    planned = _jobs(project, spec)
    if (
        report.spec != spec
        or len(report.cells) != len(planned)
        or report.unmatched != sum(pair.control is None for pair in plan.matches)
    ):
        raise ValueError("checkpoint experiment lost its registered population")
    registration = artifact_input(store.read(identity).manifest)
    if store.read(identity).manifest.inputs != (spec.selection,):
        raise ValueError("checkpoint experiment lost its selection dependency")
    if (
        CheckpointExperimentSpec.model_validate_json(store.read_bytes(identity, "spec.json"))
        != spec
    ):
        raise ValueError("checkpoint experiment registration differs")
    dependencies = {registration}
    for index, (cell, (arm, base)) in enumerate(zip(report.cells, planned, strict=True)):
        study = cell.job.study.model_copy(
            update={"critic_prediction": None, "predicted_success": None}
        )
        if cell.arm != arm or cell.job.model_copy(update={"study": study}) != base:
            raise ValueError("checkpoint experiment changed an arm's frozen policy")
        manifest = store.read(f"{identity}-cell-{index}").manifest
        if (
            ExperimentCell.model_validate_json(store.read_bytes(manifest.artifact_id, "cell.json"))
            != cell
        ):
            raise ValueError("checkpoint experiment report differs from a saved cell")
        inputs = {
            registration,
            cell.job.study.checkpoint,
            *deferred_execution(store, manifest.artifact_id),
        }
        if cell.job.study.critic_prediction is not None:
            prediction = load_critic_prediction(store, cell.job.study.critic_prediction)
            if spec.critic is None or prediction.spec != _critic(spec.critic, base):
                raise ValueError("checkpoint cell changed its registered critic")
            inputs.add(cell.job.study.critic_prediction)
        if cell.result is not None:
            if (
                cell.failure is not None
                or continuation_result_reference(project, cell.job, code_revision=code_revision)
                != cell.result
            ):
                raise ValueError("checkpoint cell result differs from its exact runtime job")
            inputs.add(cell.result)
        elif cell.failure is None:
            raise ValueError("checkpoint cell has neither a result nor a retained failure")
        if set(manifest.inputs) != inputs:
            raise ValueError("checkpoint cell lost its evidence dependencies")
        dependencies.add(artifact_input(manifest))
    if (
        set(store.read(identity + "-result").manifest.inputs) != dependencies
        or _metrics(project, report.cells) != report.metrics
    ):
        raise ValueError("checkpoint report does not reproduce its complete cell evidence")
    return report


def _recover_cell(
    project: ProjectStore,
    spec: CheckpointExperimentSpec,
    arm: str,
    job: ContinuationJob,
    *,
    code_revision: str,
) -> ExperimentCell:
    """Adopt a fully saved child after parent interruption without buying any new draw."""
    if spec.critic is not None:
        identity = _critic(spec.critic, job).identity(code_revision)
        if project.artifacts.exists(identity):
            reference = artifact_input(project.artifacts.read(identity).manifest)
            prediction = load_critic_prediction(project.artifacts, reference)
            job = job.model_copy(
                update={
                    "study": job.study.model_copy(
                        update={
                            "critic_prediction": reference,
                            "predicted_success": prediction.estimate.probability,
                        }
                    )
                }
            )
        else:
            return ExperimentCell(arm=arm, job=job, failure="UnfinishedDispatchIntent")
    if saved_continuation_job(project, job, code_revision=code_revision) is not None:
        return ExperimentCell(
            arm=arm,
            job=job,
            result=continuation_result_reference(project, job, code_revision=code_revision),
        )
    return ExperimentCell(arm=arm, job=job, failure="UnfinishedDispatchIntent")


def run_checkpoint_experiment(
    project: ProjectStore, spec: CheckpointExperimentSpec, *, code_revision: str
) -> CheckpointExperiment:
    """Run paired cells once under a lease; failed/unknown cells never disappear or redraw."""
    store = project.artifacts
    identity = spec.identity(code_revision)
    with FileLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_checkpoint_experiment(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_checkpoint_experiment(project, spec)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=(spec.selection,),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="checkpoint-experiment-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        registration = artifact_input(store.read(identity).manifest)
        cells, dependencies = [], {registration}
        for index, (arm, job) in enumerate(_jobs(project, spec)):
            key = f"{identity}-cell-{index}"
            if store.exists(key):
                cell = ExperimentCell.model_validate_json(store.read_bytes(key, "cell.json"))
            else:
                cell = ExperimentCell(arm=arm, job=job)
                if store.exists(key + "-intent") and not deferred_execution(store, key):
                    cell = _recover_cell(project, spec, arm, job, code_revision=code_revision)
                else:
                    if not store.exists(key + "-intent"):
                        store.write_json(
                            artifact_id=key + "-intent",
                            artifact_type="checkpoint-experiment-intent",
                            envelope=envelope,
                            files={"job.json": job},
                        )
                    try:
                        if spec.critic is not None:
                            reference = predict_success(
                                project, _critic(spec.critic, job), code_revision=code_revision
                            )
                            probability = load_critic_prediction(
                                store, reference
                            ).estimate.probability
                            job = job.model_copy(
                                update={
                                    "study": job.study.model_copy(
                                        update={
                                            "critic_prediction": reference,
                                            "predicted_success": probability,
                                        }
                                    )
                                }
                            )
                            cell = cell.model_copy(update={"job": job})
                        if (
                            saved_continuation_job(project, job, code_revision=code_revision)
                            is None
                        ):
                            execute_continuation_job(project, job, code_revision=code_revision)
                        cell = cell.model_copy(
                            update={
                                "result": continuation_result_reference(
                                    project, job, code_revision=code_revision
                                )
                            }
                        )
                    except Timeout:
                        record_execution_deferral(store, key)
                        raise
                    except (ValueError, OSError, RuntimeError) as exc:
                        cell = cell.model_copy(update={"failure": type(exc).__name__})
                inputs = {registration, cell.job.study.checkpoint, *deferred_execution(store, key)}
                if cell.result is not None:
                    inputs.add(cell.result)
                if cell.job.study.critic_prediction is not None:
                    inputs.add(cell.job.study.critic_prediction)
                store.write_json(
                    artifact_id=key,
                    artifact_type="checkpoint-experiment-cell",
                    envelope=envelope.model_copy(
                        update={"inputs": tuple(sorted(inputs, key=lambda ref: ref.artifact_id))}
                    ),
                    files={"cell.json": cell},
                )
            cells.append(cell)
            dependencies.add(artifact_input(store.read(key).manifest))
        plan = load_selection_plan(project, spec.selection)
        report = CheckpointExperiment(
            spec=spec,
            cells=tuple(cells),
            unmatched=sum(pair.control is None for pair in plan.matches),
            metrics=_metrics(project, tuple(cells)),
        )
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="checkpoint-experiment",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(dependencies, key=lambda ref: ref.artifact_id))}
            ),
            files={"report.json": report},
        )
        return report
