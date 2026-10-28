"""Execute matched checkpoint studies with prospective critics and complete branch accounting."""

import json
from datetime import UTC, datetime

from filelock import Timeout
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.uncertainty import missing_outcome_bounds
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.critic import (
    CriticSpec,
    critic_failure,
    load_critic_prediction,
    predict_success,
    preflight_critic,
)
from exp.optimize.evaluation.continuations.job import (
    MAXIMUM_PHYSICAL_ATTEMPTS,
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.selection import load_selection_plan, selection_jobs
from exp.optimize.evaluation.continuations.study import actor_budget_exhausted, known_outcomes
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
        failure: Sanitized setup/continuation failure if execution could not produce a study.
        critic_failure: Sanitized reason the registered critic produced no estimate, such as
            ``finish_reason:length``. The cell's continuations still run without a forecast,
            and the critic is scored as having failed on this cell rather than abstained.
    """

    arm: str
    job: ContinuationJob
    result: ArtifactInput | None = None
    failure: str | None = None
    critic_failure: str | None = Field(default=None, exclude_if=lambda value: value is None)


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
        or spec.actor_reasoning_effort != job.study.reasoning_effort
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
    if attempts > MAXIMUM_PHYSICAL_ATTEMPTS:
        raise ValueError("checkpoint experiment exceeds its combined actor/critic attempt ceiling")


def _metrics(project: ProjectStore, cells: tuple[ExperimentCell, ...]) -> JsonObject:
    """Keep missing outcomes visible and distinguish recoverability from causal attribution."""
    metrics: JsonObject = {"scope": "resumed-checkpoint measurements; correlated within one task"}
    for arm in ("proposed", "random"):
        group = [cell for cell in cells if cell.arm == arm]
        planned = sum(cell.job.study.repeats for cell in group)
        passed = known = mixed = actor_requests = exhausted = 0
        briers = []
        for cell in group:
            if cell.result is None:
                continue
            result = load_study(project.artifacts, cell.result)
            labels = known_outcomes(row.assessment for row in result.branches)
            passed += sum(labels)
            known += len(labels)
            exhausted += actor_budget_exhausted(result.branches)
            mixed += bool(labels and 0 < sum(labels) < len(labels))
            for branch in result.branches:
                calls = branch.outcome.get("model_calls", [])
                actor_requests += len(calls) if isinstance(calls, list) else 0
            brier = result.metrics()["brier"]
            if isinstance(brier, (float, int)):
                briers.append(float(brier))
        critic_failures: dict[str, int] = {}
        for cell in group:
            if cell.critic_failure is not None:
                critic_failures[cell.critic_failure] = (
                    critic_failures.get(cell.critic_failure, 0) + 1
                )
        metrics[arm] = {
            "checkpoints": len(group),
            "planned_branches": planned,
            "successes": passed,
            "assessable": known,
            "unknown": planned - known,
            "actor_budget_exhausted": exhausted,
            "actor_budget_exhausted_share": exhausted / planned if planned else None,
            "success_bounds": missing_outcome_bounds(passed, planned - known, planned),
            "checkpoints_with_both_success_and_failure": mixed,
            "critic_assessable_checkpoints": len(briers),
            "critic_failures": sum(critic_failures.values()),
            "critic_failure_kinds": dict(sorted(critic_failures.items())),
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
            if (
                spec.critic is None
                or cell.critic_failure is not None
                or prediction.spec != _critic(spec.critic, base)
            ):
                raise ValueError("checkpoint cell changed its registered critic")
            inputs.add(cell.job.study.critic_prediction)
        elif cell.critic_failure is not None:
            failure_id = manifest.artifact_id + "-critic-failure"
            if (
                spec.critic is None
                or store.exists(_critic(spec.critic, base).identity(code_revision))
                or not store.exists(failure_id)
                or _saved_critic_failure(project, failure_id) != cell.critic_failure
            ):
                raise ValueError("checkpoint cell lost its critic failure")
            inputs.add(artifact_input(store.read(failure_id).manifest))
        elif spec.critic is not None and cell.result is not None:
            raise ValueError("checkpoint cell continued without its critic estimate or failure")
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


def _saved_critic_failure(project: ProjectStore, failure_id: str) -> str:
    """Read the reason a cell's critic produced no estimate from its durable record."""
    record = json.loads(project.artifacts.read_bytes(failure_id, "failure.json"))
    if not isinstance(record, dict) or not isinstance(record.get("failure"), str):
        raise ValueError("checkpoint critic failure record is malformed")
    return record["failure"]


def _predict(
    project: ProjectStore,
    template: CriticSpec,
    job: ContinuationJob,
    key: str,
    envelope: ArtifactEnvelope,
    *,
    code_revision: str,
) -> tuple[ContinuationJob, str | None]:
    """Attach the cell's estimate, or durably record why the critic produced none.

    A failed critic, for example one that spends its output allowance before a typed estimate,
    leaves the job without a forecast so the cell's continuations still measure both arms. The
    failure is saved before any continuation runs and reused on resume, so its reason never
    changes and no critic draw is repeated.

    Args:
        project: Project owning the critic draw and the cell.
        template: Registered critic template.
        job: The cell's continuation job without a forecast.
        key: The cell's artifact identity.
        envelope: Experiment provenance for the failure record.
        code_revision: Revision the critic draw is keyed under.

    Returns:
        The job with its saved estimate attached and no failure, or the unchanged job and the
        critic failure reason.

    Raises:
        Timeout: The critic's lease is held elsewhere; the cell is deferred, not failed.
    """
    store, critic = project.artifacts, _critic(template, job)
    failure_id = key + "-critic-failure"
    if store.exists(failure_id):
        return job, _saved_critic_failure(project, failure_id)
    try:
        reference = predict_success(project, critic, code_revision=code_revision)
    except Timeout:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        reason = critic_failure(store, critic, code_revision=code_revision, error=exc)
        identity = critic.identity(code_revision)
        evidence = next(
            (
                artifact_input(store.read(identity + suffix).manifest)
                for suffix in ("-response", "-intent")
                if store.exists(identity + suffix)
            ),
            None,
        )
        store.write_json(
            artifact_id=failure_id,
            artifact_type="checkpoint-experiment-critic-failure",
            envelope=envelope.model_copy(
                update={
                    "inputs": tuple(
                        sorted(
                            {*envelope.inputs, *(() if evidence is None else (evidence,))},
                            key=lambda ref: ref.artifact_id,
                        )
                    )
                }
            ),
            files={"failure.json": {"failure": reason}},
        )
        return job, reason
    probability = load_critic_prediction(store, reference).estimate.probability
    study = job.study.model_copy(
        update={"critic_prediction": reference, "predicted_success": probability}
    )
    return job.model_copy(update={"study": study}), None


def _recover_cell(
    project: ProjectStore,
    spec: CheckpointExperimentSpec,
    arm: str,
    job: ContinuationJob,
    key: str,
    *,
    code_revision: str,
) -> ExperimentCell:
    """Adopt a fully saved child after parent interruption without buying any new draw."""
    reason = None
    if spec.critic is not None:
        identity = _critic(spec.critic, job).identity(code_revision)
        if project.artifacts.exists(key + "-critic-failure"):
            reason = _saved_critic_failure(project, key + "-critic-failure")
        elif project.artifacts.exists(identity):
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
            critic_failure=reason,
        )
    return ExperimentCell(
        arm=arm, job=job, failure="UnfinishedDispatchIntent", critic_failure=reason
    )


def run_checkpoint_experiment(
    project: ProjectStore, spec: CheckpointExperimentSpec, *, code_revision: str
) -> CheckpointExperiment:
    """Run paired cells once under a lease; failed/unknown cells never disappear or redraw.

    A critic that produces no estimate does not fail its cell: the continuations run without a
    forecast and the cell records ``critic_failure``, so selected-versus-random outcomes are
    still measured and the critic is counted as failing there, not as abstaining.
    """
    store = project.artifacts
    identity = spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
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
                    cell = _recover_cell(project, spec, arm, job, key, code_revision=code_revision)
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
                            job, reason = _predict(
                                project,
                                spec.critic,
                                job,
                                key,
                                envelope,
                                code_revision=code_revision,
                            )
                            cell = cell.model_copy(update={"job": job, "critic_failure": reason})
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
                if cell.critic_failure is not None:
                    inputs.add(artifact_input(store.read(key + "-critic-failure").manifest))
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
