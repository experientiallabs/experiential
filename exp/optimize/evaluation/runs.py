"""Durable local evaluation runs shared by terminal and programmatic applications."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    Sha256,
    assert_secret_free,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import file_write_lock
from exp.common.models import ModelCatalog
from exp.common.progress import ProgressEvent, ProgressHook, report
from exp.common.project import ProjectStore
from exp.common.project.paths import validate_local_id
from exp.common.project.records import ProjectRecords
from exp.common.tasks import TaskCase, load_task_set
from exp.common.traces import load_trace_dataset
from exp.optimize.evaluation.contracts import EvaluationBudget
from exp.optimize.evaluation.judging_resume import read_judging_revision
from exp.optimize.evaluation.prepare import (
    ModelEvaluationOptions,
    PreparedModelEvaluation,
    prepare_model_evaluation,
)
from exp.optimize.evaluation.runtime import run_prepared_model_evaluation
from exp.optimize.evaluation.service import ModelEvaluationResult
from exp.optimize.router.judging.artifacts import read_review_state
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.budget import SpendLimitReached
from exp.simulation.engines.text.errors import SimulationContentionError

logger = logging.getLogger(__name__)


class EvaluationPreparationOutdated(ValueError):
    """A saved plan needs fresh cost preparation before another paid dispatch."""


class EvaluationDefaults(ContractModel):
    """Project-local choices reused by new evaluations, separate from frozen run inputs.

    Attributes:
        models: Selected configured worker aliases.
        judge: Optional judge model override; None keeps the project judge and its calibration.
        options: Rollout budgets, parallelism, seed, and independent repeat count.
        minimum_scenarios: Required distinct scenarios, defaulting to twenty.
    """

    models: tuple[str, ...] = ()
    judge: str | None = None
    options: ModelEvaluationOptions = Field(default_factory=ModelEvaluationOptions)
    minimum_scenarios: int = Field(default=20, ge=1)


class EvaluationRun(ContractModel):
    """Saved execution plan and mutable progress index over immutable evaluation artifacts.

    Attributes:
        run_id: Stable local receipt identity.
        project_config_sha256: Exact retained project configuration used for execution and resume.
        created_at: Preparation timestamp reused on exact resume.
        code_revision: Producer revision recorded with immutable evidence.
        prepared: Frozen model, task, judge, and cost bindings.
        judging_revision: Optional explicit retry pass over unchanged saved rollouts.
        status: Current execution lifecycle state.
        spending_limit_usd: Optional total allowance, authorized by explicit launch consent.
            Explicit None keeps accounting without an aggregate spending cap. The field
            must be present in persisted records.
        required_spending_limit_usd: Advisory allowance from the last blocked request.
            Cleared on resume; only the live request ledger decides new admission.
        stage: Most recent engine progress stage.
        completed: Completed units in that stage, when available.
        total: Planned units in that stage, when available.
        report_id: Immutable report after completion.
        evaluation_id: Immutable evaluation dataset after completion.
        simulation_id: Exact simulation recipe after completion.
        simulation_cost_usd: Reconciled simulation spend, including invalid attempts.
        judge_cost_usd: Reconciled judging spend, separate from assistant task cost.
    """

    run_id: str
    project_config_sha256: Sha256
    created_at: datetime
    code_revision: str
    prepared: PreparedModelEvaluation
    judging_revision: ArtifactInput | None = None
    status: Literal["prepared", "running", "interrupted", "paused", "failed", "completed"] = (
        "prepared"
    )
    spending_limit_usd: float | None = Field(gt=0, allow_inf_nan=False)
    required_spending_limit_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    stage: str = "prepared"
    completed: int | None = None
    total: int | None = None
    report_id: str | None = None
    evaluation_id: str | None = None
    simulation_id: str | None = None
    simulation_cost_usd: float | None = None
    judge_cost_usd: float | None = None


def run_directory(project: ProjectStore, run_id: str) -> Path:
    """Resolve an exact run below the project runtime directory without accepting traversal."""
    validate_local_id(run_id, label="evaluation run ID")
    return project.paths.runtime_directory / "evaluations" / run_id


def load_defaults(project: ProjectStore) -> EvaluationDefaults:
    """Load project evaluation choices, with explicit product defaults before first use."""
    payload = project.records.read("evaluation-defaults")
    return (
        EvaluationDefaults() if payload is None else EvaluationDefaults.model_validate_json(payload)
    )


def save_defaults(project: ProjectStore, defaults: EvaluationDefaults) -> None:
    """Atomically save secret-free project choices for subsequent evaluations."""
    assert_secret_free(defaults)
    project.records.write("evaluation-defaults", canonical_json_bytes(defaults))


def evaluation_tasks(project: ProjectStore) -> tuple[TaskCase, ...]:
    """Read the project's selected immutable scenarios or give the ingestion remedy."""
    if not project.exists():
        raise ValueError(f"project is not built; run exp build {project.paths.project_id} first")
    build = project.load_project().build
    if build is None:
        raise ValueError(
            f"project has no grounded scenarios; run exp build {project.paths.project_id} "
            "--traces PATH --source chat-json"
        )
    return load_task_set(project.artifacts, build.task_set.artifact_id).tasks


def prepare_run(
    project: ProjectStore,
    catalog: ModelCatalog,
    defaults: EvaluationDefaults,
    *,
    code_revision: str,
    continuation_of: str | None = None,
    progress: ProgressHook | None = None,
) -> EvaluationRun:
    """Freeze a reproducible scenario/model/repeat matrix without provider calls.

    Args:
        project: Grounded project selected by its local name.
        catalog: Credential-free model metadata used for reservations.
        defaults: Selected models, budgets, repeat count, and required scenario coverage.
        code_revision: Exact producer revision.
        continuation_of: Optional completed incomplete matrix whose budgets will be increased.
        progress: Optional observer of scenario validation and preparation stages.

    Returns:
        A saved prepared run, ready for review and spend consent.
    """
    report(progress, "Loading scenarios")
    config = project.load_project()
    project = project.snapshot(sha256_json(config))
    tasks = evaluation_tasks(project)
    if len(tasks) < defaults.minimum_scenarios:
        raise ValueError(
            f"only {len(tasks)} distinct scenarios; this project requires "
            f"{defaults.minimum_scenarios}. "
            "Ingest more distinct traces before launching a report card."
        )
    for task in tasks:
        names = [tool.name for tool in task.tools]
        if len(names) != len(set(names)):
            raise ValueError(
                f"scenario {task.task_id} has duplicate tool definitions; repair the trace export"
            )
    if config.models is None:
        raise ValueError("project model roles are missing; configure ingestion first")
    assert config.build is not None
    report(progress, "Loading captured traces")
    traces = load_trace_dataset(project.artifacts, config.build.trace_dataset.artifact_id).traces
    by_id = {trace.trace_id: trace for trace in traces}
    report(progress, "Checking tool schemas", completed=0, total=len(tasks))
    for index, task in enumerate(tasks, start=1):
        available = {tool.name for tool in task.tools}
        for trace_id in task.source_trace_ids:
            for span in by_id[trace_id].spans:
                name = span.attributes.get("gen_ai.tool.name")
                if isinstance(name, str) and name not in available:
                    raise ValueError(
                        f"scenario {task.task_id} calls {name!r} without its tool definition; "
                        "add the original tool schema to the trace export and ingest again"
                    )
        report(progress, "Checking tool schemas", completed=index, total=len(tasks))
    report(progress, "Loading judge settings")
    selected = read_review_state(project)
    setup = selected.setup if selected else None
    calibration = (
        (selected.approved_calibration or selected.provisional_calibration) if selected else None
    )
    if selected is not None and calibration is None:
        raise ValueError(
            "project judge setup has no calibration; finish exp config judge before evaluating"
        )
    if selected is None and config.hosted_judge is not None:
        setup = config.hosted_judge.setup
        calibration = config.hosted_judge.calibration
    created_at = datetime.now(UTC)
    run_id = stable_id(
        "eval", {"project": project.paths.project_id, "created_at": created_at.isoformat()}
    )
    prepared = prepare_model_evaluation(
        project,
        catalog,
        defaults.models,
        embedder_alias=config.models.embedder,
        options=defaults.options,
        run_id=run_id,
        judge_setup=setup,
        judge_alias=defaults.judge,
        calibration_id=calibration.artifact_id if calibration else None,
        continuation_of=continuation_of,
        created_at=created_at,
        code_revision=code_revision,
        progress=progress,
    )
    run = EvaluationRun(
        run_id=run_id,
        project_config_sha256=sha256_json(config),
        created_at=created_at,
        code_revision=code_revision,
        prepared=prepared,
        spending_limit_usd=None,
    )
    report(progress, "Saving evaluation")
    save_run(project, run)
    return run


def _run_records(project: ProjectStore) -> ProjectRecords:
    """Select the durable evaluation-run namespace."""
    return ProjectRecords(project.paths.root, project.paths.project_id, "evaluation-runs")


def save_run(project: ProjectStore, run: EvaluationRun) -> None:
    """Commit progress while refusing to rewrite a run's frozen execution inputs."""
    assert_secret_free(run)
    validate_local_id(run.run_id, label="evaluation run ID")
    records = _run_records(project)
    with records.transaction():
        previous = records.read(run.run_id)
        if previous is not None:
            saved = EvaluationRun.model_validate_json(previous)
            fields = {"run_id", "created_at", "code_revision", "prepared", "project_config_sha256"}
            if saved.model_dump(include=fields) != run.model_dump(include=fields):
                raise ValueError("evaluation run inputs are immutable; prepare a new run")
            new_pass = run.judging_revision != saved.judging_revision
            if new_pass:
                if run.judging_revision is None:
                    raise ValueError("evaluation judging history cannot be removed")
                revision = read_judging_revision(project, run.prepared, run.judging_revision)
                if revision.previous != saved.judging_revision:
                    raise ValueError("judging revision must extend the saved judging history")
            if saved.status == "completed" and run != saved:
                if not new_pass:
                    raise ValueError("completed evaluation metadata is immutable")
                run = run.model_copy(
                    update={
                        "status": "prepared",
                        "stage": "prepared",
                        "completed": None,
                        "total": None,
                        "required_spending_limit_usd": None,
                        "report_id": None,
                        "evaluation_id": None,
                        "simulation_id": None,
                        "simulation_cost_usd": None,
                        "judge_cost_usd": None,
                    }
                )
        if run.status == "completed":
            records.append(f"{run.run_id}/{sha256_json(run)}", canonical_json_bytes(run))
        records.write(run.run_id, canonical_json_bytes(run))


def load_run(project: ProjectStore, run_id: str) -> EvaluationRun:
    """Load one exact saved run and reject mismatched persisted identities."""
    validate_local_id(run_id, label="evaluation run ID")
    payload = _run_records(project).read(run_id)
    if payload is None:
        raise ValueError("evaluation run does not exist; select a saved run")
    try:
        run = EvaluationRun.model_validate_json(payload)
    except ValidationError as exc:
        new_fields = {
            ("spending_limit_usd",),
            ("prepared", "cost", "captured_turns"),
            ("prepared", "cost", "measured_turns"),
            ("prepared", "cost", "estimate_basis"),
        }
        if all(error["type"] == "missing" and error["loc"] in new_fields for error in exc.errors()):
            raise EvaluationPreparationOutdated(
                "saved evaluation needs a new cost plan; choose New evaluation in exp eval. "
                "The existing build is ready to reuse."
            ) from None
        raise
    if run.run_id != run_id or run.prepared.setup.run_id != run_id:
        raise ValueError("evaluation run identity differs from its database record")
    return run


def list_runs(project: ProjectStore) -> tuple[EvaluationRun, ...]:
    """List saved runs newest first without constructing provider clients."""
    runs = []
    outdated = 0
    for run_id in _run_records(project).list_ids():
        try:
            runs.append(load_run(project, run_id))
        except EvaluationPreparationOutdated:
            outdated += 1
    if outdated:
        logger.warning(
            "%d saved evaluation(s) need a new preparation; choose New evaluation", outdated
        )
    return tuple(sorted(runs, key=lambda item: item.created_at, reverse=True))


def execute_run(
    project: ProjectStore,
    run: EvaluationRun,
    catalog: RuntimeModelCatalog,
    *,
    provider_spend_consented: bool,
    progress: ProgressHook | None = None,
) -> ModelEvaluationResult:
    """Execute or resume a saved plan under one local owner, preserving completed paid work.

    Args:
        project: Owner of the run and immutable evidence.
        run: Exact prepared run selected for execution.
        catalog: Runtime catalog constructed only after cost consent.
        provider_spend_consented: Authorization for the reviewed run spending limit.
        progress: Optional terminal or application observer.

    Returns:
        The engine's immutable report and actual provider spend.
    """
    if not provider_spend_consented:
        raise ValueError("review the evaluation estimate before launching")
    path = run_directory(project, run.run_id)
    with file_write_lock(path / "execution", what="evaluation run", timeout_s=0.1):
        saved = load_run(project, run.run_id)
        if (
            saved.prepared != run.prepared
            or saved.project_config_sha256 != run.project_config_sha256
        ):
            raise ValueError("supplied run differs from its frozen database snapshot")
        frozen_project = project.snapshot(saved.project_config_sha256)
        active = (
            saved
            if saved.status == "completed"
            else saved.model_copy(update={"status": "running", "required_spending_limit_usd": None})
        )
        save_run(project, active)

        def observe(event: ProgressEvent) -> None:
            """Persist progress before forwarding it to a transient observer."""
            nonlocal active
            if active.status == "completed":
                return
            active = active.model_copy(
                update={"stage": event.stage, "completed": event.completed, "total": event.total}
            )
            save_run(project, active)
            if progress is not None:
                progress(event)

        try:
            result = run_prepared_model_evaluation(
                frozen_project,
                active.prepared,
                catalog,
                budget=EvaluationBudget(
                    maximum_cost_usd=active.spending_limit_usd,
                    maximum_judgments=active.prepared.cost.judgment_count,
                ),
                provider_spend_consented=True,
                created_at=active.created_at,
                code_revision=active.code_revision,
                progress=observe,
                judging_revision=active.judging_revision,
            )
        except SpendLimitReached as exc:
            save_run(
                project,
                active.model_copy(
                    update={
                        "status": "paused",
                        "stage": "Spending limit reached",
                        "required_spending_limit_usd": exc.required_usd,
                    }
                ),
            )
            raise
        except SimulationContentionError:
            save_run(
                project,
                active.model_copy(
                    update={"status": "paused", "stage": "Waiting for rollout state"}
                ),
            )
            raise
        except BaseException as exc:
            if active.status == "completed":
                raise
            save_run(
                project,
                active.model_copy(
                    update={
                        "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    }
                ),
            )
            raise
        save_run(
            project,
            active.model_copy(
                update={
                    "status": "completed",
                    "stage": "report",
                    "report_id": result.report.report_id,
                    "evaluation_id": result.evaluation_id,
                    "simulation_id": result.simulation_spec.simulation_id,
                    "simulation_cost_usd": result.simulation_cost_usd,
                    "judge_cost_usd": result.judge_cost_usd,
                }
            ),
        )
        return result
