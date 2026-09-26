"""Portable report and rollout exports from verified immutable evaluation evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.core.files import write_text_atomic
from exp.common.evaluations import load_evaluation_dataset
from exp.common.evaluations.dataset import EvaluationRow
from exp.common.evaluations.model_report import ModelEvaluationReport
from exp.common.project import ProjectStore
from exp.common.rollouts import RolloutArtifact, RolloutEventKind
from exp.common.tasks import TaskCase, load_task_set
from exp.optimize.evaluation.runs import EvaluationRun, run_directory
from exp.simulation.engines.text.resume import load_rollout


class ReportEvidence(ContractModel):
    """Self-contained local report with each scenario, row, and selected rollout.

    Attributes:
        schema_version: Export structure version.
        project: Local project name.
        run_id: Exact execution receipt.
        report: Verified aggregate model comparison.
        judgment_status: Provisional or human-calibrated project judge status.
        world_model: Configured simulator alias.
        judge_model: Exact judging model identity.
        tasks: Scenarios pinned by the evaluation dataset.
        rows: Model/scenario/repeat outcomes, including invalid evidence.
        rollouts: Immutable selected rollout records.
        simulation_cost_usd: Reconciled experiment simulation spend.
        judge_cost_usd: Reconciled experiment judge spend.
    """

    schema_version: int = 1
    project: str
    run_id: str
    report: ModelEvaluationReport
    judgment_status: Literal["provisional", "human_calibrated"]
    world_model: str
    judge_model: str
    tasks: tuple[TaskCase, ...]
    rows: tuple[EvaluationRow, ...]
    rollouts: tuple[RolloutArtifact, ...]
    simulation_cost_usd: float | None = Field(default=None)
    judge_cost_usd: float | None = Field(default=None)


def load_report_evidence(project: ProjectStore, run: EvaluationRun) -> ReportEvidence:
    """Resolve and verify all report inputs without credentials or provider calls."""
    if run.report_id is None or run.evaluation_id is None:
        raise ValueError("this run has no report yet; resume its saved work")
    stored = project.artifacts.read(run.report_id)
    if stored.manifest.artifact_type != "model-evaluation-report":
        raise ValueError("run report pointer has the wrong artifact type")
    report = ModelEvaluationReport.model_validate_json(
        project.artifacts.read_bytes(run.report_id, "report.json")
    )
    if report.evaluation_id != run.evaluation_id:
        raise ValueError("run report and evaluation identities differ")
    dataset = load_evaluation_dataset(project.artifacts, run.evaluation_id)
    rollouts = tuple(
        load_rollout(project.artifacts, row.rollout_id) for row in dataset.rows if row.rollout_id
    )
    return ReportEvidence(
        project=project.paths.project_id,
        run_id=run.run_id,
        report=report,
        judgment_status=run.prepared.setup.judgment_status,
        world_model=run.prepared.setup.world_model_settings.world_model_alias,
        judge_model=run.prepared.judge_request.model.model_id,
        tasks=load_task_set(project.artifacts, dataset.manifest.task_set_id).tasks,
        rows=dataset.rows,
        rollouts=rollouts,
        simulation_cost_usd=run.simulation_cost_usd,
        judge_cost_usd=run.judge_cost_usd,
    )


def export_report(project: ProjectStore, run: EvaluationRun) -> tuple[Path, Path]:
    """Write standalone JSON and an offline interactive report beside the saved run."""
    evidence = load_report_evidence(project, run)
    root = run_directory(project, run.run_id)
    payload = evidence.model_dump_json()
    write_text_atomic(root / "report.json", payload)
    write_text_atomic(
        root / "rollouts.jsonl",
        "".join(item.model_dump_json() + "\n" for item in evidence.rollouts),
    )
    safe = payload.replace("<", "\\u003c").replace("&", "\\u0026")
    write_text_atomic(
        root / "report.html",
        Path(__file__).with_name("report.html").read_text().replace("__DATA__", safe),
    )
    return root / "report.json", root / "report.html"


def rollout_transcript(rollout: RolloutArtifact) -> str:
    """Render the last complete candidate request and output without duplicating history."""
    spans = [event for event in rollout.spans if event.kind == RolloutEventKind.AGENT_MODEL_CALL]
    if not spans:
        return "No candidate response was saved."
    return json.dumps(spans[-1].payload, ensure_ascii=False, indent=2)
