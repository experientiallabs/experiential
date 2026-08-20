"""Verified judge provenance contracts and read-only judge input verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.judging import CalibrationReport, JudgeCalibration, verify_persisted_calibration
from exp.common.models import (
    CompletionCostReservation,
    ModelCatalog,
    ModelSnapshot,
    verify_completion_reservation,
)
from exp.common.project import ProjectBuildArtifacts, ProjectStore, artifact_input
from exp.optimize.router.automatic.reservations import AutomaticRouterOptions
from exp.optimize.router.judging.artifacts import read_audit
from exp.optimize.router.judging.contracts import (
    ManualJudgeCalibrationAudit,
    ManualJudgeReviewState,
    ManualJudgeSetupArtifact,
    ProvisionalJudgeSetupArtifact,
)


class ProvisionalAutomaticJudge(ContractModel):
    """Verified zero-label judge provenance eligible for provisional optimization only."""

    judgment_status: Literal["provisional"] = "provisional"
    calibration_id: str
    calibration_input: ArtifactInput


class HumanCalibratedAutomaticJudge(ContractModel):
    """Verified human-approved judge provenance and its completed calibration audit."""

    judgment_status: Literal["human_calibrated"] = "human_calibrated"
    calibration_id: str
    calibration_input: ArtifactInput
    audit: ManualJudgeCalibrationAudit
    audit_input: ArtifactInput


AutomaticJudgeProvenance = Annotated[
    ProvisionalAutomaticJudge | HumanCalibratedAutomaticJudge,
    Field(discriminator="judgment_status"),
]


@dataclass(frozen=True)
class HostedAutomaticJudgeEvidence:
    """Prebuilt machine-only judge contract used by the noninteractive hosted path."""

    setup: ProvisionalJudgeSetupArtifact
    setup_input: ArtifactInput
    calibration_id: str
    calibration_input: ArtifactInput
    request_reservation: CompletionCostReservation


def hosted_judge_inputs(
    problems: list[str],
    project: ProjectStore,
    completed: ProjectBuildArtifacts | None,
    judge_alias: str | None,
    judge_model: ModelSnapshot | None,
    evidence: HostedAutomaticJudgeEvidence,
    catalog: ModelCatalog,
    options: AutomaticRouterOptions,
) -> tuple[
    ProvisionalJudgeSetupArtifact | None,
    ArtifactInput | None,
    str | None,
    ArtifactInput | None,
]:
    """Verify machine-only setup, provisional calibration, and request reservation.

    Args:
        problems: Mutable aggregate preflight problem list.
        project: Project-local immutable artifact store.
        completed: Selected completed build, if present.
        judge_alias: Project-frozen judge alias, if available.
        judge_model: Static current judge snapshot, if available.
        evidence: Hosted provisional setup and request reservation.
        catalog: Active transient model catalog.
        options: Active retry and request controls.

    Returns:
        Verified setup, setup pointer, calibration ID, and calibration pointer.
    """
    try:
        setup_stored = project.artifacts.read(evidence.setup_input.artifact_id)
        if setup_stored.manifest.artifact_type != "provisional-judge-setup":
            raise ValueError("setup pointer is not machine-only provisional judge evidence")
        if artifact_input(setup_stored.manifest) != evidence.setup_input:
            raise ValueError("setup manifest digest changed")
        persisted = ProvisionalJudgeSetupArtifact.model_validate_json(
            project.artifacts.read_bytes(evidence.setup_input.artifact_id, "setup.json")
        )
        if persisted != evidence.setup or persisted.setup_id != evidence.setup_input.artifact_id:
            raise ValueError("setup payload differs from its selected artifact")
        calibration, calibration_input = verify_persisted_calibration(
            project,
            evidence.calibration_id,
        )
        if calibration_input != evidence.calibration_input:
            raise ValueError("provisional calibration manifest digest changed")
        if calibration.status != "provisional" or calibration.label_count != 0:
            raise ValueError("hosted judge calibration must remain zero-label provisional evidence")
        if completed is None or (
            persisted.trace_dataset != completed.trace_dataset
            or persisted.task_set != completed.task_set
        ):
            raise ValueError("provisional judge setup differs from the completed build")
        if judge_alias is None or persisted.judge_alias != judge_alias:
            raise ValueError("provisional judge alias differs from the Project role")
        if judge_model is None or persisted.judge_model != judge_model:
            raise ValueError("provisional judge model differs from the active snapshot")
        if (
            calibration.rubric_id != persisted.rubric.artifact_id
            or calibration.judge_model != persisted.judge_model
            or calibration.judge_prompt_id != persisted.prompt_template.prompt.prompt_id
            or calibration.judge_prompt_sha256 != persisted.prompt_template.prompt.sha256
        ):
            raise ValueError("provisional calibration differs from its machine-only setup")
        configured = project.load_project().hosted_judge
        if configured is None or (
            configured.setup != evidence.setup_input
            or configured.calibration != evidence.calibration_input
            or configured.status != "provisional"
        ):
            raise ValueError("Project does not select this provisional judge evidence")
        record = catalog.models.get(judge_alias)
        capabilities = record.capabilities if record is not None else None
        if capabilities is None:
            raise ValueError("judge capability declaration is absent")
        verify_completion_reservation(
            evidence.request_reservation,
            model=persisted.judge_model,
            capabilities=capabilities,
            maximum_attempts=options.completion_maximum_attempts,
        )
        return persisted, evidence.setup_input, calibration.calibration_id, calibration_input
    except (OSError, ValueError) as exc:
        problems.append(f"hosted provisional judge: {exc}")
        return None, None, None, None


def manual_judge_inputs(
    problems: list[str],
    project: ProjectStore,
    completed: ProjectBuildArtifacts | None,
    judge_alias: str | None,
    judge_model: ModelSnapshot | None,
) -> tuple[
    ManualJudgeSetupArtifact | None,
    ArtifactInput | None,
    AutomaticJudgeProvenance | None,
]:
    """Verify approved setup and calibration bind the exact completed build and judge.

    Args:
        problems: Mutable aggregate problem list.
        project: Project-local review and artifact store.
        completed: Completed build pointers, if available.
        judge_alias: Build-frozen judge alias.
        judge_model: Exact current judge identity.

    Returns:
        Verified setup and typed calibration provenance, or absent values after a problem.
    """
    review = project.read_review()
    if not isinstance(review, dict) or review.get("manual_judge") is None:
        problems.append(
            "manual judge: run `exp config judge setup PROJECT`, then "
            "`exp config judge calibrate PROJECT --approve`"
        )
        return None, None, None
    try:
        state = ManualJudgeReviewState.model_validate(review["manual_judge"])
        setup = ManualJudgeSetupArtifact.model_validate_json(
            project.artifacts.read_bytes(state.setup.artifact_id, "setup.json")
        )
        setup_input = artifact_input(project.artifacts.read(state.setup.artifact_id).manifest)
        if setup_input != state.setup or setup.setup_id != state.setup.artifact_id:
            raise ValueError("approved setup pointer changed")
        selected_calibration = state.approved_calibration or state.provisional_calibration
        if selected_calibration is None:
            raise ValueError("calibration has neither provisional nor approved provenance")
        calibration, calibration_input = verify_persisted_calibration(
            project, selected_calibration.artifact_id
        )
        if calibration_input != selected_calibration:
            raise ValueError("selected calibration pointer changed")
        if state.approved_calibration is not None:
            if state.audit is None:
                raise ValueError("approved calibration has no completed audit")
            audit = read_audit(project, state.audit)
            _verify_manual_judge_chain(project, state, setup, audit, calibration)
            if calibration.status != "human_calibrated":
                raise ValueError("approved calibration status changed")
            provenance: AutomaticJudgeProvenance = HumanCalibratedAutomaticJudge(
                calibration_id=calibration.calibration_id,
                calibration_input=calibration_input,
                audit=audit,
                audit_input=state.audit,
            )
        else:
            if calibration.status != "provisional":
                raise ValueError("provisional calibration status changed")
            if (
                calibration.rubric_id != setup.rubric.artifact_id
                or calibration.judge_model != setup.judge_model
                or calibration.judge_prompt_id != setup.prompt_template.prompt.prompt_id
                or calibration.judge_prompt_sha256 != setup.prompt_template.prompt.sha256
            ):
                raise ValueError("provisional calibration differs from finalized judge setup")
            provenance = ProvisionalAutomaticJudge(
                calibration_id=calibration.calibration_id,
                calibration_input=calibration_input,
            )
        if completed is not None and (
            setup.trace_dataset != completed.trace_dataset or setup.task_set != completed.task_set
        ):
            raise ValueError("approved judge setup differs from the completed build")
        if judge_alias is not None and setup.judge_alias != judge_alias:
            raise ValueError("approved judge alias differs from the build-frozen judge")
        if judge_model is not None and setup.judge_model != judge_model:
            raise ValueError("approved judge model identity changed")
        return setup, setup_input, provenance
    except (OSError, ValueError) as exc:
        problems.append(f"manual judge: {exc}")
        return None, None, None


def _verify_manual_judge_chain(
    project: ProjectStore,
    state: ManualJudgeReviewState,
    setup: ManualJudgeSetupArtifact,
    audit: ManualJudgeCalibrationAudit,
    calibration: JudgeCalibration,
) -> None:
    """Cross-bind the selected audit to its exact setup and approved calibration lineage.

    The audit budget reserves only the provider calls consented for its own invocation. A
    calibration resumed from persisted trace reviews reserves fewer calls than the recorded
    judgment probes, so the budget must never reserve more calls than the recorded probes and
    its estimate must match its own reserved call count exactly.

    Args:
        project: Project-local immutable artifact store.
        state: Mutable review pointers selected for automatic optimization.
        setup: Manifest-verified finalized judge setup.
        audit: Manifest-verified completed calibration audit.
        calibration: Recursively verified approved calibration.

    Raises:
        ValueError: Any setup, report, provisional calibration, prompt, or budget pin differs.
    """
    report_stored = project.artifacts.read(audit.report.artifact_id)
    report_input = artifact_input(report_stored.manifest)
    report = CalibrationReport.model_validate_json(
        project.artifacts.read_bytes(audit.report.artifact_id, "report.json")
    )
    provisional, provisional_input = verify_persisted_calibration(
        project, audit.provisional_calibration.artifact_id
    )
    prompt = setup.prompt_template.prompt
    expected_estimate = (
        (
            audit.budget.maximum_input_tokens_per_call * audit.budget.input_usd_per_million_tokens
            + audit.budget.maximum_output_tokens_per_call
            * audit.budget.output_usd_per_million_tokens
        )
        / 1_000_000
        * audit.budget.maximum_attempts_per_call
        * audit.budget.call_count
    )
    matching_audits = []
    for artifact_id in project.artifacts.list_ids():
        stored = project.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "manual-judge-calibration-audit":
            continue
        candidate = read_audit(project, artifact_input(stored.manifest))
        if (
            candidate.setup == audit.setup
            and candidate.report == audit.report
            and candidate.provisional_calibration == audit.provisional_calibration
        ):
            matching_audits.append(candidate.audit_id)
    if (
        audit.setup != state.setup
        or matching_audits != [audit.audit_id]
        or report_input != audit.report
        or provisional_input != audit.provisional_calibration
        or provisional.status != "provisional"
        or calibration.out_of_fold_report_id != report.report_id
        or calibration.out_of_fold_report_sha256 != report_input.sha256
        or setup.rubric.artifact_id != calibration.rubric_id
        or report.rubric_id != calibration.rubric_id
        or provisional.rubric_id != calibration.rubric_id
        or setup.judge_model != calibration.judge_model
        or report.judge_model != calibration.judge_model
        or provisional.judge_model != calibration.judge_model
        or prompt.prompt_id != calibration.judge_prompt_id
        or prompt.sha256 != calibration.judge_prompt_sha256
        or report.judge_prompt_id != calibration.judge_prompt_id
        or report.judge_prompt_sha256 != calibration.judge_prompt_sha256
        or provisional.judge_prompt_id != calibration.judge_prompt_id
        or provisional.judge_prompt_sha256 != calibration.judge_prompt_sha256
        or audit.budget.call_count > sum(len(item.probes) for item in audit.judgments)
        or not math.isclose(
            audit.budget.estimated_cost_usd,
            expected_estimate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "selected judge calibration audit differs from its setup, approved lineage, or budget"
        )
