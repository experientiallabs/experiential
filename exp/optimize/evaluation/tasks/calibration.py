"""Persist source-verified task calibration through the existing artifact owner."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import DualReport, WorkflowEvidence
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.source_access import refuse_known_ineligible_sources
from exp.optimize.evaluation.harbor.export import export_harbor_task, verify_harbor_export
from exp.optimize.evaluation.tasks.admission import admit_calibration_evidence
from exp.simulation.evaluation.calibration import CalibrationJudgment, CalibrationPack, calibrate
from exp.simulation.evaluation.tasks import load_evaluation_task, normalized_outcome

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
CALIBRATION_PROTOCOL = "task-calibration-v12"


class SavedCalibrationJudge(ContractModel):
    """An actual dual-evaluator report, never caller-supplied replacement scores.

    Attributes:
        case_id: Exact calibration case associated with this report.
        report: Immutable saved dual result and its request/evidence dependency.
    """

    case_id: str
    report: ArtifactInput


class HarborExportRequest(ContractModel):
    """Explicit delivery settings for the frozen task environment.

    Attributes:
        destination: New immutable package directory outside the source checkout.
        name: Harbor organization/name identity.
        diagnostic: Permit a clearly non-qualified development package.
    """

    destination: Path
    name: str
    diagnostic: bool = Field(default=False, strict=True)


class CalibrationSpec(ContractModel):
    """Explicit frozen calibration registration with no new inference.

    Attributes:
        run_id: Caller identity for this immutable registration.
        pack: Exact task, predicates, independently labeled cases and grouped partitions.
        judgments: Existing evaluator artifacts, empty when expert judging is pending.
        harbor: Optional Harbor package to deliver after retaining the calibration result.
    """

    run_id: str = Field(min_length=1)
    pack: CalibrationPack
    judgments: tuple[SavedCalibrationJudge, ...] = ()
    harbor: HarborExportRequest | None = None

    @model_validator(mode="after")
    def unique_reports(self) -> Self:
        """Reject ambiguous assignment of judge evidence before any persistence."""
        if len({row.case_id for row in self.judgments}) != len(self.judgments):
            raise ValueError("judge case assignments must be unique")
        return self


def execute_calibration(
    project: ProjectStore, spec: CalibrationSpec, *, code_revision: str, dry_run: bool = False
) -> ArtifactInput | None:
    """Verify witnesses, compare saved results and persist a reusable scoped report.

    Args:
        project: Existing trusted project containing the task and label witnesses.
        spec: Frozen pack and actual judge references.
        code_revision: Current producing implementation identity.
        dry_run: Validate and compute without publishing artifacts.

    Returns:
        Immutable result reference, or none for a successful dry run.
    """
    qualification_sources = {spec.pack.task, *(row.report for row in spec.judgments)}
    for case in spec.pack.cases:
        qualification_sources.update(case.challenge.source_refs)
        if case.challenge.witness is not None:
            qualification_sources.add(case.challenge.witness)
    refuse_known_ineligible_sources(
        project.artifacts, tuple(qualification_sources), maximum_artifacts=100_000
    )
    task = load_evaluation_task(project.artifacts, spec.pack.task)
    if task.verifier != spec.pack.candidate:
        raise ValueError("candidate verifier differs from the exact task being calibrated")
    references = {spec.pack.task}
    for case in spec.pack.cases:
        references.update(case.challenge.source_refs)
        if case.challenge.witness is not None:
            references.add(case.challenge.witness)
    references.update(row.report for row in spec.judgments)
    for reference in references:
        if artifact_input(project.artifacts.read(reference.artifact_id).manifest) != reference:
            raise ValueError("calibration witness differs from its declared digest")
    judgments = []
    cases = {case.challenge.id: case for case in spec.pack.cases}
    for saved in spec.judgments:
        if saved.case_id not in cases:
            raise ValueError("saved judge report names an unknown calibration case")
        manifest = project.artifacts.read(saved.report.artifact_id).manifest
        if manifest.artifact_type != "workflow-dual-evaluation":
            raise ValueError("judge witness must be an actual dual-evaluator report")
        report = DualReport.model_validate_json(
            project.artifacts.read_bytes(saved.report.artifact_id, "report.json")
        )
        intents = [
            row
            for row in manifest.inputs
            if project.artifacts.read(row.artifact_id).manifest.artifact_type
            == "workflow-evaluation-intent"
        ]
        if len(intents) != 1:
            raise ValueError("judge report must retain one authoritative evaluation intent")
        intent = intents[0]
        if artifact_input(project.artifacts.read(intent.artifact_id).manifest) != intent:
            raise ValueError("judge evaluation intent differs from its exact digest")
        evidence = WorkflowEvidence.model_validate_json(
            project.artifacts.read_bytes(intent.artifact_id, "evidence.json")
        )
        if report.judge_completion_sha256 is not None:
            completion = project.artifacts.read_bytes(intent.artifact_id, "completion.json")
            if json.loads(completion).get("profile_sha256") != report.judge_completion_sha256:
                raise ValueError("judge completion differs from its saved evaluation intent")
        outcome = cases[saved.case_id].challenge.outcome
        if sha256_json(evidence) != report.evidence_sha256 or sha256_json(
            evidence if isinstance(outcome, WorkflowEvidence) else normalized_outcome(evidence)
        ) != sha256_json(outcome):
            raise ValueError("judge was not given the exact labeled outcome")
        judgments.append(
            CalibrationJudgment(
                case_id=saved.case_id,
                outcome_sha256=sha256_json(outcome),
                judge_sha256=report.judge_config_sha256,
                judge_completion_sha256=report.judge_completion_sha256,
                report=saved.report,
                axes=report.axes,
            )
        )
    admission = admit_calibration_evidence(project, spec.pack)
    result = calibrate(spec.pack, tuple(judgments), admission=admission)
    if (
        spec.harbor is not None
        and not spec.harbor.diagnostic
        and result["qualification"] != "qualified_in_scope"
    ):
        raise ValueError(
            "unqualified task requires an explicit diagnostic Harbor export"
            + "".join(
                f"; [{row['reason']}] {row['subject']}: {row['detail']}"
                for row in admission.model_dump(mode="json")["refusals"]
            )
        )
    if dry_run:
        return None
    implementation = CALIBRATION_PROTOCOL
    result["implementation_protocol"] = implementation
    identity = stable_id(
        "task-calibration",
        {
            "run_id": spec.run_id,
            "spec": spec.model_dump(mode="json"),
            "implementation": implementation,
        },
    )
    if project.artifacts.exists(identity):
        reference = artifact_input(project.artifacts.read(identity).manifest)
        saved = json.loads(project.artifacts.read_bytes(identity, "report.json"))
        if sha256_json(saved) != sha256_json(result):
            raise ValueError(
                f"calibration run {spec.run_id} was saved as {saved.get('qualification')}, but "
                f"the current evidence gives {result['qualification']}; the project's exposure "
                "ledger or generated proposals changed since, so re-run under a fresh run ID"
            )
        project.select_evaluation_task(reference)
        _deliver(project, spec, reference)
        return reference
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type="task-calibration",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(references, key=lambda row: row.artifact_id)),
        ),
        files={"spec.json": spec.model_dump(mode="json"), "report.json": result},
    )
    reference = artifact_input(manifest)
    project.select_evaluation_task(reference)
    _deliver(project, spec, reference)
    return reference


def _deliver(project: ProjectStore, spec: CalibrationSpec, calibration: ArtifactInput) -> None:
    """Complete or recover immutable package delivery without inference or overwrites."""
    if spec.harbor is None:
        return
    if spec.harbor.destination.exists():
        binding = verify_harbor_export(spec.harbor.destination)
        if binding.get("task") != spec.pack.task.model_dump(mode="json") or binding.get(
            "calibration"
        ) != calibration.model_dump(mode="json"):
            raise ValueError("Harbor destination belongs to another task or calibration")
        return
    export_harbor_task(
        project,
        spec.pack.task,
        spec.harbor.destination,
        name=spec.harbor.name,
        calibration=calibration,
        diagnostic=spec.harbor.diagnostic,
    )
