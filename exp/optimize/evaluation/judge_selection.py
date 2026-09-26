"""Run-local judge model choices that preserve the project's grading syllabus."""

from datetime import datetime

from exp.common.core.artifacts import ArtifactInput, canonical_json_bytes, stable_id
from exp.common.judging import CalibrationReport, JudgeCalibration, JudgeCalibrationService, Rubric
from exp.common.judging.provenance import read_artifact_json
from exp.common.models import ModelSnapshot
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.router.automatic.provisional import _persist_empty_label_set
from exp.optimize.router.judging.contracts import JudgeSetupArtifact, ProvisionalJudgeSetupArtifact


def select_judge_model(
    project: ProjectStore,
    source: JudgeSetupArtifact,
    calibration: JudgeCalibration,
    *,
    alias: str,
    model: ModelSnapshot,
    created_at: datetime,
    code_revision: str,
) -> tuple[ArtifactInput, str]:
    """Bind another model to the same syllabus with fresh provisional calibration.

    Args:
        project: Owner of the already verified judge evidence.
        source: Project's original executable rubric and prompt contract.
        calibration: Original verified calibration whose lineage split is retained.
        alias: Explicit model choice for this evaluation.
        model: Exact selected model snapshot.
        created_at: Timestamp for the new immutable setup.
        code_revision: Producer revision.

    Returns:
        A run-local setup pointer and zero-label calibration identity. Project defaults
        and any human calibration for the original model are unchanged.
    """
    rubric, rubric_input = read_artifact_json(
        project,
        artifact_id=source.rubric.artifact_id,
        expected_artifact_type="rubric",
        relative_path="rubric.json",
        model_type=Rubric,
    )
    report, _ = read_artifact_json(
        project,
        artifact_id=calibration.out_of_fold_report_id,
        expected_artifact_type="judge-calibration-report",
        relative_path="report.json",
        model_type=CalibrationReport,
    )
    setup_id = stable_id(
        "evaluation-judge",
        {
            "source_setup": source.setup_id,
            "alias": alias,
            "model": model.model_dump(mode="json"),
        },
    )
    setup = ProvisionalJudgeSetupArtifact(
        **source.model_dump(
            exclude={
                "setup_id",
                "judge_alias",
                "judge_model",
                "created_at",
                "code_revision",
                "status",
            }
        ),
        setup_id=setup_id,
        judge_alias=alias,
        judge_model=model,
        created_at=created_at,
        code_revision=code_revision,
    )
    _, manifest = project.artifacts.write_or_replay(
        artifact_id=setup_id,
        artifact_type="provisional-judge-setup",
        envelope=setup,
        envelope_path="setup.json",
        envelope_type=ProvisionalJudgeSetupArtifact,
        files={"setup.json": canonical_json_bytes(setup)},
    )
    labels = _persist_empty_label_set(
        project,
        rubric,
        rubric_input,
        created_at=created_at,
        code_revision=code_revision,
    )
    selected = JudgeCalibrationService().bootstrap_provisional(
        project,
        rubric_id=rubric.rubric_id,
        label_set_id=labels.label_set_id,
        router_lineage_split_id=report.router_lineage_split_id,
        judge_model=model,
        judge_prompt=setup.prompt_template.prompt,
        created_at=created_at,
        code_revision=code_revision,
    )
    return artifact_input(manifest), selected.calibration_id
