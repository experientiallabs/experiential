"""Missing state and matching mistakes cannot turn into positive competence evidence."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.admission import ContinuationAdmission
from exp.optimize.evaluation.continuations.admission_test import admitted_workbook_study
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.equivalence.evidence import equivalence_metrics, measure_pair
from exp.simulation.continuations import load_continuation_checkpoint


@pytest.mark.parametrize(
    "fault",
    ["missing_state", "changed_initial", "changed_request", "matching_failure", "interrupted"],
)
def test_material_evidence_cannot_be_inferred_from_matching_final_prose(
    tmp_path: Path, fault: str
) -> None:
    """Alter saved evidence locally and check that a common 'done' answer cannot rescue it."""
    project = ProjectStore(tmp_path, "case")
    ref = admitted_workbook_study(project)
    admission = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(ref.artifact_id, "admission.json")
    )
    study = load_study(project.artifacts, admission.study)
    checkpoint = load_continuation_checkpoint(project.artifacts, study.spec.checkpoint)
    reference = study.branches[0]
    outcome = reference.model_dump(mode="json")["outcome"]
    if fault == "missing_state":
        outcome.pop("final_state")
    elif fault == "changed_initial":
        outcome["initial_state"]["input_digest"] = "different"
    elif fault == "changed_request":
        outcome["model_calls"][0]["request"]["messages"][0]["content"] = "different task"
    elif fault == "matching_failure":
        outcome["final_state"]["rows"] = 4
        reference = reference.model_copy(update={"outcome": outcome})
    candidate = reference.model_copy(update={"outcome": outcome})
    if fault == "interrupted":
        candidate = candidate.model_copy(update={"status": "outcome_unknown"})
    measured = measure_pair(
        reference,
        candidate,
        checkpoint=checkpoint,
        actor_sha256=sha256_json(checkpoint.actor),
        verifier=study.spec.verifier,
        initial_paths=("/input_digest",),
        final_paths=("/rows",),
        case=0,
        repeat=0,
        condition="candidate",
        identity="test-pair",
    )
    if fault == "matching_failure":
        assert measured.material_agreement == "match"
        assert measured.comparison.reference_assessment.task_result == "fail"
        metrics = equivalence_metrics((measured,))["candidate"]
        assert isinstance(metrics, dict) and metrics["both_task_pass"] == 0
        pooled = equivalence_metrics(
            (
                measured,
                measured.model_copy(update={"repeat": 1}),
                measured.model_copy(update={"case": 1, "material_agreement": "different"}),
            )
        )
        cases = pooled["by_case"]
        assert isinstance(cases, dict)
        assert cases["0"]["candidate"]["agreement_bounds"] == [1, 1]
        assert cases["1"]["candidate"]["agreement_bounds"] == [0, 0]
    else:
        assert measured.material_agreement == "unassessable"
        assert measured.differences
    if fault == "interrupted":
        assert measured.comparison.candidate_assessment.task_result == "unassessable"
        assert measured.comparison.candidate.stop_reason == "outcome_unknown"
