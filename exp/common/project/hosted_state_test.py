"""Tests for write-once hosted pointer binding under the Project lock."""

from pathlib import Path

import pytest

from exp.common.project import ProjectStoreError
from exp.common.project.project import ProjectHostedJudgeEvidence
from exp.optimize.router.automatic.provisional import prepare_hosted_provisional_judge
from exp.optimize.router.automatic.service_test import _REVISION, _TIME, _completed_project
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.scenarios import AuthoredScenario, select_authored_scenarios


def test_judge_evidence_prepared_for_a_replaced_task_set_is_refused(tmp_path: Path) -> None:
    """Evidence prepared for task set A is not installed once the build selects task set B."""
    project, catalog, _ = _completed_project(tmp_path)
    original = project.load_project().build
    assert original is not None
    judge = prepare_hosted_provisional_judge(
        project,
        catalog,
        maximum_input_tokens=32_768,
        maximum_output_tokens=8_192,
        maximum_attempts=RetryPolicy().maximum_attempts,
        created_at=_TIME,
        code_revision=_REVISION,
        select_for_project=False,
    )
    evidence = ProjectHostedJudgeEvidence(
        setup=judge.setup_input, calibration=judge.calibration_input
    )
    replaced = select_authored_scenarios(
        project,
        (AuthoredScenario.model_validate({"scenario_id": "refund", "instruction": "Refund."}),),
    )
    assert replaced != original.task_set
    before = project.load_project()

    with pytest.raises(ProjectStoreError, match="different task set than the selected build"):
        project.bind_hosted_judge_evidence(evidence)
    assert project.load_project() == before
    assert before.hosted_judge is None

    # The same evidence binds once the build selects the task set it was prepared for again.
    assert select_authored_scenarios(project, ()) == original.task_set
    assert project.bind_hosted_judge_evidence(evidence).hosted_judge == evidence
