"""Judged acceptance: the admission evidence and a judged feasible run, never a verifier."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.optimize.evaluation.executable.judged_acceptance import (
    accept_judged_task,
    load_judged_acceptance,
    reward_qualification,
)
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.trial_test import judged_task
from exp.optimize.evaluation.variance.judged_test import route_judge


def test_a_judged_task_is_accepted_only_on_admission_and_a_judged_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused admission or no judged pass leaves the task diagnostic, with every reason."""
    project, task, _ = qualified_fixture(tmp_path, monkeypatch)
    reference, _, _ = judged_task(project, task)
    route_judge(monkeypatch, verdict=lambda evidence: "fail")
    gate = accept_judged_task(
        project,
        reference,
        admission=None,
        admitted=False,
        admission_reasons=("decision_variance: the choice never changed",),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    refused = load_judged_acceptance(project, gate)
    assert not refused.qualified and refused.passes == 0 and refused.assessed == refused.planned
    assert refused.refusals[0] == "decision_variance: the choice never changed"
    assert any("no feasibility branch was judged a pass" in row for row in refused.refusals)
    with pytest.raises(ValueError, match="needs a task with an episode judge"):
        accept_judged_task(
            project,
            task,
            admission=None,
            admitted=True,
            created_at=datetime.now(UTC),
            code_revision="test",
        )
    assert reward_qualification(project, gate, judged=True)[2] is False
    with pytest.raises(ValueError, match="gates only a task with an episode judge"):
        reward_qualification(project, gate, judged=False)
