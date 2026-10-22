"""A round payload names exactly one edit source."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.hillclimb.prompts import PromptEdit
from exp.optimize.evaluation.hillclimb.rounds import ClimbRound, ProposalChoice, plan_round
from exp.optimize.evaluation.hillclimb.tests.climb_test import climb_fixture


def test_a_round_needs_one_edit_and_every_earlier_round_decided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both or neither edit source refuses; round two before round one refuses without calls."""
    project, spec, actor = climb_fixture(tmp_path, monkeypatch, "records")
    edit = PromptEdit(kind="append", text="Check.", change_note="check")
    choice = ProposalChoice(proposals=ArtifactInput(artifact_id="p", sha256="0" * 64), index=0)
    with pytest.raises(ValueError, match="exactly one edit"):
        ClimbRound(climb=spec, round=1, edit=edit, choice=choice, deadline=datetime.now(UTC))
    with pytest.raises(ValueError, match="round 1 of this climb has no saved decision"):
        plan_round(project, ClimbRound(climb=spec, round=2, edit=edit, deadline=datetime.now(UTC)))
    plan = plan_round(
        project, ClimbRound(climb=spec, round=1, edit=edit, deadline=datetime.now(UTC))
    )
    assert plan.candidate.system_prompt == "Check." and plan.incumbent.system_prompt is None
    assert not actor.requests
    assert not project.artifacts.exists(plan.candidate.identity)
