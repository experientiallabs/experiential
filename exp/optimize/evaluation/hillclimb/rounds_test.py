"""A round payload names exactly one edit source."""

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.hillclimb.prompts import PromptEdit
from exp.optimize.evaluation.hillclimb.rounds import (
    ClimbRound,
    ProposalChoice,
    plan_round,
    run_climb_round,
)
from exp.optimize.evaluation.hillclimb.tests.climb_test import climb_fixture, deadline
from exp.runtime.models.providers.transport import JsonHttpResponse


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


def test_an_interrupted_round_resumes_only_its_registered_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new deadline or code revision would redraw the stages, so the resume is refused."""
    project, spec, actor = climb_fixture(tmp_path, monkeypatch, "records")
    edit = PromptEdit(kind="append", text="Check.", change_note="check")
    request = ClimbRound(climb=spec, round=1, edit=edit, deadline=deadline())
    original = actor.post

    def crash(url: str, **arguments: object) -> JsonHttpResponse:
        """Interrupt the process during the first screening dispatch."""
        del url, arguments
        raise KeyboardInterrupt

    monkeypatch.setattr(actor, "post", crash)
    with pytest.raises(KeyboardInterrupt):
        run_climb_round(project, request, code_revision="test")
    monkeypatch.setattr(actor, "post", original)
    later = request.model_copy(update={"deadline": request.deadline + timedelta(hours=1)})
    with pytest.raises(ValueError, match="registered with deadline"):
        run_climb_round(project, later, code_revision="test")
    with pytest.raises(ValueError, match="under code revision test"):
        run_climb_round(project, request, code_revision="other")
    assert not actor.requests
    # The same instant written in another offset resumes the registered panels, not new ones.
    shifted = request.deadline.astimezone(timezone(timedelta(hours=1)))
    record = run_climb_round(
        project, request.model_copy(update={"deadline": shifted}), code_revision="test"
    )
    assert record.screen.unknown > 0
