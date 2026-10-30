"""Real workbook interventions separate imposed actions from subsequent actor inference."""

from pathlib import Path

import pytest

from exp.common.models import AssistantAction, ToolCall
from exp.optimize.evaluation.continuations.intervention import (
    ActionInterventionSpec,
    capture_action_intervention,
)
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.study import run_continuation_study
from exp.optimize.evaluation.continuations.study_test import probe, workbook_runtime
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.environments.worlds import ExecutableWorld
from exp.simulation.continuations import load_continuation_checkpoint


def test_two_imposed_actions_restore_the_same_input_then_change_verified_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete-one and delete-two differ only in the scripted decision before identical suffixes."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    original = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    call = workbook_case().first
    results = []
    for count in (1, 2):
        action = call.model_copy(
            update={
                "arguments": {
                    **call.arguments,
                    "deletions": [{"sheet": "Orders", "start": 2, "count": count}],
                }
            }
        )
        spec = ActionInterventionSpec(
            base=job, action=action, rationale="Test whether the number of deleted rows matters."
        )
        intervention = capture_action_intervention(project, spec, code_revision="test")
        point = load_continuation_checkpoint(project.artifacts, intervention.checkpoint)
        assert point.task == original.task and point.actor.calls == original.actor.calls
        assert point.actor.messages[:-2] == original.actor.messages
        assert point.actor.messages[-1].content == intervention.observation.content
        assert intervention.before == probe(original.world)
        assert intervention.after["rows"] == 7 - count
        assert (
            original.world
            == load_continuation_checkpoint(project.artifacts, job.study.checkpoint).world
        )
        assert not transport.requests
        study = job.study.model_copy(
            update={
                "checkpoint": intervention.checkpoint,
                "repeats": 1,
                "selector": "fixed scripted action contrast",
                "predicted_success": None,
                "critic_prediction": None,
                "selection_plan": None,
            }
        )
        result = run_continuation_study(
            project.artifacts,
            study,
            actor=ChatAgentRuntime(maximum_model_calls=4),
            model=_Model(
                (
                    _response(AssistantAction(content="done")).model_copy(
                        update={"model": study.actor_model}
                    ),
                )
            ),
            world=workbook_runtime(original.world),
            probe=probe,
            probe_sha256=study.probe_sha256,
            code_revision="test",
        )
        results.append(result.metrics()["successes"])
        assert capture_action_intervention(project, spec, code_revision="test") == intervention
    assert results == [1, 0]
    job.runtime_file.unlink()
    assert capture_action_intervention(project, spec, code_revision="test") == intervention


def test_lost_reply_after_native_effect_cannot_repeat_the_intervention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable intent remains consumed even if the native edit completed before a lost response."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    spec = ActionInterventionSpec(base=job, action=workbook_case().first, rationale="Lost reply.")
    execute = ExecutableWorld.execute
    calls: list[ToolCall] = []

    def lose_reply(world: ExecutableWorld, action: ToolCall) -> Observation:
        """Execute the real workbook mutation, then fail before its response is captured."""
        calls.append(action)
        execute(world, action)
        raise OSError("lost tool response")

    monkeypatch.setattr(ExecutableWorld, "execute", lose_reply)
    with pytest.raises(OSError, match="lost tool response"):
        capture_action_intervention(project, spec, code_revision="test")
    with pytest.raises(ValueError, match="unknown outcome"):
        capture_action_intervention(project, spec, code_revision="test")
    assert len(calls) == 1 and not transport.requests


def test_invalid_intervention_refuses_before_import_or_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An out-of-contract scripted action cannot bypass the actor's tool visibility boundary."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    spec = ActionInterventionSpec(
        base=job,
        action=ToolCall(call_id="hidden", name="private_grade", arguments={}),
        rationale="An unavailable tool is not a permitted counterfactual.",
    )
    with pytest.raises(ValueError, match="not visible"):
        capture_action_intervention(project, spec, code_revision="test")
    assert not project.artifacts.exists(spec.identity("test")) and not transport.requests
