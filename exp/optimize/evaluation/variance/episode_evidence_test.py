"""Real continuation branches, contrast cells and trial ledgers become ordered episode evidence."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.action_contrast import run_action_contrast
from exp.optimize.evaluation.continuations.action_contrast_test import contrast_fixture
from exp.optimize.evaluation.continuations.intervention import ActionIntervention
from exp.optimize.evaluation.continuations.job import ContinuationJob, execute_continuation_job
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.executable.ledger import trial_ledger
from exp.optimize.evaluation.executable.qualification import qualify_executable_task
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.executable.trial import open_executable_harbor_trial
from exp.optimize.evaluation.executable.trial_test import _post
from exp.optimize.evaluation.harbor.bridge import serve_harbor_trial
from exp.optimize.evaluation.variance.episode_evidence import (
    ExogenousRecord,
    evidence_from_branch,
    evidence_from_contrast_cell,
    ledger_evidence,
    run_task_sha256,
    run_world_sha256,
)
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint

CONFIGURATION: JsonObject = {"runtime": "evidence-fixture"}


def _reply(index: int, content: str | None, calls: tuple[ToolCall, ...] = ()) -> JsonHttpResponse:
    """Script one chat-completions reply with optional text and tool calls."""
    message: JsonObject = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "type": "function",
                "id": call.call_id,
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ]
    return JsonHttpResponse(
        status_code=200,
        body={
            "id": f"reply-{index}",
            "model": "world-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": message,
                }
            ],
        },
    )


def _branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ContinuationJob, ContinuationBranch, ContinuationCheckpoint]:
    """Run one real workbook branch: text plus a refused call, a valid call, then a final reply."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    job = job.model_copy(update={"study": job.study.model_copy(update={"repeats": 1})})
    first = workbook_case().first
    malformed = first.model_copy(update={"call_id": "malformed", "arguments": {}})
    transport = ScriptedJsonTransport(
        (
            _reply(0, "Let me check the sheet first.", (malformed,)),
            _reply(1, None, (first,)),
            _reply(2, "Deleted the row; done."),
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: transport
    )
    study = execute_continuation_job(project, job, code_revision="test")
    checkpoint = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    return project, job, study.branches[0], checkpoint


def _evidence(
    branch: ContinuationBranch,
    checkpoint: ContinuationCheckpoint,
    exogenous: tuple[ExogenousRecord, ...] = (),
) -> EpisodeEvidence:
    """Convert a branch with fixed fixture identities."""
    return evidence_from_branch(
        branch,
        checkpoint,
        episode_id="branch-0",
        configuration=CONFIGURATION,
        checkpoint_source="checkpoint",
        branch_source="branch",
        exogenous=exogenous,
    )


def test_branch_keeps_prefix_as_context_and_every_scored_event_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Intermediate text, arguments, errors, receipts, states and the final answer all survive."""
    _, _, branch, checkpoint = _branch(tmp_path, monkeypatch)
    assert branch.status == "completed"
    evidence = _evidence(branch, checkpoint)
    assert evidence.complete and evidence.missing == ()
    assert evidence.task_sha256 == run_task_sha256(checkpoint)
    assert evidence.world_sha256 == run_world_sha256(CONFIGURATION) == sha256_json(CONFIGURATION)
    prefix = len(checkpoint.actor.messages)
    context = [event for event in evidence.events if event.scope == "context"]
    assert [event.event_id for event in context][:prefix] == [
        f"prefix.{index}" for index in range(prefix)
    ]
    assert context[-1].event_id == "state.initial" and context[-1].owner == "world"
    user = [event for event in context if event.kind == "user_message"]
    assert user and all(event.owner == "user" for event in user)
    scored = [
        (event.event_id, event.kind, event.owner) for event in evidence.events[len(context) :]
    ]
    assert scored == [
        ("call.0.text", "assistant_message", "actor"),
        ("call.0.tool.0", "tool_call", "actor"),
        ("call.0.tool.0.receipt", "receipt", "harness"),
        ("call.0.tool.0.reply", "tool_error", "world"),
        ("call.1.tool.0", "tool_call", "actor"),
        ("call.1.tool.0.receipt", "receipt", "harness"),
        ("call.1.tool.0.reply", "tool_result", "world"),
        ("call.1.tool.0.state", "state_snapshot", "world"),
        ("call.2.final", "final_answer", "actor"),
        ("stop", "harness_note", "harness"),
    ]
    events = {event.event_id: event for event in evidence.events}
    assert events["call.0.text"].content == "Let me check the sheet first."
    assert events["call.0.text"].source == "branch#outcome.model_calls[0].response.output.content"
    call = events["call.1.tool.0"].content
    assert isinstance(call, dict) and call["arguments"] == workbook_case().first.arguments
    refusal = events["call.0.tool.0.reply"].content
    assert isinstance(refusal, dict) and refusal["is_error"] is True
    assert refusal["tool_call_id"] == "malformed"
    assert events["call.2.final"].content == "Deleted the row; done."
    assert events["call.1.tool.0.state"].content == branch.outcome["final_state"]
    # Conversion is deterministic and the result round-trips as saved JSON.
    assert _evidence(branch, checkpoint) == evidence
    assert EpisodeEvidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_missing_replies_and_interrupted_branches_are_incomplete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped tool reply or unanswered actor call is missing, not silently absent."""
    _, _, branch, checkpoint = _branch(tmp_path, monkeypatch)
    steps, calls = branch.outcome["steps"], branch.outcome["model_calls"]
    assert isinstance(steps, list) and isinstance(calls, list)
    interrupted = branch.model_copy(
        update={
            "status": "outcome_unknown",
            "outcome": {
                **branch.outcome,
                "steps": steps[:1],
                "model_calls": [
                    *calls[:1],
                    calls[1],
                    {key: value for key, value in calls[2].items() if key != "response"},
                ],
                "execution_phase": "actor_model_dispatch",
            },
        }
    )
    evidence = _evidence(interrupted, checkpoint)
    assert not evidence.complete
    first = workbook_case().first.call_id
    assert f"tool call {first} has no recorded step or reply" in evidence.missing
    assert any("actor call 2 has no recorded response" in r for r in evidence.missing)
    assert any("outcome_unknown" in reason for reason in evidence.missing)
    assert "call.2.unanswered" in evidence.event_ids()
    # Exogenous events land after their step; one naming an absent step is refused.
    scheduled = ExogenousRecord(after_step=0, content={"event": "rate change"}, source="clock")
    placed = _evidence(branch, checkpoint, (scheduled,))
    order = [event.event_id for event in placed.events]
    assert order.index("exogenous.0") == order.index("call.0.tool.0.reply") + 1
    with pytest.raises(ValueError, match="exogenous"):
        _evidence(branch, checkpoint, (scheduled.model_copy(update={"after_step": 9}),))
    with pytest.raises(ValueError, match="do not match"):
        _evidence(
            branch.model_copy(update={"outcome": {**branch.outcome, "model_calls": []}}), checkpoint
        )


def test_contrast_cell_scores_the_imposed_action_as_experiment_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The imposed action is scored and not actor-sampled; the suffix follows its actual state."""
    project, spec, _ = contrast_fixture(tmp_path, monkeypatch)
    report = run_action_contrast(project, spec, code_revision="test")
    checkpoint = load_continuation_checkpoint(project.artifacts, spec.base.study.checkpoint)
    for index, cell in enumerate(report.cells):
        assert cell.intervention is not None and cell.study is not None
        intervention = ActionIntervention.model_validate_json(
            project.artifacts.read_bytes(cell.intervention.artifact_id, "intervention.json")
        )
        suffix = load_study(project.artifacts, cell.study).branches[0]
        evidence = evidence_from_contrast_cell(
            cell,
            spec,
            checkpoint,
            intervention=intervention,
            suffix=suffix,
            episode_id=f"cell-{index}",
            contrast_source="contrast",
            checkpoint_source="checkpoint",
            intervention_source=cell.intervention.artifact_id,
            suffix_source=cell.study.artifact_id,
        )
        assert evidence.complete
        assert evidence.task_sha256 == run_task_sha256(checkpoint)
        assert evidence.world_sha256 == run_world_sha256(spec.base.configuration)
        events = {event.event_id: event for event in evidence.events}
        imposed = events["imposed.call"]
        assert (imposed.kind, imposed.owner, imposed.scope) == ("tool_call", "experiment", "scored")
        action = spec.baseline if cell.arm == "baseline" else spec.alternative
        assert imposed.content == action.model_dump(mode="json")
        assert events["imposed.before"].scope == "context"
        assert events["imposed.reply"].kind == "tool_result"
        assert events["imposed.state"].content == intervention.after
        scored = [event for event in evidence.events if event.scope == "scored"]
        assert scored[0].event_id == "imposed.call"
        assert not any(event.kind == "tool_call" and event.owner == "actor" for event in scored)
        assert events["call.0.final"].content == "done"
    cell = report.cells[0]
    with pytest.raises(ValueError, match="does not match the saved cell"):
        evidence_from_contrast_cell(
            cell,
            spec,
            checkpoint,
            intervention=None,
            suffix=None,
            episode_id="cell",
            contrast_source="contrast",
            checkpoint_source="checkpoint",
        )
    unknown = cell.model_copy(update={"intervention": None, "study": None, "failure": "Timeout"})
    missing = evidence_from_contrast_cell(
        unknown,
        spec,
        checkpoint,
        intervention=None,
        suffix=None,
        episode_id="cell",
        contrast_source="contrast",
        checkpoint_source="checkpoint",
    )
    assert not missing.complete and len(missing.missing) == 2
    assert missing.events[-1].event_id == "imposed.call"


def test_trial_ledger_becomes_evidence_with_dialogue_between_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Customer-facing messages sent through the bridge are scored actor events in order."""
    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    qualification = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    first = workbook_case().first
    with open_executable_harbor_trial(
        project, task, trial_id="judged", code_revision="test", qualification=qualification
    ) as trial:
        with serve_harbor_trial(trial) as endpoint:
            actor = endpoint.actor_token
            _post(endpoint.url + "/message", actor, {"text": "Removing the duplicate row."})
            _post(endpoint.url + "/call", actor, first.model_dump(mode="json"))
            _post(endpoint.url + "/message", actor, {"text": "All set: one row removed."})
            _post(endpoint.url + "/submit", actor, {"answer": "done"})
            assert _post(endpoint.url + "/grade", endpoint.verifier_token, {})["reward"] == 1
        identity = trial.identity
    definition = load_executable_task(project, task)
    checkpoint = load_continuation_checkpoint(project.artifacts, definition.job.study.checkpoint)
    entries = trial_ledger(project, identity)
    evidence = ledger_evidence(
        entries,
        checkpoint,
        episode_id=identity,
        configuration=CONFIGURATION,
        ledger_source=identity,
        checkpoint_source="checkpoint",
    )
    assert evidence.complete
    scored = [
        (event.event_id, event.kind, event.owner)
        for event in evidence.events
        if event.scope == "scored"
    ]
    assert scored == [
        ("ledger.1", "assistant_message", "actor"),
        ("ledger.2", "tool_call", "actor"),
        ("ledger.3.receipt", "receipt", "harness"),
        ("ledger.3.reply", "tool_result", "world"),
        ("ledger.3.state", "state_snapshot", "world"),
        ("ledger.4", "assistant_message", "actor"),
        ("ledger.5", "final_answer", "actor"),
    ]
    assert evidence.events[[e.event_id for e in evidence.events].index("ledger.4")].source == (
        identity + "-ledger-4"
    )
    # An unanswered request is missing evidence; a ledger without submission says so.
    open_call = ledger_evidence(
        entries[:3],
        checkpoint,
        episode_id=identity,
        configuration=CONFIGURATION,
        ledger_source=identity,
        checkpoint_source="checkpoint",
    )
    assert not open_call.complete
    assert open_call.missing == (f"tool call {first.call_id} has no recorded reply or failure",)
    assert open_call.events[-1].event_id == "ledger.end"
    with pytest.raises(ValueError, match="contiguous"):
        ledger_evidence(
            entries[1:],
            checkpoint,
            episode_id=identity,
            configuration=CONFIGURATION,
            ledger_source=identity,
            checkpoint_source="checkpoint",
        )
