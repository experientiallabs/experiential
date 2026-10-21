"""Continuation experiments execute real workbooks and never silently redraw failed branches."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.models import AssistantAction, ModelRequest, ModelResponse, ToolCall
from exp.common.project import ArtifactStore, ProjectPaths
from exp.optimize.evaluation.continuations.study import (
    ContinuationBranch,
    ContinuationStudy,
    ContinuationStudySpec,
    known_outcomes,
    run_continuation_study,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.artifact_world import ArtifactWorldFactory
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.environments.workbook_edits import WorkbookEditAdapter, read_supported_workbook
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.runtime.models.call_evidence import current_call_intent, current_consumer
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    WorldCheckpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.continuations_test import capsule
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier, grade
from exp.simulation.reconstruction.models import resolve


def probe(checkpoint: WorldCheckpoint) -> JsonObject:
    """Read actual native XLSX bytes through a private evaluator projection."""
    assert isinstance(checkpoint, ExecutableCheckpoint)
    output = checkpoint.image.parts.get("output.xlsx")
    return {
        "rows": None if output is None else read_supported_workbook(output)["Orders"].max_row,
        "input_digest": checkpoint.image.domains["input.xlsx"],
    }


def workbook_runtime(checkpoint: WorldCheckpoint) -> ExecutableWorldRuntime:
    """Restore using captured input bytes, avoiding fresh XLSX serialization timestamps."""
    assert isinstance(checkpoint, ExecutableCheckpoint)
    case = workbook_case()
    return ExecutableWorldRuntime(
        case.runtime.manifest,
        ArtifactWorldFactory(
            WorkbookEditAdapter(), files=checkpoint.image.parts, outputs=("output.xlsx",)
        ),
        seed=7,
        events=case.runtime.events,
    )


def test_native_workbook_branches_restore_same_world_and_replay_without_calls(
    tmp_path: Path,
) -> None:
    """Every branch edits a fresh workbook; results replay without another request."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    checkpoint, actor = capsule(store)
    ref = persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )
    case = workbook_case()
    world = workbook_runtime(checkpoint.world)
    # Factory identity is based on bytes; the source fixture captured those exact files.
    spec = ContinuationStudySpec(
        checkpoint=ref,
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=TaskVerifier(
            description="Delete one data row and preserve the input.",
            outcome_schema={},
            required_paths=("/final_state/rows",),
            rules=(OutcomeRule(path="/final_state/rows", operator="equals", value=6),),
            allowed_tools=("edit_workbook",),
            preserved_paths=("/input_digest",),
        ),
        probe_sha256=sha256_json({"probe": "xlsx-row-count-v1"}),
        repeats=2,
        predicted_success=0.75,
        selector="all initial checkpoints; conformance fixture",
    )
    model = _Model(
        tuple(
            response
            for _ in range(2)
            for response in (
                _response(AssistantAction(tool_calls=(case.first,))),
                _response(AssistantAction(content="done")),
            )
        )
    )
    result = run_continuation_study(
        store,
        spec,
        actor=actor,
        model=model,
        world=world,
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    from exp.simulation.continuations import load_continuation_checkpoint

    assert all(len(branch.checkpoints) == 1 for branch in result.branches)
    for branch in result.branches:
        point = load_continuation_checkpoint(store, branch.checkpoints[0])
        assert point.actor.calls == 1 and point.actor.messages[-1].role == "tool"
        assert point.actor.messages[: len(checkpoint.actor.messages)] == checkpoint.actor.messages
        assert probe(point.world)["rows"] == 6
        assert ref in point.sources
    assert result.metrics()["successes"] == 2
    assert result.metrics()["brier"] == 0.0625
    assert len(model.requests) == 4
    assert (
        run_continuation_study(
            store,
            spec,
            actor=actor,
            model=_Model(()),
            world=world,
            probe=probe,
            probe_sha256=spec.probe_sha256,
            code_revision="test",
        )
        == result
    )
    assert result.metrics()["training_admitted"] is False


def test_failed_branches_stay_in_denominator_and_are_not_redrawn(tmp_path: Path) -> None:
    """An unavailable actor result is unknown, never a failed skill label or a fresh retry."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    checkpoint, actor = capsule(store)
    ref = persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = ContinuationStudySpec(
        checkpoint=ref,
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=TaskVerifier(description="Declared test.", outcome_schema={}),
        probe_sha256="a" * 64,
        repeats=2,
        predicted_success=0.5,
        selector="fixture",
    )
    result = run_continuation_study(
        store,
        spec,
        actor=actor,
        model=_Model(()),
        world=workbook_runtime(checkpoint.world),
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    assert result.metrics()["planned"] == 2
    assert result.metrics()["assessable"] == 0
    assert result.metrics()["brier"] is None
    assert all(row.status == "outcome_unknown" for row in result.branches)
    assert all(row.failure == "StopIteration" for row in result.branches)
    with pytest.raises(ValueError, match="changed"):
        run_continuation_study(
            store,
            spec,
            actor=actor,
            model=_Model(()),
            world=workbook_runtime(checkpoint.world),
            probe=probe,
            probe_sha256="b" * 64,
            code_revision="test",
        )


def test_held_out_checkpoint_is_refused_before_actor_dispatch(tmp_path: Path) -> None:
    """A valid held-out capsule cannot become development data through the study API."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    checkpoint, actor = capsule(store)
    task = checkpoint.task.model_copy(update={"partition": "held_out"})
    checkpoint = checkpoint.model_copy(
        update={
            "task": task,
            "actor": checkpoint.actor.model_copy(update={"task_sha256": sha256_json(task)}),
        }
    )
    ref = persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = ContinuationStudySpec(
        checkpoint=ref,
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=TaskVerifier(description="Declared test.", outcome_schema={}),
        probe_sha256="a" * 64,
        selector="fixture",
    )
    model = _Model(())
    with pytest.raises(ValueError, match="requires fit data"):
        run_continuation_study(
            store,
            spec,
            actor=actor,
            model=model,
            world=workbook_runtime(checkpoint.world),
            probe=probe,
            probe_sha256=spec.probe_sha256,
            code_revision="test",
        )
    assert not model.requests


def test_changed_actor_identity_cannot_execute_a_returned_tool_call(tmp_path: Path) -> None:
    """A provider identity mismatch is retained but its action cannot mutate the world."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    checkpoint, actor = capsule(store)
    ref = persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )
    response = _response(AssistantAction(tool_calls=(workbook_case().first,)))
    spec = ContinuationStudySpec(
        checkpoint=ref,
        actor_model=response.model.model_copy(update={"model_id": "another-model"}),
        verifier=TaskVerifier(description="Declared test.", outcome_schema={}),
        probe_sha256="a" * 64,
        repeats=1,
        selector="fixture",
    )
    model = _Model((response,))
    result = run_continuation_study(
        store,
        spec,
        actor=actor,
        model=model,
        world=workbook_runtime(checkpoint.world),
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    assert len(model.requests) == 1
    assert result.branches[0].status == "outcome_unknown"
    assert result.branches[0].outcome["steps"] == []
    assert result.metrics()["assessable"] == 0


def test_mixed_continuations_restore_both_authorities_without_prefix_reexecution(
    tmp_path: Path,
) -> None:
    """Native and simulated prefixes are restored together before each fresh actor suffix."""
    from exp.common.core.artifacts import ArtifactEnvelope
    from exp.common.project import artifact_input
    from exp.runtime.agents.chat import ChatAgentRuntime, ChatCheckpoint
    from exp.runtime.environments.mixed import MixedCheckpoint, StepResult
    from exp.runtime.environments.mixed_test import _order, _refund, _task, _world_runtime
    from exp.simulation.continuations import ContinuationCheckpoint

    store = ArtifactStore(ProjectPaths(tmp_path, "mixed"))
    source = store.write_json(
        artifact_id="origin",
        artifact_type="authored-task",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"task.json": {"origin": "authored mixed-world conformance"}},
    )
    events = []
    world = _world_runtime(events)
    task = _task("place_order", "refund").model_copy(update={"reference_answer": None})
    actor = ChatAgentRuntime(maximum_model_calls=4)
    checkpoints = []
    with world.open(task) as session:

        def capture(state: ChatCheckpoint) -> None:
            """Pair actor context with both backend states after each whole turn."""
            checkpoints.append(
                ContinuationCheckpoint(
                    task=task,
                    actor=state,
                    world=session.checkpoint(),
                    sources=(artifact_input(source),),
                )
            )

        actor.run(
            task,
            model=_Model(
                (
                    _response(AssistantAction(tool_calls=(_order("o1"), _refund("r1")))),
                    _response(AssistantAction(content="done")),
                )
            ),
            environment=session,
            on_checkpoint=capture,
        )
    checkpoint = checkpoints[1]
    assert isinstance(checkpoint.world, MixedCheckpoint)
    assert checkpoint.world.sequence == 2
    ref = persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )

    def mixed_probe(state: WorldCheckpoint) -> JsonObject:
        """Expose logical time to the verifier, with backend images remaining private."""
        assert isinstance(state, MixedCheckpoint)
        assert state.native is not None and state.candidate is not None
        assert state.native.clock == state.candidate.clock
        return {"clock": state.native.clock, "steps": state.sequence}

    spec = ContinuationStudySpec(
        checkpoint=ref,
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=TaskVerifier(
            description="One more simulated refund advances the shared clock.",
            outcome_schema={},
            rules=(OutcomeRule(path="/final_state/clock", operator="equals", value=5),),
        ),
        probe_sha256="a" * 64,
        repeats=2,
        selector="completed two-action prefix",
    )
    before = len([row for row in events if isinstance(row, StepResult)])
    result = run_continuation_study(
        store,
        spec,
        actor=actor,
        model=_Model(
            tuple(
                response
                for _ in range(2)
                for response in (
                    _response(AssistantAction(tool_calls=(_refund("r2", 9),))),
                    _response(AssistantAction(content="done")),
                )
            )
        ),
        world=world,
        probe=mixed_probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    assert result.metrics()["successes"] == 2
    assert len([row for row in events if isinstance(row, StepResult)]) - before == 2
    assert [row.outcome["initial_state"] for row in result.branches] == [
        {"clock": 3, "steps": 2}
    ] * 2
    for row in result.branches:
        assert resolve(row.outcome, "/steps/0/action/name") == "refund"
        assert "image" not in str(resolve(row.outcome, "/model_calls/0/request"))


def test_crash_keeps_original_branch_unknown_and_continues_only_unstarted_branches(
    tmp_path: Path,
) -> None:
    """A durable intent without a result cannot be laundered into a new sampled outcome."""
    from exp.common.models import ModelRequest, ModelResponse

    store = ArtifactStore(ProjectPaths(tmp_path, "crash"))
    checkpoint, actor = capsule(store)
    ref = persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = ContinuationStudySpec(
        checkpoint=ref,
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=TaskVerifier(
            description="Answer done.",
            outcome_schema={},
            rules=(OutcomeRule(path="/final_answer", operator="equals", value="done"),),
        ),
        probe_sha256="a" * 64,
        repeats=2,
        predicted_success=0.5,
        selector="fixture",
    )

    class Interrupted:
        """Represent a process killed after a provider dispatch may have occurred."""

        def complete(self, request: ModelRequest) -> ModelResponse:
            """Leave a durable branch intent without a completed branch response."""
            raise KeyboardInterrupt("process interrupted")

    world = workbook_runtime(checkpoint.world)
    with pytest.raises(KeyboardInterrupt):
        run_continuation_study(
            store,
            spec,
            actor=actor,
            model=Interrupted(),
            world=world,
            probe=probe,
            probe_sha256=spec.probe_sha256,
            code_revision="test",
        )
    model = _Model((_response(AssistantAction(content="done")),))
    resumed = run_continuation_study(
        store,
        spec,
        actor=actor,
        model=model,
        world=world,
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    assert resumed.branches[0].status == "outcome_unknown"
    assert resumed.branches[0].failure == "UnfinishedDispatchIntent"
    assert resumed.branches[1].assessment is not None
    assert resumed.branches[1].assessment.status == "pass"
    assert len(model.requests) == 1
    assert resumed.metrics()["planned"] == 2
    assert resumed.metrics()["unassessable"] == 1
    assert model.requests[0].temperature == spec.temperature
    assert model.requests[0].maximum_output_tokens == spec.maximum_output_tokens


def test_process_interruption_keeps_last_complete_tool_boundary_without_redraw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paired checkpoint is durable before the next actor call, even without a branch result."""
    from exp.common.models import ModelRequest, ModelResponse
    from exp.common.project import artifact_input
    from exp.optimize.evaluation.continuations.job_test import job_fixture
    from exp.optimize.evaluation.continuations.records import load_study
    from exp.runtime.agents.chat import ChatAgentRuntime
    from exp.simulation.continuations import load_continuation_checkpoint

    project, job, _ = job_fixture(tmp_path, monkeypatch)
    point = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    actor = ChatAgentRuntime(maximum_model_calls=job.maximum_model_calls)
    spec = job.study.model_copy(update={"repeats": 1, "selector": "interrupted capture"})

    class InterruptedModel:
        """Stop the process after a real XLSX tool operation and before its next response."""

        def __init__(self) -> None:
            """Start at the registered initial model call."""
            self.calls = 0

        def complete(self, request: ModelRequest) -> ModelResponse:
            """Simulate abrupt process exit outside the normal branch exception handler."""
            del request
            self.calls += 1
            if self.calls == 2:
                raise SystemExit("process interrupted")
            return _response(AssistantAction(tool_calls=(workbook_case().first,))).model_copy(
                update={"model": spec.actor_model}
            )

    model = InterruptedModel()
    with pytest.raises(SystemExit, match="process interrupted"):
        run_continuation_study(
            project.artifacts,
            spec,
            actor=actor,
            model=model,
            world=workbook_runtime(point.world),
            probe=probe,
            probe_sha256=spec.probe_sha256,
            code_revision="test",
        )
    recovered = run_continuation_study(
        project.artifacts,
        spec,
        actor=actor,
        model=_Model(()),
        world=workbook_runtime(point.world),
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    branch = recovered.branches[0]
    assert branch.status == "outcome_unknown" and branch.failure == "UnfinishedDispatchIntent"
    assert len(branch.checkpoints) == 1
    saved = load_continuation_checkpoint(project.artifacts, branch.checkpoints[0])
    assert probe(saved.world)["rows"] == 6 and saved.actor.calls == 1
    assert model.calls == 2
    result_id = next(
        key
        for key in project.artifacts.list_ids(artifact_type="continuation-study")
        if key.endswith("-result")
        and "interrupted capture" in project.artifacts.read_bytes(key, "record.json").decode()
    )
    assert (
        load_study(project.artifacts, artifact_input(project.artifacts.read(result_id).manifest))
        == recovered
    )


def _diagnostic_study(
    store: ArtifactStore,
) -> tuple[ContinuationStudySpec, ChatAgentRuntime, ContinuationCheckpoint]:
    """Bind one native workbook task with an explicit tool policy for boundary regressions."""
    point, actor = capsule(store)
    ref = persist_continuation_checkpoint(
        store, point, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = ContinuationStudySpec(
        checkpoint=ref,
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=TaskVerifier(
            description="Delete one row, without forbidden tools.",
            outcome_schema={},
            required_paths=("/final_state/rows",),
            rules=(OutcomeRule(path="/final_state/rows", operator="equals", value=6),),
            allowed_tools=("edit_workbook",),
        ),
        probe_sha256=sha256_json({"probe": "xlsx-row-count-v1"}),
        repeats=1,
        selector="boundary regression",
    )
    return spec, actor, point


def test_invisible_tool_attempt_is_retained_and_fails_policy(tmp_path: Path) -> None:
    """A refusal before environment dispatch is still evidence of an actor policy violation."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    spec, actor, point = _diagnostic_study(store)
    forbidden = ToolCall(call_id="forbidden", name="send_mail", arguments={"to": "fixture"})
    result = run_continuation_study(
        store,
        spec,
        actor=actor,
        model=_Model((_response(AssistantAction(tool_calls=(forbidden,))),)),
        world=workbook_runtime(point.world),
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    branch = result.branches[0]
    assert branch.assessment is not None and branch.assessment.policy.status == "fail"
    assert branch.outcome["steps"] == [
        {
            "action": forbidden.model_dump(mode="json"),
            "status": "blocked",
            "dispatch": "not_started",
            "reason": "tool_not_visible",
        }
    ]
    assert branch.failure_detail is not None
    assert branch.failure_detail.details["phase"] == "actor_tool_authorization"
    calls = branch.outcome["model_calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    assert "send_mail" not in str(calls[0]["request"])


def test_actor_request_is_durable_before_interruption_and_unknown_is_not_redrawn(
    tmp_path: Path,
) -> None:
    """A process crash cannot erase the final canonical actor input or buy it again on resume."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    spec, actor, point = _diagnostic_study(store)
    seen = []

    class InterruptModel:
        """Inspect retained input at dispatch, then simulate process death."""

        def complete(self, request: ModelRequest) -> ModelResponse:
            reference = current_call_intent()
            assert reference is not None
            saved = ModelRequest.model_validate_json(
                store.read_bytes(reference.artifact_id, "request.json")
            )
            assert saved == request
            assert request.temperature == spec.temperature
            assert request.maximum_output_tokens == spec.maximum_output_tokens
            assert current_consumer() == "actor"
            seen.append(reference)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_continuation_study(
            store,
            spec,
            actor=actor,
            model=InterruptModel(),
            world=workbook_runtime(point.world),
            probe=probe,
            probe_sha256=spec.probe_sha256,
            code_revision="test",
        )
    result = run_continuation_study(
        store,
        spec,
        actor=actor,
        model=InterruptModel(),
        world=workbook_runtime(point.world),
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    assert len(seen) == 1
    assert seen[0] in result.branches[0].actor_calls
    assert result.branches[0].status == "outcome_unknown"
    assert result.branches[0].failure_detail is not None
    assert result.branches[0].failure_detail.details["phase"] == "interrupted_branch"


def test_step_limit_stops_stay_in_the_resumed_success_denominator() -> None:
    """An arm that runs out of steps more often cannot report a higher success rate."""
    verifier = TaskVerifier(
        description="Answer done.",
        outcome_schema={"properties": {"final_answer": {"const": "done"}}},
        required_paths=("/final_answer",),
    )
    spec = ContinuationStudySpec(
        checkpoint=ArtifactInput(artifact_id="checkpoint", sha256="0" * 64),
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=verifier,
        probe_sha256=sha256_json({"probe": "none"}),
        repeats=4,
        selector="denominator scenario",
    )

    def study(*stops: tuple[str, str | None]) -> ContinuationStudy:
        """Grade completed branches with the given actor stop reason and final answer."""
        branches = []
        for index, (stop_reason, answer) in enumerate(stops):
            outcome: JsonObject = {"steps": [], "final_answer": answer, "stop_reason": stop_reason}
            branches.append(
                ContinuationBranch(
                    index=index,
                    status="completed",
                    outcome=outcome,
                    assessment=grade(verifier, outcome),
                )
            )
        return ContinuationStudy(spec=spec, branches=tuple(branches))

    exhausted = study(
        ("completed", "done"),
        ("maximum_steps", None),
        ("maximum_steps", None),
        ("length", "do"),
    ).metrics()
    finishing = study(
        ("completed", "done"),
        ("completed", "done"),
        ("completed", "wrong"),
        ("completed", "wrong"),
    ).metrics()
    assert (exhausted["assessable"], exhausted["unassessable"]) == (4, 0)
    assert exhausted["resumed_success_rate"] == 0.25
    assert finishing["resumed_success_rate"] == 0.5


def test_known_outcomes_label_passes_and_failures_and_skip_unknowns() -> None:
    """Only assessed grades become labels; missing and unassessable grades stay unknown."""
    verifier = TaskVerifier(
        description="Answer done.",
        outcome_schema={"properties": {"final_answer": {"const": "done"}}},
        required_paths=("/final_answer",),
    )
    passed = grade(verifier, {"steps": [], "final_answer": "done", "stop_reason": "completed"})
    failed = grade(verifier, {"steps": [], "final_answer": "no", "stop_reason": "completed"})
    unknown = grade(verifier, {"steps": [], "final_answer": "done", "stop_reason": "failure"})
    assert known_outcomes((passed, None, failed, unknown, passed)) == (1, 0, 1)
