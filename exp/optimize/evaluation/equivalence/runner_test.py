"""Paired execution through real XLSX worlds, admitted HTTP and saved trace-council reviews."""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from filelock import Timeout

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_bytes, sha256_json
from exp.common.evaluations.actor_equivalence import (
    ActorComparison,
    ActorEquivalenceReport,
    EquivalenceRubric,
)
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.optimize.evaluation.equivalence import runner as owner
from exp.optimize.evaluation.equivalence.evidence import PairMeasurement
from exp.optimize.evaluation.equivalence.policy_test import fixture_policy, registered_policy
from exp.optimize.evaluation.equivalence.review import ReviewModel
from exp.optimize.evaluation.equivalence.rubric_test import frozen_rubric
from exp.optimize.evaluation.equivalence.runner import (
    EquivalenceCase,
    EquivalenceSpec,
    preflight_equivalence,
    run_equivalence,
    saved_equivalence,
)
from exp.optimize.evaluation.equivalence.verdicts import policy_calibrations
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import (
    WorldCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)


def response(model: str, message: JsonObject, *, tools: bool) -> JsonHttpResponse:
    """Frame one scripted response on the real admitted provider transport."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "id": "fixture",
            "model": model,
            "choices": [
                {"index": 0, "finish_reason": "tool_calls" if tools else "stop", "message": message}
            ],
        },
    )


def proposal(value: JsonObject) -> JsonHttpResponse:
    """Return one typed judge or analyst response with original raw output retained."""
    return response(
        "judge-v1",
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "review",
                    "type": "function",
                    "function": {"name": "submit_proposal", "arguments": json.dumps(value)},
                }
            ],
        },
        tools=True,
    )


def rating() -> JsonObject:
    """Supply intentionally favorable ratings even when mechanical checks disagree."""
    return {
        name: {
            "dimension_id": name,
            "status": "assessed",
            "score": 5,
            "rationale": "Fixture rating, not calibrated evidence.",
            "evidence_refs": ["A:task_checks", "B:task_checks"],
            "critical_flags": [],
        }
        for name in ("goal", "decisions", "facts", "consequences", "completion")
    }


def simulated_workbook(project: ProjectStore, configuration: JsonObject) -> ContinuationRuntime:
    """Restore a real grounded predictor/kernel with a scripted proposal, never a native edit."""
    from exp.common.models import (
        AssistantAction,
        ModelCapabilities,
        ModelResponse,
        OperationEconomics,
        ToolCall,
    )
    from exp.runtime.environments.world_contracts import (
        ExecutableCheckpoint,
        StateDomain,
        ToolEffectPolicy,
        WorldManifest,
    )
    from exp.runtime.environments.worlds import ExecutableWorldRuntime
    from exp.simulation.world_model.executable import GroundedWorldFactory, PredictorEvent
    from exp.simulation.world_model.grounded_session import GroundedTransitionEngine
    from exp.simulation.world_model.runtime_test import _runtime

    source = load_continuation_checkpoint(
        project.artifacts, ArtifactInput.model_validate(configuration["checkpoint"])
    )
    assert isinstance(source.world, ExecutableCheckpoint)
    predictor, _, _ = _runtime("", capabilities=ModelCapabilities(supports_tools=True))
    engine = GroundedTransitionEngine(
        predictor, source.task, maximum_output_tokens=1024, output_protocol="json-tool"
    )
    manifest = WorldManifest(
        domains=(StateDomain(name="workbook", owner="simulated"),),
        tools=(ToolEffectPolicy(name="edit_workbook", reads=("workbook",), writes=("workbook",)),),
    )
    events: list[PredictorEvent] = []

    def complete(body: JsonObject, operation: str) -> JsonObject:
        """Exercise original request/parser and reject the known illegal clock-edit failure."""
        assert events[-1].stage == "intent" and events[-1].operation_id == operation
        assert "domains" in json.dumps(body)
        edits: list[JsonObject] = [
            {"op": "set", "path": ["domains", "workbook", "rows"], "value": 6}
        ]
        if configuration.get("illegal_clock"):
            edits.append({"op": "set", "path": ["environment", "logical_time"], "value": 1})
        return ModelResponse(
            model=predictor.artifact.model,
            economics=OperationEconomics(),
            output=AssistantAction(
                tool_calls=(
                    ToolCall(
                        name="submit_transition",
                        call_id="proposal",
                        arguments={
                            "observations": [
                                {
                                    "call_id": workbook_case().first.call_id,
                                    "reply": {"rows": 6},
                                    "encoding": "json",
                                }
                            ],
                            "effect": {"kind": "edits", "edits": edits},
                        },
                    ),
                )
            ),
        ).model_dump(mode="json")

    def probe(checkpoint: WorldCheckpoint) -> JsonObject:
        """Project committed simulated state with the same meanings as the native XLSX probe."""
        assert isinstance(checkpoint, ExecutableCheckpoint)
        wire = json.loads(checkpoint.image.parts["predictor.json"])
        return wire["state"]["workbook"]

    factory = GroundedWorldFactory(
        engine,
        manifest,
        {"workbook": {"rows": None, "input_digest": source.world.image.domains["input.xlsx"]}},
        complete=complete,
        record=events.append,
    )
    return ContinuationRuntime(
        ExecutableWorldRuntime(manifest, factory, owner="simulated"),
        probe,
        sha256_json({"probe": "grounded-workbook-row-count-v1"}),
    )


@pytest.mark.parametrize("illegal_clock", [False, True])
def test_native_reference_and_grounded_candidate_use_actual_world_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, illegal_clock: bool
) -> None:
    """Compare real XLSX effects to simulated commits; a simulator failure is a difference."""
    project, spec, _, reviews = equivalence_fixture(tmp_path, monkeypatch)
    reference = spec.cases[0].reference
    configuration: JsonObject = {
        "checkpoint": reference.study.checkpoint.model_dump(mode="json"),
        "illegal_clock": illegal_clock,
    }
    runtime = simulated_workbook(project, configuration)
    capsule = load_continuation_checkpoint(project.artifacts, reference.study.checkpoint)
    with runtime.world.open(capsule.task) as session:
        candidate_checkpoint = capsule.model_copy(update={"world": session.checkpoint()})
    candidate_ref = persist_continuation_checkpoint(
        project.artifacts, candidate_checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )
    source = tmp_path / "grounded_candidate.py"
    source.write_text(
        "from exp.optimize.evaluation.equivalence.runner_test import simulated_workbook\n"
        "continuation_runtime = simulated_workbook\n"
    )
    candidate = reference.model_copy(
        update={
            "runtime_file": source,
            "runtime_sha256": sha256_bytes(source.read_bytes()),
            "configuration": configuration,
            "study": reference.study.model_copy(
                update={"checkpoint": candidate_ref, "probe_sha256": runtime.probe_sha256}
            ),
        }
    )
    spec = spec.model_copy(
        update={"cases": (spec.cases[0].model_copy(update={"candidate": candidate}),)}
    )
    result = run_equivalence(project, spec, code_revision="test")
    measured = result.measurements[0]
    if illegal_clock:
        assert measured.material_agreement == "different"
        assert "different:completed_execution" in measured.differences
        assert measured.judgment is None
        assert measured.comparison.candidate_assessment.execution != "completed"
    else:
        assert measured.material_agreement == "match"
        assert measured.comparison.candidate_assessment.task_result == "pass"
        assert "simulated" in json.dumps(measured.comparison.candidate.audit_metadata)
        messages = reviews.requests[0].payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        payload = json.loads(str(messages[1]["content"]))
        assert all(
            set(row["reply"]) == {"content", "is_error"}
            for side in ("A", "B")
            for row in payload[side]["steps"]
        )
        assert "execution_receipts" in json.dumps(reviews.requests[-1].payload)
        cell = next(row for row in result.cells if row.condition == "candidate")
        assert cell.result is not None
        from exp.optimize.evaluation.continuations.records import load_study
        from exp.runtime.environments.world_contracts import ExecutableCheckpoint

        study = load_study(project.artifacts, cell.result)
        end = load_continuation_checkpoint(project.artifacts, study.branches[0].checkpoints[-1])
        assert isinstance(end.world, ExecutableCheckpoint)
        assert end.world.clock == 1
        assert "predictor.json" in end.world.image.parts
        assert "output.xlsx" not in end.world.image.parts
    assert saved_equivalence(project, spec, code_revision="test") == result


def equivalence_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, wrong_candidate: bool = False
) -> tuple[ProjectStore, EquivalenceSpec, ScriptedJsonTransport, ScriptedJsonTransport]:
    """Execute three independent copies of a native workbook under identical actor policies.

    This tests the paired execution contract, not live-model simulator accuracy. The candidate
    takes a deliberately wrong action when requested, while the reviewer still gives full marks.
    """
    project, base, _ = job_fixture(tmp_path, monkeypatch)
    base = base.model_copy(
        update={
            "study": base.study.model_copy(update={"training_specification": None, "repeats": 1})
        }
    )
    model = RuntimeModelCatalog(
        load_model_catalog(project.model_catalog_path), environment={}
    ).snapshot("judge")[0]
    reviewer = ReviewModel(alias="judge", model=model, output_tokens=1024)
    spec = EquivalenceSpec(
        rubric=frozen_rubric(project, reviewer),
        cases=(
            EquivalenceCase(
                reference=base,
                candidate=base,
                initial_state_paths=("/input_digest",),
                final_state_paths=("/rows", "/input_digest"),
            ),
        ),
        judge=reviewer,
        analysts=(reviewer,),
        maximum_review_attempts=3,
        deadline=base.deadline,
    )
    replies = []
    for index in range(3):
        call = workbook_case().first
        arguments = {
            **call.arguments,
            "deletions": [
                {"sheet": "Orders", "start": 2, "count": 2 if wrong_candidate and index == 1 else 1}
            ],
        }
        replies.extend(
            (
                response(
                    "world-v1",
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    },
                    tools=True,
                ),
                response("world-v1", {"role": "assistant", "content": "done"}, tools=False),
            )
        )
    actors = ScriptedJsonTransport(tuple(replies))
    reviews = ScriptedJsonTransport(
        (
            proposal(rating()),
            proposal(rating()),
            proposal(
                {
                    "summary": "Inspect actual state consequences; labels are provisional.",
                    "findings": [],
                    "limitations": [
                        "Scripted conformance only; no live-model equivalence measured."
                    ],
                }
            ),
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actors
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", lambda: reviews
    )
    return project, spec, actors, reviews


def test_paired_pipeline_and_trace_council_replay_without_runtime_or_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three conditions execute, private evidence stays out of actors, and replay is free."""
    project, spec, actors, reviews = equivalence_fixture(tmp_path, monkeypatch)
    preflight_equivalence(project, spec)
    assert not actors.requests and not reviews.requests
    result = run_equivalence(project, spec, code_revision="test")
    assert len(result.cells) == 3 and len(result.measurements) == 2
    assert all(row.material_agreement == "match" for row in result.measurements)
    assert all(
        row.judgment is not None and row.judgment.score_100 == 100 for row in result.measurements
    )
    assert result.analyses["case-0:judge"].findings == () and not result.analysis_failures
    assert len(actors.requests) == 6 and len(reviews.requests) == 3
    initial = [actors.requests[index].payload for index in (0, 2, 4)]
    assert initial[0] == initial[1] == initial[2]
    assert "paired_evidence" not in json.dumps([row.payload for row in actors.requests])
    analyst = json.dumps(reviews.requests[-1].payload)
    assert "paired_evidence" in analyst and "reference_control" in analyst
    assert "actual environment is unavailable" not in analyst.lower()
    messages = reviews.requests[-1].payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    packet = json.loads(str(messages[1]["content"]))["definition"]["paired_evidence"]
    executions = [
        value
        for value in packet.values()
        if isinstance(value, dict) and value.get("kind") == "execution"
    ]
    assert len(executions) == 3
    assert {value["condition"] for value in executions} == {
        "reference",
        "reference_control",
        "candidate",
    }
    assert (
        packet["pair:0:candidate:0"]["reference_execution"]
        == packet["pair:0:reference_control:0"]["reference_execution"]
    )

    assert result.metrics["source_families"] == 1
    spec.cases[0].reference.runtime_file.unlink()
    project.model_catalog_path.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    assert saved_equivalence(project, spec, code_revision="test") == result
    assert run_equivalence(project, spec, code_revision="test") == result
    assert len(actors.requests) == 6 and len(reviews.requests) == 3


def test_favorable_judge_cannot_hide_wrong_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual five-row output disagrees with six-row goal even if a model scores it 100."""
    project, spec, _, _ = equivalence_fixture(tmp_path, monkeypatch, wrong_candidate=True)
    result = run_equivalence(project, spec, code_revision="test")
    candidate, control = result.measurements
    assert candidate.material_agreement == "different"
    assert candidate.comparison.reference_assessment.task_result == "pass"
    assert candidate.comparison.candidate_assessment.task_result == "fail"
    assert candidate.judgment is not None and candidate.judgment.score_100 == 100
    assert "different:final_state/rows" in candidate.differences
    assert control.material_agreement == "match"
    metrics = result.metrics["candidate"]
    assert isinstance(metrics, dict) and metrics["both_task_pass"] == 0


@pytest.mark.parametrize("field", ["maximum_model_calls", "system_prompt", "deadline"])
def test_changed_actor_policy_refuses_before_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """The comparison cannot silently change its actor, allowance or instruction."""
    project, spec, actors, reviews = equivalence_fixture(tmp_path, monkeypatch)
    case = spec.cases[0]
    values = {"maximum_model_calls": 5, "system_prompt": "different", "deadline": datetime.now(UTC)}
    with pytest.raises(ValueError, match="paired actor policies differ"):
        EquivalenceCase.model_validate(
            {
                **case.model_dump(mode="json"),
                "candidate": case.candidate.model_copy(update={field: values[field]}),
            }
        )
    assert not actors.requests and not reviews.requests


@pytest.mark.parametrize("path", ["rows", "/rows~2", "/rows/~"])
def test_state_paths_use_the_shared_pointer_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """Relative pointers and invalid escapes refuse before any execution is registered."""
    _, spec, actors, _ = equivalence_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="pointer"):
        EquivalenceCase.model_validate(
            {**spec.cases[0].model_dump(mode="json"), "final_state_paths": [path]}
        )
    assert not actors.requests


def test_failed_candidate_stays_in_denominator_and_is_not_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A simulator setup failure keeps its planned pair and cannot buy an equivalence rating."""
    project, spec, actors, _ = equivalence_fixture(tmp_path, monkeypatch)
    from exp.optimize.evaluation.equivalence import runner as equivalence

    execute = equivalence.execute_continuation_job

    def fail_candidate(
        project: ProjectStore, job: ContinuationJob, *, code_revision: str
    ) -> ContinuationStudy:
        """Inject one known runtime failure, retaining reference and control execution."""
        if job.study.selector.endswith("candidate"):
            raise RuntimeError("candidate setup failed")
        return execute(project, job, code_revision=code_revision)

    reviews = ScriptedJsonTransport(
        (
            proposal(rating()),
            proposal(
                {
                    "summary": "Candidate unavailable.",
                    "findings": [],
                    "limitations": ["No candidate measurement."],
                }
            ),
        )
    )
    monkeypatch.setattr(equivalence, "execute_continuation_job", fail_candidate)
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", lambda: reviews
    )
    result = run_equivalence(project, spec, code_revision="test")
    assert result.measurements[0].material_agreement == "unassessable"
    assert result.measurements[0].judgment is None
    metrics = result.metrics["candidate"]
    assert isinstance(metrics, dict)
    assert metrics["planned_pairs"] == 1 and metrics["unassessable"] == 1
    assert len(reviews.requests) == 2 and len(actors.requests) == 4


def test_saved_report_rejects_changed_derived_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-consistent artifact rewrite still cannot invent agreement not in the source runs."""
    project, spec, _, _ = equivalence_fixture(tmp_path, monkeypatch, wrong_candidate=True)
    result = run_equivalence(project, spec, code_revision="test")
    assert result.measurements[0].material_agreement == "different"
    # Exercise the public rebuild rather than trusting cached assertions.
    from exp.optimize.evaluation.equivalence import runner as equivalence

    original = equivalence.equivalence_metrics

    def changed(rows: tuple[PairMeasurement, ...]) -> JsonObject:
        """Deliberately change the metric derived during independent verification."""
        return {**original(rows), "fabricated": True}

    monkeypatch.setattr(equivalence, "equivalence_metrics", changed)
    with pytest.raises(ValueError, match="does not reproduce"):
        saved_equivalence(project, spec, code_revision="test")


@pytest.mark.parametrize("child_saved", [False, True])
def test_interrupted_panel_adopts_saved_children_and_never_redraws_unknown_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child_saved: bool
) -> None:
    """A crash after intent preserves its outcome boundary, including completed child adoption."""
    from exp.optimize.evaluation.equivalence import runner as equivalence

    project, spec, actors, _ = equivalence_fixture(tmp_path, monkeypatch)
    execute = equivalence.execute_continuation_job

    def interrupt(
        project: ProjectStore, job: ContinuationJob, *, code_revision: str
    ) -> ContinuationStudy:
        """Crash after the registered intent, optionally after the child result commits."""
        if child_saved:
            execute(project, job, code_revision=code_revision)
        raise KeyboardInterrupt("fixture process interruption")

    monkeypatch.setattr(equivalence, "execute_continuation_job", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_equivalence(project, spec, code_revision="test")
    purchased = len(actors.requests)
    assert purchased == (2 if child_saved else 0)
    monkeypatch.setattr(equivalence, "execute_continuation_job", execute)
    result = run_equivalence(project, spec, code_revision="test")
    assert len(actors.requests) == purchased + 4
    first = result.cells[0]
    if child_saved:
        assert first.result is not None and first.failure is None
        assert result.measurements[0].material_agreement == "match"
    else:
        assert first.result is None and first.failure == "UnfinishedDispatchIntent"
        assert all(row.material_agreement == "unassessable" for row in result.measurements)
    assert saved_equivalence(project, spec, code_revision="test") == result


def test_same_panel_lease_excludes_another_process_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel cannot duplicate execution while another file-lock owner holds its lease."""
    from filelock import FileLock, Timeout

    project, spec, actors, reviews = equivalence_fixture(tmp_path, monkeypatch)
    lock_path = project.artifacts.project_directory / (spec.identity("test") + ".lock")
    with FileLock(lock_path, timeout=0):
        with pytest.raises(Timeout):
            run_equivalence(project, spec, code_revision="test")
    assert not actors.requests and not reviews.requests


def test_busy_source_access_defers_the_judge_and_a_later_run_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contention for the disclosure boundary sends nothing and never records a failed judgment."""
    from filelock import FileLock

    project, spec, actors, reviews = equivalence_fixture(tmp_path, monkeypatch)
    judge = owner.evaluate_actor_equivalence

    def contended(
        pair: ActorComparison,
        *,
        model_identity: str,
        complete: Callable[[JsonObject, str], JsonObject],
        rubric: EquivalenceRubric,
    ) -> ActorEquivalenceReport:
        """Judge while another process holds the source-access lock."""
        with FileLock(project.artifacts.project_directory / "source-access.lock", timeout=0):
            return judge(pair, model_identity=model_identity, complete=complete, rubric=rubric)

    with monkeypatch.context() as patch:
        patch.setattr(owner, "evaluate_actor_equivalence", contended)
        with pytest.raises(Timeout):
            run_equivalence(project, spec, code_revision="test")
    assert len(actors.requests) == 6 and not reviews.requests
    assert saved_equivalence(project, spec, code_revision="test") is None

    def unlocked() -> ScriptedJsonTransport:
        """Prove no provider transport is built while the disclosure boundary is held."""
        with FileLock(project.artifacts.project_directory / "source-access.lock", timeout=0):
            return reviews

    monkeypatch.setattr("exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", unlocked)
    result = run_equivalence(project, spec, code_revision="test")
    assert all(
        row.judgment is not None and row.judgment.status == "completed"
        for row in result.measurements
    )
    assert len(actors.requests) == 6 and len(reviews.requests) == 3
    assert saved_equivalence(project, spec, code_revision="test") == result


def test_five_actor_panel_preserves_one_source_family_and_scopes_each_analyst(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fifteen independent world executions remain five actors on one source, not five tasks."""
    project, spec, _, _ = equivalence_fixture(tmp_path, monkeypatch)
    catalog = load_model_catalog(project.model_catalog_path)
    models = dict(catalog.models)
    for index in range(5):
        models[f"actor-{index}"] = models["world"].model_copy(update={"model": f"actor-{index}-v1"})
    catalog = catalog.model_copy(update={"models": models})
    write_model_catalog(project.model_catalog_path, catalog)
    registry = RuntimeModelCatalog(catalog, environment={})
    cases, responses = [], []
    base = spec.cases[0].reference
    for index in range(5):
        alias = f"actor-{index}"
        job = base.model_copy(
            update={
                "actor_alias": alias,
                "catalog_sha256": sha256_json(catalog.model_dump(mode="json")),
                "study": base.study.model_copy(update={"actor_model": registry.snapshot(alias)[0]}),
            }
        )
        cases.append(spec.cases[0].model_copy(update={"reference": job, "candidate": job}))
        for _ in range(3):
            call = workbook_case().first
            responses.extend(
                (
                    response(
                        f"{alias}-v1",
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call.call_id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": json.dumps(call.arguments),
                                    },
                                }
                            ],
                        },
                        tools=True,
                    ),
                    response(f"{alias}-v1", {"role": "assistant", "content": "done"}, tools=False),
                )
            )
    spec = spec.model_copy(update={"cases": tuple(cases), "maximum_review_attempts": 15})
    actors = ScriptedJsonTransport(tuple(responses))
    reviews = ScriptedJsonTransport(
        tuple(
            [proposal(rating()) for _ in range(10)]
            + [
                proposal(
                    {
                        "summary": "This case completed.",
                        "findings": [],
                        "limitations": ["Scripted conformance."],
                    }
                )
                for _ in range(5)
            ]
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actors
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", lambda: reviews
    )
    result = run_equivalence(project, spec, code_revision="test")
    assert len(result.cells) == 15 and len(result.measurements) == 10
    assert result.metrics["actor_models"] == 5 and result.metrics["source_families"] == 1
    assert len(result.analyses) == 5 and not result.analysis_failures
    assert len(actors.requests) == 30 and len(reviews.requests) == 15
    for case, request in enumerate(reviews.requests[-5:]):
        packet = json.dumps(request.payload)
        assert f"pair:{case}:candidate:0" in packet
        assert f"pair:{(case + 1) % 5}:candidate:0" not in packet


def test_deferred_child_resumes_the_same_registered_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known contention stays pending and retry reuses the child's original execution identity."""
    project, spec, transport, _ = equivalence_fixture(tmp_path, monkeypatch)
    original = owner.execute_continuation_job
    blocked = []

    def busy_once(
        project: ProjectStore, job: ContinuationJob, *, code_revision: str
    ) -> ContinuationStudy:
        """Inject one child lease refusal before delegating to durable native execution."""
        if not blocked:
            blocked.append(job)
            raise Timeout("child.lock")
        return original(project, job, code_revision=code_revision)

    monkeypatch.setattr(owner, "execute_continuation_job", busy_once)
    with pytest.raises(Timeout):
        run_equivalence(project, spec, code_revision="test")
    assert not project.artifacts.exists(spec.identity("test") + "-cell-0")
    report = run_equivalence(project, spec, code_revision="test")
    assert report.cells[0].result is not None and report.cells[0].failure is None
    assert run_equivalence(project, spec, code_revision="test") == report


def test_registered_policy_calibrates_the_report_over_task_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned policy's margins are applied per dimension; one family stays underpowered."""
    project, spec, _, _ = equivalence_fixture(tmp_path, monkeypatch)
    policy = registered_policy(project.artifacts)
    spec = spec.model_copy(update={"policy": policy})
    result = run_equivalence(project, spec, code_revision="test")
    assert result.calibration is not None and result.calibration.policy == policy
    assert all(len(row.dimensions) == 6 for row in result.measurements)
    lineage = load_continuation_checkpoint(
        project.artifacts, spec.cases[0].reference.study.checkpoint
    ).task.lineage_group_id
    goal = result.calibration.dimension("goal_success")
    assert goal.units == (f"family:{lineage}",)
    assert goal.status == "unassessable" and "underpowered: 1" in goal.reason
    assert result.calibration.dimension("recovery").inapplicable_pairs["candidate"] == 1
    identity = spec.identity("test")
    assert policy in project.artifacts.read(identity).manifest.inputs
    assert saved_equivalence(project, spec, code_revision="test") == result
    report = artifact_input(project.artifacts.read(identity + "-result").manifest)
    (calibration,) = policy_calibrations(project, (report,), policy)
    assert calibration.calibration == result.calibration
    other = registered_policy(project.artifacts, fixture_policy(margin=0.4))
    with pytest.raises(ValueError, match="was not registered under admission policy"):
        policy_calibrations(project, (report,), other)


def test_a_pinned_policy_with_a_later_clock_still_precedes_the_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registration cites the saved policy, so a policy clock an hour ahead is not refused."""
    project, spec, _, _ = equivalence_fixture(tmp_path, monkeypatch)
    skewed = registered_policy(
        project.artifacts,
        fixture_policy(margin=0.5),
        at=datetime.now(UTC) + timedelta(hours=1),
    )
    result = run_equivalence(
        project, spec.model_copy(update={"policy": skewed}), code_revision="test"
    )
    assert result.calibration is not None and result.calibration.policy == skewed
