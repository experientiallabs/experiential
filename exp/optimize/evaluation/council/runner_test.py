"""Council actors use isolated durable worlds, retained providers and a read-only analyst."""

import json
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from shutil import which

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_bytes
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.council import runner as trace_council
from exp.optimize.evaluation.fidelity import runner as fidelity_owner
from exp.optimize.evaluation.fidelity.runner import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity.runner_test import reply, source_project, transition_reply
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    investigation,
    judged,
    memory_project,
    predicted,
    trajectory,
)
from exp.runtime.agents.factory import ActorHarness
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.simulation.actor_context import ActorContextFragment
from exp.simulation.evaluation.analysis import CouncilTrace
from exp.simulation.world_model.experience_memory import is_reserved, memory_head
from exp.simulation.world_model.session_research_test import payload as prediction_context
from exp.simulation.world_model.session_research_test import structured_reply


def spec_project(
    tmp_path: Path,
) -> tuple[ProjectStore, trace_council.TraceCouncilSpec, FidelitySpec]:
    """Reuse the real source loader and reconstructed artifact, without native execution."""
    project, fidelity = source_project(tmp_path)
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["model"]
    catalog = catalog.model_copy(
        update={
            "models": {**catalog.models, "other": record.model_copy(update={"model": "fixture-v2"})}
        }
    )
    write_model_catalog(project.model_catalog_path, catalog)
    spec = trace_council.TraceCouncilSpec(
        run_id="council-smoke",
        world_model=fidelity.world_model,
        tasks=(fidelity.task,),
        actors=(
            trace_council.CouncilActor(alias="model", tier="operator-smaller"),
            trace_council.CouncilActor(alias="other", tier="operator-larger"),
        ),
        analysts=("model",),
        judge_alias="model",
        judge=fidelity.judge,
        deadline=fidelity.deadline,
        maximum_physical_attempts=12,
        maximum_actor_turns=2,
        actor_output_tokens=4096,
        simulator_output_tokens=4096,
        judge_output_tokens=4096,
        analyst_output_tokens=4096,
    )
    return project, spec, fidelity


def actor_reply(*, final: bool, model: str) -> JsonHttpResponse:
    """Emit a real OpenAI-compatible actor tool call or final answer."""
    message: JsonObject = {"role": "assistant", "content": "Done" if final else None}
    if not final:
        message["tool_calls"] = [
            {
                "id": "actor-read",
                "type": "function",
                "function": {"name": "inspect", "arguments": '{"key":"current"}'},
            }
        ]
    return JsonHttpResponse(
        status_code=200,
        body={
            "model": model,
            "choices": [{"finish_reason": "stop" if final else "tool_calls", "message": message}],
        },
    )


def sequence(fidelity: FidelitySpec) -> list[JsonHttpResponse]:
    """Two actor episodes, their simulator consequences, judges and the final analyst."""
    responses = []
    for model in ("fixture-v1", "fixture-v2"):
        responses += [
            actor_reply(final=False, model=model),
            transition_reply(
                {
                    "observations": [
                        {"call_id": "actor-read", "reply": "known", "encoding": "json"}
                    ],
                    "effect": {
                        "kind": "edits",
                        "edits": [{"op": "set", "path": ["last"], "value": "current"}],
                    },
                }
            ),
            actor_reply(final=True, model=model),
            judged(fidelity),
        ]
    responses.append(
        structured_reply(
            {
                "summary": "The simulator-only council ran.",
                "findings": [],
                "limitations": ["No actual environment or native outcomes."],
            }
        )
    )
    return responses


def test_installed_pi_runs_every_council_actor_with_isolated_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed Pi traverses the same five-model wire, durable worlds and retained judgments."""
    installed = which("pi")
    if installed is None:
        pytest.skip("external Pi executable is not installed")
    executable = Path(installed).absolute()
    project, spec, fidelity = spec_project(tmp_path)
    catalog = load_model_catalog(project.model_catalog_path)
    models = dict(catalog.models)
    actors, responses = [], []
    for index in range(5):
        alias, model = f"actor-{index}", f"fixture-actor-{index}"
        models[alias] = models["model"].model_copy(update={"model": model})
        actors.append(trace_council.CouncilActor(alias=alias, tier=f"declared-{index}"))
        responses.extend(
            (
                actor_reply(final=False, model=model),
                sequence(fidelity)[1],
                actor_reply(final=True, model=model),
                judged(fidelity),
            )
        )
    responses.append(sequence(fidelity)[-1])
    write_model_catalog(project.model_catalog_path, catalog.model_copy(update={"models": models}))
    spec = spec.model_copy(
        update={
            "actors": tuple(actors),
            "maximum_physical_attempts": 30,
            "actor_harness": ActorHarness(
                kind="pi",
                executable=executable,
                executable_sha256=sha256_bytes(executable.read_bytes()),
                version="installed-conformance-probe",
                timeout_seconds=20,
            ),
            "tasks": (
                spec.tasks[0].model_copy(
                    update={
                        "initial_context": {"public_fact": "inspect current before answering"},
                    }
                ),
            ),
        }
    )
    wire = ScriptedJsonTransport(tuple(responses))
    monkeypatch.setenv("FIDELITY_TEST_KEY", "inert-fixture-key")
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: wire)
    prepared = trace_council.prepare_trace_council(project, spec, code_revision="test")
    result = trace_council.execute_trace_council(project, prepared)
    rows = result["traces"]
    assert isinstance(rows, list) and len(rows) == 5
    for row in rows:
        trace = CouncilTrace.model_validate(row)
        assert trace.error_type is None, trace.model_dump_json()
        assert trace.workflow is not None and trace.evaluation is not None
        assert len(trace.workflow.evidence.steps) == 1
        assert trace.workflow.evidence.steps[0].before == {}
    assert result["analysis_failures"] == {}
    assert len(wire.requests) == result["physical_attempts"] == 21
    for index in range(5):
        assert "public_fact" in json.dumps(wire.requests[4 * index].payload)
        assert "mechanical_check_source" not in json.dumps(wire.requests[4 * index].payload)
    assert trace_council.execute_trace_council(project, prepared) == result
    assert len(wire.requests) == 21


@pytest.mark.parametrize("judge_output_protocol", ["tool", "json"])
def test_public_tui_runs_council_and_replays_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, judge_output_protocol: str
) -> None:
    """Exercise actual CLI, durable states, physical admission and hosted wire parsing."""
    project, spec, fidelity = spec_project(tmp_path)
    if judge_output_protocol == "json":
        catalog = load_model_catalog(project.model_catalog_path)
        record = catalog.models["model"]
        assert record.capabilities is not None
        judge_record = record.model_copy(
            update={
                "capabilities": record.capabilities.model_copy(update={"supports_tools": False})
            }
        )
        write_model_catalog(
            project.model_catalog_path,
            catalog.model_copy(update={"models": {**catalog.models, "json-judge": judge_record}}),
        )
        spec = spec.model_copy(update={"judge_alias": "json-judge"})
    spec = spec.model_copy(update={"judge_output_protocol": judge_output_protocol})
    responses = sequence(fidelity)
    if judge_output_protocol == "json":
        for index in (3, 7):
            responses[index] = reply(
                json.dumps(
                    {
                        "axes": {
                            d.dimension_id: {
                                "status": "assessed",
                                "score": 3,
                                "rationale": "Fixture",
                                "evidence_refs": ["step-0"],
                                "critical_flags": [],
                            }
                            for d in fidelity.judge.definition.dimensions
                        }
                    }
                )
            )
    wire = ScriptedJsonTransport(responses)
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: wire)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    path = tmp_path / "council.json"
    path.write_text(spec.model_dump_json())
    args = [
        "eval",
        "fidelity-test",
        "--root",
        str(tmp_path),
        "--trace-analysis",
        str(path),
        "--yes",
        "--non-interactive",
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "actual environment unavailable" in first.output
    prepared = trace_council.prepare_trace_council(
        project, spec, code_revision=installed_release_revision()
    )
    result = trace_council.saved_trace_council(project, prepared)
    assert result is not None
    assert len(wire.requests) == 9
    assert result["physical_attempts"] == 9
    assert result["analysis_failures"] == {}
    traces = result["traces"]
    assert isinstance(traces, list) and len(traces) == 2
    for value in traces:
        trace = CouncilTrace.model_validate(value)
        assert trace.workflow is not None and trace.evaluation is not None
        assert trace.workflow.evidence.initial_state == {}
        assert trace.evaluation.mechanical["workflow_joint_exact"] is None
        assert trace.evaluation.judge_status == "completed"
        assert trace.evaluation.judge_completion_sha256 is not None
        assert trace.workflow.evidence.steps[0].before == {}
    for index in (3, 7):
        assert ("tools" in wire.requests[index].payload) == (judge_output_protocol == "tool")
    actor_requests = [wire.requests[i].payload for i in (0, 2, 4, 6)]
    assert "mechanical_check_source" not in json.dumps(actor_requests)
    assert "REFERENCE_ONLY_SENTINEL" not in json.dumps(wire.requests[0].payload)
    assert "mechanical_check_source" in json.dumps(wire.requests[-1].payload)
    monkeypatch.delenv("FIDELITY_TEST_KEY")
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 0 and "No provider calls" in again.output
    assert len(wire.requests) == 9


def test_invented_analyst_evidence_is_recorded_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed analysis remains in the report instead of silently becoming a clean audit."""
    project, spec, fidelity = spec_project(tmp_path)
    responses = sequence(fidelity)
    responses[-1] = structured_reply(
        {
            "summary": "Unsupported",
            "limitations": ["Unknown native truth"],
            "findings": [
                {
                    "category": "answer_leakage",
                    "attribution": "undetermined",
                    "simulator_assessment": "unresolved",
                    "simulator_explanation": "Simulator causation cannot be separated here.",
                    "fidelity_assessment": "No measured tool supplied.",
                    "fidelity_refs": [],
                    "counterfactual_support": "unavailable",
                    "counterfactual_explanation": "No applicable archived reply.",
                    "counterfactual_refs": [],
                    "severity": "high",
                    "basis": "hypothesis",
                    "explanation": "Invented native evidence",
                    "evidence_refs": ["native:secret"],
                    "proposed_change": "Inspect",
                    "validation_probe": "Check supplied records",
                    "native_validation_required": True,
                }
            ],
        }
    )
    monkeypatch.setattr(
        trace_council, "HttpxJsonTransport", lambda: ScriptedJsonTransport(responses)
    )
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    prepared = trace_council.prepare_trace_council(project, spec, code_revision="fixture")
    result = trace_council.execute_trace_council(project, prepared)
    assert result["analysis_failures"] == {"model": "ReturnedOutputRejected"}
    assert result["planned_actor_cells"] == 2
    assert result["categories"] == {}


def test_no_hidden_reference_or_changed_registration_is_accepted(tmp_path: Path) -> None:
    """The council cannot acquire native truth by smuggling a task reference into context."""
    project, spec, _ = spec_project(tmp_path)
    value = spec.model_dump(mode="json")
    value["tasks"][0]["reference_answer"] = "hidden"
    with pytest.raises(ValueError, match="cannot include reference"):
        trace_council.TraceCouncilSpec.model_validate(value)
    trace_council.prepare_trace_council(project, spec, code_revision="fixture")
    changed = spec.model_copy(update={"maximum_actor_turns": 3})
    with pytest.raises(ValueError, match="registration changed"):
        trace_council.prepare_trace_council(project, changed, code_revision="fixture")


def test_exhausted_shared_cap_keeps_every_planned_cell_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One admitted actor call cannot erase subsequent unavailable tools, judges or analysts."""
    project, base, fidelity = spec_project(tmp_path)
    spec = base.model_copy(update={"maximum_physical_attempts": 1})
    wire = ScriptedJsonTransport(sequence(fidelity))
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: wire)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    prepared = trace_council.prepare_trace_council(project, spec, code_revision="fixture")
    result = trace_council.execute_trace_council(project, prepared)
    assert len(wire.requests) == 1 and result["physical_attempts"] == 1
    assert result["planned_actor_cells"] == 2
    traces = result["traces"]
    assert isinstance(traces, list) and len(traces) == 2
    for value in traces:
        trace = CouncilTrace.model_validate(value)
        assert trace.error_type is not None
    assert result["analysis_failures"] and result["categories"] == {}


def test_frozen_memory_reaches_actor_simulator_without_feedback_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actor path reads a disjoint checkpoint through JEV and never learns its answers."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    learning = trajectory(base, "learn-alpha", "alpha", "learn")
    learning = learning.model_copy(
        update={"task": learning.task.model_copy(update={"lineage_group_id": "alpha-owner"})}
    )
    training_wire = RoutedTransport(
        [
            predicted("learn-alpha-0", "wrong"),
            investigation("alpha"),
            predicted("learn-alpha-1", "alpha-reply-1"),
            judged(learning),
        ]
    )
    monkeypatch.setattr(fidelity_owner, "HttpxJsonTransport", lambda: training_wire)
    execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, learning, code_revision="fixture")
    )
    before = memory_head(project.artifacts, "fixture-memory")
    assert before is not None and before.version == 1
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["model"]
    catalog = catalog.model_copy(
        update={
            "models": {**catalog.models, "other": record.model_copy(update={"model": "fixture-v2"})}
        }
    )
    write_model_catalog(project.model_catalog_path, catalog)
    assert learning.retrieval is not None
    actor_spec = trace_council.TraceCouncilSpec.model_validate(
        {
            "run_id": "memory-actor",
            "world_model": base.world_model.model_dump(mode="json"),
            "tasks": [
                base.task.model_copy(update={"lineage_group_id": "reserved-beta"}).model_dump(
                    mode="json"
                )
            ],
            "actors": [{"alias": "model", "tier": "first"}, {"alias": "other", "tier": "second"}],
            "analysts": ["model"],
            "judge_alias": "model",
            "judge": base.judge.model_dump(mode="json"),
            "deadline": base.deadline.isoformat(),
            "maximum_physical_attempts": 24,
            "maximum_actor_turns": 2,
            "actor_output_tokens": 4096,
            "simulator_output_tokens": 4096,
            "judge_output_tokens": 4096,
            "analyst_output_tokens": 4096,
            "retrieval": learning.retrieval.model_dump(mode="json"),
            "memory": {"lineage": "fixture-memory", "checkpoint": 1, "overview_lines": 2},
        }
    )
    prepared = trace_council.prepare_trace_council(project, actor_spec, code_revision="fixture")
    wire = RoutedTransport(sequence(base))
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: wire)
    result = trace_council.execute_trace_council(project, prepared)
    assert result["analysis_failures"] == {}
    assert memory_head(project.artifacts, "fixture-memory") == before
    assert is_reserved(project.artifacts, "fixture-memory", "reserved-beta")
    payloads = [prediction_context(wire, index) for index in (1, 5)]
    assert len(payloads) == 2
    for payload in payloads:
        assert "alpha-reply-0" in json.dumps(payload["memory_observations"])
        assert "alpha entries may use a different value" in json.dumps(payload["research_history"])
        assert "REFERENCE_ONLY_SENTINEL" not in json.dumps(payload)
        assert payload["environment_state"] == {}
    assert len(wire.decisions) == 2
    monkeypatch.delenv("FIDELITY_TEST_KEY")
    assert trace_council.execute_trace_council(project, prepared) == result
    assert len(wire.requests) == 9


def test_native_counterfactual_and_context_are_analyst_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delivered actor fragments are captured while native replies enter only analysis."""
    project, base, fidelity = spec_project(tmp_path)
    source = project.artifacts.write_json(
        artifact_id="native-record",
        artifact_type="archived-native-episode",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), inputs=(), code_revision="fixture"
        ),
        files={
            "episode.json": {
                "episode_id": "native-record",
                "task_lineage": base.tasks[0].lineage_group_id,
                "initial_context": base.tasks[0].initial_context,
                "steps": [
                    {
                        "action": {"name": "inspect", "arguments": {"key": "current"}},
                        "reply": "NATIVE_ONLY_COUNTERFACTUAL",
                    }
                ],
            }
        },
    )
    spec = base.model_copy(
        update={
            "analyst_native_episodes": (artifact_input(source),),
            "actor_context": (
                ActorContextFragment(
                    fragment_id="context-a",
                    source="system_prompt",
                    role="system",
                    content="Inspect before answering.",
                ),
                ActorContextFragment(
                    fragment_id="context-b", source="hook", content="Never inspect."
                ),
            ),
        }
    )
    wire = ScriptedJsonTransport(sequence(fidelity))
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: wire)
    result = trace_council.execute_trace_council(
        project, trace_council.prepare_trace_council(project, spec, code_revision="fixture")
    )
    for index in (0, 2, 4, 6):
        assert "NATIVE_ONLY_COUNTERFACTUAL" not in json.dumps(wire.requests[index].payload)
    assert "NATIVE_ONLY_COUNTERFACTUAL" in json.dumps(wire.requests[-1].payload)
    for index in (1, 5):
        assert "Never inspect." not in json.dumps(wire.requests[index].payload)
    traces = result["traces"]
    assert isinstance(traces, list)
    for row in traces:
        trace = CouncilTrace.model_validate(row)
        fragments = trace.context_manifest["fragments"]
        assert isinstance(fragments, list)
        assert {fragment["origin"] for fragment in fragments if isinstance(fragment, dict)} >= {
            "context-a",
            "context-b",
        }
    evidence = json.loads(
        project.artifacts.read_bytes(str(result["analysis_evidence"]), "evidence.json")
    )
    assert len(evidence["counterfactuals"]) == 2
    assert all(row["applicable"] for row in evidence["counterfactuals"])


class ParallelCouncilWire(ScriptedJsonTransport):
    """Two actor calls must overlap; each simulator still advances its own state in order."""

    def __init__(self, fidelity: FidelitySpec) -> None:
        """Bind a two-party barrier and record detached request observations."""
        super().__init__()
        self.fidelity = fidelity
        self.barrier = threading.Barrier(2)
        self.guard = threading.Lock()
        self.actor_calls: dict[str, int] = {}
        self.prestate_counts: list[int] = []
        self.analysis_started = False

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Route real wire requests without depending on thread completion order."""
        with self.guard:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
        if payload.get("tool_choice") == "auto":
            model = str(payload["model"])
            with self.guard:
                ordinal = self.actor_calls.get(model, 0)
                self.actor_calls[model] = ordinal + 1
            if ordinal == 0:
                self.barrier.wait(timeout=3)
            return actor_reply(final=ordinal == 2, model=model)
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        content = json.loads(str(messages[1]["content"]))
        if "candidate_response" in content:
            state = content["environment_state"]
            count = state.get("count", 0)
            with self.guard:
                self.prestate_counts.append(count)
            return transition_reply(
                {
                    "observations": [
                        {"call_id": "actor-read", "reply": "known", "encoding": "json"}
                    ],
                    "effect": {
                        "kind": "edits",
                        "edits": [{"op": "set", "path": ["count"], "value": count + 1}],
                    },
                }
            )
        if "You analyze an evaluation conducted entirely" in json.dumps(messages):
            with self.guard:
                assert self.actor_calls == {"fixture-v1": 3, "fixture-v2": 3}
                assert sorted(self.prestate_counts) == [0, 0, 1, 1]
                self.analysis_started = True
            return structured_reply(
                {
                    "summary": "Parallel fixture.",
                    "findings": [],
                    "limitations": ["No native environment."],
                }
            )
        return judged(self.fidelity)


def test_parallel_cells_overlap_but_keep_state_sequential_and_output_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A barrier proves overlap while each actor owns an independent carried world."""
    project, base, fidelity = spec_project(tmp_path)
    spec = trace_council.TraceCouncilSpec.model_validate(
        {
            **base.model_dump(mode="json"),
            "maximum_concurrency": 2,
            "maximum_actor_turns": 3,
            "maximum_physical_attempts": 13,
        }
    )
    wire = ParallelCouncilWire(fidelity)
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: wire)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    prepared = trace_council.prepare_trace_council(project, spec, code_revision="parallel")
    result = trace_council.execute_trace_council(project, prepared)
    assert wire.analysis_started
    assert result["physical_attempts"] == len(wire.requests) == 13
    assert result["analysis_failures"] == {}
    rows = result["traces"]
    assert isinstance(rows, list)
    assert [row["actor_alias"] for row in rows if isinstance(row, dict)] == ["model", "other"]
    assert all(isinstance(row, dict) and row["error_type"] is None for row in rows)
    for row in rows:
        assert isinstance(row, dict)
        workflow = row["workflow"]
        assert isinstance(workflow, dict)
        evidence = workflow["evidence"]
        assert isinstance(evidence, dict)
        steps = evidence["steps"]
        assert isinstance(steps, list) and len(steps) == 2
        first, second = steps
        assert isinstance(first, dict) and isinstance(second, dict)
        assert first["before"] == {} and first["after"] == {"count": 1}
        assert second["before"] == first["after"] and second["after"] == {"count": 2}


def test_parallel_cells_cannot_exceed_shared_physical_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent actors share the actual dispatch cap instead of getting separate quotas."""
    project, base, fidelity = spec_project(tmp_path)
    spec = trace_council.TraceCouncilSpec.model_validate(
        {
            **base.model_dump(mode="json"),
            "maximum_concurrency": 2,
            "maximum_actor_turns": 3,
            "maximum_physical_attempts": 2,
        }
    )
    wire = ParallelCouncilWire(fidelity)
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: wire)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    result = trace_council.execute_trace_council(
        project, trace_council.prepare_trace_council(project, spec, code_revision="parallel-cap")
    )
    assert result["physical_attempts"] == len(wire.requests) == 2
    assert result["planned_actor_cells"] == 2
    rows = result["traces"]
    assert isinstance(rows, list) and len(rows) == 2
    assert all(isinstance(row, dict) and row["error_type"] is not None for row in rows)
