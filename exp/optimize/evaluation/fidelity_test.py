"""Trace-backed predictions, reference isolation, durable replay and visible failures."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.common.evaluations.reconstruction import EvaluationReferences, ReferenceStep, default_judge
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ToolCall,
    write_model_catalog,
)
from exp.common.project import ProjectConfig, ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.common.traces import Trace, TraceDataset, TraceSource, TraceSpan
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.evaluation import FixedWorkflow
from exp.simulation.retrieval import RAGLineageBinding, persist_trace_rag
from exp.simulation.world_model.construction import ground_world_model_from_traces


def source_project(
    root: Path, served_model_id: str | None = None
) -> tuple[ProjectStore, FidelitySpec]:
    """Persist a real source dataset and retrieval index without model construction.

    Args:
        root: Project root.
        served_model_id: Optional catalog pin for the provider-reported model ID.
    """
    project = ProjectStore(root, "fidelity-test")
    project.initialize(ProjectConfig(project_id="fidelity-test"))
    now = datetime.now(UTC)
    catalog = ModelCatalog(
        connections={
            "http": ConnectionConfig(
                provider="openai-compatible",
                base_url="https://fixture.invalid/v1",
                api_key_env="FIDELITY_TEST_KEY",
            )
        },
        models={
            "model": ModelRecord(
                connection="http",
                model="fixture-v1",
                served_model_id=served_model_id,
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=ModelCapabilities(
                    supports_tools=True,
                    supports_structured_output=True,
                    context_window_tokens=200_000,
                    maximum_output_tokens=4096,
                ),
            )
        },
    )
    write_model_catalog(project.model_catalog_path, catalog)
    model, _ = RuntimeModelCatalog(catalog).snapshot("model")
    shared = {"gen_ai.tool.name": "inspect", "gen_ai.tool.call.id": "source-call"}
    trace = Trace(
        trace_id="observed",
        task="Read an entry",
        source=TraceSource(
            identity=SourceIdentity(kind="file", source_id="fixture"),
            semantic_convention_version="1.37.0",
        ),
        spans=(
            TraceSpan(
                span_id="request",
                name="request",
                started_at=now,
                ended_at=now,
                attributes={**shared, "gen_ai.tool.call.arguments": '{"key":"old"}'},
            ),
            TraceSpan(
                span_id="result",
                name="result",
                started_at=now + timedelta(seconds=1),
                ended_at=now + timedelta(seconds=1),
                attributes={
                    **shared,
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.message": "old-value",
                },
            ),
        ),
    )
    payload = trace.model_dump_json().encode() + b"\n"
    dataset = TraceDataset(
        schema_version=1,
        created_at=now,
        code_revision="fixture",
        source=trace.source.identity,
        dataset_id="source",
        semantic_convention_version="1.37.0",
        traces_path="traces.jsonl",
        traces_sha256=hashlib.sha256(payload).hexdigest(),
        issues_path="issues.json",
        issues_sha256=hashlib.sha256(b"[]").hexdigest(),
        trace_ids=(trace.trace_id,),
        invalid_trace_count=0,
    )
    manifest = project.artifacts.write(
        artifact_id="source",
        artifact_type="trace-dataset",
        envelope=dataset,
        files={
            "trace-dataset.json": dataset.model_dump_json().encode(),
            "traces.jsonl": payload,
            "issues.json": b"[]",
        },
    )
    rag = persist_trace_rag(
        project.artifacts,
        (artifact_input(manifest),),
        (RAGLineageBinding(trace_id=trace.trace_id, lineage_id="source", partition="fit"),),
        created_at=now,
        code_revision="fixture",
    )
    built = ground_world_model_from_traces(
        (trace,),
        store=project.artifacts,
        serving_rag=artifact_input(rag.manifest),
        model_alias="model",
        model=model,
        created_at=now,
        code_revision="fixture",
        top_k=1,
    )
    task = TaskCase(
        task_id="test",
        origin="authored",
        lineage_group_id="other",
        partition="held_out",
        instruction="Inspect current entries",
        tools=built.tools,
        initial_context={"public_prefix": [{"key": "current", "value": "known"}]},
        workload_weight=1,
    )
    actions = tuple(
        ToolCall(call_id=f"call-{i}", name="inspect", arguments={"key": "current"})
        for i in range(2)
    )
    references = EvaluationReferences(
        capability="recorded_reference_only",
        description="REFERENCE_ONLY_SENTINEL",
        steps=tuple(
            ReferenceStep(
                step=i,
                action={"name": action.name, "arguments": action.arguments},
                reply_available=True,
                reply=f"reply-{i}",
            )
            for i, action in enumerate(actions)
        ),
    )
    return project, FidelitySpec(
        run_id="fixture",
        world_model=artifact_input(built.simulator.manifest),
        task=task,
        workload=FixedWorkflow(id="fixture", task=task.instruction, actions=actions),
        references=references,
        judge=default_judge("fixture-v1", protocol="keyed-v3"),
        judge_alias="model",
        deadline=now + timedelta(minutes=5),
        maximum_physical_attempts=9,
        maximum_output_tokens=4096,
        request_reservation_usd=1,
    )


def reply(content: str) -> JsonHttpResponse:
    """Return an actual Chat Completions wire response for the maintained client."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "model": "fixture-v1",
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
        },
    )


def transition_reply(proposal: JsonObject) -> JsonHttpResponse:
    """Return a typed proposal on the real provider tool-call wire."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "model": "fixture-v1",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "transition",
                                "type": "function",
                                "function": {
                                    "name": "submit_transition",
                                    "arguments": json.dumps(proposal),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )


def judge_reply(axes: JsonObject) -> JsonHttpResponse:
    """Return the judge's submitted axes on the real provider tool-call wire."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "model": "fixture-v1",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "judge",
                                "type": "function",
                                "function": {
                                    "name": "submit_proposal",
                                    "arguments": json.dumps({"axes": axes}),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )


@pytest.mark.parametrize(
    ("json_output", "output_protocol"),
    [(False, "text"), (True, "text"), (False, "json-tool")],
)
def test_predictions_references_and_completed_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
    output_protocol: str,
) -> None:
    """Two real grounded calls carry state; the judge alone receives reference outcomes."""
    project, spec = source_project(tmp_path)
    spec = FidelitySpec.model_validate(
        {
            **spec.model_dump(mode="json"),
            "json_object_output": json_output,
            "output_protocol": output_protocol,
        }
    )
    axes: JsonObject = {
        d.dimension_id: {
            "status": "assessed",
            "score": 5,
            "rationale": "Fixture match",
            "evidence_refs": ["step-0"],
            "critical_flags": [],
        }
        for d in spec.judge.definition.dimensions
    }
    predictions = [
        reply(
            json.dumps(
                {
                    "message": "",
                    "terminal": False,
                    "state": {"seen": i + 1},
                    "tool_results": [{"call_id": f"call-{i}", "content": f"reply-{i}"}],
                }
            )
        )
        for i in range(2)
    ]
    native_reply = {"text": 'a"b\\c\n', "amount": 3.0, "nested": [None, True, {"id": "001"}]}
    if output_protocol == "json-tool":
        spec = spec.model_copy(
            update={
                "references": spec.references.model_copy(
                    update={
                        "steps": tuple(
                            reference.model_copy(update={"reply": native_reply})
                            for reference in spec.references.steps
                        )
                    }
                )
            }
        )
        predictions = [
            transition_reply(
                {
                    "observations": [
                        {"call_id": f"call-{i}", "reply": native_reply, "encoding": "json"}
                    ],
                    "effect": {
                        "kind": "edits",
                        "edits": [{"op": "set", "path": ["seen"], "value": i + 1}],
                    },
                    "terminal": False,
                }
            )
            for i in range(2)
        ]
    wire = ScriptedJsonTransport(predictions + [judge_reply(axes)])
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    assert not wire.requests
    result = execute_fidelity_evaluation(project, prepared)
    assert result.judge_status == "completed" and result.planned_steps == 2
    assert result.mechanical["valid_steps"] == 2
    assert len(wire.requests) == 3
    assert wire.requests[0].payload.get("response_format") == (
        {"type": "json_object"} if json_output else None
    )
    assert all("REFERENCE_ONLY_SENTINEL" not in json.dumps(r.payload) for r in wire.requests[:2])
    messages = wire.requests[1].payload["messages"]
    assert isinstance(messages, list)
    message = messages[1]
    assert isinstance(message, dict)
    content = message["content"]
    assert isinstance(content, str)
    second = json.loads(content)
    assert second["environment_state"] == {"seen": 1}
    if output_protocol == "json-tool":
        prior_result = next(m for m in second["visible_conversation"] if m["role"] == "tool")
        assert json.loads(prior_result["content"]) == native_reply
        assert wire.requests[0].payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "submit_transition"},
        }
    monkeypatch.delenv("FIDELITY_TEST_KEY")
    assert execute_fidelity_evaluation(project, prepared) == result
    assert len(wire.requests) == 3
    spec_path = tmp_path / "fidelity.json"
    spec_path.write_text(spec.model_dump_json())
    replay = CliRunner().invoke(
        app,
        [
            "eval",
            "fidelity-test",
            "--root",
            str(tmp_path),
            "--fidelity",
            str(spec_path),
            "--non-interactive",
            "--verbose",
        ],
    )
    assert replay.exit_code == 0, replay.output
    assert "Saved result. No provider calls." in replay.output
    assert "2/2 planned" in replay.output and "unmeasured" in replay.output
    assert len(wire.requests) == 3
    changed = spec.model_copy(update={"maximum_output_tokens": 2048})
    with pytest.raises(ValueError, match="inputs changed"):
        prepare_fidelity_evaluation(project, changed, code_revision="fixture")


@pytest.mark.parametrize("typed", [False, True])
def test_invalid_returned_proposal_is_not_committed_or_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, typed: bool
) -> None:
    """A completed malformed response remains an output failure with a blocked suffix."""
    project, spec = source_project(tmp_path)
    malformed = reply("not a transition")
    if typed:
        spec = spec.model_copy(update={"output_protocol": "json-tool"})
        malformed = transition_reply(
            {
                "observations": [
                    {"call_id": "wrong", "reply": {"value": "uncommitted"}, "encoding": "json"}
                ],
                "effect": {
                    "kind": "edits",
                    "edits": [{"op": "set", "path": ["uncommitted"], "value": True}],
                },
                "terminal": False,
            }
        )
    wire = ScriptedJsonTransport([malformed, reply("not a judge")])
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    result = execute_fidelity_evaluation(project, prepared)
    assert result.mechanical["valid_steps"] == 0
    assert result.mechanical["failed_steps"] == 1
    assert result.mechanical["blocked_steps"] == 1
    assert result.judge_status == "failed"
    assert len(wire.requests) == 2


def test_a_provider_answering_under_its_served_model_id_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Predictions and judging accept the catalog's served model ID, and only that one."""
    project, spec = source_project(tmp_path, served_model_id="fixture-2026-10-01")
    axes: JsonObject = {
        d.dimension_id: {
            "status": "assessed",
            "score": 5,
            "rationale": "Fixture match",
            "evidence_refs": ["step-0"],
            "critical_flags": [],
        }
        for d in spec.judge.definition.dimensions
    }
    responses = [
        *(
            reply(
                json.dumps(
                    {
                        "message": "",
                        "terminal": False,
                        "state": {"seen": i + 1},
                        "tool_results": [{"call_id": f"call-{i}", "content": f"reply-{i}"}],
                    }
                )
            )
            for i in range(2)
        ),
        judge_reply(axes),
    ]
    for response in responses:
        response.body["model"] = "fixture-2026-10-01"
    wire = ScriptedJsonTransport(responses)
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    result = execute_fidelity_evaluation(project, prepared)
    assert result.judge_status == "completed" and result.mechanical["valid_steps"] == 2
