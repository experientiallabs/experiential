"""The reconstructed path cannot inherit native behavior through task tool metadata."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from exp.common.core.artifacts import JsonObject, SourceIdentity, sha256_json
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelCapabilities,
    ModelSnapshot,
    ToolCall,
)
from exp.common.project import ProjectConfig, ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.common.traces import Trace, TraceDataset, TraceSource, TraceSpan
from exp.simulation.engines.text.prompt import build_world_model_request
from exp.simulation.reconstruction.library import LibraryEntry, retrieve_library
from exp.simulation.retrieval import RAGLineageBinding, persist_trace_rag
from exp.simulation.world_model.construction import construct_world_model


def test_bound_task_exposes_inferred_tools_without_native_documentation(tmp_path: Path) -> None:
    """Drive construction, persistence and task binding with no supplied native contracts."""
    now = datetime(2026, 10, 14, tzinfo=UTC)
    origin = SourceIdentity(kind="file", source_id="test-observations")
    trace = Trace(
        trace_id="episode",
        task="Read a value",
        source=TraceSource(identity=origin, semantic_convention_version="1.37.0"),
        spans=(
            TraceSpan(
                span_id="action",
                name="request",
                started_at=now,
                ended_at=now,
                attributes={
                    "gen_ai.tool.name": "read",
                    "gen_ai.tool.call.id": "a",
                    "gen_ai.tool.call.arguments": "{}",
                },
            ),
            TraceSpan(
                span_id="reply",
                name="result",
                started_at=now + timedelta(seconds=1),
                ended_at=now + timedelta(seconds=1),
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "read",
                    "gen_ai.tool.call.id": "a",
                    "gen_ai.tool.message": "null",
                },
            ),
        ),
    )
    project = ProjectStore(tmp_path, "construction")
    project.initialize(ProjectConfig(project_id="construction"))
    traces = (trace, trace.model_copy(update={"trace_id": "second-episode"}))
    payload = b"".join(t.model_dump_json().encode() + b"\n" for t in traces)
    dataset = TraceDataset(
        schema_version=1,
        created_at=now,
        code_revision="test",
        source=origin,
        dataset_id="source",
        semantic_convention_version="1.37.0",
        traces_path="traces.jsonl",
        traces_sha256=hashlib.sha256(payload).hexdigest(),
        issues_path="issues.json",
        issues_sha256=hashlib.sha256(b"[]").hexdigest(),
        trace_ids=tuple(t.trace_id for t in traces),
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
        tuple(
            RAGLineageBinding(trace_id=t.trace_id, lineage_id=t.trace_id, partition="fit")
            for t in traces
        ),
        created_at=now,
        code_revision="test",
        default_top_k=1,
    )

    dispatched: dict[str, JsonObject] = {}

    def complete(body: JsonObject, operation: str) -> JsonObject:
        """Provide a deterministic constructor proposal for the observed read interface."""
        assert "Native-policy-secret" not in json.dumps(body)
        dispatched[operation] = body
        return {
            "interface": {
                "tools": [
                    {
                        "name": "read",
                        "description": "Observed read",
                        "parameters": {"type": "object"},
                    }
                ],
                "state_schema": {"type": "object"},
            },
            "claims": [
                {
                    "statement": "Unobserved behavior is unknown",
                    "status": "unknown",
                    "origin": "model_prior",
                    "scope": "unseen calls",
                }
            ],
            "text_rules": "Retain uncertainty about unseen behavior.",
        }

    caps = ModelCapabilities()
    model = ModelSnapshot(
        provider="fixture",
        model_id="constructor",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256=caps.identity_sha256(),
        connection_sha256=sha256_json({"fixture": True}),
    )
    built = construct_world_model(
        traces,
        store=project.artifacts,
        serving_rag=artifact_input(rag.manifest),
        complete=complete,
        model_alias="constructor",
        model=model,
        run_id="cold",
        clock=lambda: now,
        deadline=now + timedelta(minutes=1),
        code_revision="test",
        top_k=1,
        construction_guidance="Prefer fields the observations show.",
    )
    for operation, body in dispatched.items():
        saved = json.loads(project.artifacts.read_bytes(operation + "-intent", "request.json"))
        assert saved["request"] == body
        assert {"role": "developer", "content": "Prefer fields the observations show."} in (
            saved["request"]["messages"]
        )
    native_task = TaskCase(
        task_id="target",
        lineage_group_id="target-user",
        partition="held_out",
        instruction="Read",
        tools=(
            ToolSchema(
                name="read", description="Native-policy-secret", input_schema={"type": "object"}
            ),
            ToolSchema(
                name="undiscovered", description="Private behavior", input_schema={"type": "object"}
            ),
        ),
        workload_weight=1,
        source_trace_ids=("target-observation",),
    )
    task = built.bind_task(native_task)
    knowledge = built.simulator.artifact.knowledge
    assert knowledge is not None
    assert knowledge.claims == built.reconstruction.claims
    request = build_world_model_request(
        task,
        visible_messages=(),
        grounded_examples=(),
        maximum_output_tokens=512,
        candidate_response=AssistantAction(
            tool_calls=(ToolCall(call_id="current-read", name="read", arguments={}),)
        ),
        state={},
        knowledge=knowledge,
    )
    assert request.messages[1].content is not None
    delivered = json.loads(request.messages[1].content)
    assert delivered["behavioral_knowledge"]["claims"] == [
        claim.model_dump(mode="json") for claim in built.reconstruction.claims
    ]
    assert [tool.name for tool in task.tools] == ["read"]
    assert task.tools[0].description == "Observed read"
    assert "Native-policy-secret" not in task.model_dump_json()
    assert native_task.tools[0].description == "Native-policy-secret"
    assert built.reconstruction.sources.published_contracts is None

    originals = [entry for entry in built.library.entries if entry.kind == "trajectory"]
    assert len(originals) == 2
    assert all(
        len(
            json.loads(project.artifacts.read_bytes(entry.artifact.artifact_id, entry.path))[
                "traces"
            ]
        )
        == 1
        for entry in originals
    )

    seen: list[str] = []

    def select(query: JsonObject, entries: tuple[LibraryEntry, ...]) -> tuple[str, ...]:
        """A constructor must not retrieve originals it already receives in full."""
        seen.extend(entry.entry_id for entry in entries)
        return tuple(entry.entry_id for entry in entries)

    context = retrieve_library(project.artifacts, built.library, traces, select=select, top_k=2)
    assert seen == [] and context["selected"] == []
    excluded = context["excluded_exact_target_originals"]
    assert isinstance(excluded, list) and len(excluded) == 2
