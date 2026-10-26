"""Direct trace grounding has no semantic construction or native schema dependency."""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import SourceIdentity, sha256_json
from exp.common.models import BillingSource, ModelCapabilities, ModelSnapshot
from exp.common.project import ProjectConfig, ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.common.traces import Trace, TraceDataset, TraceSource, TraceSpan
from exp.simulation.retrieval import RAGLineageBinding, persist_trace_rag
from exp.simulation.world_model.construction import ground_world_model_from_traces


def _episode(now: datetime) -> Trace:
    """Record an observed argument without documenting its type or requiredness."""
    return Trace(
        trace_id="episode",
        task="Inspect an entry",
        source=TraceSource(
            identity=SourceIdentity(kind="file", source_id="observations"),
            semantic_convention_version="1.37.0",
        ),
        tools=(
            ToolSchema(
                name="hidden",
                description="Native policy must not leak",
                input_schema={"type": "object"},
            ),
        ),
        spans=(
            TraceSpan(
                span_id="a",
                name="request",
                started_at=now,
                ended_at=now,
                attributes={
                    "gen_ai.tool.name": "inspect",
                    "gen_ai.tool.call.id": "call",
                    "gen_ai.tool.call.arguments": '{"key":"historical-id"}',
                },
            ),
            TraceSpan(
                span_id="b",
                name="result",
                started_at=now + timedelta(seconds=1),
                ended_at=now + timedelta(seconds=1),
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "inspect",
                    "gen_ai.tool.call.id": "call",
                    "gen_ai.tool.message": '{"value":1}',
                },
            ),
        ),
    )


def test_direct_trace_grounding_uses_observed_calls_and_preserves_exact_source_binding(
    tmp_path: Path,
) -> None:
    """A real persisted RAG builds a simulator without any constructor callback."""
    now = datetime(2026, 10, 14, tzinfo=UTC)
    trace = _episode(now)
    project = ProjectStore(tmp_path, "direct")
    project.initialize(ProjectConfig(project_id="direct"))
    payload = trace.model_dump_json().encode() + b"\n"
    dataset = TraceDataset(
        schema_version=1,
        created_at=now,
        code_revision="test",
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
        (RAGLineageBinding(trace_id=trace.trace_id, lineage_id="source-user", partition="fit"),),
        created_at=now,
        code_revision="test",
    )
    caps = ModelCapabilities()
    model = ModelSnapshot(
        provider="fixture",
        model_id="predictor",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256=caps.identity_sha256(),
        connection_sha256=sha256_json({"provider": "fixture"}),
    )
    built = ground_world_model_from_traces(
        (trace,),
        store=project.artifacts,
        serving_rag=artifact_input(rag.manifest),
        model_alias="predictor",
        model=model,
        created_at=now,
        code_revision="test",
        top_k=3,
    )
    assert tuple(t.name for t in built.tools) == ("inspect",)
    assert built.tools[0].input_schema == {
        "type": "object",
        "properties": {"key": {}},
        "additionalProperties": True,
    }
    knowledge = built.simulator.artifact.knowledge
    assert (
        knowledge is not None
        and knowledge.rules == ""
        and knowledge.source_trace_ids == ("episode",)
    )
    assert "historical-id" not in knowledge.model_dump_json()
    task = TaskCase(
        task_id="task",
        lineage_group_id="other-user",
        partition="held_out",
        instruction="Inspect",
        tools=trace.tools,
        workload_weight=1,
        source_trace_ids=("other-episode",),
    )
    assert built.bind_task(task).tools == built.tools
    assert "Native policy" not in built.bind_task(task).model_dump_json()
    changed = trace.model_copy(update={"task": "Different source contents"})
    with pytest.raises(ValueError, match="trace contents differ"):
        ground_world_model_from_traces(
            (changed,),
            store=project.artifacts,
            serving_rag=artifact_input(rag.manifest),
            model_alias="predictor",
            model=model,
            created_at=now,
            code_revision="test",
            top_k=3,
        )
