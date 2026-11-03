"""The reconstructed path cannot inherit native behavior through task tool metadata."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, SourceIdentity, sha256_json
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelCapabilities,
    ModelSnapshot,
    ToolCall,
)
from exp.common.project import (
    ArtifactStore,
    ProjectConfig,
    ProjectPaths,
    ProjectStore,
    artifact_input,
)
from exp.common.tasks import TaskCase, ToolSchema
from exp.common.traces import Trace, TraceDataset, TraceSource, TraceSpan
from exp.simulation.engines.text.prompt import build_world_model_request
from exp.simulation.reconstruction.library import LibraryEntry, retrieve_library
from exp.simulation.retrieval import CorpusScope, RAGLineageBinding, persist_trace_rag
from exp.simulation.retrieval.tests.retrieval_test import _persist_trace_values
from exp.simulation.world_model.artifact import WORLD_MODEL_ARTIFACT_PATH
from exp.simulation.world_model.construction import (
    TraceGroundedWorldModel,
    construct_world_model,
    ground_world_model_from_traces,
)
from exp.simulation.world_model.runtime import (
    load_grounded_world_model_artifact,
    persist_scoped_world_model,
)


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


_SCOPED_AT = datetime(2026, 10, 14, tzinfo=UTC)


def _tool_episode(index: int) -> Trace:
    """Record one episode whose only observed call uses a tool named after its index.

    Args:
        index: Episode position; trace ``trace-{index}`` calls ``tool-{index}``.

    Returns:
        The canonical episode.
    """
    tool = f"tool-{index}"
    return Trace(
        trace_id=f"trace-{index}",
        task=f"Inspect entry {index}",
        source=TraceSource(
            identity=SourceIdentity(kind="file", source_id="scoped-observations"),
            semantic_convention_version="1.37.0",
        ),
        spans=(
            TraceSpan(
                span_id="a",
                name="request",
                started_at=_SCOPED_AT,
                ended_at=_SCOPED_AT,
                attributes={
                    "gen_ai.tool.name": tool,
                    "gen_ai.tool.call.id": "call",
                    "gen_ai.tool.call.arguments": '{"key":"k"}',
                },
            ),
            TraceSpan(
                span_id="b",
                name="result",
                started_at=_SCOPED_AT + timedelta(seconds=1),
                ended_at=_SCOPED_AT + timedelta(seconds=1),
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool,
                    "gen_ai.tool.call.id": "call",
                    "gen_ai.tool.message": '{"value":1}',
                },
            ),
        ),
    )


def _bindings(traces: tuple[Trace, ...]) -> tuple[RAGLineageBinding, ...]:
    """Bind traces 0-2 to their own fit lineages and trace 3 to a held-out lineage.

    Args:
        traces: The four fixture episodes.

    Returns:
        The complete lineage assignment.
    """
    return tuple(
        RAGLineageBinding(
            trace_id=trace.trace_id,
            lineage_id=f"lineage-{index}",
            partition="held_out" if index == 3 else "fit",
        )
        for index, trace in enumerate(traces)
    )


@dataclass(frozen=True)
class _ScopedCorpus:
    """Four persisted episodes, their bindings and the build's serving index.

    Attributes:
        store: Project-local artifact store.
        source: Trace dataset pointer.
        traces: Episodes ``trace-0`` to ``trace-3`` in order.
        bindings: Complete lineage assignment; ``trace-3`` is held out.
        serving: Serving index over fit and held-out evidence.
    """

    store: ArtifactStore
    source: ArtifactInput
    traces: tuple[Trace, ...]
    bindings: tuple[RAGLineageBinding, ...]
    serving: ArtifactInput


def _scoped_corpus(tmp_path: Path) -> _ScopedCorpus:
    """Persist four tool episodes and a serving index over fit and held-out evidence.

    Args:
        tmp_path: Temporary project root.

    Returns:
        The persisted corpus.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="scoped"))
    source, traces = _persist_trace_values(store, tuple(_tool_episode(i) for i in range(4)))
    bindings = _bindings(traces)
    serving = persist_trace_rag(
        store,
        (source,),
        bindings,
        created_at=_SCOPED_AT,
        code_revision="test",
        included_partitions=frozenset({"fit", "held_out"}),
    )
    return _ScopedCorpus(store, source, traces, bindings, artifact_input(serving.manifest))


def _scope(kind: Literal["lineage", "trace"], *ids: str) -> CorpusScope:
    """Declare a construction scope bound to a fixed plan digest.

    Args:
        kind: Scope kind.
        *ids: Admitted identities.

    Returns:
        The canonical scope.
    """
    return CorpusScope.declare(
        name="world-learning-construction", kind=kind, ids=ids, source_plan="d" * 64
    )


def _ground(
    corpus: _ScopedCorpus,
    traces: tuple[Trace, ...],
    *,
    serving: ArtifactInput | None = None,
    scope: CorpusScope | None = None,
    bindings: tuple[RAGLineageBinding, ...] | None = None,
) -> TraceGroundedWorldModel:
    """Ground a world model directly from traces over the corpus.

    Args:
        corpus: Persisted fixture corpus.
        traces: Construction episodes.
        serving: Index to ground over; defaults to the build's serving index.
        scope: Optional construction scope.
        bindings: Optional lineage assignment for the scope.

    Returns:
        The grounded world model.
    """
    return ground_world_model_from_traces(
        traces,
        store=corpus.store,
        serving_rag=serving or corpus.serving,
        model_alias="predictor",
        model=_predictor(),
        created_at=_SCOPED_AT,
        code_revision="test",
        top_k=3,
        construction_scope=scope,
        lineage_bindings=bindings,
    )


def _predictor() -> ModelSnapshot:
    """Return the frozen fixture predictor identity.

    Returns:
        The model snapshot.
    """
    return ModelSnapshot(
        provider="fixture",
        model_id="predictor",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256=ModelCapabilities().identity_sha256(),
        connection_sha256=sha256_json({"provider": "fixture"}),
    )


def test_scoped_grounding_reads_exactly_the_in_scope_fit_traces(tmp_path: Path) -> None:
    """Under a scope only in-scope fit traces are read, and the scope is recorded."""
    corpus = _scoped_corpus(tmp_path)
    t0, _, t2, _ = corpus.traces
    scope = _scope("trace", "trace-0", "trace-2")

    built = _ground(corpus, (t0, t2), scope=scope)
    assert [tool.name for tool in built.tools] == ["tool-0", "tool-2"]
    artifact = load_grounded_world_model_artifact(
        corpus.store, artifact_input(built.simulator.manifest)
    )
    assert artifact.construction_scope == scope
    assert artifact.knowledge is not None
    assert artifact.knowledge.source_trace_ids == ("trace-0", "trace-2")
    assert artifact.serving_rag == corpus.serving
    lineage = _scope("lineage", "lineage-0", "lineage-2")
    by_lineage = _ground(corpus, (t0, t2), scope=lineage)
    assert by_lineage.simulator.artifact.construction_scope == lineage


def test_scoped_grounding_refuses_held_out_unknown_and_mismatched_traces(
    tmp_path: Path,
) -> None:
    """Held-out or unknown scope ids, and traces other than the in-scope set, are refused."""
    corpus = _scoped_corpus(tmp_path)
    t0, t1, t2, t3 = corpus.traces
    with pytest.raises(ValueError, match="admits held-out trace trace-3"):
        _ground(corpus, (t0, t3), scope=_scope("trace", "trace-0", "trace-3"))
    with pytest.raises(ValueError, match="admits held-out trace trace-3"):
        _ground(corpus, (t3,), scope=_scope("lineage", "lineage-3"))
    with pytest.raises(ValueError, match="names trace ghost outside"):
        _ground(corpus, (t0,), scope=_scope("trace", "ghost", "trace-0"))
    with pytest.raises(ValueError, match="names lineage lineage-9 outside"):
        _ground(corpus, (t0,), scope=_scope("lineage", "lineage-0", "lineage-9"))
    with pytest.raises(ValueError, match="construction trace trace-3 is not an in-scope"):
        _ground(corpus, (t0, t3), scope=_scope("trace", "trace-0"))
    with pytest.raises(ValueError, match="construction trace trace-1 is not an in-scope"):
        _ground(corpus, (t0, t1), scope=_scope("trace", "trace-0"))
    with pytest.raises(ValueError, match="omits in-scope fit trace trace-2"):
        _ground(corpus, (t0,), scope=_scope("trace", "trace-0", "trace-2"))
    changed = t0.model_copy(update={"task": "Different source contents"})
    with pytest.raises(ValueError, match="trace contents differ"):
        _ground(corpus, (changed, t2), scope=_scope("trace", "trace-0", "trace-2"))


def test_scoped_grounding_places_traces_outside_a_scoped_index_by_their_bindings(
    tmp_path: Path,
) -> None:
    """A retrieval-scoped index places only its own traces; bindings place the others."""
    corpus = _scoped_corpus(tmp_path)
    t0, t1, t2, _ = corpus.traces
    retrieval = CorpusScope.declare(
        name="world-learning-retrieval", kind="trace", ids=("trace-1",), source_plan="d" * 64
    )
    scoped = artifact_input(
        persist_trace_rag(
            corpus.store,
            (corpus.source,),
            corpus.bindings,
            created_at=_SCOPED_AT,
            code_revision="test",
            corpus_scope=retrieval,
        ).manifest
    )
    construction = _scope("trace", "trace-0", "trace-1", "trace-2")
    with pytest.raises(ValueError, match="pass the build's lineage bindings"):
        _ground(corpus, (t0, t1, t2), serving=scoped, scope=construction)
    built = _ground(
        corpus, (t0, t1, t2), serving=scoped, scope=construction, bindings=corpus.bindings
    )
    assert [tool.name for tool in built.tools] == ["tool-1"]
    assert built.simulator.artifact.construction_scope == construction
    leaked = tuple(
        binding.model_copy(update={"partition": "held_out"})
        if binding.trace_id == "trace-1"
        else binding
        for binding in corpus.bindings
    )
    with pytest.raises(ValueError, match="lineage binding of trace trace-1 differs"):
        _ground(corpus, (t0, t1, t2), serving=scoped, scope=construction, bindings=leaked)
    with pytest.raises(ValueError, match="every source trace exactly once"):
        _ground(
            corpus, (t0, t1, t2), serving=scoped, scope=construction, bindings=corpus.bindings[:3]
        )


def test_unscoped_grounding_keeps_the_complete_corpus_contract(tmp_path: Path) -> None:
    """Without a scope the index's whole corpus is required and bindings are refused."""
    corpus = _scoped_corpus(tmp_path)
    t0, t1, t2, _ = corpus.traces
    with pytest.raises(ValueError, match="trace contents differ"):
        _ground(corpus, (t0, t2))
    with pytest.raises(ValueError, match="resolve only a declared construction scope"):
        _ground(corpus, corpus.traces, bindings=corpus.bindings)
    built = _ground(corpus, corpus.traces)
    assert built.simulator.artifact.construction_scope is None
    assert b"construction_scope" not in corpus.store.read_bytes(
        built.simulator.manifest.artifact_id, WORLD_MODEL_ARTIFACT_PATH
    )


def test_scoped_construction_records_the_scope_through_retrieval_rebinding(
    tmp_path: Path,
) -> None:
    """Constructing under one scope and retrieving under another keeps both recorded."""
    corpus = _scoped_corpus(tmp_path)
    t0, t1, _, _ = corpus.traces
    construction = _scope("lineage", "lineage-0", "lineage-1")

    def complete(body: JsonObject, operation: str) -> JsonObject:
        """Return a fixed constructor proposal.

        Args:
            body: Constructor request.
            operation: Operation name.

        Returns:
            The proposal.
        """
        assert "tool-2" not in json.dumps(body) and "tool-3" not in json.dumps(body)
        return {
            "interface": {
                "tools": [{"name": "tool-0", "description": "Observed", "parameters": {}}],
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
            "text_rules": "Retain uncertainty.",
        }

    built = construct_world_model(
        (t0, t1),
        store=corpus.store,
        serving_rag=corpus.serving,
        complete=complete,
        model_alias="constructor",
        model=_predictor(),
        run_id="scoped-cold",
        clock=lambda: _SCOPED_AT,
        deadline=_SCOPED_AT + timedelta(minutes=1),
        code_revision="test",
        top_k=1,
        construction_scope=construction,
    )
    world = artifact_input(built.simulator.manifest)
    assert built.simulator.artifact.construction_scope == construction
    retrieval = CorpusScope.declare(
        name="world-learning-retrieval",
        kind="trace",
        ids=("trace-1", "trace-2"),
        source_plan="d" * 64,
    )
    scoped = artifact_input(
        persist_trace_rag(
            corpus.store,
            (corpus.source,),
            corpus.bindings,
            created_at=_SCOPED_AT,
            code_revision="test",
            corpus_scope=retrieval,
        ).manifest
    )
    rebound = persist_scoped_world_model(corpus.store, world, scoped, retrieval_scope=retrieval)
    assert rebound == persist_scoped_world_model(
        corpus.store, world, scoped, retrieval_scope=retrieval, construction_scope=construction
    )
    artifact = load_grounded_world_model_artifact(corpus.store, rebound)
    assert artifact.construction_scope == construction
    assert artifact.serving_rag == scoped
    with pytest.raises(ValueError, match="records construction scope"):
        persist_scoped_world_model(
            corpus.store,
            world,
            scoped,
            retrieval_scope=retrieval,
            construction_scope=_scope("lineage", "lineage-0", "lineage-1", "lineage-2"),
        )
