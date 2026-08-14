"""End-to-end tests for immutable real-trace RAG construction and retrieval."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    SourceIdentity,
    canonical_json_bytes,
    sha256_json,
)
from wmo.common.models import Embedding, ModelCapabilities, ModelSnapshot
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactFile,
    ArtifactManifest,
    ArtifactStore,
    ProjectPaths,
    artifact_input,
)
from wmo.common.traces import Trace, TraceSource, TraceSpan
from wmo.simulation.ingest import (
    TraceNormalizationResult,
    persist_trace_dataset,
)
from wmo.simulation.retrieval import (
    RAGAction,
    RAGEmbedderBinding,
    RAGLineageBinding,
    RAGQuery,
    TraceRAGRetriever,
    load_rag_index,
    persist_trace_rag,
)

_CREATED_AT = datetime(2026, 8, 13, tzinfo=UTC)
SourceKind = Literal["file", "otlp", "production", "simulation", "manual", "generated"]


class _ConstantEmbedder:
    """Return one fixed unit vector for deterministic tie tests."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        return tuple(Embedding(values=(1.0, 0.0)) for _ in texts)


def test_one_trace_and_more_than_one_thousand_traces_are_valid(tmp_path: Path) -> None:
    """Trace-count guidance never becomes an enforced product boundary."""
    for count, project_id in ((1, "one-trace"), (1_001, "many-traces")):
        store = _store(tmp_path / project_id, project_id)
        source_input, traces = _persist_traces(store, count=count)
        result = persist_trace_rag(
            store,
            (source_input,),
            _bindings(traces),
            created_at=_CREATED_AT,
            code_revision="revision-a",
        )
        assert result.index.transition_count == count


def test_build_reload_is_deterministic_and_terminal_turns_are_excluded(tmp_path: Path) -> None:
    store = _store(tmp_path, "reload")
    source_input, traces = _persist_traces(store, count=2)
    first = persist_trace_rag(
        store,
        (source_input,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
    )
    replay = persist_trace_rag(
        store,
        (source_input,),
        tuple(reversed(_bindings(traces))),
        created_at=_CREATED_AT,
        code_revision="revision-a",
    )
    loaded = load_rag_index(store, first.index.rag_id)

    assert replay.index == first.index == loaded.index
    assert replay.transitions == first.transitions == loaded.transitions
    assert replay.vectors == first.vectors == loaded.vectors
    assert all(item.action_span_id.endswith("-request") for item in loaded.transitions)
    assert all(item.observation_span_id.endswith("-answer") for item in loaded.transitions)


def test_tool_calls_require_real_following_results(tmp_path: Path) -> None:
    store = _store(tmp_path, "tools")
    trace = _tool_trace(with_result=True)
    terminal = _tool_trace(with_result=False, index=2)
    source_input, _ = _persist_trace_values(store, (trace, terminal))
    result = persist_trace_rag(
        store,
        (source_input,),
        _bindings((trace, terminal)),
        created_at=_CREATED_AT,
        code_revision="revision-a",
    )

    assert len(result.transitions) == 1
    transition = result.transitions[0]
    assert transition.action.kind == "tool_call"
    assert transition.action.tool_name == "lookup_account"
    assert transition.observation.kind == "tool_result"
    assert transition.observation.content == "account found"


def test_held_out_lineages_never_enter_the_index(tmp_path: Path) -> None:
    store = _store(tmp_path, "fit-only")
    source_input, traces = _persist_traces(store, count=2)
    bindings = (
        RAGLineageBinding(
            trace_id=traces[0].trace_id,
            lineage_id="lineage-fit",
            partition="fit",
        ),
        RAGLineageBinding(
            trace_id=traces[1].trace_id,
            lineage_id="lineage-held-out",
            partition="held_out",
        ),
    )

    result = persist_trace_rag(
        store,
        (source_input,),
        bindings,
        created_at=_CREATED_AT,
        code_revision="revision-a",
    )

    assert result.index.fit_lineage_ids == ("lineage-fit",)
    assert {item.trace_id for item in result.transitions} == {traces[0].trace_id}


def test_held_out_only_index_is_not_a_supported_fit_evidence_artifact(tmp_path: Path) -> None:
    """Every retrieval artifact retains fit evidence when serving also includes held out."""
    store = _store(tmp_path, "held-out-only")
    source_input, traces = _persist_traces(store, count=1)

    with pytest.raises(ValueError, match="must contain fit"):
        persist_trace_rag(
            store,
            (source_input,),
            (
                RAGLineageBinding(
                    trace_id=traces[0].trace_id,
                    lineage_id="lineage-held-out",
                    partition="held_out",
                ),
            ),
            created_at=_CREATED_AT,
            code_revision="revision-a",
            included_partitions=frozenset({"held_out"}),
        )


def test_retrieve_filters_lineage_before_stable_tie_break_and_never_mutates(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, "retrieve")
    source_input, traces = _persist_traces(store, count=2)
    embedder = _constant_binding()
    persisted = persist_trace_rag(
        store,
        (source_input,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
        embedder=embedder,
    )
    loaded = load_rag_index(store, persisted.index.rag_id)
    retriever = TraceRAGRetriever(loaded, embedder=embedder)
    excluded = loaded.transitions[0].lineage_id
    query = RAGQuery(
        task="reset password 0",
        action=RAGAction(kind="message", content="Which email is on the account?"),
        excluded_lineage_ids=(excluded,),
    )
    query_before = query.model_dump_json()
    index_before = loaded
    artifact_before = store.read_bytes(persisted.index.rag_id, "vectors.jsonl")

    matches = retriever.retrieve(query)

    assert [item.transition.lineage_id for item in matches] == [loaded.transitions[1].lineage_id]
    assert query.model_dump_json() == query_before
    assert loaded == index_before
    assert store.read_bytes(persisted.index.rag_id, "vectors.jsonl") == artifact_before


def test_equal_scores_use_transition_id_order(tmp_path: Path) -> None:
    store = _store(tmp_path, "ties")
    source_input, traces = _persist_traces(store, count=3)
    embedder = _constant_binding()
    persisted = persist_trace_rag(
        store,
        (source_input,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
        embedder=embedder,
    )
    query = RAGQuery(
        task="anything",
        action=RAGAction(kind="message", content="anything"),
        top_k=2,
    )

    matches = TraceRAGRetriever(
        load_rag_index(store, persisted.index.rag_id), embedder=embedder
    ).retrieve(query)

    assert [match.transition.transition_id for match in matches] == list(
        persisted.index.transition_ids[:2]
    )


def test_omitted_query_limit_uses_index_default_and_explicit_limit_overrides(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, "persisted-top-k")
    source_input, traces = _persist_traces(store, count=3)
    embedder = _constant_binding()
    persisted = persist_trace_rag(
        store,
        (source_input,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
        embedder=embedder,
        default_top_k=1,
    )
    retriever = TraceRAGRetriever(load_rag_index(store, persisted.index.rag_id), embedder=embedder)
    action = RAGAction(kind="message", content="anything")

    inherited = retriever.retrieve(RAGQuery(task="anything", action=action))
    overridden = retriever.retrieve(RAGQuery(task="anything", action=action, top_k=2))

    assert len(inherited) == 1
    assert len(overridden) == 2


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [("default_top_k", 1), ("code_revision", "revision-b")],
)
def test_replay_sensitive_fields_produce_distinct_artifact_ids(
    tmp_path: Path,
    changed_field: str,
    changed_value: str | int,
) -> None:
    store = _store(tmp_path, f"identity-{changed_field}")
    source_input, traces = _persist_traces(store, count=2)
    kwargs: dict[str, str | int] = {
        "code_revision": "revision-a",
        "default_top_k": 5,
    }
    kwargs[changed_field] = changed_value
    original = persist_trace_rag(
        store,
        (source_input,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
        default_top_k=5,
    )
    changed = persist_trace_rag(
        store,
        (source_input,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision=str(kwargs["code_revision"]),
        default_top_k=int(kwargs["default_top_k"]),
    )

    assert changed.index.rag_id != original.index.rag_id
    assert load_rag_index(store, changed.index.rag_id).index == changed.index


@pytest.mark.parametrize("artifact_type", ["simulation", "teacher-rollout", "judgment"])
def test_non_trace_artifact_sources_are_forbidden(tmp_path: Path, artifact_type: str) -> None:
    store = _store(tmp_path, "forbidden-artifact")
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=_CREATED_AT,
        code_revision="revision-a",
    )
    manifest = store.write(
        artifact_id=f"forbidden-{artifact_type}",
        artifact_type=artifact_type,
        envelope=envelope,
        files={"evidence.json": b"{}"},
    )

    with pytest.raises(ValueError, match="forbidden artifact type"):
        persist_trace_rag(
            store,
            (artifact_input(manifest),),
            (),
            created_at=_CREATED_AT,
            code_revision="revision-a",
        )


@pytest.mark.parametrize("kind", ["generated", "simulation", "manual"])
def test_non_real_trace_provenance_is_forbidden(tmp_path: Path, kind: SourceKind) -> None:
    store = _store(tmp_path, f"forbidden-{kind}")
    trace = _message_trace(0, source_kind=kind)
    source_input, _ = _persist_trace_values(store, (trace,))

    with pytest.raises(ValueError, match="forbidden provenance"):
        persist_trace_rag(
            store,
            (source_input,),
            _bindings((trace,)),
            created_at=_CREATED_AT,
            code_revision="revision-a",
        )


def test_loader_fails_closed_on_payload_hash_change(tmp_path: Path) -> None:
    store, rag_id = _built_index(tmp_path, "bad-hash")
    path = store.project_directory / "artifacts" / rag_id / "transitions.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ArtifactCorruptionError, match="digest mismatch"):
        load_rag_index(store, rag_id)


def test_loader_fails_closed_on_envelope_identity_change(tmp_path: Path) -> None:
    store, rag_id = _built_index(tmp_path, "bad-id")
    loaded = load_rag_index(store, rag_id)
    mismatched = loaded.index.model_copy(update={"rag_id": "trace-rag-other"})
    store.write(
        artifact_id="trace-rag-mismatched",
        artifact_type="trace-rag-index",
        envelope=mismatched,
        files={
            "rag-index.json": canonical_json_bytes(mismatched),
            "transitions.jsonl": store.read_bytes(rag_id, "transitions.jsonl"),
            "vectors.jsonl": store.read_bytes(rag_id, "vectors.jsonl"),
        },
    )

    with pytest.raises(ArtifactCorruptionError, match="does not match artifact"):
        load_rag_index(store, "trace-rag-mismatched")


def test_loader_fails_closed_on_vector_dimension_change(tmp_path: Path) -> None:
    store, rag_id = _built_index(tmp_path, "bad-dimension")
    loaded = load_rag_index(store, rag_id)
    vector_payload = (
        canonical_json_bytes(loaded.vectors[0].model_copy(update={"values": (1.0,)})) + b"\n"
    )
    changed = loaded.index.model_copy(
        update={"vectors_sha256": hashlib.sha256(vector_payload).hexdigest()}
    )
    store.write(
        artifact_id="trace-rag-bad-dimension",
        artifact_type="trace-rag-index",
        envelope=changed.model_copy(update={"rag_id": "trace-rag-bad-dimension"}),
        files={
            "rag-index.json": canonical_json_bytes(
                changed.model_copy(update={"rag_id": "trace-rag-bad-dimension"})
            ),
            "transitions.jsonl": store.read_bytes(rag_id, "transitions.jsonl"),
            "vectors.jsonl": vector_payload,
        },
    )

    with pytest.raises(ArtifactCorruptionError, match="dimension"):
        load_rag_index(store, "trace-rag-bad-dimension")


def test_loader_fails_closed_on_nan_vector(tmp_path: Path) -> None:
    store, rag_id = _built_index(tmp_path, "bad-nan")
    directory = store.project_directory / "artifacts" / rag_id
    vector_path = directory / "vectors.jsonl"
    value = json.loads(vector_path.read_text(encoding="utf-8").splitlines()[0])
    value["values"][0] = float("nan")
    payload = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    vector_path.write_bytes(payload)
    manifest_path = directory / "manifest.json"
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_bytes())
    files = tuple(
        ArtifactFile(
            path=item.path,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
        if item.path == "vectors.jsonl"
        else item
        for item in manifest.files
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest.model_copy(update={"files": files})))

    with pytest.raises(ArtifactCorruptionError):
        load_rag_index(store, rag_id)


def _built_index(tmp_path: Path, project_id: str) -> tuple[ArtifactStore, str]:
    """Build one valid local index and return its store and identity."""
    store = _store(tmp_path, project_id)
    source_input, traces = _persist_traces(store, count=1)
    persisted = persist_trace_rag(
        store,
        (source_input,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
    )
    return store, persisted.index.rag_id


def _store(root: Path, project_id: str) -> ArtifactStore:
    """Create a project-scoped artifact store for a test."""
    return ArtifactStore(ProjectPaths(root=root, project_id=project_id))


def _persist_traces(
    store: ArtifactStore,
    *,
    count: int,
) -> tuple[ArtifactInput, tuple[Trace, ...]]:
    """Persist a requested positive number of canonical real message traces."""
    traces = tuple(_message_trace(index) for index in range(count))
    source_input, _ = _persist_trace_values(store, traces)
    return source_input, traces


def _persist_trace_values(
    store: ArtifactStore,
    traces: tuple[Trace, ...],
) -> tuple[ArtifactInput, tuple[Trace, ...]]:
    """Persist exact canonical traces and return their verified manifest input."""
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=traces, issues=()),
        store,
        created_at=_CREATED_AT,
        code_revision="revision-a",
    )
    return artifact_input(persisted.manifest), persisted.traces


def _bindings(traces: tuple[Trace, ...]) -> tuple[RAGLineageBinding, ...]:
    """Assign one deterministic fit lineage per source trace."""
    return tuple(
        RAGLineageBinding(
            trace_id=trace.trace_id,
            lineage_id=f"lineage-{index}",
            partition="fit",
        )
        for index, trace in enumerate(traces)
    )


def _message_trace(index: int, *, source_kind: SourceKind = "otlp") -> Trace:
    """Create one real two-call transcript with one nonterminal observed transition."""
    start = _CREATED_AT + timedelta(seconds=index * 10)
    request = f"reset password {index}"
    question = "Which email is on the account?"
    answer = f"customer-{index}@example.com"
    return Trace(
        trace_id=f"trace-{index}",
        conversation_id=f"conversation-{index}",
        task=request,
        spans=(
            TraceSpan(
                span_id=f"span-{index}-request",
                name="chat",
                started_at=start,
                ended_at=start + timedelta(seconds=1),
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.input.messages": [{"role": "user", "content": request}],
                    "gen_ai.completion": question,
                },
            ),
            TraceSpan(
                span_id=f"span-{index}-answer",
                name="chat",
                started_at=start + timedelta(seconds=2),
                ended_at=start + timedelta(seconds=3),
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.input.messages": [
                        {"role": "user", "content": request},
                        {"role": "assistant", "content": question},
                        {"role": "user", "content": answer},
                    ],
                    "gen_ai.completion": "A reset link is on the way.",
                },
            ),
        ),
        source=TraceSource(
            identity=SourceIdentity(
                kind=source_kind,
                source_id="trace-fixture",
                sha256="a" * 64,
            ),
            semantic_convention_version="1.37.0",
        ),
    )


def _tool_trace(*, with_result: bool, index: int = 1) -> Trace:
    """Create a tool call with an optional real following tool observation."""
    start = _CREATED_AT + timedelta(seconds=index * 10)
    spans = [
        TraceSpan(
            span_id=f"span-{index}-tool-call",
            name="chat",
            started_at=start,
            ended_at=start + timedelta(seconds=1),
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.tool.name": "lookup_account",
                "gen_ai.tool.call.id": f"call-{index}",
                "gen_ai.tool.call.arguments": {"customer": index},
            },
        )
    ]
    if with_result:
        spans.append(
            TraceSpan(
                span_id=f"span-{index}-tool-result",
                name="execute_tool lookup_account",
                started_at=start + timedelta(seconds=2),
                ended_at=start + timedelta(seconds=3),
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "lookup_account",
                    "gen_ai.tool.call.id": f"call-{index}",
                    "gen_ai.tool.message": "account found",
                },
            )
        )
    return Trace(
        trace_id=f"tool-trace-{index}",
        conversation_id=f"tool-conversation-{index}",
        task="find the account",
        spans=tuple(spans),
        source=TraceSource(
            identity=SourceIdentity(
                kind="otlp",
                source_id="trace-fixture",
                sha256="a" * 64,
            ),
            semantic_convention_version="1.37.0",
        ),
    )


def _constant_binding() -> RAGEmbedderBinding:
    """Return a semantic-style explicit constant embedding binding."""
    capabilities = ModelCapabilities(supports_embeddings=True)
    return RAGEmbedderBinding(
        client=_ConstantEmbedder(),
        snapshot=ModelSnapshot(
            provider="test",
            model_id="constant",
            revision="1",
            capabilities_sha256=sha256_json(capabilities),
            connection_sha256="b" * 64,
        ),
    )
