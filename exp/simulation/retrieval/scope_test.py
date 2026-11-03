"""Tests for corpus scopes and scoped fit-only RAG indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import canonical_json_bytes, sha256_json
from exp.common.project import ArtifactCorruptionError, ArtifactStore, artifact_input
from exp.common.traces import Trace
from exp.simulation.retrieval import (
    CorpusScope,
    RAGIndex,
    RAGLineageBinding,
    load_fit_rag_retriever,
    load_rag_index,
    persist_trace_rag,
)
from exp.simulation.retrieval.build import PersistedRAGIndex
from exp.simulation.retrieval.contracts import RAG_INDEX_PATH
from exp.simulation.retrieval.tests.retrieval_test import (
    _CREATED_AT,
    _persist_traces,
    _store,
)

_PLAN = "c" * 64


def _scope(kind: Literal["lineage", "trace"], *ids: str) -> CorpusScope:
    """Declare a retrieval scope bound to a fixed plan digest."""
    return CorpusScope.declare(
        name="world-learning-retrieval", kind=kind, ids=ids, source_plan=_PLAN
    )


def _bindings(traces: tuple[Trace, ...]) -> tuple[RAGLineageBinding, ...]:
    """Bind traces 0-2 to fit lineages and trace 3 to a held-out lineage."""
    return tuple(
        RAGLineageBinding(
            trace_id=trace.trace_id,
            lineage_id=f"lineage-{index}",
            partition="held_out" if index == 3 else "fit",
        )
        for index, trace in enumerate(traces)
    )


def _build(
    store: ArtifactStore,
    scope: CorpusScope | None,
    *,
    partitions: frozenset[Literal["fit", "held_out"]] = frozenset({"fit"}),
) -> tuple[PersistedRAGIndex, tuple[Trace, ...]]:
    """Persist four traces and build an index over them, optionally scoped."""
    source, traces = _persist_traces(store, count=4)
    built = persist_trace_rag(
        store,
        (source,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
        included_partitions=partitions,
        corpus_scope=scope,
    )
    return built, traces


def test_declare_sorts_deduplicates_and_hashes_identities() -> None:
    """A declared scope is canonical regardless of input order."""
    scope = _scope("trace", "b", "a", "b")

    assert scope.ids == ("a", "b")
    assert scope.ids_sha256 == sha256_json(["a", "b"])
    assert scope == _scope("trace", "a", "b")
    assert scope.admits(trace_id="a", lineage_id="x")
    assert not scope.admits(trace_id="c", lineage_id="a")
    assert _scope("lineage", "x").admits(trace_id="c", lineage_id="x")


def test_scope_refuses_unsorted_empty_or_misdigested_identities() -> None:
    """A hand-built scope must carry sorted identities and their exact digest."""
    with pytest.raises(ValidationError, match="sorted and unique"):
        CorpusScope(
            name="s",
            kind="trace",
            ids=("b", "a"),
            ids_sha256=sha256_json(["b", "a"]),
            source_plan=_PLAN,
        )
    with pytest.raises(ValidationError, match="digest"):
        CorpusScope(name="s", kind="trace", ids=("a",), ids_sha256="0" * 64, source_plan=_PLAN)
    with pytest.raises(ValidationError):
        _scope("trace")


def test_unscoped_index_identity_and_envelope_carry_no_scope(tmp_path: Path) -> None:
    """Without a scope the index envelope and identity are exactly the unscoped contract."""
    store = _store(tmp_path, "unscoped")
    built, _ = _build(store, None)

    assert built.index.corpus_scope is None
    assert b"corpus_scope" not in store.read_bytes(built.index.rag_id, RAG_INDEX_PATH)
    assert built.index.fit_lineage_ids == ("lineage-0", "lineage-1", "lineage-2")


def test_trace_scoped_index_holds_exactly_the_in_scope_fit_transitions(tmp_path: Path) -> None:
    """A trace scope keeps only its traces, records itself and changes the identity."""
    store = _store(tmp_path, "trace-scope")
    unscoped, traces = _build(store, None)
    scope = _scope("trace", traces[0].trace_id, traces[2].trace_id)
    scoped, _ = _build(store, scope)

    assert {item.trace_id for item in scoped.transitions} == {
        traces[0].trace_id,
        traces[2].trace_id,
    }
    assert scoped.index.fit_lineage_ids == ("lineage-0", "lineage-2")
    assert scoped.index.included_lineage_ids == scoped.index.fit_lineage_ids
    assert scoped.index.corpus_scope == scope
    assert scoped.index.rag_id != unscoped.index.rag_id
    loaded = load_rag_index(store, scoped.index.rag_id)
    assert loaded.index.corpus_scope == scope
    assert loaded.transitions == scoped.transitions
    retriever = load_fit_rag_retriever(store, artifact_input(loaded.manifest))
    assert retriever.index.corpus_scope == scope


def test_lineage_scoped_index_covers_exactly_its_lineages(tmp_path: Path) -> None:
    """A lineage scope admits every trace of its lineages and nothing else."""
    store = _store(tmp_path, "lineage-scope")
    scoped, traces = _build(store, _scope("lineage", "lineage-1"))

    assert {item.trace_id for item in scoped.transitions} == {traces[1].trace_id}
    assert scoped.index.fit_lineage_ids == ("lineage-1",)


@pytest.mark.parametrize(("kind", "identity"), [("trace", "trace-3"), ("lineage", "lineage-3")])
def test_scope_admitting_held_out_evidence_is_refused(
    tmp_path: Path, kind: Literal["lineage", "trace"], identity: str
) -> None:
    """A scoped index can never hold a held-out lineage, however the scope names it."""
    store = _store(tmp_path, f"held-out-{kind}")
    with pytest.raises(ValueError, match="held-out"):
        _build(store, _scope(kind, "trace-0" if kind == "trace" else "lineage-0", identity))


def test_scope_naming_unknown_evidence_is_refused(tmp_path: Path) -> None:
    """A scope over traces the build does not bind fails instead of silently shrinking."""
    store = _store(tmp_path, "unknown")
    with pytest.raises(ValueError, match="outside the build's lineage bindings"):
        _build(store, _scope("trace", "trace-0", "trace-99"))


def test_scoped_index_must_be_fit_only(tmp_path: Path) -> None:
    """A scope cannot be combined with a serving index that includes held out."""
    store = _store(tmp_path, "serving")
    with pytest.raises(ValueError, match="fit-only"):
        _build(store, _scope("trace", "trace-0"), partitions=frozenset({"fit", "held_out"}))


def test_scoped_envelope_refuses_held_out_partitions_and_loader_refuses_tampering(
    tmp_path: Path,
) -> None:
    """The envelope and loader independently refuse evidence outside fit or the scope."""
    store = _store(tmp_path, "tamper")
    scoped, _ = _build(store, _scope("trace", "trace-0", "trace-1"))
    payload = scoped.index.model_dump(mode="json")

    with pytest.raises(ValidationError, match="only the fit partition"):
        RAGIndex.model_validate({**payload, "included_partitions": ["fit", "held_out"]})
    narrowed = scoped.index.model_copy(
        update={"rag_id": "trace-rag-narrowed", "corpus_scope": _scope("trace", "trace-0")}
    )
    store.write(
        artifact_id=narrowed.rag_id,
        artifact_type="trace-rag-index",
        envelope=narrowed,
        files={
            RAG_INDEX_PATH: canonical_json_bytes(narrowed),
            "transitions.jsonl": store.read_bytes(scoped.index.rag_id, "transitions.jsonl"),
            "vectors.jsonl": store.read_bytes(scoped.index.rag_id, "vectors.jsonl"),
        },
    )
    with pytest.raises(ArtifactCorruptionError, match="outside the index's corpus scope"):
        load_rag_index(store, narrowed.rag_id)
