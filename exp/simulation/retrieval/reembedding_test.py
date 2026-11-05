"""Offline native-index coverage for cold re-embedding inventories."""

from pathlib import Path

import pytest

from exp.common.project import artifact_input
from exp.simulation.retrieval import load_rag_index, persist_trace_rag
from exp.simulation.retrieval.embedding_inputs import embedding_batches, plan_rag_embedding_inputs
from exp.simulation.retrieval.reembedding import plan_rag_reembedding
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.retrieval.scope_test import _bindings
from exp.simulation.retrieval.tests.retrieval_test import (
    _CREATED_AT,
    _persist_traces,
    _store,
)


def test_native_root_and_scoped_fit_membership_are_preserved(tmp_path: Path) -> None:
    """Shared new vectors must not pool scope membership or admit held-out evidence."""
    store = _store(tmp_path, "reembedding")
    source, traces = _persist_traces(store, count=4)
    bindings = _bindings(traces)
    root = persist_trace_rag(
        store,
        (source,),
        bindings,
        created_at=_CREATED_AT,
        code_revision="fixture",
        included_partitions=frozenset({"fit", "held_out"}),
    )
    fit = persist_trace_rag(
        store,
        (source,),
        bindings,
        created_at=_CREATED_AT,
        code_revision="fixture",
    )
    fit_trace = next(row.trace_id for row in bindings if row.partition == "fit")
    scope = CorpusScope.declare(
        name="same-fit-scope", kind="trace", ids=(fit_trace,), source_plan="c" * 64
    )
    scoped = persist_trace_rag(
        store,
        (source,),
        bindings,
        created_at=_CREATED_AT,
        code_revision="fixture",
        corpus_scope=scope,
    )
    loaded = tuple(load_rag_index(store, row.index.rag_id) for row in (root, fit, scoped))
    before = {row.index.rag_id: store.read(row.index.rag_id).manifest for row in loaded}
    plan = plan_rag_reembedding(loaded, maximum_batch_inputs=2, attempts_per_batch=3)
    expected = plan_rag_embedding_inputs(tuple(row.key_text for row in loaded[0].transitions))
    assert plan.unique_inputs == len(expected.texts)
    assert plan.batches == len(tuple(embedding_batches(expected.texts, maximum_inputs=2)))
    assert plan.physical_attempt_bound == plan.batches * 3
    assert plan.admission == "not_admitted"
    assert {row.source for row in plan.indexes} == {artifact_input(row.manifest) for row in loaded}
    assert sum(row.included_partitions == ("fit",) for row in plan.indexes) == 2
    assert {row.index.rag_id: store.read(row.index.rag_id).manifest for row in loaded} == before
    assert (
        plan_rag_reembedding(tuple(reversed(loaded)), maximum_batch_inputs=2, attempts_per_batch=3)
        == plan
    )


def test_complete_old_vectors_do_not_reduce_cold_target_inventory(tmp_path: Path) -> None:
    """Existing completed source vectors are evidence, never target-model cache hits."""
    store = _store(tmp_path, "cold")
    source, traces = _persist_traces(store, count=2)
    built = persist_trace_rag(
        store,
        (source,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="fixture",
        included_partitions=frozenset({"fit", "held_out"}),
    )
    loaded = load_rag_index(store, built.index.rag_id)
    plan = plan_rag_reembedding((loaded,))
    assert loaded.matrix.size > 0
    assert plan.unique_inputs > 0 and plan.physical_attempt_bound > 0


def test_no_full_root_is_refused(tmp_path: Path) -> None:
    """A fit-only artifact cannot silently stand in for the original whole serving graph."""
    store = _store(tmp_path, "missing-root")
    source, traces = _persist_traces(store, count=2)
    built = persist_trace_rag(
        store,
        (source,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="fixture",
    )
    with pytest.raises(ValueError, match="full original serving root"):
        plan_rag_reembedding((load_rag_index(store, built.index.rag_id),))


@pytest.mark.parametrize("attempts", [0, -1, True])
def test_invalid_physical_allowance_refuses_before_planning(
    attempts: int,
) -> None:
    """An invalid allowance cannot be mistaken for a zero-cost rebuild."""
    with pytest.raises(ValueError, match="positive physical attempt"):
        plan_rag_reembedding((), attempts_per_batch=attempts)


@pytest.mark.parametrize("change", ["dimension", "top_k"])
def test_native_mixed_representation_graph_is_refused(
    tmp_path: Path,
    change: str,
) -> None:
    """Individually valid indexes cannot disguise changed vector geometry or retrieval limits."""
    from exp.simulation.retrieval import RAGEmbedderBinding
    from exp.simulation.retrieval.tests.retrieval_test import _ConstantEmbedder

    store = _store(tmp_path, "mixed-representation")
    source, traces = _persist_traces(store, count=4)
    root = persist_trace_rag(
        store,
        (source,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="fixture",
        included_partitions=frozenset({"fit", "held_out"}),
    )
    # The constant vectors are a deliberate negative fixture, not Qwen output/readiness.
    binding = RAGEmbedderBinding(client=_ConstantEmbedder(), snapshot=root.index.embedder)
    child_store = store
    if change == "dimension":
        child_store = _store(tmp_path / "independent-cold-cache", "mixed-representation")
        other_source, _ = _persist_traces(child_store, count=4)
        assert other_source == source
    child = persist_trace_rag(
        child_store,
        (source,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="fixture",
        embedder=binding if change == "dimension" else None,
        default_top_k=root.index.default_top_k + (change == "top_k"),
    )
    original = load_rag_index(store, root.index.rag_id)
    changed = load_rag_index(child_store, child.index.rag_id)
    with pytest.raises(ValueError, match="original serving representation"):
        plan_rag_reembedding((original, changed))
