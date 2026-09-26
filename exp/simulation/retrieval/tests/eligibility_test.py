"""Excluded grounding corpora never require a paid query embedding."""

from pathlib import Path
from unittest.mock import patch

import pytest

from exp.common.models import EmbeddingCostReservation
from exp.simulation.retrieval import (
    RAGAction,
    RAGQuery,
    TraceRAGRetriever,
    load_rag_index,
    persist_trace_rag,
)
from exp.simulation.retrieval.retriever import RAGQueryInputLimitError
from exp.simulation.retrieval.tests.retrieval_test import (
    _CREATED_AT,
    _bindings,
    _constant_binding,
    _persist_traces,
    _store,
)


def test_excluded_corpus_skips_query_embedding_and_its_input_ceiling(tmp_path: Path) -> None:
    """A long excluded query costs zero; an eligible query still enforces its reservation."""
    store = _store(tmp_path, "excluded-queries")
    source, traces = _persist_traces(store, count=1)
    embedder = _constant_binding()
    loaded = persist_trace_rag(
        store,
        (source,),
        _bindings(traces),
        embedder=embedder,
        created_at=_CREATED_AT,
        code_revision="test",
    )
    retriever = TraceRAGRetriever(load_rag_index(store, loaded.index.rag_id), embedder=embedder)
    query = RAGQuery(
        task="long task " * 10_000,
        action=RAGAction(kind="message", content="next action"),
        excluded_lineage_ids=loaded.index.included_lineage_ids,
    )
    reservation = EmbeddingCostReservation(
        model=embedder.snapshot,
        maximum_attempts=embedder.maximum_attempts,
        input_usd_per_million_tokens=embedder.input_usd_per_million_tokens,
        maximum_input_tokens=10,
    )
    with patch("exp.simulation.retrieval.retriever.embed_rag_texts") as embedding:
        cost = retriever.estimate_query_economics(query, reservation)
        assert cost.cost_usd is not None and cost.cost_usd.value == 0
        assert retriever.retrieve(query) == ()
        embedding.assert_not_called()
    with pytest.raises(RAGQueryInputLimitError):
        retriever.estimate_query_economics(
            query.model_copy(update={"excluded_lineage_ids": ()}), reservation
        )
    with pytest.raises(ValueError, match="retry bound"):
        retriever.estimate_query_economics(
            query, reservation.model_copy(update={"maximum_attempts": 50})
        )
