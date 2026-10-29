"""Tests for exact retrieval pricing, including turns without visible actions."""

from unittest.mock import MagicMock

import pytest

from exp.common.models import (
    BillingSource,
    EmbeddingCostReservation,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
)
from exp.simulation.engines.text.grounding import estimate_retrieval_economics
from exp.simulation.retrieval import RAGAction, RAGQuery, TraceRAGRetriever


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("known_cost", [False, True])
def test_retrieval_cost_counts_queries_without_masking_unknown_spend(
    count: int, known_cost: bool
) -> None:
    """Only an empty query batch has a zero bound; missing paid-query cost stays unknown."""
    reservation = EmbeddingCostReservation(
        model=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="test",
            model_id="embedder",
            capabilities_sha256="a" * 64,
            connection_sha256="b" * 64,
        ),
        input_usd_per_million_tokens=1,
        maximum_attempts=2,
        maximum_input_tokens=1000,
    )
    queries = tuple(
        RAGQuery(
            task="Answer the question.",
            initial_context={},
            action=RAGAction(kind="message", content=f"Answer {index}"),
            excluded_lineage_ids=(),
            top_k=1,
        )
        for index in range(count)
    )
    retriever = MagicMock(spec=TraceRAGRetriever)
    retriever.estimate_query_economics.return_value = OperationEconomics(
        cost_usd=NumericMeasurement(value=0.1, provenance="estimated") if known_cost else None
    )

    economics = estimate_retrieval_economics(queries, retriever, reservation)

    assert retriever.estimate_query_economics.call_count == count
    if count == 0:
        assert economics.cost_usd == NumericMeasurement(value=0, provenance="estimated")
    elif known_cost:
        assert economics.cost_usd == NumericMeasurement(value=count * 0.1, provenance="estimated")
    else:
        assert economics.cost_usd is None
