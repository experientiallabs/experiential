"""Market usage pricing never inherits retry reservations or provider promotions."""

from datetime import timedelta
from pathlib import Path

import pytest

from exp.common.evaluations.build_test import (
    _persist_rollout,
    _production_rollout,
    _snapshot,
    _store,
    _task,
)
from exp.common.evaluations.model_report_test import _row
from exp.common.evaluations.operating_cost import candidate_usage_cost, operating_row
from exp.common.models import CandidateTokenPrice, Usage
from exp.common.models.catalog_prices import GatewayLongContextTier
from exp.common.models.model import OperationEconomics
from exp.common.models.token_cost_test import prices


def test_actual_tokens_price_without_retry_budget_and_with_known_cache_splits() -> None:
    """Successful tokens cost the same irrespective of the rollout's authorized retry budget."""
    price = CandidateTokenPrice(
        candidate_alias="worker",
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=2,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=1.5,
    )
    ordinary = candidate_usage_cost(Usage(input_tokens=8, output_tokens=4), price)
    assert ordinary is not None and ordinary.value == pytest.approx(0.000016)
    split = candidate_usage_cost(
        Usage(input_tokens=8, output_tokens=4, cached_input_tokens=2, cache_write_input_tokens=2),
        price,
    )
    assert split is not None and split.value == pytest.approx(0.000016)
    assert candidate_usage_cost(None, price) is None
    assert (
        candidate_usage_cost(
            Usage(input_tokens=8, output_tokens=4, cached_input_tokens=2),
            price.model_copy(update={"cached_input_usd_per_million_tokens": None}),
        )
        is None
    )


def test_scheduled_report_prices_each_saved_call_and_excludes_world(tmp_path: Path) -> None:
    """Two short calls cannot become one long-context request, or inherit world spend."""
    store = _store(tmp_path)
    row = _row("task", 0, 1, 999)
    rollout = _production_rollout(
        row.rollout_id or "missing",
        cell_id=row.cell_id,
        task=_task("task", partition="fit"),
        candidate=_snapshot("worker"),
        world_cost=999,
    )
    first = rollout.spans[0].model_copy(
        update={
            "usage": Usage(
                input_tokens=60,
                output_tokens=10,
                cached_input_tokens=0,
                cache_write_input_tokens=0,
                reasoning_tokens=0,
            )
        }
    )
    second = first.model_copy(
        update={
            "span_id": "second",
            "started_at": first.ended_at,
            "ended_at": first.ended_at + timedelta(seconds=2),
        }
    )
    rollout = rollout.model_copy(
        update={
            "spans": (first, second),
            "candidate_economics": OperationEconomics(
                usage=Usage(input_tokens=120, output_tokens=20)
            ),
        }
    )
    _persist_rollout(store, rollout)
    before = store.read_bytes(rollout.rollout_id, "rollout.json")
    card = prices().model_copy(
        update={
            "long_context": GatewayLongContextTier(
                input_threshold_tokens=100,
                input_nano_usd_per_million_tokens=10_000_000_000,
                output_nano_usd_per_million_tokens=40_000_000_000,
            )
        }
    )
    price = CandidateTokenPrice(
        candidate_alias="worker",
        token_prices=card,
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.1,
        cache_write_usd_per_million_tokens=2,
    )
    result = operating_row(store, row, price)
    assert result.candidate_cost_usd is not None
    assert result.candidate_cost_usd.value == pytest.approx(0.0002)
    assert result.candidate_latency_seconds is not None
    assert result.candidate_latency_seconds.value == 3
    assert store.read_bytes(rollout.rollout_id, "rollout.json") == before
    assert (
        operating_row(
            store, row.model_copy(update={"status": "incomplete"}), price
        ).candidate_cost_usd
        == row.candidate_cost_usd
    )
