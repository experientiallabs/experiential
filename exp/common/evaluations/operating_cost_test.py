"""Market usage pricing never inherits retry reservations or provider promotions."""

import pytest

from exp.common.evaluations.operating_cost import candidate_usage_cost
from exp.common.models import CandidateTokenPrice, Usage


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
