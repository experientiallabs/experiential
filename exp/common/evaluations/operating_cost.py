"""Price successful assistant usage independently of experiment budget reservations."""

import math

from exp.common.evaluations.dataset import EvaluationRow
from exp.common.evaluations.evidence import read_rollout
from exp.common.models import CandidateTokenPrice, NumericMeasurement, Usage
from exp.common.models.token_cost import schedule_usage_cost_nano_usd
from exp.common.project import ArtifactStore
from exp.common.rollouts import RolloutEventKind


def candidate_usage_cost(
    usage: Usage | None, price: CandidateTokenPrice
) -> NumericMeasurement | None:
    """Apply frozen catalog rates to recorded tokens without hypothetical retry charges.

    Args:
        usage: Successful rollout's cumulative assistant usage, excluding invalid attempts.
        price: Frozen ordinary, cached-read, cache-write, and output list prices.

    Returns:
        Catalog-priced cost, or unavailable when required usage or cache rates are absent.
        Undeclared cache tokens receive the ordinary input rate; no discount is invented.
    """
    if usage is None:
        return None
    if price.token_prices is not None:
        cost = schedule_usage_cost_nano_usd(price.token_prices, usage)
        return (
            None
            if cost is None
            else NumericMeasurement(value=cost / 1_000_000_000, provenance="estimated")
        )
    cached = usage.cached_input_tokens or 0
    written = usage.cache_write_input_tokens or 0
    if cached + written > usage.input_tokens:
        return None
    if (cached and price.cached_input_usd_per_million_tokens is None) or (
        written and price.cache_write_usd_per_million_tokens is None
    ):
        return None
    cost = (
        (usage.input_tokens - cached - written) * price.input_usd_per_million_tokens
        + cached * (price.cached_input_usd_per_million_tokens or 0)
        + written * (price.cache_write_usd_per_million_tokens or 0)
        + usage.output_tokens * price.output_usd_per_million_tokens
    ) / 1_000_000
    return NumericMeasurement(value=cost, provenance="estimated")


def operating_row(
    store: ArtifactStore, row: EvaluationRow, price: CandidateTokenPrice
) -> EvaluationRow:
    """Create a reporting view without mutating the experiment's conservative spend ledger."""
    if row.status != "completed" or row.rollout_id is None:
        return row
    rollout, _ = read_rollout(store, row.rollout_id)
    latency = row.candidate_latency_seconds
    spans = [span for span in rollout.spans if span.kind == RolloutEventKind.AGENT_MODEL_CALL]
    if price.token_prices is None:
        cost = candidate_usage_cost(rollout.candidate_economics.usage, price)
    else:
        calls = [candidate_usage_cost(span.usage, price) for span in spans]
        cumulative = rollout.candidate_economics.usage
        covers_usage = cumulative is not None and (
            sum(span.usage.input_tokens for span in spans if span.usage is not None)
            == cumulative.input_tokens
            and sum(span.usage.output_tokens for span in spans if span.usage is not None)
            == cumulative.output_tokens
        )
        # Long-context rates apply to each request, not to the cumulative episode.
        # A missing per-call meter leaves cost unknown rather than pricing a partial total.
        cost = (
            NumericMeasurement(
                value=math.fsum(item.value for item in calls if item is not None),
                provenance="estimated",
            )
            if covers_usage and calls and all(item is not None for item in calls)
            else None
        )
    if latency is None:
        if spans:
            latency = NumericMeasurement(
                value=math.fsum(
                    (span.ended_at - span.started_at).total_seconds() for span in spans
                ),
                provenance="observed",
            )
    return row.model_copy(
        update={
            "candidate_cost_usd": cost,
            "candidate_latency_seconds": latency,
        }
    )
