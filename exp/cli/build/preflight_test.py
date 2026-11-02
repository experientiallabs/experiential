"""Tests for the build's credential-free cost quotes."""

from __future__ import annotations

from exp.cli.build.preflight import tagging_cost_ceiling
from exp.common.models import ModelCapabilities
from exp.simulation.mining_cost import TaggingWork


def _work(
    *,
    candidates: int = 10,
    questions: int = 4,
    batches: int = 1,
    view_tokens: int = 1_000,
    question_tokens: int = 40,
) -> TaggingWork:
    """Return one counted tagging workload.

    Args:
        candidates: Distinct cases to tag.
        questions: Questions the bank declares.
        batches: Requests each case takes.
        view_tokens: Rendered-view tokens over every case.
        question_tokens: Question-text tokens over one bank pass.

    Returns:
        The workload those counts describe.
    """
    return TaggingWork(
        bank_id="probe-v1",
        candidates=candidates,
        questions=questions,
        batches=batches,
        view_tokens=view_tokens,
        question_tokens=question_tokens,
    )


def test_the_quote_prices_every_batch_of_every_case_at_the_retry_allowance() -> None:
    """The ceiling is an allowance, so it prices retries and ignores the cache that may avoid them.

    The view is re-sent with every batch, because one request carries at most the surface's
    question limit and each of them has to describe the episode again. That is why the batch
    count multiplies the view tokens and not the question tokens.
    """
    work = _work(candidates=10, batches=1, view_tokens=1_000, question_tokens=40)
    capabilities = ModelCapabilities(input_cost_per_million_tokens_usd=3.0)

    assert work.input_tokens == 1_400
    assert tagging_cost_ceiling(work, capabilities) == 1_400 * 3 * 3.0 / 1_000_000


def test_a_second_batch_re_sends_the_view_and_not_the_whole_bank() -> None:
    """A bank wider than one request costs another copy of the episode, not another bank pass."""
    one = _work(batches=1, questions=32)
    two = _work(batches=2, questions=40)

    assert two.input_tokens - one.input_tokens == one.view_tokens
    assert two.calls == 2 * two.candidates


def test_an_unpriced_tagger_is_quoted_as_undefined_rather_than_as_free() -> None:
    """A model with no catalog price has an unknown cost, and zero is not a conservative guess."""
    assert tagging_cost_ceiling(_work(), ModelCapabilities()) is None
    assert (
        tagging_cost_ceiling(_work(), ModelCapabilities(input_cost_per_million_tokens_usd=0.0))
        == 0.0
    )
