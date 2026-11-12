"""Tests for the front-door slot executors' author draw."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from exp.optimize.evaluation.variance import frontdoor_executors
from exp.runtime.models.providers.errors import ProviderRetryableResponseError
from exp.simulation.mining.environment_review import ReviewerError


class _Settings:
    output_tokens = 16000


def _draw_raising(monkeypatch: pytest.MonkeyPatch, error: Exception) -> Callable[..., object]:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(frontdoor_executors, "author_response", refuse)
    return frontdoor_executors.author_draw(
        object(),  # type: ignore[arg-type]
        _Settings(),  # type: ignore[arg-type]
        "revision",
    )


def test_spent_output_allowance_fails_only_that_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saved length-stop rejection becomes a draw failure the authoring loop records."""
    stop = ProviderRetryableResponseError("spent", hit_length_limit=True)
    draw = _draw_raising(monkeypatch, stop)
    with pytest.raises(ReviewerError, match="16000-token output allowance") as raised:
        draw(object(), identity="tag-author-draw-x")
    assert raised.value.__cause__ is stop


def test_other_unusable_responses_still_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the known output-allowance stop is softened; other malformed output is not."""
    malformed = ProviderRetryableResponseError("malformed")
    draw = _draw_raising(monkeypatch, malformed)
    with pytest.raises(ProviderRetryableResponseError) as raised:
        draw(object(), identity="tag-author-draw-x")
    assert raised.value is malformed
