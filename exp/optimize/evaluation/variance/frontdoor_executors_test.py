"""Tests for the front-door slot executors' author draw."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, ModelSnapshot
from exp.common.project.paths import ProjectPaths
from exp.optimize.evaluation.variance import frontdoor_executors
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.gateway.reservation_tokenizer import count_reservation_tokens
from exp.runtime.models.providers.errors import ProviderRetryableResponseError
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    ScriptedJsonTransport,
)
from exp.simulation.mining.environment_review import ReviewerError
from exp.simulation.mining.questions import Question, QuestionBank
from exp.simulation.mining.tagging import DecisionCache
from exp.simulation.mining.views import Unit


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


def _bank() -> QuestionBank:
    return QuestionBank(
        bank_id="fixture-v1",
        version=1,
        questions=(Question(question_id="sit.q0", family="situation", text="Did it?"),),
    )


def _snapshot() -> ModelSnapshot:
    return ModelSnapshot(
        provider="fixture",
        model_id="decisions-fixture",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def test_a_view_that_fits_is_sent_unchanged() -> None:
    """Fitting never touches a state under the ceiling, so its saved answers keep replaying."""
    unit = Unit("trace-1", "observations", "short episode " * 50)
    assert frontdoor_executors.fit_decision_unit(unit, maximum_tokens=1_000) is unit


def test_an_overlong_view_is_sent_as_its_head_and_tail(tmp_path: Path) -> None:
    """A state over the ceiling keeps its start and end, fits, keeps its row, and replays."""
    text = "HEAD " + " ".join(f"w{index}" for index in range(5_000)) + " TAIL"
    unit = Unit("trace-1", "observations", text)
    sent: list[DecisionRequest] = []

    def decide(request: DecisionRequest) -> JsonObject:
        sent.append(request)
        return {"answers": {"sit.q0": {"type": "noul", "noul": 0.7}}, "usage": {}}

    class _Small(frontdoor_executors.FittedDecisionsTagger):
        maximum_state_tokens = 500

    cache = DecisionCache(ProjectPaths(root=tmp_path / ".exp", project_id="p"))
    tagger = _Small(decide, _snapshot(), cache=cache)
    matrix = tagger.tag([unit], _bank())
    tagger.tag([unit], _bank())

    assert len(sent) == 1
    state = sent[0].state
    assert isinstance(state, str)
    assert state.startswith("HEAD ") and state.endswith(" TAIL") and "elided" in state
    assert count_reservation_tokens(state) <= 500
    assert matrix.rows[0].unit_hash == unit.unit_hash
    assert matrix.rows[0].values == (0.7,)


def test_a_refused_decision_names_the_providers_reason() -> None:
    """A bare 400 cannot be diagnosed, so the provider's stated reason reaches the message."""
    refusal = {"error": {"type": "invalid_request_error", "message": "too long", "x": "y"}}
    transport = frontdoor_executors.ReasonedDecisionTransport(
        ScriptedJsonTransport([JsonHttpResponse(400, refusal)])
    )
    with pytest.raises(
        ProviderTransportError, match="HTTP 400: type=invalid_request_error; message=too long"
    ) as raised:
        transport.post("https://example.test", headers={}, payload={}, timeout_seconds=1)
    assert raised.value.status_code == 400


def test_an_answered_decision_passes_through() -> None:
    """A 200 object response is returned unchanged."""
    answered = JsonHttpResponse(200, {"answers": {}})
    transport = frontdoor_executors.ReasonedDecisionTransport(ScriptedJsonTransport([answered]))
    response = transport.post("https://example.test", headers={}, payload={}, timeout_seconds=1)
    assert response.body == {"answers": {}}


def test_an_unknown_author_outcome_fails_only_that_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A draw left mid-dispatch by a stopped run fails as a ReviewerError, not the slot."""
    import pytest

    from exp.optimize.evaluation.variance import frontdoor_executors as module

    def refuse(*args: object, **kwargs: object) -> NoReturn:
        raise ValueError("author dispatch has an unknown outcome; no implicit redraw")

    monkeypatch.setattr(module, "author_response", refuse)
    draw = module.author_draw(object(), object(), "rev")
    with pytest.raises(module.ReviewerError):
        draw(object(), identity="x")


def test_other_author_value_errors_still_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the unknown-outcome refusal is narrowed to one draw."""
    import pytest

    from exp.optimize.evaluation.variance import frontdoor_executors as module

    def refuse(*args: object, **kwargs: object) -> NoReturn:
        raise ValueError("author deadline expired before dispatch")

    monkeypatch.setattr(module, "author_response", refuse)
    draw = module.author_draw(object(), object(), "rev")
    with pytest.raises(ValueError, match="deadline"):
        draw(object(), identity="x")
