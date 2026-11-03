"""Reviews retain invalid outputs and unknown dispatches without purchasing another draw."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.equivalence.review import ReviewUnavailable, review_call
from exp.optimize.evaluation.equivalence.runner_test import equivalence_fixture, proposal
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport


@pytest.mark.parametrize("invalid", [False, True])
def test_saved_raw_review_replays_after_deadline_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: bool,
) -> None:
    """A malformed saved result remains malformed; a valid saved result requires no new call."""
    project, spec, _, _ = equivalence_fixture(tmp_path, monkeypatch)
    transport = ScriptedJsonTransport(
        (proposal({"answer": True, **({"extra": 1} if invalid else {})}),)
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", lambda: transport
    )
    body: JsonObject = {
        "messages": [{"role": "user", "content": "Check supplied evidence."}],
        "response_schema": {
            "type": "object",
            "properties": {"answer": {"type": "boolean"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }

    def call(deadline: datetime) -> JsonObject:
        """Read or dispatch one frozen review slot under its single shared budget."""
        return review_call(
            project,
            spec.judge,
            body,
            identity="review",
            panel_id="panel",
            catalog_sha256=spec.cases[0].reference.catalog_sha256,
            maximum_attempts=1,
            maximum_call_attempts=1,
            deadline=deadline,
            sources=(spec.cases[0].reference.study.checkpoint,),
            code_revision="test",
            allow_dispatch=True,
            consumer="analyst",
        )

    if invalid:
        with pytest.raises(ValueError, match="response schema"):
            call(spec.deadline)
    else:
        assert call(spec.deadline) == {"answer": True}
    assert len(transport.requests) == 1
    project.model_catalog_path.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    past = datetime.now(UTC) - timedelta(days=1)
    if invalid:
        with pytest.raises(ValueError, match="response schema"):
            call(past)
    else:
        assert call(past) == {"answer": True}
    assert len(transport.requests) == 1


def test_uncertain_review_does_not_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumed intent with no model response stays explicitly unavailable on replay."""
    project, spec, _, _ = equivalence_fixture(tmp_path, monkeypatch)
    transport = ScriptedJsonTransport(
        (JsonHttpResponse(status_code=500, body={"error": "unavailable"}),)
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", lambda: transport
    )
    body: JsonObject = {
        "messages": [{"role": "user", "content": "Check."}],
        "response_schema": {"type": "object"},
    }
    for _ in range(2):
        with pytest.raises(ReviewUnavailable):
            review_call(
                project,
                spec.judge,
                body,
                identity="review",
                panel_id="panel",
                catalog_sha256=spec.cases[0].reference.catalog_sha256,
                maximum_attempts=1,
                maximum_call_attempts=1,
                deadline=spec.deadline,
                sources=(),
                code_revision="test",
                allow_dispatch=True,
                consumer="analyst",
            )
    assert len(transport.requests) == 1
    assert project.artifacts.exists("review-intent") and project.artifacts.exists("review-failure")


def test_retrying_judge_cannot_consume_the_next_analysts_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An early retry stops at its allocation, leaving the next planned reviewer a real call."""
    project, spec, _, _ = equivalence_fixture(tmp_path, monkeypatch)
    transport = ScriptedJsonTransport(
        (
            JsonHttpResponse(status_code=500, body={"error": "unavailable"}),
            proposal({"answer": True}),
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", lambda: transport
    )
    body: JsonObject = {
        "messages": [{"role": "user", "content": "Check."}],
        "response_schema": {"type": "object"},
    }

    def call(identity: str) -> JsonObject:
        """Use distinct immutable requests with one shared two-slot physical ledger."""
        return review_call(
            project,
            spec.judge,
            body,
            identity=identity,
            panel_id="panel",
            catalog_sha256=spec.cases[0].reference.catalog_sha256,
            maximum_attempts=2,
            maximum_call_attempts=1,
            deadline=spec.deadline,
            sources=(),
            code_revision="test",
            allow_dispatch=True,
            consumer="analyst",
        )

    with pytest.raises(ReviewUnavailable, match="per-review"):
        call("judge")
    assert len(transport.requests) == 1
    assert call("analyst") == {"answer": True}
    assert len(transport.requests) == 2
    with pytest.raises(ReviewUnavailable, match="per-review"):
        call("judge")
    assert len(transport.requests) == 2
