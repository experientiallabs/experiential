"""Unknown-outcome re-dispatch: only when resuming, recorded first, bounded, replay-first."""

import pytest

from exp.common.models.unknown_outcome import (
    MAXIMUM_REDISPATCHES,
    DispatchOutcomeUnknown,
    resume_unknown,
)


def test_an_unknown_outcome_is_an_error_outside_a_resumption() -> None:
    """Without a recorder the lost request is never sent again."""
    sent: list[str] = []

    def call(identity: str) -> str:
        """The original request's outcome is unknown."""
        sent.append(identity)
        raise DispatchOutcomeUnknown("lost")

    with pytest.raises(DispatchOutcomeUnknown):
        resume_unknown(call, "step", None)
    assert sent == ["step"]


def test_a_resumption_records_each_redispatch_before_sending_it() -> None:
    """The record precedes the dispatch, so a crash in between still leaves it flagged."""
    events: list[str] = []

    def call(identity: str) -> str:
        """The original and first re-dispatch were lost; the second answers."""
        events.append("send " + identity)
        if identity != "step-retry-2":
            raise DispatchOutcomeUnknown("lost")
        return "answer"

    assert resume_unknown(call, "step", lambda identity: events.append("record " + identity)) == (
        "answer"
    )
    assert events == [
        "send step",
        "record step-retry-1",
        "send step-retry-1",
        "record step-retry-2",
        "send step-retry-2",
    ]


def test_redispatches_are_bounded() -> None:
    """A request that keeps losing its outcome stops after the allowed re-dispatches."""
    recorded: list[str] = []

    def call(identity: str) -> str:
        """Every attempt is lost."""
        raise DispatchOutcomeUnknown(identity)

    with pytest.raises(DispatchOutcomeUnknown):
        resume_unknown(call, "step", recorded.append)
    assert len(recorded) == MAXIMUM_REDISPATCHES
