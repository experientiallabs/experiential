"""Effect evidence survives refusal, rollback and independently fired events."""

from collections.abc import Callable

import pytest

from exp.runtime.environments.tests.world_conformance_test import (
    WorldCase,
    ticket_case,
    workbook_case,
)
from exp.runtime.environments.world_contracts import UndeclaredEffectError, WorldEvent
from exp.runtime.environments.worlds_test import _call, _runtime, _task


@pytest.mark.parametrize("make_case", [ticket_case, workbook_case])
def test_known_refusal_and_committed_effect_have_distinct_receipts(
    make_case: Callable[[], WorldCase],
) -> None:
    """The same evidence contract works with SQLite records and binary workbook artifacts."""
    case = make_case()
    with case.runtime.open(case.task) as world:
        with pytest.raises(UndeclaredEffectError):
            world.execute(case.outside_policy)
        receipt = world.last_execution
        assert receipt is not None and receipt.effects == "not_started"
        assert not receipt.dispatched and receipt.before_sha256 == receipt.after_sha256
        world.execute(case.first)
        receipt = world.last_execution
        assert receipt is not None and receipt.effects == "committed" and receipt.dispatched
        assert receipt.error_type is None and receipt.before_sha256 != receipt.after_sha256


def test_rollback_does_not_claim_that_scheduled_events_never_happened() -> None:
    """A failed tool rolls back to after the due event, not to the episode's earlier state."""
    event = WorldEvent(
        event_id="due", due=0, kind="action", action=_call("set", domain="a", value="event")
    )
    with _runtime(events=(event,)).open(_task()) as world:
        with pytest.raises(RuntimeError, match="crashed"):
            world.execute(_call("boom"))
        receipt = world.last_execution
        assert receipt is not None and receipt.effects == "rolled_back" and receipt.dispatched
        assert receipt.before_sha256 != receipt.after_sha256
        assert world.checkpoint().image.parts["a"] == b"event"
        assert len(world.fired) == 1
