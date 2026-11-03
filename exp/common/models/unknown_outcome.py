"""Re-dispatch a logical request whose earlier dispatch has an unknown outcome, transparently.

A request ledger reserves each logical request before it is sent and saves the response after
it returns. A reservation without a saved response means the request may have reached the
provider: it may have been answered, billed and logged, while nothing was saved here. Such a
request is never replayed and never silently bought again. Only a caller that is explicitly
resuming an interrupted run may send it again, under a new logical identity, and only after it
has recorded that it does so: the new answer is transparent but not provably uninfluenced,
because the lost one may have been seen.
"""

from collections.abc import Callable

MAXIMUM_REDISPATCHES = 3


class DispatchOutcomeUnknown(ValueError):
    """An earlier dispatch of this logical request saved no response; its outcome is unknown."""


def redispatch_identity(logical_id: str, retry: int) -> str:
    """The logical identity of one re-dispatch; retry 0 is the original request.

    Args:
        logical_id: Original logical request identity.
        retry: Re-dispatch number, from 0.

    Returns:
        ``logical_id`` itself for retry 0, else ``<logical_id>-retry-<retry>``.
    """
    return logical_id if retry == 0 else f"{logical_id}-retry-{retry}"


def resume_unknown[ResultT](
    call: Callable[[str], ResultT],
    logical_id: str,
    retried: Callable[[str], None] | None,
) -> ResultT:
    """Send or replay one logical request, re-dispatching after unknown outcomes when resuming.

    Each attempt replays its own saved response when there is one, so a re-dispatch that
    already returned is never bought twice.

    Args:
        call: Sends or replays the request under the given logical identity.
        logical_id: Original logical request identity.
        retried: Records one re-dispatch, by its new logical identity, before it is sent;
            None outside an explicit resumption, where an unknown outcome stays an error.

    Returns:
        The saved or new response.

    Raises:
        DispatchOutcomeUnknown: The outcome is unknown and this is not a resumption, or
            every allowed re-dispatch also ended with an unknown outcome.
    """
    retry = 0
    while True:
        try:
            return call(redispatch_identity(logical_id, retry))
        except DispatchOutcomeUnknown:
            if retried is None or retry >= MAXIMUM_REDISPATCHES:
                raise
            retry += 1
            retried(redispatch_identity(logical_id, retry))
