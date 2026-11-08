"""Declared lifecycles of the environment-family compatibility review, draw and group.

A transport-level failure is never a review verdict. Two small machines say so explicitly:

- :data:`DRAW_MACHINE`, per reviewer draw identity: ``pending`` moves to ``answered`` when a
  completed model response is saved, or to ``transient_failed`` when the draw failed at the
  transport level. A ``transient_failed`` draw is ``retried`` back to ``pending`` on the next run
  under a fresh attempt identity; its failure is kept as a receipt, never as an answer. A draw
  whose model exhausted its output allowance is ``refused``: a failed draw of that one review,
  replayed as failed from its saved rejection, so its group goes on with its other variants.
- :data:`GROUP_MACHINE`, per reviewed cluster: ``pending`` moves to ``reviewed`` when every draw
  of its review completed (whatever the verdict, a genuine ``uncertain`` included), or to
  ``transient_failed`` when one of them failed transiently, or to ``unresolved`` when a draw
  spent the review's attempt allowance (:data:`ALLOWANCE_EXHAUSTED`) or the reviewer failed
  unrecoverably. Only ``reviewed`` groups are saved; a ``transient_failed`` group is ``retried``
  back to ``pending`` on the next run and holds the record back (:class:`FamiliesPending`). An
  ``unresolved`` group is reported with its reason in the record while every other cluster goes
  on; it is not saved either, so a later review draws it again.

:func:`is_transient` names the transport-level failures: what the shared retry classifier calls
retryable (transport errors, retryable HTTP statuses such as 409 and 503, timeouts, OS errors),
and an attempt budget exhausted on them.
"""

from __future__ import annotations

from enum import StrEnum

from exp.common.state_machine import StateMachine
from exp.runtime.models.providers.transport import classify_retry
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError


class DrawState(StrEnum):
    """States of one reviewer draw identity."""

    PENDING = "pending"
    ANSWERED = "answered"
    TRANSIENT_FAILED = "transient_failed"
    REFUSED = "refused"


DRAW_MACHINE = StateMachine(
    name="environment-review-draw",
    states=DrawState,
    initial=frozenset({DrawState.PENDING}),
    transitions=(
        (DrawState.PENDING, "answered", DrawState.ANSWERED),
        (DrawState.PENDING, "failed_transiently", DrawState.TRANSIENT_FAILED),
        (DrawState.TRANSIENT_FAILED, "retried", DrawState.PENDING),
        (DrawState.PENDING, "refused", DrawState.REFUSED),
    ),
)
"""One reviewer draw: answered, refused (output allowance exhausted), or failed transiently and
retried on the next run."""


ALLOWANCE_EXHAUSTED = "attempt_budget_exhausted"
"""The :func:`is_transient` reason of a draw that spent its whole attempt allowance."""


class GroupState(StrEnum):
    """States of one reviewed cluster."""

    PENDING = "pending"
    REVIEWED = "reviewed"
    TRANSIENT_FAILED = "transient_failed"
    UNRESOLVED = "unresolved"


GROUP_MACHINE = StateMachine(
    name="environment-review-group",
    states=GroupState,
    initial=frozenset({GroupState.PENDING}),
    transitions=(
        (GroupState.PENDING, "reviewed", GroupState.REVIEWED),
        (GroupState.PENDING, "failed_transiently", GroupState.TRANSIENT_FAILED),
        (GroupState.TRANSIENT_FAILED, "retried", GroupState.PENDING),
        (GroupState.PENDING, "unresolved", GroupState.UNRESOLVED),
    ),
)
"""One cluster's review: reviewed with its verdict, failed transiently and retried, or left
unresolved (allowance spent or reviewer failed unrecoverably) while the other clusters go on."""


class TransientReviewFailure(RuntimeError):
    """A review draw failed at the transport level; it is retried, never saved as a verdict.

    Attributes:
        identity: The draw identity that failed.
        reason: The retry classification of the failure.
    """

    def __init__(self, identity: str, reason: str) -> None:
        """Name the failed draw and why."""
        super().__init__(
            f"review draw {identity[:24]} failed transiently ({reason}); retried next run"
        )
        self.identity = identity
        self.reason = reason


class FamiliesPending(RuntimeError):
    """Some groups failed transiently; nothing is saved for them and the record is not written.

    Attributes:
        pending: Why each pending cluster failed, by cluster.
    """

    def __init__(self, pending: dict[str, str]) -> None:
        """Name every pending cluster."""
        super().__init__(
            f"{len(pending)} environment groups failed transiently and stay pending: "
            + "; ".join(f"{cluster}: {reason}" for cluster, reason in sorted(pending.items()))
        )
        self.pending = dict(pending)


def is_transient(error: BaseException) -> str | None:
    """Return the transport-level reason of ``error`` or of any cause, or None.

    Args:
        error: The draw's failure.

    Returns:
        The retry classification (``transport``, ``http_503``, ``timeout`` and so on), or
        ``attempt_budget_exhausted``, or None for a failure that is not transport-level.
    """
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, DispatchCapExhaustedError):
            return ALLOWANCE_EXHAUSTED
        if isinstance(current, Exception):
            classification = classify_retry(current)
            if classification.retryable:
                return classification.reason
        current = current.__cause__
    return None
