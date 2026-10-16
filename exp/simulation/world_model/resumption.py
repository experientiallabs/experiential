"""The explicit authority to continue an interrupted session instead of starting a new one.

A session opened without it stops on any step or investigation a stopped process left open.
With it, the session continues from its first uncommitted step: saved inputs and outputs are
reused, and a request that may have reached the provider without a saved response (an unknown
outcome) is sent again only after the caller has recorded that it does so.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

RecoveryStage = Literal["prediction", "investigation", "judge", "retrieval"]
"""Where an unknown outcome was re-dispatched. ``retrieval`` is a step whose interrupted
preparation (query embeddings and memory-recall decisions) had to be prepared again."""


@dataclass(frozen=True)
class Resumption:
    """Continue an interrupted session in a new execution window.

    Attributes:
        window: Deadline of the resumed execution window. It extends, never shortens, the
            session's registered deadline; the session binding and every saved identity keep
            the registered deadline.
        retried: Durably records one unknown-outcome re-dispatch as ``(stage, operation,
            logical id)`` before the request is sent; recording the same logical id twice
            must be idempotent.
    """

    window: datetime
    retried: Callable[[RecoveryStage, str, str], None]

    def deadline(self, registered: datetime) -> datetime:
        """The effective deadline: the later of the registered one and this window."""
        return max(registered, self.window)

    def recorder(self, stage: RecoveryStage, operation: str) -> Callable[[str], None]:
        """Record re-dispatches of one operation's requests at one stage."""
        return lambda logical: self.retried(stage, operation, logical)
