"""The explicit authority to continue an interrupted session instead of starting a new one.

A session opened without it stops on any step or investigation a stopped process left open.
With it, the session continues from its first uncommitted step: saved inputs and outputs are
reused, and a request that may have reached the provider without a saved response (an unknown
outcome) is sent again only after the caller has recorded that it does so.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from exp.common.core.artifacts import JsonObject
from exp.common.models.client import RequestAdmissionRefused
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown, resume_unknown
from exp.common.project import ArtifactStore

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


def dispatch_reader(
    artifacts: ArtifactStore,
    save: Callable[[str, JsonObject], None],
    identity: str,
    operation: str,
    logical: str,
    call: Callable[[], JsonObject],
    resumption: Resumption | None,
) -> JsonObject:
    """Replay saved reader judgments and qualify explicitly resumed unknown outcomes.

    Args:
        artifacts: Immutable investigation artifacts.
        save: Investigation owner's artifact writer, retaining exact JSON records.
        identity: Owning investigation identity.
        operation: Original tool operation used for retry reporting.
        logical: Digest binding the reader request, owner and round.
        call: Already admitted reader dispatch.
        resumption: Explicit retry authority, absent for fresh executions.

    Returns:
        Exact saved or newly returned decision. Saved responses are never redrawn.
    """
    base = identity + "-reader-" + logical[:16]

    def invoke(attempt: str) -> JsonObject:
        """Claim one reader intent before dispatch and retain its exact response."""
        original = attempt
        admission = 0
        while artifacts.exists(attempt + "-not-dispatched"):
            admission += 1
            attempt = original + f"-admission-{admission}"
        result_id = attempt + "-response"
        if artifacts.exists(result_id):
            return json.loads(artifacts.read_bytes(result_id, "record.json"))["response"]
        if artifacts.exists(attempt):
            raise DispatchOutcomeUnknown("reader intent has no saved response")
        save(attempt, {"request_sha256": logical})
        try:
            response = call()
        except RequestAdmissionRefused:
            save(attempt + "-not-dispatched", {"dispatch": "not_started"})
            raise
        save(result_id, {"response": response})
        return response

    retried = None if resumption is None else resumption.recorder("investigation", operation)
    return resume_unknown(invoke, base, retried)
