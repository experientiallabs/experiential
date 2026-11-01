"""The claim a run writes before it buys a decision, and how a later reader judges it.

A decision costs money, so only one run may be buying any one of them at a time. The claim record
is how that is arranged without holding a lock across the call: a run writes its claim under the
project lock, releases the lock, buys the decision, then saves the answer and drops the claim
together. A reader that finds somebody else's claim therefore has one question to answer, and
answering it honestly is the whole of this module. Is that claim a call still being made, or the
mark an interrupted call left behind?

The two have opposite right answers. A call in flight will commit an answer in a moment, so the
thing to do is wait for it and replay what it saved. An interrupted call has an unknown outcome
and may already have been billed, so the thing to do is refuse to buy it again and name it, which
is what :meth:`~exp.simulation.mining.tagging.DecisionCache.discard_intent` releases once a human
has established the call was never billed. Reading the first as the second tells a user to discard
a claim another run is about to commit; reading the second as the first waits for an answer that
is never coming.

A claim is therefore explicit about which it is:

- **In flight.** The run that wrote it was buying the decision when it last said anything, so a
  reader waits for it to resolve.
- **Abandoned.** Its call committed no answer, because it raised, because the record is not one
  this module can account for, or because the claim is older than any single call can legitimately
  take. A reader refuses it by name.

A failing call marks its own claim, so the common failure is explicit immediately. A run killed
outright marks nothing, which is what :data:`CLAIM_LEASE_SECONDS` is for: a claim nobody resolved
within it is abandoned by age alone. That case costs a reader waiting rather than money, and the
waiting is bounded by :data:`CLAIM_WAIT_SECONDS`.

Nothing here touches the records. This module encodes a claim, judges one that was read, and
writes the sentences a user sees; the cache owns the lock, the records, and the waiting.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, replace
from enum import Enum

from exp.common.core.artifacts import canonical_json_bytes

CLAIM_RECORD_VERSION = 1
"""Shape of the claim record.

A record of any other shape, including one written before claims carried an owner and a time, is
not a claim this module can account for, so it reads as abandoned. That is the safe direction and
it needs no migration: an unaccountable claim is exactly the interrupted call a run must refuse by
name rather than buy again.
"""

CLAIM_LEASE_SECONDS = 1_200.0
"""How long a claim may be in flight before its age alone makes it abandoned.

Twenty minutes is deliberately longer than any one decision can legitimately take: a call is
bounded by the per-attempt ceiling in :mod:`exp.simulation.mining.decisions` and the three
attempts the shared retry policy allows, which is a quarter of an hour of timeouts in the worst
case. A shorter lease would eventually call a live call's claim abandoned, which costs money by
sending its owner's user to discard a claim that is about to be used. The lease is only ever
reached by a run that was killed outright, since any call that returns marks its own claim.
"""

CLAIM_WAIT_SECONDS = 120.0
"""How long a run waits for another run's in-flight claim before giving up on it.

This bounds patience, not correctness: a run that gives up refuses without buying anything and
says so in a sentence that sends nobody to ``discard_intent``. Two minutes covers an ordinary
call comfortably, and keeps the one case that cannot resolve (an owner killed outright, whose
claim stays in flight until the lease runs out) to a bounded wait instead of a quarter of an hour.
"""

CLAIM_POLL_SECONDS = 0.5
"""Gap between locked re-reads while waiting for another run's claim to resolve.

Each poll is two short reads under the project lock, so this is cheap next to the seconds a
decision spends on the wire, and it bounds how long a waiter sits on an answer already committed.
"""


class ClaimState(Enum):
    """What a reader may do about the claim record it just read."""

    ABSENT = "absent"
    IN_FLIGHT = "in_flight"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class DecisionClaim:
    """One run's claim on one decision, as the record stores it.

    Attributes:
        key: Content identity of the claimed decision.
        owner: Identity of the run holding the claim, so a run can recognize its own.
        pid: Process that wrote it, so a refusal names something a user can look for.
        claimed_unix: Wall-clock second the claim was written, which the lease is measured from.
        abandoned: The owning run established that its call committed no answer.
    """

    key: str
    owner: str
    pid: int
    claimed_unix: float
    abandoned: bool = False

    def payload(self) -> bytes:
        """Return the exact bytes this claim is stored as, the same bytes on every call."""
        return canonical_json_bytes(
            {
                "version": CLAIM_RECORD_VERSION,
                "key": self.key,
                "owner": self.owner,
                "pid": self.pid,
                "claimed_unix": self.claimed_unix,
                "abandoned": self.abandoned,
            }
        )

    def abandon(self) -> DecisionClaim:
        """Return this claim marked as a call that committed no answer."""
        return replace(self, abandoned=True)


@dataclass(frozen=True)
class ClaimReading:
    """One reader's judgment of the claim record it just read.

    Attributes:
        state: What the reader may do about it.
        claim: The claim behind the judgment, absent when there was no record or it could not be
            read as a claim.
        age_seconds: How long ago a readable claim was written, for the sentence a user reads.
    """

    state: ClaimState
    claim: DecisionClaim | None = None
    age_seconds: float = 0.0


def new_owner() -> str:
    """Return an identity for one run's claims, distinct from every other run's."""
    return uuid.uuid4().hex[:12]


def new_claim(key: str, owner: str, *, now: float) -> DecisionClaim:
    """Return the claim one run writes before buying one decision.

    Args:
        key: Content identity of the decision about to be bought.
        owner: Identity of the claiming run.
        now: Wall-clock second the claim is written.

    Returns:
        The in-flight claim.
    """
    return DecisionClaim(key=key, owner=owner, pid=os.getpid(), claimed_unix=now)


def read_claim(
    raw: bytes | None,
    *,
    key: str,
    now: float,
    lease_seconds: float = CLAIM_LEASE_SECONDS,
) -> ClaimReading:
    """Judge one stored claim record.

    A record this module cannot read as a claim of this exact key is abandoned. So is one that is
    older than the lease. Both are claims nobody can account for, and the safe reading of an
    unaccountable claim is that a call was made and its outcome is unknown.

    Args:
        raw: Exact stored bytes, or None when there is no claim record.
        key: Content identity the claim must name.
        now: Current wall-clock second, which the claim's age is measured against.
        lease_seconds: How long a claim may be in flight before its age alone abandons it.

    Returns:
        What a reader may do about it, with the claim itself when there was a readable one.
    """
    if raw is None:
        return ClaimReading(ClaimState.ABSENT)
    claim = _parsed(raw, key)
    if claim is None:
        return ClaimReading(ClaimState.ABANDONED)
    age_seconds = max(0.0, now - claim.claimed_unix)
    if claim.abandoned or age_seconds >= lease_seconds:
        return ClaimReading(ClaimState.ABANDONED, claim=claim, age_seconds=age_seconds)
    return ClaimReading(ClaimState.IN_FLIGHT, claim=claim, age_seconds=age_seconds)


def abandoned_claim_refusal(key: str) -> str:
    """Return the refusal naming a decision whose call left an unknown outcome behind.

    This wording is the money-safety contract a user acts on, so it is fixed: the outcome is
    unknown, tagging will not buy it again, and releasing it takes one named call after the user
    has established that nothing was billed.
    """
    return (
        f"decision {key[:12]} was dispatched and its outcome is unknown; it may "
        "already have been billed, so tagging will not buy it again. Confirm the "
        "call was not billed, then call DecisionCache.discard_intent with that key"
    )


def in_flight_claim_refusal(
    key: str, reading: ClaimReading, *, lease_seconds: float = CLAIM_LEASE_SECONDS
) -> str:
    """Return the refusal for a claim another run may still be buying.

    It deliberately sends nobody to ``discard_intent``. Discarding a claim whose call is in flight
    lets two runs buy one decision, and from here the difference between a slow run and a run that
    was killed outright is a matter of waiting, which is what the sentence says.
    """
    owner = (
        "another run"
        if reading.claim is None
        else f"run {reading.claim.owner} (pid {reading.claim.pid})"
    )
    return (
        f"decision {key[:12]} was claimed {reading.age_seconds:.0f}s ago by {owner}, which has "
        "committed no answer yet, so tagging will not buy it a second time. Wait for that run to "
        "finish and run again to replay its answer. If no tagging run is active, that one was "
        f"killed outright and its claim is refusable by name {lease_seconds:.0f}s after it was "
        "made."
    )


def own_claim_discard_refusal(key: str) -> str:
    """Return the refusal for discarding a claim this very run is still buying."""
    return (
        f"decision {key[:12]} is being bought by this run right now, so its claim is not an "
        "interrupted call; wait for that call to finish instead of discarding its claim"
    )


def absent_claim_discard_refusal(key: str) -> str:
    """Return the refusal for discarding a claim that is not there to discard."""
    return (
        f"decision {key[:12]} has no retained claim; it was never dispatched, or its answer is "
        "already committed, or the claim has already been discarded"
    )


def _parsed(raw: bytes, key: str) -> DecisionClaim | None:
    """Return the claim these exact bytes hold, or None when they are not one for this key.

    Every field is required to be the type it was written as rather than coerced into it. A
    record that needs coercing is not one this module wrote, and the next step after trusting it
    spends money.
    """
    try:
        stored = _decoded(raw)
    except ValueError:
        return None
    if stored.get("version") != CLAIM_RECORD_VERSION or stored.get("key") != key:
        return None
    owner = stored.get("owner")
    pid = stored.get("pid")
    claimed_unix = stored.get("claimed_unix")
    if isinstance(pid, bool) or isinstance(claimed_unix, bool):
        return None
    if not isinstance(owner, str) or not isinstance(pid, int):
        return None
    if not isinstance(claimed_unix, (int, float)):
        return None
    return DecisionClaim(
        key=key,
        owner=owner,
        pid=pid,
        claimed_unix=float(claimed_unix),
        abandoned=stored.get("abandoned") is True,
    )


def _decoded(raw: bytes) -> dict[str, object]:
    """Decode exactly one JSON object from stored bytes, rejecting any other shape.

    Args:
        raw: Stored record bytes.

    Returns:
        The decoded object.

    Raises:
        ValueError: The bytes are not JSON, or are JSON that is not an object.
    """
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("a claim record is a JSON object")
    return decoded
