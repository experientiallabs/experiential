"""Tests for the claim a run writes before it buys a decision."""

from __future__ import annotations

import json
import os

from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.mining.decisions import DEFAULT_DECISION_TIMEOUT_SECONDS
from exp.simulation.mining.intents import (
    CLAIM_LEASE_SECONDS,
    CLAIM_WAIT_SECONDS,
    ClaimState,
    abandoned_claim_refusal,
    absent_claim_discard_refusal,
    in_flight_claim_refusal,
    new_claim,
    new_owner,
    own_claim_discard_refusal,
    read_claim,
)

_KEY = "a" * 64

_ABANDONED_REFUSAL = (
    "decision aaaaaaaaaaaa was dispatched and its outcome is unknown; it may already have been "
    "billed, so tagging will not buy it again. Confirm the call was not billed, then call "
    "DecisionCache.discard_intent with that key"
)


def test_the_refusal_by_name_is_worded_exactly_as_it_always_was() -> None:
    """This sentence is the money-safety contract a user acts on, so it is pinned literally."""
    assert abandoned_claim_refusal(_KEY) == _ABANDONED_REFUSAL


def test_a_claim_round_trips_through_its_stored_bytes() -> None:
    """The stored bytes are matched exactly on commit, so they must be stable and readable."""
    claim = new_claim(_KEY, "owner-1", now=1000.0)

    assert claim.payload() == claim.payload()
    reading = read_claim(claim.payload(), key=_KEY, now=1000.5)
    assert reading.state is ClaimState.IN_FLIGHT
    assert reading.claim == claim
    assert reading.claim is not None
    assert reading.claim.pid == os.getpid()
    assert reading.age_seconds == 0.5


def test_no_record_is_a_decision_nobody_has_claimed() -> None:
    """An absent claim is the ordinary case: this run may take it and buy the decision."""
    assert read_claim(None, key=_KEY, now=1000.0).state is ClaimState.ABSENT


def test_a_claim_its_owner_marked_is_abandoned_whatever_its_age() -> None:
    """A call that returned no answer says so, so the next reader need not wait out the lease."""
    claim = new_claim(_KEY, "owner-1", now=1000.0).abandon()

    reading = read_claim(claim.payload(), key=_KEY, now=1000.1)

    assert reading.state is ClaimState.ABANDONED
    assert reading.claim is not None
    assert reading.claim.abandoned is True


def test_a_claim_older_than_the_lease_is_abandoned_by_age_alone() -> None:
    """A run killed outright marks nothing, so age is the only thing left to judge it by."""
    claim = new_claim(_KEY, "owner-1", now=1000.0)

    assert read_claim(claim.payload(), key=_KEY, now=1000.0 + CLAIM_LEASE_SECONDS - 1).state is (
        ClaimState.IN_FLIGHT
    )
    assert read_claim(claim.payload(), key=_KEY, now=1000.0 + CLAIM_LEASE_SECONDS).state is (
        ClaimState.ABANDONED
    )


def test_the_lease_outlasts_the_longest_call_a_run_can_legitimately_make() -> None:
    """A lease shorter than a live call would call that call abandoned, which costs money.

    Sending the owner's user to discard a claim whose answer is about to be committed is how two
    runs end up buying one decision, so the lease is tied to the ceiling a call cannot exceed.
    """
    longest_call = RetryPolicy().maximum_attempts * DEFAULT_DECISION_TIMEOUT_SECONDS

    assert CLAIM_LEASE_SECONDS >= longest_call
    assert CLAIM_WAIT_SECONDS < CLAIM_LEASE_SECONDS


def test_a_record_of_any_other_shape_is_abandoned_rather_than_trusted() -> None:
    """A claim nobody can account for is exactly the interrupted call a run must refuse."""
    older_shape = json.dumps({"schema_version": 1, "key": _KEY}).encode()
    foreign_key = new_claim("b" * 64, "owner-1", now=1000.0).payload()

    for raw in (older_shape, foreign_key, b"", b"{", b"[]", b'"a"', b"null"):
        assert read_claim(raw, key=_KEY, now=1000.0).state is ClaimState.ABANDONED


def test_a_claim_with_an_unreadable_field_is_abandoned() -> None:
    """Partial trust in a record is no trust at all when the next step spends money."""
    broken = json.dumps(
        {"version": 1, "key": _KEY, "owner": "o", "pid": "not-a-pid", "claimed_unix": 1.0}
    ).encode()

    assert read_claim(broken, key=_KEY, now=1000.0).state is ClaimState.ABANDONED


def test_the_in_flight_refusal_never_sends_anyone_to_discard_a_live_claim() -> None:
    """Discarding a call that is in flight is how one decision gets bought twice."""
    reading = read_claim(new_claim(_KEY, "owner-1", now=1000.0).payload(), key=_KEY, now=1042.0)

    message = in_flight_claim_refusal(_KEY, reading)

    assert "discard" not in message
    assert "claimed 42s ago by run owner-1" in message
    assert f"pid {os.getpid()}" in message
    assert "1200s after it was made" in message


def test_the_in_flight_refusal_still_says_something_about_an_unreadable_claim() -> None:
    """A refusal is part of the interface even when the record behind it is not readable."""
    assert "another run" in in_flight_claim_refusal(_KEY, read_claim(b"{", key=_KEY, now=1.0))


def test_each_discard_refusal_says_what_to_do_instead() -> None:
    """These two are the only ways a discard is refused, so both have to be actionable."""
    assert "wait for that call to finish" in own_claim_discard_refusal(_KEY)
    assert "already committed" in absent_claim_discard_refusal(_KEY)


def test_every_run_claims_under_its_own_identity() -> None:
    """A run recognizes its own claims, which is the one case it can judge with certainty."""
    assert new_owner() != new_owner()
