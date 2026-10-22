"""Step-row helpers shared by the session store's transactions: claims, replay and audit."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from exp.common.core.artifacts import JsonObject, canonical_json_bytes, sha256_json
from exp.runtime.sessions.errors import (
    ClaimLostError,
    SessionStoreError,
    StepAmbiguousError,
    StepConflictError,
    StepInFlightError,
    StepRejectedError,
)
from exp.runtime.sessions.records import CompletedStep, StepClaim, StepDecision, StepStatus


def identity_text(value: JsonObject) -> str:
    """Render a JSON object as canonical UTF-8 text, for identity-bearing values only."""
    return canonical_json_bytes(value).decode("utf-8")


def stored_object(text: str) -> JsonObject:
    """Decode stored canonical JSON text into an object.

    Args:
        text: Text produced by ``_text``.

    Returns:
        The decoded object.

    Raises:
        SessionStoreError: If the stored value is not an object.
    """
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise SessionStoreError("stored JSON value is not an object")
    return decoded


def utc_now() -> datetime:
    """Return the current aware UTC time for commit authorization."""
    return datetime.now(UTC)


def placeholders(values: tuple[str, ...]) -> str:
    """Return comma-separated SQL placeholders, one per value."""
    return ", ".join("?" for _ in values)


def set_status(connection: sqlite3.Connection, key: tuple[str, str], status: str) -> None:
    """Set one step's status inside the caller's transaction."""
    connection.execute(
        "UPDATE steps SET status = ? WHERE session_id = ? AND step_id = ?",
        (status, key[0], key[1]),
    )


def replay_or_refuse(
    session_id: str, step_id: str, request_sha256: str, prior: sqlite3.Row
) -> StepDecision:
    """Return the stored outcome of a known step, or raise the matching refusal.

    Args:
        session_id: Session of the step.
        step_id: Step identifier.
        request_sha256: Digest of the caller's current request.
        prior: The stored step row.

    Returns:
        A completed decision for a completed step with the same request.

    Raises:
        StepConflictError: If the request differs or the step was released.
        StepAmbiguousError: If the step is blocked as ambiguous.
        StepRejectedError: If the step was rejected.
        StepInFlightError: If the step is still open.
    """
    if prior["request_sha256"] != request_sha256:
        raise StepConflictError(f"step {step_id!r} was already used with a different request")
    status = StepStatus(prior["status"])
    if status is StepStatus.COMPLETED:
        return StepDecision(
            kind="completed",
            completed=CompletedStep(
                response=stored_object(prior["response_json"]),
                resulting_version=prior["resulting_version"],
            ),
        )
    if status is StepStatus.AMBIGUOUS:
        raise StepAmbiguousError(f"step {step_id!r} may have reached the provider; review it")
    if status is StepStatus.REJECTED:
        reason = stored_object(prior["response_json"]).get("reason", "")
        raise StepRejectedError(f"step {step_id!r} was rejected: {reason}")
    if status is StepStatus.RELEASED:
        raise StepConflictError(f"step {step_id!r} was released; use a new step ID")
    raise StepInFlightError(f"step {step_id!r} in session {session_id!r} is still open")


def audit(
    connection: sqlite3.Connection,
    session_id: str,
    step_id: str | None,
    event: str,
    detail: JsonObject,
) -> None:
    """Append one audit row. The detail is stored as a digest, never as content.

    Args:
        connection: Connection with an open transaction.
        session_id: Session the event belongs to.
        step_id: Step the event belongs to, when applicable.
        event: Stable event name.
        detail: Content whose digest is recorded.
    """
    connection.execute(
        "INSERT INTO audit (session_id, step_id, event, detail_sha256, recorded_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (session_id, step_id, event, sha256_json(detail), datetime.now(UTC).isoformat()),
    )


def require_open_claim(
    connection: sqlite3.Connection,
    claim: StepClaim,
    *,
    expected_status: StepStatus | None,
) -> None:
    """Verify the claim is the open claim for its step, optionally at one status.

    Args:
        connection: Connection with an open transaction.
        claim: Claim presented by the caller.
        expected_status: Required status, or None for any open status.

    Raises:
        ClaimLostError: If the step is missing, not open, has another token, or differs in status.
    """
    row = connection.execute(
        "SELECT status, claim_token FROM steps WHERE session_id = ? AND step_id = ?",
        (claim.session_id, claim.step_id),
    ).fetchone()
    if row is None or row["claim_token"] != claim.claim_token:
        raise ClaimLostError(f"claim for step {claim.step_id!r} is not the open claim")
    status = StepStatus(row["status"])
    if status not in (StepStatus.CLAIMED, StepStatus.PROVIDER_STARTED):
        raise ClaimLostError(f"step {claim.step_id!r} is {status.value}, not open")
    if expected_status is not None and status is not expected_status:
        raise ClaimLostError(f"step {claim.step_id!r} is {status.value}")
