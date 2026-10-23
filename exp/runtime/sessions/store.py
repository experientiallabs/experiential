"""Durable SQLite session store: immutable binding, step idempotency, expected-version state.

A session pins one immutable ``SessionBinding`` (target, receipt, knowledge, corpus, and
representation digests) and one candidate state object. Only the state and version change
during a session. A step is one caller operation keyed by ``step_id``:

1. ``begin_step`` claims the step against an expected version, or replays the stored
   terminal outcome of the same request.
2. ``mark_provider_started`` records that a provider call may run. A crash after this point
   makes the step ambiguous rather than releasable.
3. ``complete_step`` commits the next state, the terminal response, and an audit row in one
   transaction. ``reject_step`` closes a step whose reply was refused, with no state change.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ContractModel,
    JsonObject,
    Sha256,
    canonical_json_bytes,
    sha256_json,
)
from exp.common.sqlite.connection import connect_database

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
"""Schema identity stored in ``PRAGMA user_version``. Mismatches fail closed."""


class SessionStoreError(RuntimeError):
    """Base class for refused session-store operations."""


class SessionNotFoundError(SessionStoreError):
    """The requested session does not exist in this database."""


class SessionExistsError(SessionStoreError):
    """A session with this identifier already exists."""


class VersionConflictError(SessionStoreError):
    """The expected session version does not match the stored version."""


class StepConflictError(SessionStoreError):
    """A step ID was reused with a different request digest."""


class StepInFlightError(SessionStoreError):
    """Another step on the same session is open, so this step must wait."""


class StepAmbiguousError(SessionStoreError):
    """A step reached the provider without a persisted outcome and is blocked for review."""


class StepRejectedError(SessionStoreError):
    """The step was closed without a state change. Its stored reason is in the message."""


class SessionBindingMismatchError(SessionStoreError):
    """The request binding differs from the binding pinned at session creation."""


class ProviderBudgetExhaustedError(SessionStoreError):
    """The session has used its provider-call budget, so no further dispatch may start."""


class StepDeadlineExceededError(SessionStoreError):
    """A returned proposal was retained but cannot commit after its session deadline."""


class SessionDrainingError(SessionStoreError):
    """The session is draining and refuses new steps."""


class ClaimLostError(SessionStoreError):
    """The claim is not the open claim for its step."""


class StepStatus(StrEnum):
    """Durable lifecycle state of one step row."""

    CLAIMED = "claimed"
    PROVIDER_STARTED = "provider_started"
    COMPLETED = "completed"
    REJECTED = "rejected"
    RELEASED = "released"
    AMBIGUOUS = "ambiguous"


_OPEN_STATUSES = ("claimed", "provider_started", "ambiguous")


class SessionBinding(ContractModel):
    """Immutable identity of the evidence and execution policy a session runs under.

    Attributes:
        deadline: Absolute commit deadline, fixed for the entire session.
        target_id: Target name with its deployed profile version, as ``name@version``.
        initialization_receipt_sha256: Digest of the public initialization receipt.
        knowledge_sha256: Digest of the frozen knowledge or rulebook.
        corpus_sha256: Digest of the retrieval corpus snapshot.
        representation_sha256: Digest of the state representation contract.
        execution_policy_sha256: Digest of the ``ExecutionPolicy`` that bounds model
            settings, provider-call caps, and inspection.
    """

    deadline: AwareDatetime
    target_id: str = Field(pattern=r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$")
    initialization_receipt_sha256: Sha256
    knowledge_sha256: Sha256
    corpus_sha256: Sha256
    representation_sha256: Sha256
    execution_policy_sha256: Sha256


class SessionRecord(ContractModel):
    """One session's binding, durable state, and version.

    Attributes:
        session_id: Caller-chosen stable identifier.
        binding: Immutable identity, fixed at creation.
        version: Count of committed transitions; starts at 0.
        state: The candidate state object.
        draining: Whether new steps are refused.
    """

    session_id: str = Field(min_length=1)
    binding: SessionBinding
    version: int = Field(ge=0)
    state: JsonObject
    draining: bool


class StepClaim(ContractModel):
    """An open claim for exactly one provider attempt.

    Attributes:
        session_id: Session the step belongs to.
        step_id: Caller-chosen step identifier, unique within the session.
        claim_token: Opaque token required by every later call on this claim.
        base_version: Session version the claim was made against.
    """

    session_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    claim_token: str = Field(min_length=1)
    base_version: int = Field(ge=0)


class RequestPreflightReceipt(ContractModel):
    """Content-free audit receipt for preparation before provider dispatch.

    Attributes:
        request_sha256: Order-preserving digest of the complete prepared wire body.
        capacity_sha256: Digest of the declared model, capacity, counter and renderer policy.
        input_token_upper_bound: Conservative input bound, not measured provider usage.
        output_tokens: Declared output reservation, not actual output usage.
        accepted: Whether the request passed the declared capacity check.
    """

    request_sha256: Sha256
    capacity_sha256: Sha256
    input_token_upper_bound: int = Field(ge=0, strict=True)
    output_tokens: int = Field(gt=0, strict=True)
    accepted: bool


class CompletedStep(ContractModel):
    """The stored terminal response of a completed step, replayed verbatim.

    Attributes:
        response: Exact response object returned by the original attempt.
        resulting_version: Session version after the step committed.
    """

    response: JsonObject
    resulting_version: int = Field(ge=0)


class StepDecision(ContractModel):
    """Outcome of ``begin_step``: a fresh claim or a replayed completion.

    Attributes:
        kind: ``"claimed"`` for a new attempt, ``"completed"`` for a replay.
        claim: Set only when ``kind`` is ``"claimed"``.
        completed: Set only when ``kind`` is ``"completed"``.
    """

    kind: Literal["claimed", "completed"]
    claim: StepClaim | None = None
    completed: CompletedStep | None = None

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> Self:
        """Require exactly the payload matching ``kind``.

        Returns:
            The validated decision.

        Raises:
            ValueError: If the payload does not match ``kind``.
        """
        if (self.kind == "claimed") != (self.claim is not None):
            raise ValueError("claim is required exactly when kind is claimed")
        if (self.kind == "completed") != (self.completed is not None):
            raise ValueError("completed is required exactly when kind is completed")
        return self


class RecoveryReport(ContractModel):
    """Step keys whose status recovery changed.

    Attributes:
        released: Claims that never reached the provider.
        ambiguous: Claims that may have reached the provider; blocked until reviewed.
    """

    released: tuple[tuple[str, str], ...] = ()
    ambiguous: tuple[tuple[str, str], ...] = ()


_SCHEMA_STATEMENTS = (
    """CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        binding_json TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version >= 0),
        state_json TEXT NOT NULL,
        draining INTEGER NOT NULL CHECK (draining IN (0, 1))
    ) STRICT""",
    """CREATE TRIGGER sessions_binding_immutable BEFORE UPDATE OF binding_json ON sessions
    BEGIN SELECT RAISE(ABORT, 'session binding is immutable'); END""",
    """CREATE TABLE steps (
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        step_id TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        base_version INTEGER NOT NULL,
        status TEXT NOT NULL,
        claim_token TEXT NOT NULL,
        proposal_json TEXT,
        response_json TEXT,
        resulting_version INTEGER,
        PRIMARY KEY (session_id, step_id)
    ) STRICT""",
    """CREATE UNIQUE INDEX steps_one_open_per_session
        ON steps(session_id) WHERE status IN ('claimed', 'provider_started', 'ambiguous')""",
    """CREATE TABLE audit (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        step_id TEXT,
        event TEXT NOT NULL,
        detail_sha256 TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    ) STRICT""",
)


def _text(value: JsonObject) -> str:
    """Render a JSON object as canonical UTF-8 text, for identity-bearing values only."""
    return canonical_json_bytes(value).decode("utf-8")


def _declared_text(value: JsonObject) -> str:
    """Render a JSON object as UTF-8 text that keeps its declared key order.

    Used for state, proposals, and responses, whose key order is data the provider and client
    declared. Identity digests never use this encoding.
    """
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _object(text: str) -> JsonObject:
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


def _utc_now() -> datetime:
    """Return the current aware UTC time for commit authorization."""
    return datetime.now(UTC)


class SessionStore:
    """Transactional single-file session store.

    Each mutation runs in one ``BEGIN IMMEDIATE`` transaction, so concurrent writers serialize
    and version checks see the state the write will change.

    Args:
        path: SQLite file, created with its schema when absent.

    Raises:
        SessionStoreError: If an existing file has a different schema version.
    """

    def __init__(self, path: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        """Open or initialize the database at ``path``.

        Args:
            path: SQLite database file.
            clock: Aware wall clock sampled under the commit transaction.

        Raises:
            SessionStoreError: If the schema version is not ``SCHEMA_VERSION``.
        """
        self._clock = clock or _utc_now
        self._path = path
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.execute("COMMIT")
            elif version != SCHEMA_VERSION:
                raise SessionStoreError(
                    f"session database schema {version} is not {SCHEMA_VERSION}; "
                    "use a fresh database path"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a fresh configured connection, closed on exit.

        Yields:
            Connection with foreign keys, WAL, and a bounded busy timeout.
        """
        connection = connect_database(self._path)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside an immediate write transaction.

        Yields:
            Connection whose writes commit or roll back together.
        """
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    def create_session(
        self, session_id: str, binding: SessionBinding, initial_state: JsonObject
    ) -> SessionRecord:
        """Create a session at version 0 with its immutable binding.

        Args:
            session_id: New stable identifier.
            binding: Evidence identity fixed for the session's lifetime.
            initial_state: Starting candidate state.

        Returns:
            The created record.

        Raises:
            SessionExistsError: If the identifier is already used.
        """
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone():
                raise SessionExistsError(f"session {session_id!r} already exists")
            connection.execute(
                "INSERT INTO sessions (session_id, binding_json, version, state_json, draining)"
                " VALUES (?, ?, 0, ?, 0)",
                (session_id, _text(binding.model_dump(mode="json")), _declared_text(initial_state)),
            )
            _audit(
                connection,
                session_id,
                None,
                "session_created",
                {"binding_sha256": sha256_json(binding)},
            )
        return self.read_session(session_id)

    def read_session(self, session_id: str) -> SessionRecord:
        """Return the binding, committed state, and version of one session.

        Args:
            session_id: Session identifier.

        Returns:
            The current record.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_id, binding_json, version, state_json, draining"
                " FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"session {session_id!r} does not exist")
        return SessionRecord(
            session_id=row["session_id"],
            binding=SessionBinding.model_validate_json(row["binding_json"]),
            version=row["version"],
            state=_object(row["state_json"]),
            draining=bool(row["draining"]),
        )

    def begin_step(
        self,
        session_id: str,
        step_id: str,
        request_sha256: Sha256,
        expected_version: int,
        binding: SessionBinding,
    ) -> StepDecision:
        """Claim a step, or replay the stored outcome of the same request.

        A completed step replays even after the session version has advanced, because the
        request digest already binds the original expected version. Only a new step is checked
        against the current version.

        Args:
            session_id: Session the step acts on.
            step_id: Caller-chosen identifier.
            request_sha256: Digest of the full request, including binding and expected version.
            expected_version: Version the caller computed its request against.

        Returns:
            A fresh claim, or the stored completion.

        Raises:
            SessionNotFoundError: If the session does not exist.
            StepConflictError: If ``step_id`` was used for a different request or was released.
            StepInFlightError: If another step on the session is open.
            StepAmbiguousError: If this step is blocked as ambiguous.
            StepRejectedError: If this step was closed as rejected.
            SessionDrainingError: If the session is draining.
            VersionConflictError: If a new step's expected version is stale.
        """
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT version, draining, binding_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise SessionNotFoundError(f"session {session_id!r} does not exist")
            if SessionBinding.model_validate_json(session["binding_json"]) != binding:
                raise SessionBindingMismatchError(
                    f"request binding differs from session {session_id!r}"
                )
            prior = connection.execute(
                "SELECT request_sha256, status, response_json, resulting_version FROM steps"
                " WHERE session_id = ? AND step_id = ?",
                (session_id, step_id),
            ).fetchone()
            if prior is not None:
                return _replay_or_refuse(session_id, step_id, request_sha256, prior)
            if bool(session["draining"]):
                raise SessionDrainingError(f"session {session_id!r} is draining")
            open_step = connection.execute(
                "SELECT step_id FROM steps WHERE session_id = ?"
                f" AND status IN ({_placeholders(_OPEN_STATUSES)})",
                (session_id, *_OPEN_STATUSES),
            ).fetchone()
            if open_step is not None:
                raise StepInFlightError(
                    f"session {session_id!r} has open step {open_step['step_id']!r}"
                )
            if session["version"] != expected_version:
                raise VersionConflictError(
                    f"expected version {expected_version}, stored {session['version']}"
                )
            claim = StepClaim(
                session_id=session_id,
                step_id=step_id,
                claim_token=uuid4().hex,
                base_version=expected_version,
            )
            connection.execute(
                "INSERT INTO steps (session_id, step_id, request_sha256, base_version,"
                " status, claim_token) VALUES (?, ?, ?, ?, 'claimed', ?)",
                (session_id, step_id, request_sha256, expected_version, claim.claim_token),
            )
            _audit(connection, session_id, step_id, "step_claimed", {"request": request_sha256})
        return StepDecision(kind="claimed", claim=claim)

    def record_request_preflight(self, claim: StepClaim, receipt: RequestPreflightReceipt) -> None:
        """Persist the digest of a typed preflight receipt for a still-claimed step.

        Repeating the same receipt is idempotent. A changed receipt or expired claim is refused.
        Only the audit log changes; this operation consumes no dispatch budget and never changes
        candidate state or version. A refused capacity receipt remains distinguishable from an
        accepted request when its receipt is independently retained. The audit stores only a
        digest, so it cannot reconstruct the receipt's fields.

        Args:
            claim: Open claim that has not reached provider-start accounting.
            receipt: Typed prepared-wire capacity receipt, without prompt or source content.

        Raises:
            ClaimLostError: The claim is no longer open at claimed status.
            StepConflictError: A different receipt was already stored for this step.
        """
        payload = receipt.model_dump(mode="json")
        with self._transaction() as connection:
            _require_open_claim(connection, claim, expected_status=StepStatus.CLAIMED)
            prior = connection.execute(
                "SELECT detail_sha256 FROM audit WHERE session_id = ? AND step_id = ?"
                " AND event = 'request_preflight'",
                (claim.session_id, claim.step_id),
            ).fetchone()
            if prior is not None:
                if prior["detail_sha256"] != sha256_json(payload):
                    raise StepConflictError(
                        "prepared request receipt differs from the stored receipt"
                    )
                return
            _audit(connection, claim.session_id, claim.step_id, "request_preflight", payload)

    def mark_provider_started(self, claim: StepClaim, maximum_provider_calls: int) -> None:
        """Durably record a provider dispatch, refusing it once the session budget is spent.

        The dispatch count and the status change happen in one transaction. Every
        ``provider_started`` audit row is one dispatch, including attempts later rejected or
        left ambiguous, so those attempts consume budget.

        Args:
            claim: The open claim for this attempt.
            maximum_provider_calls: Hard budget of dispatches for the whole session.

        Raises:
            ClaimLostError: If the claim is not the open claim for its step.
            ProviderBudgetExhaustedError: If the session already used its budget.
        """
        with self._transaction() as connection:
            _require_open_claim(connection, claim, expected_status=StepStatus.CLAIMED)
            used = connection.execute(
                "SELECT COUNT(*) AS n FROM audit"
                " WHERE session_id = ? AND event = 'provider_started'",
                (claim.session_id,),
            ).fetchone()["n"]
            if used >= maximum_provider_calls:
                raise ProviderBudgetExhaustedError(
                    f"session {claim.session_id!r} used {used} of {maximum_provider_calls} calls"
                )
            connection.execute(
                "UPDATE steps SET status = 'provider_started' WHERE session_id = ? AND step_id = ?",
                (claim.session_id, claim.step_id),
            )
            _audit(connection, claim.session_id, claim.step_id, "provider_started", {})

    def record_proposal(self, claim: StepClaim, proposal: JsonObject) -> None:
        """Durably store the paid provider output before any state commit.

        A failed commit then leaves the proposal on disk for inspection instead of a lost paid
        response. The proposal is stored verbatim; the reply string is never re-encoded.

        Args:
            claim: The open claim whose provider call has returned.
            proposal: The exact provider output object.

        Raises:
            ClaimLostError: If the claim is not open at ``provider_started``.
        """
        with self._transaction() as connection:
            _require_open_claim(connection, claim, expected_status=StepStatus.PROVIDER_STARTED)
            connection.execute(
                "UPDATE steps SET proposal_json = ? WHERE session_id = ? AND step_id = ?",
                (_declared_text(proposal), claim.session_id, claim.step_id),
            )
            _audit(connection, claim.session_id, claim.step_id, "proposal_recorded", proposal)

    def proposal(self, session_id: str, step_id: str) -> JsonObject | None:
        """Return the durably stored provider output for one step, if any.

        Args:
            session_id: Session of the step.
            step_id: Step identifier.

        Returns:
            The stored proposal object, or None when no proposal was recorded.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT proposal_json FROM steps WHERE session_id = ? AND step_id = ?",
                (session_id, step_id),
            ).fetchone()
        if row is None or row["proposal_json"] is None:
            return None
        return _object(row["proposal_json"])

    def complete_step(
        self, claim: StepClaim, next_state: JsonObject, response: JsonObject
    ) -> SessionRecord:
        """Publish one state revision or durably reject a late proposal.

        Args:
            claim: Current provider-started owner of the logical operation.
            next_state: Fully validated candidate state, serialized before locking.
            response: Ordered observations to replay with the committed state.

        Returns:
            Session at the newly committed revision.

        Raises:
            ClaimLostError: Ownership or the saved proposal is missing.
            VersionConflictError: The proposal was based on a stale revision.
            StepDeadlineExceededError: Deadline passed; proposal and rejection remain durable.
        """
        state_text, response_text = _declared_text(next_state), _declared_text(response)
        expired = False
        with self._transaction() as connection:
            _require_open_claim(connection, claim, expected_status=StepStatus.PROVIDER_STARTED)
            proposal_row = connection.execute(
                "SELECT proposal_json FROM steps WHERE session_id = ? AND step_id = ?",
                (claim.session_id, claim.step_id),
            ).fetchone()
            if proposal_row["proposal_json"] is None:
                raise ClaimLostError("complete requires a recorded proposal first")
            current = connection.execute(
                "SELECT version, binding_json FROM sessions WHERE session_id = ?",
                (claim.session_id,),
            ).fetchone()
            if current["version"] != claim.base_version:
                raise VersionConflictError("proposal is based on a stale session revision")
            binding = SessionBinding.model_validate_json(current["binding_json"])
            now = self._clock()
            if now.tzinfo is None:
                raise ValueError("commit clock must return an aware datetime")
            expired = now >= binding.deadline
            if expired:
                connection.execute(
                    "UPDATE steps SET status = 'rejected', response_json = ? "
                    "WHERE session_id = ? AND step_id = ?",
                    (
                        _declared_text({"reason": "deadline_stopped"}),
                        claim.session_id,
                        claim.step_id,
                    ),
                )
                _audit(connection, claim.session_id, claim.step_id, "deadline_stopped", {})
            else:
                next_version = claim.base_version + 1
                connection.execute(
                    "UPDATE sessions SET version = ?, state_json = ? WHERE session_id = ?",
                    (next_version, state_text, claim.session_id),
                )
                connection.execute(
                    "UPDATE steps SET status = 'completed', response_json = ?, "
                    "resulting_version = ? WHERE session_id = ? AND step_id = ?",
                    (response_text, next_version, claim.session_id, claim.step_id),
                )
                _audit(
                    connection,
                    claim.session_id,
                    claim.step_id,
                    "step_completed",
                    {"v": next_version},
                )
        if expired:
            raise StepDeadlineExceededError("proposal retained; session deadline forbids commit")
        return self.read_session(claim.session_id)

    def reject_step(self, claim: StepClaim, reason: str) -> None:
        """Close an open step with no state change because its reply was refused.

        Args:
            claim: The open claim whose reply failed validation or commit checks.
            reason: Short, content-free explanation stored with the rejection.

        Raises:
            ClaimLostError: If the claim is not open for its step.
        """
        with self._transaction() as connection:
            _require_open_claim(connection, claim, expected_status=None)
            connection.execute(
                "UPDATE steps SET status = 'rejected', response_json = ? WHERE session_id = ?"
                " AND step_id = ?",
                (_declared_text({"reason": reason}), claim.session_id, claim.step_id),
            )
            _audit(connection, claim.session_id, claim.step_id, "step_rejected", {})

    def mark_ambiguous(self, claim: StepClaim, reason: str) -> None:
        """Record an unresolved dispatched outcome without releasing or repurchasing it.

        Args:
            claim: Claim whose completion callback may have reached the provider.
            reason: Content-free exception classification, never credentials or raw error text.
        """
        with self._transaction() as connection:
            _require_open_claim(connection, claim, expected_status=StepStatus.PROVIDER_STARTED)
            _set_status(connection, (claim.session_id, claim.step_id), "ambiguous")
            _audit(
                connection,
                claim.session_id,
                claim.step_id,
                "dispatch_outcome_unknown",
                {"reason": reason},
            )

    def release_step(self, claim: StepClaim) -> None:
        """Release a claim whose provider call never started.

        Args:
            claim: The open claim to release.

        Raises:
            ClaimLostError: If the claim is not open, or the provider already started.
        """
        with self._transaction() as connection:
            _require_open_claim(connection, claim, expected_status=StepStatus.CLAIMED)
            connection.execute(
                "UPDATE steps SET status = 'released' WHERE session_id = ? AND step_id = ?",
                (claim.session_id, claim.step_id),
            )
            _audit(connection, claim.session_id, claim.step_id, "step_released", {})

    def recover(self) -> RecoveryReport:
        """Resolve open claims left by a crash or restart.

        Claims that never reached the provider are released. Claims that did reach it become
        ambiguous and keep blocking the session until reviewed.

        Returns:
            The step keys whose status changed.
        """
        released: list[tuple[str, str]] = []
        ambiguous: list[tuple[str, str]] = []
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT session_id, step_id, status FROM steps WHERE status IN"
                " ('claimed', 'provider_started')"
            ).fetchall()
            for row in rows:
                key = (row["session_id"], row["step_id"])
                if row["status"] == StepStatus.CLAIMED:
                    _set_status(connection, key, "released")
                    released.append(key)
                    _audit(connection, key[0], key[1], "recovered_released", {})
                else:
                    _set_status(connection, key, "ambiguous")
                    ambiguous.append(key)
                    _audit(connection, key[0], key[1], "recovered_ambiguous", {})
        logger.info("session recovery released=%d ambiguous=%d", len(released), len(ambiguous))
        return RecoveryReport(released=tuple(released), ambiguous=tuple(ambiguous))

    def set_draining(self, session_id: str, draining: bool) -> SessionRecord:
        """Start or stop refusing new steps for a session.

        Args:
            session_id: Session to change.
            draining: True to refuse new steps.

        Returns:
            The updated record.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET draining = ? WHERE session_id = ?",
                (int(draining), session_id),
            )
            if cursor.rowcount == 0:
                raise SessionNotFoundError(f"session {session_id!r} does not exist")
            _audit(connection, session_id, None, "draining_set", {"draining": draining})
        return self.read_session(session_id)

    def open_steps(self, session_id: str) -> tuple[str, ...]:
        """Return step IDs that are open or ambiguous for one session.

        Args:
            session_id: Session to inspect.

        Returns:
            Sorted step IDs.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT step_id FROM steps WHERE session_id = ?"
                f" AND status IN ({_placeholders(_OPEN_STATUSES)}) ORDER BY step_id",
                (session_id, *_OPEN_STATUSES),
            ).fetchall()
        return tuple(row["step_id"] for row in rows)


def _placeholders(values: tuple[str, ...]) -> str:
    """Return comma-separated SQL placeholders, one per value."""
    return ", ".join("?" for _ in values)


def _set_status(connection: sqlite3.Connection, key: tuple[str, str], status: str) -> None:
    """Set one step's status inside the caller's transaction."""
    connection.execute(
        "UPDATE steps SET status = ? WHERE session_id = ? AND step_id = ?",
        (status, key[0], key[1]),
    )


def _replay_or_refuse(
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
                response=_object(prior["response_json"]),
                resulting_version=prior["resulting_version"],
            ),
        )
    if status is StepStatus.AMBIGUOUS:
        raise StepAmbiguousError(f"step {step_id!r} may have reached the provider; review it")
    if status is StepStatus.REJECTED:
        reason = _object(prior["response_json"]).get("reason", "")
        raise StepRejectedError(f"step {step_id!r} was rejected: {reason}")
    if status is StepStatus.RELEASED:
        raise StepConflictError(f"step {step_id!r} was released; use a new step ID")
    raise StepInFlightError(f"step {step_id!r} in session {session_id!r} is still open")


def _audit(
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


def _require_open_claim(
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
