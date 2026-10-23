"""Durable SQLite session store: immutable binding, step idempotency, incremental state.

A session pins one immutable ``SessionBinding`` (target, receipt, knowledge, corpus, and
representation digests) and one candidate state object. Only the state and version change
during a session. A step is one caller operation keyed by ``step_id``:

1. ``begin_step`` claims the step against an expected version, or replays the stored
   terminal outcome of the same request.
2. ``mark_provider_started`` records that a provider call may run. A crash after this point
   makes the step ambiguous rather than releasable.
3. ``complete_step`` publishes the step's structural state edits, its event, the terminal
   response, and an audit row in one transaction. ``reject_step`` closes a step whose reply was
   refused, with no state change.

State is stored incrementally (see ``projection``): each revision is an immutable event plus a
root of content-addressed nodes, so a commit writes only the containers its edits change, and
unchanged subtrees are shared across revisions. Callers hold revision-bound ``StateRef`` values
and materialize JSON explicitly with ``read_state`` or ``read_session``. Historical revisions
stay readable while a retention pin owns them; ``collect_garbage`` reclaims the rest.

One connection per thread and database is reused across operations, so a step pays one durable
WAL sync per transaction rather than a connection setup and a checkpoint on every close. Every
transaction still commits with ``synchronous = FULL`` before the call returns.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject, Sha256, sha256_json
from exp.common.sqlite.connection import persistent_connection
from exp.runtime.sessions.edits import StateEdit
from exp.runtime.sessions.errors import (
    ClaimLostError,
    ProviderBudgetExhaustedError,
    SessionBindingMismatchError,
    SessionDrainingError,
    SessionExistsError,
    SessionNotFoundError,
    SessionStoreError,
    StateReferenceError,
    StateUnavailableError,
    StepConflictError,
    StepDeadlineExceededError,
    StepInFlightError,
    VersionConflictError,
)
from exp.runtime.sessions.nodes import EncodedNode, declared_text, encode
from exp.runtime.sessions.projection import (
    StateRef,
    commit_edits,
    edits_effect,
    event_root,
    genesis_effect,
    insert_nodes,
    prepare_edits,
    read_value,
    record_event,
    replay_roots,
)
from exp.runtime.sessions.records import (
    OPEN_STATUSES,
    RecoveryReport,
    RequestPreflightReceipt,
    SessionBinding,
    SessionHead,
    SessionRecord,
    StepClaim,
    StepDecision,
    StepStatus,
)
from exp.runtime.sessions.retention import (
    CollectionPreview,
    CollectionReport,
    RetentionPin,
    collect,
    pin_revisions,
    preview_collection,
    registered_pins,
    release_pin,
)
from exp.runtime.sessions.schema import open_schema, refuse_stale_schema
from exp.runtime.sessions.steps import (
    audit,
    identity_text,
    placeholders,
    replay_or_refuse,
    require_open_claim,
    set_status,
    stored_object,
    utc_now,
)

logger = logging.getLogger(__name__)


class SessionStore:
    """Transactional single-file session store.

    Each mutation runs in one ``BEGIN IMMEDIATE`` transaction, so concurrent writers serialize
    and version checks see the state the write will change. Each read runs in one snapshot.

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
            SessionStoreError: If the database has another schema version or codec.
        """
        self._clock = clock or utc_now
        self._path = path
        refuse_stale_schema(path)
        with self._connect() as connection:
            self.store_id = open_schema(connection, path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield this thread's reusable configured connection to the store.

        Yields:
            Connection with foreign keys, WAL, full commit sync, and a bounded busy timeout.
        """
        with persistent_connection(self._path) as connection:
            yield connection

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
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    @contextmanager
    def _snapshot(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a read transaction, so every query sees one revision.

        Yields:
            Connection whose reads share one consistent database snapshot.
        """
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                yield connection
            finally:
                connection.execute("COMMIT")

    def create_session(
        self, session_id: str, binding: SessionBinding, initial_state: JsonObject
    ) -> SessionHead:
        """Create a session at version 0 with its immutable binding and recorded genesis.

        Args:
            session_id: New stable identifier.
            binding: Evidence identity fixed for the session's lifetime.
            initial_state: Starting candidate state, recorded as the replayable genesis.

        Returns:
            The created head.

        Raises:
            SessionExistsError: If the identifier is already used.
        """
        nodes: dict[bytes, EncodedNode] = {}
        root = encode(initial_state, nodes)
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone():
                raise SessionExistsError(f"session {session_id!r} already exists")
            insert_nodes(connection, nodes)
            connection.execute(
                "INSERT INTO sessions (session_id, binding_json, version, root, draining)"
                " VALUES (?, ?, 0, ?, 0)",
                (session_id, identity_text(binding.model_dump(mode="json")), root),
            )
            record_event(connection, session_id, 0, None, genesis_effect(initial_state), root)
            audit(
                connection,
                session_id,
                None,
                "session_created",
                {"binding_sha256": sha256_json(binding)},
            )
        return self.read_head(session_id)

    def read_head(self, session_id: str) -> SessionHead:
        """Return the binding, version and current state reference, without reading the state.

        Args:
            session_id: Session identifier.

        Returns:
            The current head.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        with self._connect() as connection:
            row = _head_row(connection, session_id)
        return self._head(row)

    def read_session(self, session_id: str) -> SessionRecord:
        """Return the binding, complete committed state, and version of one session.

        This materializes the whole state; callers that need part of it use ``read_head`` and
        ``read_state`` with a path.

        Args:
            session_id: Session identifier.

        Returns:
            The current record.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        with self._snapshot() as connection:
            row = _head_row(connection, session_id)
            state = read_value(connection, bytes(row["root"]), ())
        if not isinstance(state, dict):
            raise SessionStoreError("stored session state is not an object")
        return SessionRecord(
            session_id=row["session_id"],
            binding=SessionBinding.model_validate_json(row["binding_json"]),
            version=row["version"],
            state=state,
            draining=bool(row["draining"]),
        )

    def read_state(self, ref: StateRef, path: Sequence[str | int] = ()) -> JsonValue:
        """Materialize one revision's state, or the value at ``path`` inside it, as fresh JSON.

        Only the containers on ``path`` and the selected subtree are read. The reference must
        come from this store, and the revision must still be materialized: the head always is,
        and older revisions are while a retention pin owns them.

        Args:
            ref: Revision-bound state reference issued by this store.
            path: Object keys and array indexes inside the state; empty for the whole state.

        Returns:
            A detached JSON value with its declared key order and exact scalars.

        Raises:
            StateReferenceError: If the reference is foreign or names another root.
            StateUnavailableError: If the revision was reclaimed.
            KeyError: If ``path`` does not exist in the revision.
        """
        if ref.store_id != self.store_id:
            raise StateReferenceError("state reference belongs to another session store")
        with self._snapshot() as connection:
            root = event_root(connection, ref.session_id, ref.revision)
            if root is None or root.hex() != ref.root_sha256:
                raise StateReferenceError(
                    f"session {ref.session_id!r} has no revision {ref.revision} with that root"
                )
            return read_value(connection, root, tuple(path))

    def _head(self, row: sqlite3.Row) -> SessionHead:
        """Build a head from one ``sessions`` row."""
        return SessionHead(
            session_id=row["session_id"],
            binding=SessionBinding.model_validate_json(row["binding_json"]),
            version=row["version"],
            state=StateRef(
                store_id=self.store_id,
                session_id=row["session_id"],
                revision=row["version"],
                root_sha256=bytes(row["root"]).hex(),
            ),
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
                return replay_or_refuse(session_id, step_id, request_sha256, prior)
            if bool(session["draining"]):
                raise SessionDrainingError(f"session {session_id!r} is draining")
            open_step = connection.execute(
                "SELECT step_id FROM steps WHERE session_id = ?"
                f" AND status IN ({placeholders(OPEN_STATUSES)})",
                (session_id, *OPEN_STATUSES),
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
            audit(connection, session_id, step_id, "step_claimed", {"request": request_sha256})
        return StepDecision(kind="claimed", claim=claim)

    def reclaim_step(
        self, session_id: str, step_id: str, request_sha256: Sha256, binding: SessionBinding
    ) -> tuple[StepClaim, StepStatus]:
        """Take over a step a stopped process left open, for an explicit resumption only.

        A claim that never reached the provider reopens as claimed. One that may have reached
        it stays ``provider_started`` (its dispatch is already counted) and is never released:
        the caller replays its saved output or re-dispatches it as an unknown outcome.

        Args:
            session_id: Session of the step.
            step_id: Interrupted step.
            request_sha256: The caller's request digest; it must equal the saved one.
            binding: The caller's binding; it must equal the session's.

        Returns:
            A fresh claim on the step, and the status the step was left in.

        Raises:
            SessionNotFoundError: If the session does not exist.
            SessionBindingMismatchError: If the binding differs.
            StepConflictError: If the step is not open or was opened for another request.
            VersionConflictError: If the session advanced past the step's base version.
        """
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT version, binding_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise SessionNotFoundError(f"session {session_id!r} does not exist")
            if SessionBinding.model_validate_json(session["binding_json"]) != binding:
                raise SessionBindingMismatchError(f"binding differs from session {session_id!r}")
            row = connection.execute(
                "SELECT request_sha256, status, base_version FROM steps"
                " WHERE session_id = ? AND step_id = ?",
                (session_id, step_id),
            ).fetchone()
            if row is None or row["status"] not in OPEN_STATUSES:
                raise StepConflictError(f"step {step_id!r} is not an interrupted open step")
            if row["request_sha256"] != request_sha256:
                raise StepConflictError(f"step {step_id!r} was opened for a different request")
            if session["version"] != row["base_version"]:
                raise VersionConflictError(f"step {step_id!r} is based on a stale version")
            left = StepStatus(row["status"])
            status = "claimed" if left is StepStatus.CLAIMED else "provider_started"
            claim = StepClaim(
                session_id=session_id,
                step_id=step_id,
                claim_token=uuid4().hex,
                base_version=row["base_version"],
            )
            connection.execute(
                "UPDATE steps SET status = ?, claim_token = ? WHERE session_id = ? AND step_id = ?",
                (status, claim.claim_token, session_id, step_id),
            )
            audit(connection, session_id, step_id, "step_reclaimed", {"from": left.value})
        return claim, left

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
            require_open_claim(connection, claim, expected_status=StepStatus.CLAIMED)
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
            audit(connection, claim.session_id, claim.step_id, "request_preflight", payload)

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
            require_open_claim(connection, claim, expected_status=StepStatus.CLAIMED)
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
            audit(connection, claim.session_id, claim.step_id, "provider_started", {})

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
            require_open_claim(connection, claim, expected_status=StepStatus.PROVIDER_STARTED)
            connection.execute(
                "UPDATE steps SET proposal_json = ? WHERE session_id = ? AND step_id = ?",
                (declared_text(proposal), claim.session_id, claim.step_id),
            )
            audit(connection, claim.session_id, claim.step_id, "proposal_recorded", proposal)

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
        return stored_object(row["proposal_json"])

    def complete_step(
        self,
        claim: StepClaim,
        edits: Sequence[StateEdit],
        response: JsonObject,
        *,
        window: datetime | None = None,
    ) -> SessionHead:
        """Publish one state revision or durably reject a late proposal.

        The edits, the event recording them, the changed nodes, the new head and the replay
        response commit in one transaction. Edits that do not apply publish nothing and leave
        the step open.

        Args:
            claim: Current provider-started owner of the logical operation.
            edits: Ordered structural edits of the fully validated proposal; their container
                values are encoded before locking.
            response: Ordered observations to replay with the committed state.
            window: Deadline of an explicitly resumed execution window; it extends, never
                shortens, the binding's commit deadline. The binding itself is unchanged.

        Returns:
            Head at the newly committed revision.

        Raises:
            ClaimLostError: Ownership or the saved proposal is missing.
            VersionConflictError: The proposal was based on a stale revision.
            StepDeadlineExceededError: Deadline passed; proposal and rejection remain durable.
            StateEditError: An edit does not apply to the base revision.
        """
        nodes: dict[bytes, EncodedNode] = {}
        prepared = prepare_edits(edits, nodes)
        effect, response_text = edits_effect(prepared.edits), declared_text(response)
        expired = False
        with self._transaction() as connection:
            require_open_claim(connection, claim, expected_status=StepStatus.PROVIDER_STARTED)
            proposal_row = connection.execute(
                "SELECT proposal_json FROM steps WHERE session_id = ? AND step_id = ?",
                (claim.session_id, claim.step_id),
            ).fetchone()
            if proposal_row["proposal_json"] is None:
                raise ClaimLostError("complete requires a recorded proposal first")
            current = connection.execute(
                "SELECT version, binding_json, root FROM sessions WHERE session_id = ?",
                (claim.session_id,),
            ).fetchone()
            if current["version"] != claim.base_version:
                raise VersionConflictError("proposal is based on a stale session revision")
            binding = SessionBinding.model_validate_json(current["binding_json"])
            now = self._clock()
            if now.tzinfo is None:
                raise ValueError("commit clock must return an aware datetime")
            expired = now >= max(binding.deadline, window or binding.deadline)
            if expired:
                connection.execute(
                    "UPDATE steps SET status = 'rejected', response_json = ? "
                    "WHERE session_id = ? AND step_id = ?",
                    (
                        declared_text({"reason": "deadline_stopped"}),
                        claim.session_id,
                        claim.step_id,
                    ),
                )
                audit(connection, claim.session_id, claim.step_id, "deadline_stopped", {})
            else:
                next_version = claim.base_version + 1
                root, changed = commit_edits(connection, bytes(current["root"]), prepared, nodes)
                insert_nodes(connection, changed)
                record_event(
                    connection, claim.session_id, next_version, claim.step_id, effect, root
                )
                connection.execute(
                    "UPDATE sessions SET version = ?, root = ? WHERE session_id = ?",
                    (next_version, root, claim.session_id),
                )
                connection.execute(
                    "UPDATE steps SET status = 'completed', response_json = ?, "
                    "resulting_version = ? WHERE session_id = ? AND step_id = ?",
                    (response_text, next_version, claim.session_id, claim.step_id),
                )
                audit(
                    connection,
                    claim.session_id,
                    claim.step_id,
                    "step_completed",
                    {"v": next_version},
                )
        if expired:
            raise StepDeadlineExceededError("proposal retained; session deadline forbids commit")
        return self.read_head(claim.session_id)

    def reject_step(self, claim: StepClaim, reason: str) -> None:
        """Close an open step with no state change because its reply was refused.

        Args:
            claim: The open claim whose reply failed validation or commit checks.
            reason: Short, content-free explanation stored with the rejection.

        Raises:
            ClaimLostError: If the claim is not open for its step.
        """
        with self._transaction() as connection:
            require_open_claim(connection, claim, expected_status=None)
            connection.execute(
                "UPDATE steps SET status = 'rejected', response_json = ? WHERE session_id = ?"
                " AND step_id = ?",
                (declared_text({"reason": reason}), claim.session_id, claim.step_id),
            )
            audit(connection, claim.session_id, claim.step_id, "step_rejected", {})

    def mark_ambiguous(self, claim: StepClaim, reason: str) -> None:
        """Record an unresolved dispatched outcome without releasing or repurchasing it.

        Args:
            claim: Claim whose completion callback may have reached the provider.
            reason: Content-free exception classification, never credentials or raw error text.
        """
        with self._transaction() as connection:
            require_open_claim(connection, claim, expected_status=StepStatus.PROVIDER_STARTED)
            set_status(connection, (claim.session_id, claim.step_id), "ambiguous")
            audit(
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
            require_open_claim(connection, claim, expected_status=StepStatus.CLAIMED)
            connection.execute(
                "UPDATE steps SET status = 'released' WHERE session_id = ? AND step_id = ?",
                (claim.session_id, claim.step_id),
            )
            audit(connection, claim.session_id, claim.step_id, "step_released", {})

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
                    set_status(connection, key, "released")
                    released.append(key)
                    audit(connection, key[0], key[1], "recovered_released", {})
                else:
                    set_status(connection, key, "ambiguous")
                    ambiguous.append(key)
                    audit(connection, key[0], key[1], "recovered_ambiguous", {})
        logger.info("session recovery released=%d ambiguous=%d", len(released), len(ambiguous))
        return RecoveryReport(released=tuple(released), ambiguous=tuple(ambiguous))

    def set_draining(self, session_id: str, draining: bool) -> SessionHead:
        """Start or stop refusing new steps for a session.

        Args:
            session_id: Session to change.
            draining: True to refuse new steps.

        Returns:
            The updated head.

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
            audit(connection, session_id, None, "draining_set", {"draining": draining})
        return self.read_head(session_id)

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
                f" AND status IN ({placeholders(OPEN_STATUSES)}) ORDER BY step_id",
                (session_id, *OPEN_STATUSES),
            ).fetchall()
        return tuple(row["step_id"] for row in rows)

    def pin_revisions(
        self, owner_id: str, session_id: str, start_revision: int, end_revision: int
    ) -> RetentionPin:
        """Durably retain revisions ``[start_revision, end_revision)`` for one named owner.

        Register the pin before exposing a historical revision or beginning to read it. It lasts
        until ``release_pin``; a stopped process does not release it.

        Args:
            owner_id: Stable owner, such as a checkpoint, fork or frozen evaluation identity.
            session_id: Session whose history is retained.
            start_revision: First retained revision.
            end_revision: First revision after the range; at most the head version plus one.

        Returns:
            The registered pin.

        Raises:
            SessionNotFoundError: If the session does not exist.
            RetentionConflictError: If the owner pins another range or the range is invalid.
            StateUnavailableError: If a revision in the range was already reclaimed.
        """
        pin = RetentionPin(
            owner_id=owner_id,
            session_id=session_id,
            start_revision=start_revision,
            end_revision=end_revision,
        )
        with self._transaction() as connection:
            return pin_revisions(connection, pin)

    def release_pin(self, owner_id: str, session_id: str) -> bool:
        """Release one owner's pin; other owners and the head keep what they retain.

        Args:
            owner_id: Owner that registered the pin.
            session_id: Pinned session.

        Returns:
            Whether a pin was released.
        """
        with self._transaction() as connection:
            return release_pin(connection, owner_id, session_id)

    def ref_at(self, session_id: str, revision: int) -> StateRef:
        """Return the reference to one committed revision, whether or not it is still retained.

        Args:
            session_id: Session identifier.
            revision: Committed revision; 0 is genesis.

        Returns:
            The revision-bound reference.

        Raises:
            StateReferenceError: If the session has no such revision.
        """
        with self._connect() as connection:
            root = event_root(connection, session_id, revision)
        if root is None:
            raise StateReferenceError(f"session {session_id!r} has no revision {revision}")
        return StateRef(
            store_id=self.store_id,
            session_id=session_id,
            revision=revision,
            root_sha256=root.hex(),
        )

    def collect_garbage(
        self, *, batch_size: int = 500, maximum_batches: int | None = None
    ) -> CollectionReport:
        """Reclaim state nodes that no session head and no retention pin reaches.

        Events are never reclaimed. An interrupted or bounded run resumes from its durable
        cursor on the next call.

        Args:
            batch_size: Node rows examined per write transaction.
            maximum_batches: Stop after this many batches; None runs to completion.

        Returns:
            Reclaimed and retained counts for this run.
        """
        if batch_size < 1:
            raise ValueError("collection batches examine at least one node")
        with self._connect() as connection:
            report = collect(
                connection,
                self._transaction,
                batch_size=batch_size,
                maximum_batches=maximum_batches,
            )
        logger.info(
            "session state collection reclaimed=%d bytes=%d complete=%s",
            report.reclaimed_nodes,
            report.reclaimed_bytes,
            report.complete,
        )
        return report

    def pins(self) -> tuple[RetentionPin, ...]:
        """Return every registered retention pin, ordered by owner and session."""
        with self._snapshot() as connection:
            return registered_pins(connection)

    def preview_collection(self) -> CollectionPreview:
        """Report what ``collect_garbage`` would reclaim and what each pin holds; write nothing.

        Returns:
            Reclaimable and retained counts and every pin's exclusive holding, from one snapshot.
        """
        with self._snapshot() as connection:
            return preview_collection(connection)

    def vacuum(self) -> int:
        """Rewrite the database file so pages freed by collection return to the file system.

        SQLite runs the rewrite as one transaction, so an interrupted vacuum leaves the store
        unchanged; it waits for, and then excludes, concurrent writers.

        Returns:
            Bytes by which the database and its write-ahead log shrank.
        """
        before = _file_bytes(self._path)
        with self._connect() as connection:
            connection.execute("VACUUM")
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return max(0, before - _file_bytes(self._path))

    def verify_history(self, session_id: str) -> tuple[StateRef, ...]:
        """Rebuild every revision from the recorded genesis and edits alone, and check it.

        The replay uses no stored state nodes and calls no model, tool or clock, so it proves
        that the event log reproduces each published root.

        Args:
            session_id: Session to verify.

        Returns:
            One reference per verified revision, from genesis to the head.

        Raises:
            SessionNotFoundError: If the session does not exist.
            StateUnavailableError: If a replayed root differs from the recorded one.
        """
        with self._snapshot() as connection:
            head = _head_row(connection, session_id)
            recorded = [
                bytes(row["root"])
                for row in connection.execute(
                    "SELECT root FROM state_events WHERE session_id = ? ORDER BY revision",
                    (session_id,),
                )
            ]
            replayed = replay_roots(connection, session_id)
        if (
            replayed != recorded
            or len(recorded) != head["version"] + 1
            or recorded[-1] != bytes(head["root"])
        ):
            raise StateUnavailableError(f"session {session_id!r} event log does not reproduce it")
        return tuple(
            StateRef(
                store_id=self.store_id,
                session_id=session_id,
                revision=revision,
                root_sha256=root.hex(),
            )
            for revision, root in enumerate(recorded)
        )


def _file_bytes(path: Path) -> int:
    """Return the size of a database file and its write-ahead log."""
    wal = path.with_name(path.name + "-wal")
    return path.stat().st_size + (wal.stat().st_size if wal.exists() else 0)


def _head_row(connection: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    """Return one ``sessions`` row.

    Raises:
        SessionNotFoundError: If the session does not exist.
    """
    row = connection.execute(
        "SELECT session_id, binding_json, version, root, draining FROM sessions"
        " WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise SessionNotFoundError(f"session {session_id!r} does not exist")
    return row
