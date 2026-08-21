"""Group-commit write-behind ledger batching durable writes on one writer thread.

Concurrent request-path ledger writes are funneled through a single dedicated
writer thread that drains queued operations into one SQLite transaction per
batch. Every caller awaits its own operation's durable commit before
proceeding, so acceptance, budget reservation, and terminal settlement keep
their exact fail-closed semantics while the per-request fsync cost is
amortized across all operations sharing a batch. There is no flush window:
no caller observes success before its write is durable on disk.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AttemptId,
    AuthorizationSnapshot,
    ExecutionSnapshot,
    GatewayEvent,
    GatewayFailure,
)
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.sqlite.migrations import connect_database

_logger = logging.getLogger(__name__)

_DEFAULT_MAX_BATCH_SIZE = 128

_T = TypeVar("_T")


@dataclass(frozen=True)
class _PendingWrite:
    """One queued ledger operation and the future resolved after durable commit."""

    apply: Callable[[sqlite3.Connection], object]
    future: concurrent.futures.Future[object]


class GroupCommitAttemptLedger:
    """Async attempt ledger committing queued writes in shared durable batches.

    Each public method enqueues one operation and resolves only after the
    writer thread has committed the batch containing it, so callers keep
    write-through durability while concurrent requests share one fsync.
    A failed operation rolls back to its own savepoint without disturbing
    the surrounding batch, and its original exception is re-raised to the
    awaiting caller.
    """

    def __init__(
        self,
        core: SQLiteAttemptLedger,
        *,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    ) -> None:
        """Start the single writer thread over an existing synchronous ledger.

        Args:
            core: Synchronous SQLite ledger owning schema and write logic.
            max_batch_size: Maximum queued operations drained into one commit.
        """
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least one")
        self.core = core
        self._max_batch_size = max_batch_size
        self._queue: queue.SimpleQueue[_PendingWrite | None] = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="gateway-ledger-writer",
            daemon=True,
        )
        self._thread.start()

    async def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Durably persist accepted authority before route selection or dispatch.

        Args:
            authorization: Frozen authority and request identity.
        """
        await self._submit(
            lambda connection: self.core.apply_accept_request(
                connection, authorization=authorization
            )
        )

    async def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
    ) -> AttemptId:
        """Durably reserve budget and record one dispatch before provider work.

        Args:
            snapshot: Route-bound immutable request plan.
            deployment: Exact deployment about to receive the request.
            attempt_ordinal: Zero-based physical dispatch position for this request.
            route_depth: Zero-based operational route position.
            maximum_cost_micro_usd: Conservative charge reserved before dispatch.
            route_reason: Optional learned-selection reason code.
            fallback_reason: Optional embedding or router fallback reason code.

        Returns:
            Stable new attempt ID.
        """
        return await self._submit(
            lambda connection: self.core.apply_start_attempt(
                connection,
                snapshot=snapshot,
                deployment=deployment,
                attempt_ordinal=attempt_ordinal,
                route_depth=route_depth,
                maximum_cost_micro_usd=maximum_cost_micro_usd,
                route_reason=route_reason,
                fallback_reason=fallback_reason,
            )
        )

    async def finish_attempt(
        self,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
    ) -> None:
        """Durably settle one attempt with normalized content-free fields.

        Args:
            attempt_id: Stable attempt ID.
            terminal_event: Provider terminal event, possibly carrying usage.
            failure: Sanitized failure when no successful terminal event exists.
            finalize_request: Whether this attempt is the final route for its parent request.
        """
        await self._submit(
            lambda connection: self.core.apply_finish_attempt(
                connection,
                attempt_id=attempt_id,
                terminal_event=terminal_event,
                failure=failure,
                finalize_request=finalize_request,
            )
        )

    async def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Durably terminalize accepted work that never reached dispatch.

        Args:
            authorization: Frozen authority identifying the accepted request.
            failure: Sanitized pre-dispatch terminal failure.
        """
        await self._submit(
            lambda connection: self.core.apply_finish_request(
                connection, authorization=authorization, failure=failure
            )
        )

    async def flush(self) -> None:
        """Resolve after every previously enqueued operation is durably committed."""
        await self._submit(lambda connection: None)

    def close(self) -> None:
        """Stop the writer thread after draining every queued operation."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=30)

    def _run(self) -> None:
        """Drain queued operations into one durable SQLite transaction per batch."""
        connection = connect_database(
            self.core.database_path,
            busy_timeout_ms=self.core.busy_timeout_ms,
        )
        try:
            stopping = False
            while not stopping:
                item = self._queue.get()
                if item is None:
                    break
                batch = [item]
                while len(batch) < self._max_batch_size:
                    try:
                        extra = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if extra is None:
                        stopping = True
                        break
                    batch.append(extra)
                self._commit_batch(connection, batch)
        finally:
            connection.close()

    @staticmethod
    def _commit_batch(
        connection: sqlite3.Connection,
        batch: list[_PendingWrite],
    ) -> None:
        """Apply each queued operation under its own savepoint and commit once.

        A failing operation rolls back to its savepoint so its effects never
        commit, while sibling operations in the same batch stay intact. Only
        after the shared COMMIT succeeds are successful futures resolved, so a
        caller never observes success for a write that is not durable.

        Args:
            connection: Writer-owned configured SQLite connection.
            batch: Queued operations resolved after this shared commit.
        """
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            for pending in batch:
                pending.future.set_exception(exc)
            return
        outcomes: list[tuple[_PendingWrite, object, BaseException | None]] = []
        for index, pending in enumerate(batch):
            savepoint = f"gateway_op_{index}"
            connection.execute(f"SAVEPOINT {savepoint}")
            try:
                value = pending.apply(connection)
            except Exception as exc:  # noqa: BLE001 - the caller re-raises its own failure.
                connection.execute(f"ROLLBACK TO {savepoint}")
                connection.execute(f"RELEASE {savepoint}")
                outcomes.append((pending, None, exc))
            else:
                connection.execute(f"RELEASE {savepoint}")
                outcomes.append((pending, value, None))
        try:
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                _logger.exception("gateway ledger batch rollback failed after commit failure")
            for pending in batch:
                pending.future.set_exception(exc)
            return
        for pending, value, error in outcomes:
            if error is not None:
                pending.future.set_exception(error)
            else:
                pending.future.set_result(value)

    async def _submit(self, apply: Callable[[sqlite3.Connection], _T]) -> _T:
        """Enqueue one operation and await its durable batch commit.

        Args:
            apply: Operation run on the writer connection inside the batch transaction.

        Returns:
            The operation's return value after its batch has committed.
        """
        if self._closed:
            raise RuntimeError("gateway ledger writer is closed")
        future: concurrent.futures.Future[_T] = concurrent.futures.Future()
        self._queue.put(
            _PendingWrite(
                apply=apply,
                future=cast("concurrent.futures.Future[object]", future),
            )
        )
        return await asyncio.wrap_future(future)
