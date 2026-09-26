"""Ledger-side preparation of per-operation SQLite model-chain authority."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.interfaces import GatewayClock
from exp.runtime.gateway.model_chain_authority import (
    ChainOperation,
    SnapshotClassificationMemo,
    SQLiteChainAuthorityObservation,
    SQLiteChainPreflight,
    SQLiteChainWitness,
    prepare_sqlite_chain_authority,
)


@contextmanager
def prepare_ledger_chain_authority(
    authorization: AuthorizationSnapshot,
    operation: ChainOperation,
    *,
    connect: Callable[[], AbstractContextManager[sqlite3.Connection]],
    clock: GatewayClock,
    serving_snapshot_max_bytes: int,
    classification_memo: SnapshotClassificationMemo,
    connection: sqlite3.Connection | None = None,
    observation: SQLiteChainAuthorityObservation | None = None,
) -> Iterator[SQLiteChainPreflight | None]:
    """Classify before a write transaction from a fresh read or captured rows.

    Args:
        authorization: Frozen request authority and its absolute deadline.
        operation: Ledger operation covered by the live proof.
        connect: Ledger connection factory.
        clock: Ledger clock used to compute remaining time.
        serving_snapshot_max_bytes: Serving limit for each catalog file.
        classification_memo: Bounded parser memo owned by the gateway composition.
        connection: Optional open connection used for the initial authority observation.
        observation: Optional exact rows previously observed on the writer thread.

    Yields:
        An open operation-bound preflight retained until its consumer finishes.
    """
    witness = authorization._local_sqlite_chain_witness
    if (
        observation is None
        and isinstance(witness, SQLiteChainWitness)
        and witness.maximum_bytes == serving_snapshot_max_bytes
    ):
        with prepare_sqlite_chain_authority(
            None,
            authorization.organization_id,
            authorization.alias_revision_id,
            request_id=authorization.request_id,
            operation=operation,
            maximum_bytes=serving_snapshot_max_bytes,
            remaining_seconds=authorization.deadline_monotonic - clock.monotonic(),
            classification_memo=classification_memo,
            witness=witness,
        ) as proof:
            yield proof
        return
    if observation is not None:
        with prepare_sqlite_chain_authority(
            None,
            authorization.organization_id,
            authorization.alias_revision_id,
            request_id=authorization.request_id,
            operation=operation,
            maximum_bytes=serving_snapshot_max_bytes,
            remaining_seconds=authorization.deadline_monotonic - clock.monotonic(),
            classification_memo=classification_memo,
            observation=observation,
        ) as proof:
            yield proof
        return
    with (
        connect() if connection is None else nullcontext(connection) as connection,
        prepare_sqlite_chain_authority(
            connection,
            authorization.organization_id,
            authorization.alias_revision_id,
            request_id=authorization.request_id,
            operation=operation,
            maximum_bytes=serving_snapshot_max_bytes,
            remaining_seconds=authorization.deadline_monotonic - clock.monotonic(),
            classification_memo=classification_memo,
        ) as proof,
    ):
        yield proof
