"""Typed SQLite transaction boundary for immutable alias activation."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from wmo.common.core.artifacts import Sha256
from wmo.runtime.gateway.contracts import DirectTarget, GatewayTarget, ProjectTarget

ConnectionFactory = Callable[[], AbstractContextManager[sqlite3.Connection]]


class AliasActivationOutcomeUnknownError(ValueError):
    """An alias activation could not prove whether its COMMIT landed."""

    def __init__(self, *, alias_id: str, revision_id: str) -> None:
        """Create a content-free recovery error for one immutable alias revision.

        Args:
            alias_id: Stable public alias identifier.
            revision_id: Immutable revision whose commit outcome is unknown.
        """
        self.alias_id = alias_id
        self.revision_id = revision_id
        super().__init__(
            "operation_outcome_unknown: inspect alias "
            f"{alias_id!r} revision {revision_id!r} before retrying"
        )


@contextmanager
def alias_activation_transaction(
    *,
    connect: ConnectionFactory,
    organization_id: str,
    alias_id: str,
    alias_name: str,
    revision_id: str,
    target: GatewayTarget,
    snapshot_ref: str,
    catalog_sha256: Sha256,
    refusal_failover: bool,
) -> Iterator[sqlite3.Connection]:
    """Run activation while typing only ambiguous COMMIT acknowledgements.

    Args:
        connect: Factory for configured SQLite connection contexts.
        organization_id: Owning tenant.
        alias_id: Stable alias resource identifier.
        alias_name: Public model name.
        revision_id: Immutable revision identifier.
        target: Direct pool or project target.
        snapshot_ref: Registered catalog snapshot reference.
        catalog_sha256: Exact normalized catalog digest.
        refusal_failover: Whether typed precommit refusals may advance.

    Yields:
        Open immediate transaction for the activation body.

    Raises:
        AliasActivationOutcomeUnknownError: COMMIT failed and a fresh read was inconclusive.
        ValueError: COMMIT was proven not to contain the desired activation.
    """
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        try:
            connection.execute("COMMIT")
        except BaseException as commit_error:  # noqa: BLE001 - COMMIT may land before interruption
            try:
                outcome = reconcile_alias_activation(
                    connect=connect,
                    organization_id=organization_id,
                    alias_id=alias_id,
                    alias_name=alias_name,
                    revision_id=revision_id,
                    target=target,
                    snapshot_ref=snapshot_ref,
                    catalog_sha256=catalog_sha256,
                    refusal_failover=refusal_failover,
                )
            except BaseException as reconciliation_error:  # noqa: BLE001 - never undo possible COMMIT
                raise AliasActivationOutcomeUnknownError(
                    alias_id=alias_id,
                    revision_id=revision_id,
                ) from reconciliation_error
            if outcome is True:
                return
            if outcome is False:
                raise ValueError("alias activation did not commit") from commit_error
            raise AliasActivationOutcomeUnknownError(
                alias_id=alias_id,
                revision_id=revision_id,
            ) from commit_error


def reconcile_alias_activation(
    *,
    connect: ConnectionFactory,
    organization_id: str,
    alias_id: str,
    alias_name: str,
    revision_id: str,
    target: GatewayTarget,
    snapshot_ref: str,
    catalog_sha256: Sha256,
    refusal_failover: bool,
) -> bool | None:
    """Return whether the exact activation committed, or ``None`` if unreadable.

    Args:
        connect: Factory for a fresh configured SQLite connection context.
        organization_id: Owning tenant.
        alias_id: Stable alias resource identifier.
        alias_name: Public model name.
        revision_id: Immutable revision identifier.
        target: Direct pool or project target.
        snapshot_ref: Registered catalog snapshot reference.
        catalog_sha256: Exact normalized catalog digest.
        refusal_failover: Whether typed precommit refusals may advance.

    Returns:
        ``True`` for the exact active revision, ``False`` for a proven absence or mismatch, and
        ``None`` when a fresh SQLite read is unavailable.
    """
    try:
        with connect() as connection:
            row = connection.execute(
                """
                SELECT a.alias_name, r.target_kind, r.pool_id, r.project_ref, r.activation_ref,
                       r.snapshot_ref, r.catalog_sha256, r.refusal_failover
                FROM alias_revisions AS r
                JOIN gateway_aliases AS a
                  ON a.organization_id = r.organization_id AND a.alias_id = r.alias_id
                WHERE r.organization_id = ? AND r.alias_id = ? AND r.revision_id = ?
                """,
                (organization_id, alias_id, revision_id),
            ).fetchone()
    except Exception:  # noqa: BLE001 - every ordinary read failure leaves COMMIT ambiguous
        return None
    if row is None:
        return False
    expected = (
        alias_name,
        target.kind,
        target.pool_id if isinstance(target, DirectTarget) else None,
        target.project_ref if isinstance(target, ProjectTarget) else None,
        target.activation_ref if isinstance(target, ProjectTarget) else None,
        snapshot_ref,
        catalog_sha256,
        int(refusal_failover),
    )
    return tuple(row[index] for index in range(8)) == expected
