"""Atomic SQLite authority mutations used by interactive gateway setup."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from typing import TYPE_CHECKING, cast

from exp.common.core.artifacts import Sha256
from exp.runtime.gateway.auth import (
    IssuedVirtualKey,
    fingerprint_virtual_key,
    issue_key_material,
    utc_text,
)
from exp.runtime.gateway.contracts import DirectTarget
from exp.runtime.gateway.sqlite.alias_activation import (
    AliasActivationOutcomeUnknownError,
    alias_activation_transaction,
    register_catalog_snapshot_in_transaction,
)
from exp.runtime.gateway.sqlite.provider_authority import (
    ProviderConnectionBinding,
    ProviderConnectionMutation,
    upsert_provider_connection,
)

if TYPE_CHECKING:
    from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore


def upsert_provider_connections_and_activate_direct_alias(
    store: SQLiteGatewayStore,
    *,
    organization_id: str,
    alias_id: str,
    alias_name: str,
    revision_id: str,
    pool_id: str,
    snapshot_ref: str,
    catalog_sha256: Sha256,
    provider_connections: tuple[ProviderConnectionMutation, ...],
    replace: bool,
    refusal_failover: bool,
    activate_alias_revision: Callable[..., None],
) -> None:
    """Atomically revise providers, register a snapshot, and activate one direct alias.

    Args:
        store: SQLite gateway store owning the transaction connection.
        organization_id: Owning tenant.
        alias_id: Stable alias resource ID.
        alias_name: Public model string.
        revision_id: Immutable alias revision ID.
        pool_id: Direct target pool identifier.
        snapshot_ref: Content-addressed catalog snapshot reference.
        catalog_sha256: Exact normalized catalog digest.
        provider_connections: Desired provider connection revisions.
        replace: Whether differing active provider metadata may be revised.
        refusal_failover: Whether typed precommit refusals may advance.
        activate_alias_revision: Alias activation seam supplied by the store module.

    Raises:
        ValueError: The requested authority violates an existing invariant.
    """
    target = DirectTarget(pool_id=pool_id)
    now = utc_text(store._clock.now())
    connect = cast(
        Callable[[], AbstractContextManager[sqlite3.Connection]],
        store._connect,
    )
    with alias_activation_transaction(
        connect=connect,
        organization_id=organization_id,
        alias_id=alias_id,
        alias_name=alias_name,
        revision_id=revision_id,
        target=target,
        snapshot_ref=snapshot_ref,
        catalog_sha256=catalog_sha256,
        refusal_failover=refusal_failover,
    ) as connection:
        bindings = _stage_provider_connections(
            connection,
            organization_id=organization_id,
            provider_connections=provider_connections,
            replace=replace,
            now=now,
        )
        register_catalog_snapshot_in_transaction(
            connection,
            organization_id=organization_id,
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            now=now,
            store_error=store._store_error,
        )
        activate_alias_revision(
            connection,
            organization_id=organization_id,
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            target=target,
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            provider_connections=bindings,
            refusal_failover=refusal_failover,
            now=now,
            store_error=store._store_error,
        )


def configure_direct_alias_with_identity(
    store: SQLiteGatewayStore,
    *,
    organization_id: str,
    alias_id: str,
    alias_name: str,
    revision_id: str,
    pool_id: str,
    snapshot_ref: str,
    catalog_sha256: Sha256,
    provider_connections: tuple[ProviderConnectionMutation, ...],
    replace: bool,
    identity_id: str,
    identity_display_name: str,
    key_id: str,
    activate_alias_revision: Callable[..., None],
) -> tuple[bool, IssuedVirtualKey]:
    """Atomically configure serving authority and the setup caller's credentials.

    Args:
        store: SQLite gateway store owning the transaction connection.
        organization_id: Owning tenant.
        alias_id: Stable alias resource ID.
        alias_name: Public model string.
        revision_id: Immutable alias revision ID.
        pool_id: Direct target pool identifier.
        snapshot_ref: Content-addressed catalog snapshot reference.
        catalog_sha256: Exact normalized catalog digest.
        provider_connections: Desired provider connection revisions.
        replace: Whether differing active provider metadata may be revised.
        identity_id: Stable setup identity identifier.
        identity_display_name: Operator-facing setup identity name.
        key_id: Non-secret identifier for the newly issued key.
        activate_alias_revision: Alias activation seam supplied by the store module.

    Returns:
        Whether the identity was created, and the one-time key receipt.

    Raises:
        ValueError: The requested authority violates an existing invariant.
    """
    target = DirectTarget(pool_id=pool_id)
    now = utc_text(store._clock.now())
    issued, fingerprint_version, fingerprint = _prepare_key(
        store,
        organization_id=organization_id,
        identity_id=identity_id,
        key_id=key_id,
        created_at=now,
    )
    connect = cast(
        Callable[[], AbstractContextManager[sqlite3.Connection]],
        store._connect,
    )
    identity_created = False
    try:
        with alias_activation_transaction(
            connect=connect,
            organization_id=organization_id,
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            target=target,
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            refusal_failover=False,
        ) as connection:
            identity_created = _ensure_identity(
                connection,
                organization_id=organization_id,
                identity_id=identity_id,
                display_name=identity_display_name,
                now=now,
                store_error=store._store_error,
            )
            bindings = _stage_provider_connections(
                connection,
                organization_id=organization_id,
                provider_connections=provider_connections,
                replace=replace,
                now=now,
            )
            register_catalog_snapshot_in_transaction(
                connection,
                organization_id=organization_id,
                snapshot_ref=snapshot_ref,
                catalog_sha256=catalog_sha256,
                now=now,
                store_error=store._store_error,
            )
            activate_alias_revision(
                connection,
                organization_id=organization_id,
                alias_id=alias_id,
                alias_name=alias_name,
                revision_id=revision_id,
                target=target,
                snapshot_ref=snapshot_ref,
                catalog_sha256=catalog_sha256,
                provider_connections=bindings,
                refusal_failover=False,
                now=now,
                store_error=store._store_error,
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO identity_alias_grants (
                    organization_id, identity_id, alias_id, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (organization_id, identity_id, alias_id, now),
            )
            _persist_key(
                connection,
                store=store,
                issued=issued,
                fingerprint_version=fingerprint_version,
                fingerprint=fingerprint,
                created_at=now,
            )
    except AliasActivationOutcomeUnknownError as activation_error:
        raise AliasActivationOutcomeUnknownError(
            alias_id=alias_id,
            revision_id=revision_id,
            issued=issued,
        ) from activation_error
    return identity_created, issued


def _stage_provider_connections(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    provider_connections: tuple[ProviderConnectionMutation, ...],
    replace: bool,
    now: str,
) -> tuple[ProviderConnectionBinding, ...]:
    """Stage provider revisions and return bindings for the new alias revision."""
    authorities = []
    for mutation in provider_connections:
        _changed, authority = upsert_provider_connection(
            connection,
            organization_id=organization_id,
            connection_id=mutation.connection_id,
            revision_id=mutation.revision_id,
            config=mutation.config,
            replace=replace,
            now=now,
        )
        authorities.append(authority)
    return tuple(
        ProviderConnectionBinding(
            connection_id=authority.connection_id,
            connection_revision_id=authority.revision_id,
            connection_sha256=authority.connection_sha256,
        )
        for authority in authorities
    )


def _ensure_identity(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    identity_id: str,
    display_name: str,
    now: str,
    store_error: type[ValueError],
) -> bool:
    """Create or validate the active identity inside the setup transaction."""
    row = connection.execute(
        "SELECT organization_id, active FROM identities WHERE identity_id = ?",
        (identity_id,),
    ).fetchone()
    if row is not None:
        if str(row["organization_id"]) != organization_id:
            raise store_error("identity ID belongs to another organization")
        if not bool(row["active"]):
            raise store_error(f"identity {identity_id!r} is disabled")
        return False
    connection.execute(
        """
        INSERT INTO identities (
            identity_id, organization_id, display_name, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (identity_id, organization_id, display_name, now, now),
    )
    return True


def _prepare_key(
    store: SQLiteGatewayStore,
    *,
    organization_id: str,
    identity_id: str,
    key_id: str,
    created_at: str,
) -> tuple[IssuedVirtualKey, int, Sha256]:
    """Generate setup key material before entering the ambiguous transaction boundary."""
    prefix, raw_key = issue_key_material()
    pepper = store._pepper.current()
    fingerprint = fingerprint_virtual_key(raw_key, pepper)
    return (
        IssuedVirtualKey(
            key_id=key_id,
            organization_id=organization_id,
            identity_id=identity_id,
            prefix=prefix,
            raw_key=raw_key,
            expires_at=None,
            created_at=datetime.fromisoformat(created_at),
        ),
        pepper.version,
        fingerprint,
    )


def _persist_key(
    connection: sqlite3.Connection,
    *,
    store: SQLiteGatewayStore,
    issued: IssuedVirtualKey,
    fingerprint_version: int,
    fingerprint: Sha256,
    created_at: str,
) -> None:
    """Persist prepared setup key material inside the serving transaction."""
    try:
        connection.execute(
            """
            INSERT INTO virtual_keys (
                key_id, organization_id, identity_id, prefix, fingerprint_version,
                fingerprint_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issued.key_id,
                issued.organization_id,
                issued.identity_id,
                issued.prefix,
                fingerprint_version,
                fingerprint,
                created_at,
            ),
        )
    except sqlite3.IntegrityError:
        raise store._store_error(
            "virtual key issuance conflicts with existing gateway authority"
        ) from None
