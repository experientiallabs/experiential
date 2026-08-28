"""Versioned SQLite authority for secret-free serving provider connections."""

from __future__ import annotations

import sqlite3
from typing import Literal, cast

from exp.common.core.artifacts import ContractModel, Sha256, stable_id
from exp.common.models import ConnectionConfig


class ProviderAuthorityError(ValueError):
    """Provider connection state conflicts with immutable serving authority."""


class ProviderConnectionBinding(ContractModel):
    """One alias-revision binding to an exact provider connection revision."""

    connection_id: str
    connection_revision_id: str
    connection_sha256: Sha256


class ProviderConnectionMutation(ContractModel):
    """One provider connection revision staged inside an alias activation transaction."""

    connection_id: str
    revision_id: str
    config: ConnectionConfig


class ProviderConnectionAuthority(ContractModel):
    """One active or revision-pinned secret-free provider connection."""

    connection_id: str
    revision_id: str
    revision_number: int
    connection_sha256: Sha256
    config: ConnectionConfig
    active: bool = True


def provider_connection_revision_id(connection_id: str, config: ConnectionConfig) -> str:
    """Derive one immutable revision from canonical serving authority."""
    canonical = config.canonicalized()
    return stable_id(
        "provider-connection-revision",
        {
            "connection_id": connection_id,
            "config": canonical.model_dump(mode="json", exclude_none=False),
        },
    )


def upsert_provider_connection(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    connection_id: str,
    revision_id: str,
    config: ConnectionConfig,
    replace: bool,
    now: str,
) -> tuple[bool, ProviderConnectionAuthority]:
    """Create or explicitly revise one provider connection.

    Args:
        connection: Open SQLite transaction.
        organization_id: Owning tenant.
        connection_id: Stable operator-facing connection name.
        revision_id: Immutable revision identity for a changed configuration.
        config: Validated secret-free connection metadata.
        replace: Whether an existing different configuration may be revised.
        now: Canonical transaction timestamp.

    Returns:
        Change flag and the exact active authority.

    Raises:
        ProviderAuthorityError: Identity, replacement, or revision invariants conflict.
    """
    config = config.canonicalized()
    current = _active_row(
        connection,
        organization_id=organization_id,
        connection_id=connection_id,
    )
    digest = config.identity_sha256()
    if current is not None:
        authority = _authority(current)
        if authority.config == config and authority.connection_sha256 == digest:
            return False, authority
        if not replace:
            raise ProviderAuthorityError(
                "provider connection already exists with different metadata; pass --replace"
            )
        revision_number = authority.revision_number + 1
    else:
        existing_id = connection.execute(
            """
            SELECT organization_id FROM provider_connections
            WHERE connection_id = ?
            """,
            (connection_id,),
        ).fetchone()
        if existing_id is not None and str(existing_id["organization_id"]) != organization_id:
            raise ProviderAuthorityError("provider connection ID belongs to another organization")
        connection.execute(
            """
            INSERT INTO provider_connections (
                connection_id, organization_id, active, created_at, updated_at
            ) VALUES (?, ?, 1, ?, ?)
            """,
            (connection_id, organization_id, now, now),
        )
        revision_number = 1
    revision_owner = connection.execute(
        """
        SELECT organization_id, connection_id
        FROM provider_connection_revisions WHERE revision_id = ?
        """,
        (revision_id,),
    ).fetchone()
    if revision_owner is not None:
        raise ProviderAuthorityError("provider connection revision ID is already in use")
    connection.execute(
        """
        INSERT INTO provider_connection_revisions (
            revision_id, organization_id, connection_id, revision_number,
            provider, base_url, api_key_env, api_version, azure_api_surface, region,
            aws_access_key_id_env, bedrock_auth_mode, connection_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            revision_id,
            organization_id,
            connection_id,
            revision_number,
            config.provider,
            config.base_url,
            config.api_key_env,
            config.api_version,
            config.azure_api_surface,
            config.region,
            config.aws_access_key_id_env,
            config.bedrock_auth_mode,
            digest,
            now,
        ),
    )
    connection.execute(
        """
        UPDATE provider_connections
        SET active_revision_id = ?, active = 1, updated_at = ?
        WHERE organization_id = ? AND connection_id = ?
        """,
        (revision_id, now, organization_id, connection_id),
    )
    return (
        True,
        ProviderConnectionAuthority(
            connection_id=connection_id,
            revision_id=revision_id,
            revision_number=revision_number,
            connection_sha256=digest,
            config=config,
        ),
    )


def active_provider_connections(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
) -> tuple[ProviderConnectionAuthority, ...]:
    """Return active provider authorities in stable connection-name order."""
    rows = connection.execute(
        f"""
        {_SELECT_AUTHORITY}
        WHERE c.organization_id = ? AND c.active = 1
        ORDER BY c.connection_id
        """,
        (organization_id,),
    ).fetchall()
    return tuple(_authority(row) for row in rows)


def bound_provider_connections(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    alias_id: str,
    alias_revision_id: str,
) -> tuple[ProviderConnectionAuthority, ...]:
    """Return exact connection revisions frozen into one alias revision."""
    rows = connection.execute(
        """
        SELECT c.connection_id, r.revision_id, r.revision_number,
               r.provider, r.base_url, r.api_key_env, r.api_version,
               r.azure_api_surface, r.region,
               r.aws_access_key_id_env, r.bedrock_auth_mode,
               r.connection_sha256, c.active
        FROM alias_revision_provider_connections AS b
        JOIN provider_connections AS c
          ON c.organization_id = b.organization_id
         AND c.connection_id = b.connection_id
        JOIN provider_connection_revisions AS r
          ON r.organization_id = b.organization_id
         AND r.connection_id = b.connection_id
         AND r.revision_id = b.connection_revision_id
        WHERE b.organization_id = ? AND b.alias_id = ?
          AND b.alias_revision_id = ?
        ORDER BY c.connection_id
        """,
        (organization_id, alias_id, alias_revision_id),
    ).fetchall()
    return tuple(_authority(row) for row in rows)


def bind_alias_provider_connections(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    alias_id: str,
    alias_revision_id: str,
    bindings: tuple[ProviderConnectionBinding, ...],
    now: str,
) -> None:
    """Bind one alias revision to current exact provider revisions atomically."""
    names = tuple(item.connection_id for item in bindings)
    if len(set(names)) != len(names):
        raise ProviderAuthorityError("alias provider connection bindings must not repeat")
    for binding in bindings:
        row = connection.execute(
            """
            SELECT c.active_revision_id, c.active, r.connection_sha256
            FROM provider_connections AS c
            JOIN provider_connection_revisions AS r
              ON r.organization_id = c.organization_id
             AND r.connection_id = c.connection_id
             AND r.revision_id = c.active_revision_id
            WHERE c.organization_id = ? AND c.connection_id = ?
            """,
            (organization_id, binding.connection_id),
        ).fetchone()
        if (
            row is None
            or not bool(row["active"])
            or str(row["active_revision_id"]) != binding.connection_revision_id
            or str(row["connection_sha256"]) != binding.connection_sha256
        ):
            raise ProviderAuthorityError(
                "alias provider connection binding differs from active SQLite authority"
            )
        connection.execute(
            """
            INSERT INTO alias_revision_provider_connections (
                organization_id, alias_id, alias_revision_id, connection_id,
                connection_revision_id, connection_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                organization_id,
                alias_id,
                alias_revision_id,
                binding.connection_id,
                binding.connection_revision_id,
                binding.connection_sha256,
                now,
            ),
        )


def disable_provider_connection(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    connection_id: str,
    now: str,
) -> bool:
    """Disable an unreferenced provider connection without deleting its revisions."""
    referenced = connection.execute(
        """
        SELECT 1
        FROM alias_revision_provider_connections AS b
        JOIN gateway_aliases AS a
          ON a.organization_id = b.organization_id
         AND a.alias_id = b.alias_id
         AND a.active_revision_id = b.alias_revision_id
        WHERE b.organization_id = ? AND b.connection_id = ? AND a.active = 1
        LIMIT 1
        """,
        (organization_id, connection_id),
    ).fetchone()
    if referenced is not None:
        raise ProviderAuthorityError("provider connection is bound to an active alias revision")
    result = connection.execute(
        """
        UPDATE provider_connections SET active = 0, updated_at = ?
        WHERE organization_id = ? AND connection_id = ? AND active = 1
        """,
        (now, organization_id, connection_id),
    )
    return result.rowcount == 1


def _active_row(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    connection_id: str,
) -> sqlite3.Row | None:
    """Read one current provider authority inside the caller transaction."""
    return connection.execute(
        f"""
        {_SELECT_AUTHORITY}
        WHERE c.organization_id = ? AND c.connection_id = ? AND c.active = 1
        """,
        (organization_id, connection_id),
    ).fetchone()


def _authority(row: sqlite3.Row) -> ProviderConnectionAuthority:
    """Validate one SQLite authority row as a typed provider record."""
    config = ConnectionConfig(
        provider=str(row["provider"]),
        base_url=None if row["base_url"] is None else str(row["base_url"]),
        api_key_env=None if row["api_key_env"] is None else str(row["api_key_env"]),
        api_version=None if row["api_version"] is None else str(row["api_version"]),
        azure_api_surface=cast(
            Literal["openai_deployments", "model_inference"] | None,
            None if row["azure_api_surface"] is None else str(row["azure_api_surface"]),
        ),
        region=None if row["region"] is None else str(row["region"]),
        aws_access_key_id_env=(
            None if row["aws_access_key_id_env"] is None else str(row["aws_access_key_id_env"])
        ),
        bedrock_auth_mode=(
            None
            if row["bedrock_auth_mode"] is None
            else cast('Literal["access_key_pair", "api_key"]', str(row["bedrock_auth_mode"]))
        ),
    )
    digest = str(row["connection_sha256"])
    if config.identity_sha256() != digest:
        raise ProviderAuthorityError("provider connection revision digest is invalid")
    return ProviderConnectionAuthority(
        connection_id=str(row["connection_id"]),
        revision_id=str(row["revision_id"]),
        revision_number=int(row["revision_number"]),
        connection_sha256=digest,
        config=config,
        active=bool(row["active"]),
    )


_SELECT_AUTHORITY = """
SELECT c.connection_id, r.revision_id, r.revision_number,
       r.provider, r.base_url, r.api_key_env, r.api_version,
       r.azure_api_surface, r.region,
       r.aws_access_key_id_env, r.bedrock_auth_mode,
       r.connection_sha256, c.active
FROM provider_connections AS c
JOIN provider_connection_revisions AS r
  ON r.organization_id = c.organization_id
 AND r.connection_id = c.connection_id
 AND r.revision_id = c.active_revision_id
"""
