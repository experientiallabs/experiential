"""Data repairs that must run inside guarded gateway schema migrations."""

import sqlite3


def deactivate_aliases_with_inferred_bedrock_auth(connection: sqlite3.Connection) -> None:
    """Deactivate aliases whose immutable snapshot predates explicit Bedrock auth modes.

    Bound aliases can be matched to the affected revision exactly. Unbound
    aliases remain untouched because no immutable evidence connects them to the
    repaired Bedrock authority; normal management startup can bind them from
    their catalog snapshot. Aliases bound only to unaffected revisions remain
    active.
    """
    connection.execute(
        """
        DELETE FROM project_activation_bindings AS projects
        WHERE EXISTS (
            SELECT 1
            FROM gateway_aliases AS aliases
            JOIN alias_revision_provider_connections AS bindings
              ON bindings.organization_id = aliases.organization_id
             AND bindings.alias_id = aliases.alias_id
             AND bindings.alias_revision_id = aliases.active_revision_id
            JOIN provider_connection_revisions AS revisions
              ON revisions.organization_id = bindings.organization_id
             AND revisions.connection_id = bindings.connection_id
             AND revisions.revision_id = bindings.connection_revision_id
            WHERE aliases.organization_id = projects.organization_id
              AND aliases.alias_id = projects.alias_id
              AND aliases.active = 1
              AND revisions.provider = 'bedrock'
              AND revisions.api_key_env IS NOT NULL
              AND revisions.bedrock_auth_mode IS NULL
        )
        """
    )
    connection.execute(
        """
        UPDATE gateway_aliases AS aliases
        SET active = 0,
            active_revision_id = NULL,
            updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
        WHERE aliases.active = 1
          AND EXISTS (
                SELECT 1
                FROM alias_revision_provider_connections AS bindings
                JOIN provider_connection_revisions AS revisions
                  ON revisions.organization_id = bindings.organization_id
                 AND revisions.connection_id = bindings.connection_id
                 AND revisions.revision_id = bindings.connection_revision_id
                WHERE bindings.organization_id = aliases.organization_id
                  AND bindings.alias_id = aliases.alias_id
                  AND bindings.alias_revision_id = aliases.active_revision_id
                  AND revisions.provider = 'bedrock'
                  AND revisions.api_key_env IS NOT NULL
                  AND revisions.bedrock_auth_mode IS NULL
            )
        """
    )
