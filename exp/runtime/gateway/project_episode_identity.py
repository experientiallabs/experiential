"""Content-addressed episode identity for tenant-scoped project routing."""

from exp.common.core.artifacts import ArtifactId, stable_id


def project_episode_identity(
    namespace: tuple[ArtifactId, ArtifactId, ArtifactId, str],
) -> str:
    """Encode tenant-scoped episode components without delimiter collisions.

    Args:
        namespace: Organization, identity, alias revision, and caller episode key.

    Returns:
        Stable content-addressed identity with explicit component boundaries.
    """
    organization_id, identity_id, alias_revision_id, episode_key = namespace
    return stable_id(
        "gateway-project-episode",
        {
            "organization_id": organization_id,
            "identity_id": identity_id,
            "alias_revision_id": alias_revision_id,
            "episode_key": episode_key,
        },
    )
