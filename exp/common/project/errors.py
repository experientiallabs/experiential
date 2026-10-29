"""Failures at project configuration and immutable evidence boundaries."""


class ArtifactStoreError(RuntimeError):
    """Base error for immutable local artifact storage failures."""


class ArtifactAlreadyExistsError(ArtifactStoreError):
    """A completed artifact ID was reused instead of creating a new artifact."""


class ArtifactCorruptionError(ArtifactStoreError):
    """A completed artifact no longer matches its immutable manifest."""


class ProjectStoreError(RuntimeError):
    """Project initialization or local configuration persistence failed."""
