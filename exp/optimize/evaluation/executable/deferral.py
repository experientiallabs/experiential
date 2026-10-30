"""Durable lease-contention evidence, distinct from an unknown execution outcome.

An orchestration owner may retry the same child after recorded contention. The child's own
dispatch intents still govern recovery; this receipt never authorizes a new draw or generation.
"""

from datetime import UTC, datetime

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.project import ArtifactStore, artifact_input


def deferred_execution(store: ArtifactStore, cell: str) -> tuple[ArtifactInput, ...]:
    """Return a verified contention receipt for this exact cell.

    Args:
        store: Immutable execution evidence owner.
        cell: Frozen parent cell identity, without the intent suffix.

    Returns:
        The original receipt, or an empty tuple without known contention. This permits
        re-entering the same child owner, not bypassing its dispatch or recovery checks.
    """
    identity = cell + "-deferred"
    if not store.exists(identity):
        return ()
    manifest = store.read(identity).manifest
    intent = store.read(cell + "-intent").manifest
    if (
        manifest.artifact_type != "execution-deferral"
        or manifest.inputs != (artifact_input(intent),)
        or manifest.code_revision != intent.code_revision
        or store.read_bytes(identity, "reason.txt") != b"child lease busy"
    ):
        raise ValueError("execution deferral differs from its exact parent intent")
    return (artifact_input(manifest),)


def record_execution_deferral(store: ArtifactStore, cell: str) -> None:
    """Record known contention without deleting or replacing any execution intent.

    Args:
        store: Immutable execution evidence owner.
        cell: Frozen cell whose owner holds its parent lease and caught a child FileLock
            timeout. Unknown provider or native outcomes must never use this operation.
    """
    if deferred_execution(store, cell):
        return
    intent = store.read(cell + "-intent").manifest
    store.write(
        artifact_id=cell + "-deferred",
        artifact_type="execution-deferral",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=intent.code_revision,
            inputs=(artifact_input(intent),),
        ),
        files={"reason.txt": b"child lease busy"},
    )
