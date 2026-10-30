"""Validate and persist disclosed trajectory observations before memory publication."""

from datetime import datetime

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input


def consumed_trajectory_digest(trajectory: JsonObject) -> Sha256:
    """Identify a completed trajectory by its public actions and disclosed outcomes only."""
    return sha256_json(trajectory)


def require_consumed_trajectory(trajectory: JsonObject) -> None:
    """Refuse a trajectory whose ownership or inventory a later reader could not trust."""
    lineage = trajectory.get("task_lineage")
    if not isinstance(lineage, str) or not lineage.strip():
        raise ValueError("a consumed trajectory requires its task lineage")
    actions, replies = trajectory.get("actions"), trajectory.get("observed_replies")
    if not isinstance(actions, list) or not isinstance(replies, list):
        raise ValueError("a consumed trajectory requires action and reply lists")
    if len(actions) != len(replies):
        raise ValueError("a consumed trajectory requires one disclosed reply per action")
    for action in actions:
        if (
            not isinstance(action, dict)
            or set(action) != {"name", "arguments"}
            or not isinstance(action["name"], str)
            or not action["name"]
            or not isinstance(action["arguments"], dict)
        ):
            raise ValueError("each consumed action is exactly {name, arguments}")


def persist_trajectory_source(
    store: ArtifactStore, trajectory: JsonObject, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Persist disclosed source observations without publishing a memory version."""
    require_consumed_trajectory(trajectory)
    digest = consumed_trajectory_digest(trajectory)
    source_id = stable_id("consumed-trajectory", {"digest": digest})
    payload: JsonObject = {
        "trajectory": trajectory,
        "authority": "Disclosed native replies after prediction; not a behavioral rule",
    }
    if not store.exists(source_id):
        store.write_json(
            artifact_id=source_id,
            artifact_type="consumed-trajectory",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision
            ),
            files={"trajectory.json": payload},
        )
    return artifact_input(store.read(source_id).manifest)
