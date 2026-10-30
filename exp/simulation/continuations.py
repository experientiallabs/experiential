"""Immutable, paired actor/world checkpoints for development continuation studies."""

import json
from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import TaskCase
from exp.runtime.agents.chat import ChatCheckpoint
from exp.runtime.environments.mixed import MixedCheckpoint
from exp.runtime.environments.world_artifacts import (
    WorldCheckpoint,
    load_world_image,
    persist_world_image,
)


class ContinuationCheckpoint(ContractModel):
    """A completed actor turn and its corresponding executable world.

    Attributes:
        task: Exact public task; private reference answers are forbidden.
        environment_task: Original task used to restore the world when actor wording changes.
            Tools, partition and source lineage must remain identical; None uses task.
        actor: Complete visible state of the built-in stateless chat actor.
        world: Native or whole mixed world, including events, clocks and random state.
        sources: Immutable origin evidence and any frozen simulator/memory dependencies.
    """

    task: TaskCase
    environment_task: TaskCase | None = Field(default=None, exclude_if=lambda value: value is None)
    actor: ChatCheckpoint
    world: WorldCheckpoint
    sources: tuple[ArtifactInput, ...]

    @property
    def world_task(self) -> TaskCase:
        """Resolve the exact environment initialization contract separately from actor wording."""
        return self.environment_task or self.task

    @model_validator(mode="after")
    def paired(self) -> Self:
        """Bind actor state to the exact public task and retain source provenance."""
        if self.environment_task is not None and any(
            getattr(self.task, key) != getattr(self.environment_task, key)
            for key in ("tools", "partition", "lineage_group_id", "source_trace_ids", "origin")
        ):
            raise ValueError("actor and environment tasks must share tools and source partition")
        if self.task.reference_answer is not None or self.world_task.reference_answer is not None:
            raise ValueError("continuation task contains private answers; capture a public task")
        if self.actor.task_sha256 != sha256_json(self.task):
            raise ValueError("actor checkpoint belongs to another public task")
        if (
            isinstance(self.world, MixedCheckpoint)
            and self.world.task_id != self.world_task.task_id
        ):
            raise ValueError("mixed checkpoint belongs to another task")
        if not self.sources:
            raise ValueError(
                "continuation needs immutable origin evidence; persist its source first"
            )
        if len({source.artifact_id for source in self.sources}) != len(self.sources):
            raise ValueError("continuation source identities must be unique")
        return self


def persist_continuation_checkpoint(
    store: ArtifactStore,
    checkpoint: ContinuationCheckpoint,
    *,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Save a paired checkpoint after verifying every declared source artifact.

    Args:
        store: Trusted artifact store, never exposed to the actor.
        checkpoint: Actor and world captured at the same completed-turn boundary.
        created_at: Publication time.
        code_revision: Producing implementation identity.

    Returns:
        Immutable reference. Repeated publication of the same checkpoint reuses its bytes.
    """
    for source in checkpoint.sources:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("continuation source digest changed; recapture from frozen evidence")
    identity = stable_id(
        "continuation-checkpoint",
        {
            "checkpoint": sha256_json(checkpoint),
            "code_revision": code_revision,
        },
    )
    world = persist_world_image(
        store, checkpoint.world, created_at=created_at, code_revision=code_revision
    )
    wire = checkpoint.model_dump(mode="json")
    wire["world"] = world.model_dump(mode="json")
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        inputs=tuple(sorted({*checkpoint.sources, world}, key=lambda ref: ref.artifact_id)),
    )
    _, manifest = store.write_or_replay(
        artifact_id=identity,
        artifact_type="continuation-checkpoint",
        envelope=envelope,
        envelope_type=ArtifactEnvelope,
        envelope_path="envelope.json",
        files={
            "checkpoint.json": canonical_json_bytes(wire),
            "envelope.json": canonical_json_bytes(envelope),
        },
    )
    return artifact_input(manifest)


def load_continuation_checkpoint(
    store: ArtifactStore, reference: ArtifactInput
) -> ContinuationCheckpoint:
    """Verify saved checkpoint bytes and source bindings before restoration."""
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "continuation-checkpoint":
        raise ValueError("continuation checkpoint differs from its registered artifact")
    wire = json.loads(store.read_bytes(reference.artifact_id, "checkpoint.json"))
    world_ref = ArtifactInput.model_validate(wire["world"])
    wire["world"] = load_world_image(store, world_ref).model_dump(mode="json")
    checkpoint = ContinuationCheckpoint.model_validate(wire)
    if set(manifest.inputs) != {*checkpoint.sources, world_ref}:
        raise ValueError("continuation manifest does not bind its declared sources and world")
    for source in checkpoint.sources:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("continuation source differs from its saved digest")
    return checkpoint
