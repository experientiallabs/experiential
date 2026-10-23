"""Capture an initial paired executable world and actor context through one frozen definition."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.executable.loader import load_runtime_definition
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.development_scope import require_development_task


class CaptureDefinition(ContractModel):
    """Initial capture with explicit public context, evidence and trusted implementation.

    Attributes:
        task: Public development task without private expected answers.
        sources: Frozen simulator, memory, native inputs or authored source evidence.
        runtime_file: Explicit trusted factory exporting continuation_runtime.
        runtime_sha256: Exact factory source digest; imported dependencies remain host-owned.
        configuration: Frozen factory inputs, excluding secrets.
        probe_sha256: Expected private state projection definition.
        maximum_model_calls: Actor's original policy allowance, with zero calls spent at capture.
        system_prompt: Exact actor instruction, also required by the subsequent continuation job.
    """

    task: TaskCase
    sources: tuple[ArtifactInput, ...] = Field(min_length=1)
    runtime_file: Path
    runtime_sha256: Sha256
    configuration: JsonObject = Field(default_factory=dict)
    probe_sha256: Sha256
    maximum_model_calls: int = Field(default=20, ge=1, le=1000)
    system_prompt: str | None = None

    def identity(self, code_revision: str) -> str:
        """Bind startup semantics, evidence and code before the world is initialized."""
        return stable_id("continuation-capture", {"spec": sha256_json(self), "code": code_revision})


def preflight_capture(project: ProjectStore, definition: CaptureDefinition) -> None:
    """Validate provenance and file identity without importing runtime code or loading secrets."""
    require_development_task(project.artifacts, definition.task)
    if definition.task.reference_answer is not None:
        raise ValueError("capture cannot put private expected answers into an actor task")
    if len(set(row.artifact_id for row in definition.sources)) != len(definition.sources):
        raise ValueError("capture sources must be unique")
    for source in definition.sources:
        if artifact_input(project.artifacts.read(source.artifact_id).manifest) != source:
            raise ValueError("capture source differs from its frozen digest")
    if sha256_bytes(definition.runtime_file.read_bytes()) != definition.runtime_sha256:
        raise ValueError("runtime source changed before initial capture")
    ChatAgentRuntime(
        maximum_model_calls=definition.maximum_model_calls, system_prompt=definition.system_prompt
    ).initial_checkpoint(definition.task)


def saved_capture(
    project: ProjectStore, definition: CaptureDefinition, *, code_revision: str
) -> ArtifactInput | None:
    """Replay an exact saved capture without re-importing a runtime or initializing a world."""
    identity = definition.identity(code_revision) + "-result"
    store = project.artifacts
    if not store.exists(identity):
        return None
    reference = ArtifactInput.model_validate_json(store.read_bytes(identity, "checkpoint.json"))
    checkpoint = load_continuation_checkpoint(store, reference)
    registration = artifact_input(store.read(definition.identity(code_revision)).manifest)
    if store.read(identity).manifest.inputs != tuple(
        sorted((registration, reference), key=lambda row: row.artifact_id)
    ):
        raise ValueError("capture result lost its registration or checkpoint")
    if checkpoint.task != definition.task or set(checkpoint.sources) != {
        *definition.sources,
        registration,
    }:
        raise ValueError("saved capture does not reproduce its registered sources and task")
    require_development_task(store, checkpoint.task)
    return reference


def capture_initial_world(
    project: ProjectStore, definition: CaptureDefinition, *, code_revision: str
) -> ArtifactInput:
    """Capture once under an exclusive lease; unknown startup outcomes are never repeated.

    Capture performs no actor inference or setup actions. An explicitly selected factory may
    initialize local resources; its unknown startup intent requires a new, explicit definition.
    """
    store = project.artifacts
    identity = definition.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_capture(project, definition, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_capture(project, definition)
        if store.exists(identity):
            raise ValueError(
                "initial capture has an unfinished startup intent; no implicit restart"
            )
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(definition.sources, key=lambda row: row.artifact_id)),
        )
        registration = artifact_input(
            store.write_json(
                artifact_id=identity,
                artifact_type="continuation-capture",
                envelope=envelope,
                files={"definition.json": definition},
            )
        )
        actor = ChatAgentRuntime(
            maximum_model_calls=definition.maximum_model_calls,
            system_prompt=definition.system_prompt,
        )
        with (
            load_runtime_definition(
                project,
                runtime_file=definition.runtime_file,
                runtime_sha256=definition.runtime_sha256,
                configuration=definition.configuration,
                probe_sha256=definition.probe_sha256,
            ) as runtime,
            runtime.world.open(definition.task) as world,
        ):
            checkpoint = ContinuationCheckpoint(
                task=definition.task,
                actor=actor.initial_checkpoint(definition.task),
                world=world.checkpoint(),
                sources=(*definition.sources, registration),
            )
            reference = persist_continuation_checkpoint(
                store, checkpoint, created_at=envelope.created_at, code_revision=code_revision
            )
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="continuation-capture-result",
            envelope=envelope.model_copy(
                update={
                    "inputs": tuple(
                        sorted((registration, reference), key=lambda row: row.artifact_id)
                    )
                }
            ),
            files={"checkpoint.json": reference},
        )
        return reference
