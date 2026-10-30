"""Frozen executable task contracts shared by continuation studies and Harbor delivery."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_bytes,
    sha256_json,
    stable_id,
    validate_artifact_file_path,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.simulation.capability_briefs import load_capability_brief
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint
from exp.simulation.development_scope import require_development_task


class ExecutableTask(ContractModel):
    """Executable world and observable grading contract without a second actor framework.

    Attributes:
        intent: Capability and user outcome the task is intended to measure.
        output_paths: Exact public native files returned only after final submission.
        job: Existing exact runtime factory, world checkpoint and private probe/verifier.
            The captured actor prefix becomes public context; Harbor supplies its own actor.
        maximum_tool_calls: Finite tool-attempt ceiling per trial, frozen before execution.
        capability_brief: Optional private intent; unresolved assumptions prevent qualification.
        known_limits: Public qualifications that survive export.
        harbor_transport: Opt-in Unix IPC with no actor or verifier container networking.
    """

    intent: str = Field(min_length=1)
    job: ContinuationJob
    output_paths: tuple[str, ...] = ()
    maximum_tool_calls: int = Field(default=50, ge=1, le=1000)
    capability_brief: ArtifactInput | None = None
    harbor_transport: Literal["http", "unix-no-network"] = Field(
        default="http", exclude_if=lambda value: value == "http"
    )
    known_limits: tuple[str, ...] = (
        "Observable predicates do not certify unobserved effects or host containment.",
        "Authored or simulated behavior does not establish fidelity to a native production system.",
    )

    @field_validator("output_paths")
    @classmethod
    def safe_outputs(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        """Expose only unique, explicitly named relative files within a finite inventory."""
        if len(set(paths)) != len(paths) or len(paths) > 32:
            raise ValueError("output inventory is repeated or exceeds 32 files")
        for path in paths:
            validate_artifact_file_path(path)
        return paths

    def profile(self) -> str:
        """Bind the world, probe, verifier, actor context and execution allowance."""
        return sha256_json(self)


def preflight_executable_task(
    project: ProjectStore, task: ExecutableTask
) -> ContinuationCheckpoint:
    """Check the checkpoint, intent and runtime bytes without importing code or publishing.

    Actual runtime construction and probe compatibility still require execution. Harbor owns its
    actor, so this check does not load the continuation actor's credentials or consume its budget.
    """
    if sha256_bytes(task.job.runtime_file.read_bytes()) != task.job.runtime_sha256:
        raise ValueError("runtime source changed; register the exact reviewed file digest")
    checkpoint = load_continuation_checkpoint(project.artifacts, task.job.study.checkpoint)
    require_development_task(project.artifacts, checkpoint.task)
    if checkpoint.task.partition != "fit":
        raise ValueError("executable development packages require a fit checkpoint")
    if task.capability_brief is not None:
        brief = load_capability_brief(project.artifacts, task.capability_brief)
        brief.require_task(checkpoint.task, task.intent)
    return checkpoint


def persist_executable_task(
    project: ProjectStore, task: ExecutableTask, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Validate frozen sources and publish task bytes without invoking runtime code."""
    preflight_executable_task(project, task)
    inputs = (task.job.study.checkpoint,) + (
        (task.capability_brief,) if task.capability_brief else ()
    )
    identity = stable_id("executable-task", {"task": task.profile(), "code": code_revision})
    store = project.artifacts
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type="executable-task",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={"task.json": task},
        )
    return artifact_input(store.read(identity).manifest)


def load_executable_task(project: ProjectStore, reference: ArtifactInput) -> ExecutableTask:
    """Verify task bytes, exact references and the paired checkpoint before serving."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "executable-task":
        raise ValueError("executable task reference differs")
    task = ExecutableTask.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "task.json")
    )
    checkpoint = load_continuation_checkpoint(project.artifacts, task.job.study.checkpoint)
    require_development_task(project.artifacts, checkpoint.task)
    if task.capability_brief is not None:
        load_capability_brief(project.artifacts, task.capability_brief).require_task(
            checkpoint.task, task.intent
        )
    expected_id = stable_id(
        "executable-task", {"task": task.profile(), "code": manifest.code_revision}
    )
    if reference.artifact_id != expected_id:
        raise ValueError("executable task content does not match its frozen identity")
    expected = {task.job.study.checkpoint} | (
        {task.capability_brief} if task.capability_brief else set()
    )
    if set(manifest.inputs) != expected:
        raise ValueError("executable task lost its frozen inputs")
    return task
