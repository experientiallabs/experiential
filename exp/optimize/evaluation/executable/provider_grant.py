"""Explicit per-trial provider authority, separate from a frozen world's semantics."""

import json

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256
from exp.common.project import ProjectStore, artifact_input


class ProviderLimits(ContractModel):
    """One authorized simulator allowance, including embeddings and memory-reader attempts.

    Attributes:
        maximum_physical_attempts: Finite lifetime ceiling for this trial, never replenished.
        deadline: Absolute cutoff for fresh provider calls; saved replies remain replayable.
    """

    maximum_physical_attempts: int = Field(ge=1, le=100_000, strict=True)
    deadline: AwareDatetime


class ProviderGrant(ContractModel):
    """Limits bound to a durable trial registration, not an actor-supplied scope identifier.

    Attributes:
        registration: Exact immutable trial-start receipt carrying these limits.
        limits: Authorized operational limits; no model, state or prompt overrides are allowed.
    """

    registration: ArtifactInput
    limits: ProviderLimits

    def verify(self, project: ProjectStore) -> None:
        """Require a matching owner receipt before constructing any admitted transport."""
        manifest = project.artifacts.read(self.registration.artifact_id).manifest
        if (
            artifact_input(manifest) != self.registration
            or manifest.artifact_type != "executable-trial"
        ):
            raise ValueError("provider grant requires its exact trial-start receipt")
        value = json.loads(
            project.artifacts.read_bytes(self.registration.artifact_id, "record.json")
        )
        if not self.registration.artifact_id.endswith("-start") or value.get(
            "provider_limits"
        ) != self.limits.model_dump(mode="json"):
            raise ValueError("provider limits differ from the durable trial registration")

    def require_configuration(
        self,
        project: ProjectStore,
        configuration: JsonObject,
        *,
        runtime_sha256: Sha256 | None = None,
    ) -> None:
        """Refuse a different predictor under the same paid trial authority."""
        self.verify(project)
        receipt = json.loads(
            project.artifacts.read_bytes(self.registration.artifact_id, "record.json")
        )
        task = ArtifactInput.model_validate(receipt.get("task"))
        manifest = project.artifacts.read(task.artifact_id).manifest
        if (
            artifact_input(manifest) != task
            or manifest.artifact_type != "executable-task"
            or task not in project.artifacts.read(self.registration.artifact_id).manifest.inputs
        ):
            raise ValueError("provider grant lost its exact executable task")
        job = json.loads(project.artifacts.read_bytes(task.artifact_id, "task.json"))["job"]
        if runtime_sha256 is not None and runtime_sha256 != job["runtime_sha256"]:
            raise ValueError("provider grant cannot change the registered runtime implementation")
        saved = job["configuration"]
        if configuration != saved and configuration != saved.get("simulated"):
            raise ValueError("provider grant cannot change the frozen runtime configuration")
