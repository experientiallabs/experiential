"""Typed, local artifact transformations with immutable inputs and explicit projections."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    validate_artifact_file_path,
)
from exp.common.models import ToolCall

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024


class NativeArtifactBinding(ContractModel):
    """Public contract frozen before the simulator proposes native edits.

    Attributes:
        input_artifact: Exact immutable initial file set, never a private oracle.
        adapter_sha256: Digest of the adapter implementation and dependency definition.
        output_paths: Exact relative files the adapter must produce.
        plan_key: Top-level candidate state key containing the typed plan.
    """

    input_artifact: ArtifactInput
    adapter_sha256: Sha256
    output_paths: tuple[str, ...] = Field(min_length=1, max_length=32)
    plan_key: str = Field(default="native_artifact_plan", min_length=1)

    @model_validator(mode="after")
    def paths(self) -> Self:
        """Require canonical, non-overlapping output paths before execution."""
        if len(set(self.output_paths)) != len(self.output_paths):
            raise ValueError("native output paths must be unique")
        for path in self.output_paths:
            if validate_artifact_file_path(path).as_posix() != path:
                raise ValueError("native output paths must be canonical")
            if any(other.startswith(path + "/") for other in self.output_paths):
                raise ValueError("native output file paths cannot contain each other")
        return self


class NativeArtifactPlan(ContractModel):
    """Simulator-authored edits and a separately checkable predicted projection.

    Attributes:
        input_sha256: Exact initial artifact manifest digest to which these edits apply.
        operations: Bounded typed operations; arbitrary shell is not an artifact edit.
        expected_projection: Predicted artifact facts, compared with actual native output.
    """

    input_sha256: Sha256
    operations: tuple[ToolCall, ...] = Field(min_length=1, max_length=32)
    expected_projection: JsonObject

    @model_validator(mode="after")
    def unique_operations(self) -> Self:
        """Reject ambiguous operation identities instead of applying them twice."""
        if len({call.call_id for call in self.operations}) != len(self.operations):
            raise ValueError("native operation IDs must be unique")
        return self


@dataclass(frozen=True)
class NativeArtifactOutput:
    """Detached output bytes and observations from a trusted artifact adapter.

    Attributes:
        files: Complete declared output file set, with no host filesystem paths.
        projection: Native observations in the adapter's declared projection schema.
    """

    files: Mapping[str, bytes]
    projection: JsonObject


class ArtifactEditAdapter(Protocol):
    """Application-selected pure artifact transformer, never selected by model output.

    Implementations must operate on detached input bytes without network, provider calls,
    arbitrary actor code, or external effects. Such effects belong to EnvironmentRuntime.
    """

    @property
    def definition(self) -> JsonObject:
        """Return the implementation, dependency and operation-contract identity."""

    def apply(
        self,
        files: Mapping[str, bytes],
        operations: tuple[ToolCall, ...],
        output_paths: tuple[str, ...],
    ) -> NativeArtifactOutput:
        """Apply admitted edits to an isolated copy and return only declared output bytes."""
