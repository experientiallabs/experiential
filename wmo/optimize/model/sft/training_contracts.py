"""Typed contracts for append-only managed offline Tinker SFT runs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Literal, Protocol

from pydantic import Field, TypeAdapter, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from wmo.common.models import NumericMeasurement, Usage
from wmo.optimize.model.sft.contracts import SFTExample


class TinkerSFTError(RuntimeError):
    """The frozen dataset or append-only local SFT run cannot proceed safely."""


class TinkerSFTResumeError(TinkerSFTError):
    """An existing run directory cannot be resumed without mixing immutable state."""


class TinkerSFTAmbiguousStepError(TinkerSFTResumeError):
    """A remote optimizer step may have run and cannot safely be replayed."""


class TinkerSFTBudgetExceeded(TinkerSFTError):
    """Conservative spend accounting prevents another managed training step."""


class TinkerSFTSpec(ContractModel):
    """Frozen base-model, LoRA, schedule, checkpoint, and managed-spend settings."""

    base_model: str = Field(min_length=1, max_length=512)
    lora_rank: int = Field(default=32, gt=0, le=4096)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    checkpoint_every_steps: int = Field(gt=0)
    maximum_steps: int | None = Field(default=None, gt=0)
    maximum_datum_tokens: int | None = Field(default=None, gt=1)
    maximum_cost_usd: float | None = Field(default=None, gt=0)

    @field_validator("learning_rate", "maximum_cost_usd")
    @classmethod
    def _require_finite_float(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Tinker SFT numeric settings must be finite")
        return value


class TrainerBatchResult(ContractModel):
    """Facts returned by one completed backend-managed cross-entropy update."""

    loss: float | None = None
    gradient_norm: float | None = None
    usage: Usage | None = None
    cost_usd: NumericMeasurement | None = None

    @field_validator("loss", "gradient_norm")
    @classmethod
    def _require_finite_metric(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Tinker SFT backend metrics must be finite")
        return value


class TrainerDatum(Protocol):
    """The datum identity retained after a backend renders one frozen example."""

    @property
    def example_id(self) -> str:
        """Return the exact frozen W12 example identifier rendered into this datum."""
        ...

    @property
    def supervised_token_count(self) -> int:
        """Return the nonzero cross-entropy token count after context-only truncation."""
        ...


class TrainerSession(Protocol):
    """One managed training client already created or restored by its backend."""

    def render_examples(self, examples: Sequence[SFTExample]) -> tuple[TrainerDatum, ...]:
        """Render complete frozen assistant-action targets into cross-entropy datums."""
        ...

    def train_batch(
        self, datums: Sequence[TrainerDatum], *, learning_rate: float
    ) -> TrainerBatchResult:
        """Dispatch and complete one managed cross-entropy optimizer update."""
        ...

    def save_state(self, checkpoint_name: str) -> str:
        """Persist a resumable managed state and return its provider resource ID."""
        ...

    def save_sampling_handle(self, model_name: str) -> str:
        """Persist completed weights and return their non-secret provider resource ID."""
        ...


class TrainerBackend(Protocol):
    """The narrow seam for the concrete Tinker adapter or deterministic test fake."""

    def conservative_step_cost(
        self, spec: TinkerSFTSpec, *, batch_example_count: int
    ) -> NumericMeasurement | None:
        """Return a pre-client upper cost bound, or None when the backend cannot prove one."""
        ...

    def open(self, spec: TinkerSFTSpec, resume_state_path: str | None) -> TrainerSession:
        """Create a session, restoring state before renderer or training work when set."""
        ...


class TinkerSFTRunManifest(ArtifactEnvelope):
    """Immutable local identity for one append-only run directory."""

    run_id: ArtifactId
    dataset_id: ArtifactId
    dataset_manifest_sha256: Sha256
    dataset_build_sha256: Sha256
    dataset_examples_sha256: Sha256
    spec: TinkerSFTSpec
    spec_sha256: Sha256

    @model_validator(mode="after")
    def _require_exact_dataset_and_spec_bindings(self) -> TinkerSFTRunManifest:
        expected_inputs = (
            ArtifactInput(artifact_id=self.dataset_id, sha256=self.dataset_manifest_sha256),
        )
        if self.inputs != expected_inputs:
            raise ValueError("Tinker SFT manifests must name exactly their frozen dataset build")
        if self.spec_sha256 != sha256_json(self.spec):
            raise ValueError("Tinker SFT manifest spec digest does not match settings")
        expected_run_id = stable_id(
            "tinker-sft-run",
            {
                "dataset_id": self.dataset_id,
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
                "dataset_build_sha256": self.dataset_build_sha256,
                "dataset_examples_sha256": self.dataset_examples_sha256,
                "spec_sha256": self.spec_sha256,
            },
        )
        if self.run_id != expected_run_id:
            raise ValueError("Tinker SFT manifest run ID is not content-addressed")
        return self


class TinkerSFTMetric(ContractModel):
    """One observed managed batch without a task-behavior or quality claim."""

    record_type: Literal["metric"] = "metric"
    run_id: ArtifactId
    attempt_id: int = Field(ge=1)
    step: int = Field(ge=1)
    epoch: int = Field(ge=1)
    batch_index: int = Field(ge=1)
    batch_example_count: int = Field(gt=0)
    example_ids: tuple[ArtifactId, ...] = Field(min_length=1)
    supervised_token_count: int = Field(gt=0)
    loss: float | None = None
    gradient_norm: float | None = None
    usage: Usage | None = None
    cost_usd: NumericMeasurement | None = None
    cumulative_cost_usd: NumericMeasurement | None = None

    @field_validator("loss", "gradient_norm")
    @classmethod
    def _require_finite_metric(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Tinker SFT metrics must be finite")
        return value


class TinkerSFTCheckpoint(ContractModel):
    """One durable provider state tied to a completed prefix of training batches."""

    record_type: Literal["checkpoint"] = "checkpoint"
    checkpoint_id: ArtifactId
    run_id: ArtifactId
    attempt_id: int = Field(ge=1)
    step: int = Field(ge=1)
    state_path: str = Field(min_length=1, max_length=2048)
    metric_count: int = Field(gt=0)
    cumulative_cost_usd: NumericMeasurement | None = None


class TinkerSFTResumeEvent(ContractModel):
    """An append-only record that a new attempt starts from a durable checkpoint."""

    record_type: Literal["resume"] = "resume"
    run_id: ArtifactId
    attempt_id: int = Field(ge=2)
    checkpoint_id: ArtifactId
    resumed_from_step: int = Field(ge=1)


class TinkerSFTStepIntent(ContractModel):
    """Durable proof that one non-idempotent optimizer update is about to dispatch."""

    run_id: ArtifactId
    attempt_id: int = Field(ge=1)
    step: int = Field(ge=1)
    epoch: int = Field(ge=1)
    batch_index: int = Field(ge=1)
    example_ids: tuple[ArtifactId, ...] = Field(min_length=1)
    conservative_cost_upper_bound_usd: NumericMeasurement | None = None


class TinkerSFTCheckpointIntent(ContractModel):
    """A pre-call marker preventing an ambiguous checkpoint save from being repeated."""

    run_id: ArtifactId
    checkpoint_id: ArtifactId
    attempt_id: int = Field(ge=1)
    step: int = Field(ge=1)
    checkpoint_name: str = Field(min_length=1, max_length=512)


class TinkerSFTModelIntent(ContractModel):
    """A pre-call marker for the non-idempotent terminal sampling-handle save."""

    run_id: ArtifactId
    model_name: str = Field(min_length=1, max_length=512)
    checkpoint_id: ArtifactId


class TinkerSFTModelArtifact(ArtifactEnvelope):
    """Completed handle plus its exact dataset, event log, and checkpoint lineage."""

    model_id: ArtifactId
    run_id: ArtifactId
    dataset_id: ArtifactId
    dataset_manifest_sha256: Sha256
    dataset_build_sha256: Sha256
    events_sha256: Sha256
    final_checkpoint_id: ArtifactId
    final_checkpoint_state_path: str = Field(min_length=1, max_length=2048)
    sampling_handle: str = Field(min_length=1, max_length=2048)
    training_step_count: int = Field(gt=0)
    training_metric_count: int = Field(gt=0)
    total_cost_usd: NumericMeasurement | None = None

    @model_validator(mode="after")
    def _require_exact_dataset_binding(self) -> TinkerSFTModelArtifact:
        expected_inputs = (
            ArtifactInput(artifact_id=self.dataset_id, sha256=self.dataset_manifest_sha256),
        )
        if self.inputs != expected_inputs:
            raise ValueError("Tinker SFT models must name exactly their frozen dataset build")
        expected_model_id = stable_id(
            "tinker-sft-model",
            {
                "run_id": self.run_id,
                "checkpoint_id": self.final_checkpoint_id,
                "sampling_handle": self.sampling_handle,
            },
        )
        if self.model_id != expected_model_id:
            raise ValueError("Tinker SFT model ID is not content-addressed")
        return self


class TinkerSFTResult(ArtifactEnvelope):
    """Terminal result naming the exact model, event log, and training facts."""

    result_id: ArtifactId
    run_id: ArtifactId
    dataset_id: ArtifactId
    dataset_manifest_sha256: Sha256
    dataset_build_sha256: Sha256
    model_id: ArtifactId
    model_sha256: Sha256
    events_sha256: Sha256
    final_checkpoint_id: ArtifactId
    training_step_count: int = Field(gt=0)
    training_metric_count: int = Field(gt=0)
    checkpoint_count: int = Field(gt=0)
    total_cost_usd: NumericMeasurement | None = None

    @model_validator(mode="after")
    def _require_exact_terminal_inputs(self) -> TinkerSFTResult:
        expected_inputs = tuple(
            sorted(
                (
                    ArtifactInput(
                        artifact_id=self.dataset_id,
                        sha256=self.dataset_manifest_sha256,
                    ),
                    ArtifactInput(artifact_id=self.model_id, sha256=self.model_sha256),
                ),
                key=lambda item: item.artifact_id,
            )
        )
        if self.inputs != expected_inputs:
            raise ValueError("Tinker SFT results must name their dataset and terminal model")
        expected_result_id = stable_id(
            "tinker-sft-result",
            {"run_id": self.run_id, "model_sha256": self.model_sha256},
        )
        if self.result_id != expected_result_id:
            raise ValueError("Tinker SFT result ID is not content-addressed")
        return self


type TinkerSFTEvent = Annotated[
    TinkerSFTMetric | TinkerSFTCheckpoint | TinkerSFTResumeEvent,
    Field(discriminator="record_type"),
]
TINKER_SFT_EVENT_ADAPTER = TypeAdapter(TinkerSFTEvent)
