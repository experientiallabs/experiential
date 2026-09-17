"""Portable, token-exact training jobs for isolated application LoRA adapters.

A caller opens a backend with a frozen spec, trains a bound batch, then retains the
returned checkpoint. Sampling and promotion remain outside this training boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from exp.common.claas import ClaasScope, Experience
from exp.common.claas.contracts import Identifier
from exp.common.core.artifacts import ContractModel, Sha256, sha256_json

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class ClaasTrainingError(RuntimeError):
    """A job cannot safely execute or its completion cannot be established."""


class ClaasTrainingSpec(ContractModel):
    """Immutable model, adapter, exact objective, and finite execution limits.

    The worker supports dense text causal models on one CUDA device. SDPO uses
    full-vocabulary generalized Jensen-Shannon divergence with an EMA LoRA
    teacher. Scalar REINFORCE uses clipped, detached importance weights. Hybrid
    explicitly adds those objectives; no provider silently changes the recipe.
    """

    scope: ClaasScope
    adapter_id: Identifier
    base_model: Identifier
    model_revision: Identifier
    tokenizer_id: Identifier
    tokenizer_revision: Identifier
    initial_policy_revision: Identifier
    objective: Literal["sdpo", "reinforce", "hybrid"] = "sdpo"
    lora_rank: int = Field(default=16, strict=True, ge=1, le=256)
    lora_alpha: int = Field(default=32, strict=True, ge=1, le=512)
    target_modules: tuple[Identifier, ...] = ("q_proj", "v_proj")
    learning_rate: FiniteFloat = Field(default=1e-5, gt=0, le=1)
    sdpo_alpha: FiniteFloat = Field(default=0.5, ge=0, le=1)
    teacher_update_rate: FiniteFloat = Field(default=0.01, gt=0, le=1)
    importance_ratio_cap: FiniteFloat = Field(default=2.0, ge=1, le=100)
    scalar_loss_weight: FiniteFloat = Field(default=1.0, gt=0, le=100)
    max_gradient_norm: FiniteFloat = Field(default=1.0, gt=0, le=100)
    max_sequence_tokens: int = Field(default=8192, strict=True, ge=2, le=131072)
    max_batch_tokens: int = Field(default=32768, strict=True, ge=2)
    max_batch_examples: int = Field(default=32, strict=True, ge=1, le=4096)
    max_policy_lag: int = Field(default=2, strict=True, ge=0, le=1000)
    seed: int = Field(default=0, strict=True, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def _validate_modules(self) -> ClaasTrainingSpec:
        """Require an explicit unique set of LoRA target modules."""
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must be a nonempty unique set")
        return self


class TrainingExample(ContractModel):
    """A captured exact policy action with externally supplied scalar or text feedback."""

    experience: Experience
    scalar_reward: FiniteFloat | None = Field(default=None, ge=-1, le=1)
    text_feedback: str | None = Field(default=None, min_length=1, max_length=65536)

    @model_validator(mode="after")
    def _require_feedback(self) -> TrainingExample:
        """Reject examples without a usable learning signal."""
        if self.text_feedback is not None and not self.text_feedback.strip():
            raise ValueError("text_feedback must not be blank")
        if self.scalar_reward is None and self.text_feedback is None:
            raise ValueError("provide scalar_reward or text_feedback")
        return self


def teacher_feedback_text(feedback: str) -> str:
    """Format only newly supplied teacher context, without rewriting original rollout tokens."""
    return (
        "\n\nFeedback from a previous attempt:\n"
        + feedback
        + "\n\nUsing this feedback, produce the best response to the original request.\n"
    )


class TrainingBatch(ContractModel):
    """One optimizer update bound to the currently served policy revision."""

    batch_id: Identifier
    expected_policy_revision: Identifier
    examples: tuple[TrainingExample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_experiences(self) -> TrainingBatch:
        """Prevent repeated examples from silently changing a batch's weighting."""
        ids = tuple(item.experience.experience_id for item in self.examples)
        if len(set(ids)) != len(ids):
            raise ValueError("training batches must not repeat experience IDs")
        return self


class TrainingCheckpoint(ContractModel):
    """Immutable on-disk student, EMA teacher, and optimizer state for one application."""

    scope: ClaasScope
    adapter_id: Identifier
    policy_revision: Identifier
    policy_history: tuple[Identifier, ...] = Field(min_length=1)
    step: int = Field(strict=True, ge=0)
    path: str = Field(min_length=1)
    manifest_sha256: Sha256


class TrainingResult(ContractModel):
    """Completed optimizer evidence, without an evaluation or promotion claim."""

    checkpoint: TrainingCheckpoint
    consumed_experience_ids: tuple[Identifier, ...] = Field(min_length=1)
    metrics: dict[str, FiniteFloat]


class TrainingJob(ContractModel):
    """Serializable unit executed identically by a local process or a cloud adapter."""

    spec: ClaasTrainingSpec
    batch: TrainingBatch
    checkpoint_root: str = Field(min_length=1)
    resume_checkpoint: TrainingCheckpoint | None = None
    lineage_id: Identifier = "main"

    @model_validator(mode="after")
    def _validate_job(self) -> TrainingJob:
        """Validate all provider-independent invariants before creating compute."""
        if not self.lineage_id.strip():
            raise ValueError("lineage_id must not be blank")
        validate_training_batch(self.spec, self.batch, self.resume_checkpoint)
        if not Path(self.checkpoint_root).is_absolute():
            raise ValueError("checkpoint_root must be an absolute durable directory")
        return self


class TrainingSession(Protocol):
    """Lifecycle for one application adapter; callers serialize optimizer updates."""

    @property
    def policy_revision(self) -> str:
        """Return the exact revision required for the next training batch."""
        ...

    async def train(self, batch: TrainingBatch) -> TrainingResult:
        """Complete one optimizer update or fail without claiming a checkpoint."""
        ...

    async def checkpoint(self) -> TrainingCheckpoint:
        """Return the last complete checkpoint; a fresh untrained session has none."""
        ...

    async def close(self) -> None:
        """Release owned compute and prevent further updates."""
        ...


class ClaasTrainingBackend(Protocol):
    """Bind portable CLaaS jobs to a trainer that owns updates and native state."""

    async def open(
        self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None
    ) -> TrainingSession:
        """Bind a frozen adapter spec and optionally resume verified state."""
        ...


def validate_training_batch(
    spec: ClaasTrainingSpec, batch: TrainingBatch, resume: TrainingCheckpoint | None
) -> None:
    """Reject cross-application, stale, reconstructed, or unsupported training data.

    Arbitrary provider traffic can seed simulation but cannot stand in for an
    on-policy rollout. No student prompt or response is re-tokenized or truncated.
    """
    current = resume.policy_revision if resume else spec.initial_policy_revision
    recent = resume.policy_history[: spec.max_policy_lag + 1] if resume else (current,)
    if resume and (resume.scope != spec.scope or resume.adapter_id != spec.adapter_id):
        raise ValueError("checkpoint belongs to another application or adapter")
    if batch.expected_policy_revision != current:
        raise ValueError(
            "batch expected policy is stale; bind dispatch to the current adapter revision"
        )
    if len(batch.examples) > spec.max_batch_examples:
        raise ValueError("batch exceeds max_batch_examples; split it before dispatch")
    total_tokens = 0
    for item in batch.examples:
        experience = item.experience
        if experience.scope != spec.scope:
            raise ValueError("experience belongs to another user or application")
        tokens = experience.exact_tokens
        if tokens is None:
            raise ValueError(
                "original tokens/logprobs are missing; use this traffic as simulation seed"
            )
        if (
            tokens.sampling_temperature != 1.0
            or tokens.sampling_top_p != 1.0
            or tokens.sampling_top_k is not None
        ):
            raise ValueError(
                "training requires temperature=1, top_p=1, and no top_k sampling filter"
            )
        if (
            tokens.model_id != spec.base_model
            or tokens.model_revision != spec.model_revision
            or tokens.policy_revision not in recent
            or tokens.tokenizer_id != spec.tokenizer_id
            or tokens.tokenizer_revision != spec.tokenizer_revision
        ):
            raise ValueError(
                "rollout model, tokenizer, or policy differs; sample with the bound adapter"
            )
        length = len(tokens.prompt_token_ids) + len(tokens.response_token_ids)
        if length > spec.max_sequence_tokens:
            raise ValueError(
                "exact rollout exceeds max_sequence_tokens; truncation is not supported"
            )
        total_tokens += length
        if spec.objective in {"sdpo", "hybrid"} and item.text_feedback is None:
            raise ValueError(
                "SDPO and hybrid require text_feedback; select reinforce for scalar-only feedback"
            )
        if spec.objective in {"reinforce", "hybrid"} and item.scalar_reward is None:
            raise ValueError(
                "REINFORCE and hybrid require scalar_reward; select sdpo for text-only feedback"
            )
    if total_tokens > spec.max_batch_tokens:
        raise ValueError("batch exceeds max_batch_tokens; split it before dispatch")


def next_policy_revision(job: TrainingJob) -> str:
    """Bind a resulting revision to the exact spec, previous state, and consumed batch."""
    return "claas-" + sha256_json(
        {
            "spec": job.spec.model_dump(mode="json"),
            "batch": job.batch.model_dump(mode="json"),
            "lineage_id": job.lineage_id,
        }
    )
