"""Strict local application boundaries for captured continual-learning experience."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject

Identifier = Annotated[str, Field(min_length=1, max_length=512, pattern=r"\S")]
TokenId = Annotated[int, Field(strict=True, ge=0)]


class ClaasScope(ContractModel):
    """One local user and agent application whose learning data stays isolated."""

    user_id: Identifier
    application_id: Identifier


class CapturePolicy(ContractModel):
    """Explicit content-capture consent and finite retention for one application.

    Storage bytes bound retained serialized payloads; SQLite indexes and its write
    journal add overhead. Retention is enforced while the capture writer runs,
    and readers exclude expired records even when the writer is stopped.
    """

    scope: ClaasScope
    enabled: bool = False
    maximum_experiences: int = Field(default=10_000, strict=True, ge=1, le=1_000_000)
    maximum_storage_bytes: int = Field(default=268_435_456, strict=True, ge=1)
    maximum_experience_bytes: int = Field(default=1_048_576, strict=True, ge=1)
    retention_seconds: int = Field(default=604_800, strict=True, ge=1)

    @model_validator(mode="after")
    def _validate_storage_bounds(self) -> CapturePolicy:
        """Reject a per-experience ceiling larger than the entire retained store."""
        if self.maximum_experience_bytes > self.maximum_storage_bytes:
            raise ValueError("maximum_experience_bytes must not exceed maximum_storage_bytes")
        return self


class ExperienceProvenance(ContractModel):
    """The generating model and source evidence, without provider credentials."""

    source_kind: Literal["traffic", "simulation", "import"]
    source_id: Identifier
    model_id: Identifier
    model_revision: Identifier | None = None
    deployment_id: Identifier | None = None
    policy_revision: Identifier | None = None
    source_experience_ids: tuple[Identifier, ...] = ()

    @field_validator("source_experience_ids")
    @classmethod
    def _unique_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicated evidence links rather than silently changing lineage."""
        if len(value) != len(set(value)):
            raise ValueError("source_experience_ids must be unique")
        return value


class ExactTokenEvidence(ContractModel):
    """Original sampled tokens and rollout probabilities from a bound policy.

    These values must come from the generating backend. Retokenizing provider text
    does not supply this evidence and must leave an experience's field absent.
    """

    model_id: Identifier
    model_revision: Identifier
    policy_revision: Identifier
    tokenizer_id: Identifier
    tokenizer_revision: Identifier
    sampling_temperature: float = Field(strict=True, gt=0, allow_inf_nan=False)
    sampling_top_p: float = Field(strict=True, gt=0, le=1, allow_inf_nan=False)
    sampling_top_k: Annotated[int, Field(strict=True, ge=1)] | None
    prompt_token_ids: tuple[TokenId, ...] = Field(min_length=1)
    response_token_ids: tuple[TokenId, ...] = Field(min_length=1)
    response_logprobs: tuple[Annotated[float, Field(strict=True)], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_probabilities(self) -> ExactTokenEvidence:
        """Require one finite nonpositive rollout log probability per sampled token."""
        if len(self.response_token_ids) != len(self.response_logprobs):
            raise ValueError("response_token_ids and response_logprobs must have matching lengths")
        if any(not math.isfinite(value) or value > 0 for value in self.response_logprobs):
            raise ValueError("response_logprobs must be finite and nonpositive")
        return self


class Experience(ContractModel):
    """One completed protocol exchange available to asynchronous learning consumers.

    Request and response contain protocol body fields only. Transport headers and
    credentials are not part of this contract. Optional token evidence is never
    inferred from an arbitrary provider's textual output.
    """

    schema_version: Literal[1] = 1
    experience_id: Identifier
    response_id: Identifier
    episode_id: Identifier | None = None
    parent_response_id: Identifier | None = None
    scope: ClaasScope
    protocol: Literal["chat_completions", "responses"]
    captured_at: AwareDatetime
    request: JsonObject
    response: JsonObject
    provenance: ExperienceProvenance
    exact_tokens: ExactTokenEvidence | None = None

    @model_validator(mode="after")
    def _validate_token_provenance(self) -> Experience:
        """Prevent training evidence from naming another model or policy revision."""
        if self.exact_tokens is not None and (
            self.exact_tokens.model_id != self.provenance.model_id
            or self.exact_tokens.model_revision != self.provenance.model_revision
            or self.exact_tokens.policy_revision != self.provenance.policy_revision
        ):
            raise ValueError("exact_tokens must match the experience model and policy revision")
        return self
