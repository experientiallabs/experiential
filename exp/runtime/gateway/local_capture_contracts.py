"""Identity-scoped records and retention contracts for local gateway traffic."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject

Identifier = Annotated[str, Field(min_length=1, max_length=512, pattern=r"\S")]


class LocalCaptureScope(ContractModel):
    """The authenticated identity and application owning a captured exchange.

    Attributes:
        user_id: Gateway identity ID derived from authentication, not request metadata.
        application_id: Application selected by the explicit local capture policy.
    """

    user_id: Identifier
    application_id: Identifier


class CapturePolicy(ContractModel):
    """Explicit content-capture consent and finite retention for one application.

    SQLite indexes and journals add overhead to the serialized payload bound.
    Readers exclude expired records even while the gateway writer is stopped.

    Attributes:
        scope: Authenticated identity/application partition.
        enabled: Whether to retain content; false until the local caller enables it.
        maximum_experiences: Retained record count, from 1 to 1,000,000; default 10,000.
        maximum_storage_bytes: Positive serialized payload budget; default 256 MiB.
        maximum_experience_bytes: Positive per-record ceiling; default 1 MiB.
        retention_seconds: Positive lifetime for each record; default seven days.
    """

    scope: LocalCaptureScope
    enabled: bool = False
    maximum_experiences: int = Field(default=10_000, strict=True, ge=1, le=1_000_000)
    maximum_storage_bytes: int = Field(default=268_435_456, strict=True, ge=1)
    maximum_experience_bytes: int = Field(default=1_048_576, strict=True, ge=1)
    retention_seconds: int = Field(default=604_800, strict=True, ge=1)

    @model_validator(mode="after")
    def _validate_storage_bounds(self) -> CapturePolicy:
        """Reject a per-record ceiling larger than the entire retained store."""
        if self.maximum_experience_bytes > self.maximum_storage_bytes:
            raise ValueError("maximum_experience_bytes must not exceed maximum_storage_bytes")
        return self


class CaptureProvenance(ContractModel):
    """Observed gateway request and serving model, without provider credentials.

    Attributes:
        source_id: Native gateway request ID linking this record to its exchange.
        model_id: Selected root model; actual fallback winner is separate output evidence.
        deployment_id: Observed provider deployment ID, or None when unavailable.
    """

    source_id: Identifier
    model_id: Identifier
    deployment_id: Identifier | None = None


class CapturedExchange(ContractModel):
    """One completed gateway exchange with its submitted or expanded conversation.

    Attributes:
        schema_version: Exact capture payload schema, currently 1.
        experience_id: Stable record ID derived from authenticated request provenance.
        response_id: Observed public response ID.
        episode_id: Optional explicit caller session label, scoped by authentication.
        parent_response_id: Explicit Responses continuation link, if present.
        scope: Authenticated identity/application owning the record.
        protocol: Captured HTTP surface: Chat Completions, Responses, or Messages.
        captured_at: Timezone-aware capture timestamp.
        request: Effective request and response evidence; never transport credentials.
        response: Complete public response with exact protocol content preserved.
        provenance: Observed request, model and deployment identifiers.
    """

    schema_version: Literal[1] = 1
    experience_id: Identifier
    response_id: Identifier
    episode_id: Identifier | None = None
    parent_response_id: Identifier | None = None
    scope: LocalCaptureScope
    protocol: Literal["chat_completions", "responses", "messages"]
    captured_at: AwareDatetime
    request: JsonObject
    response: JsonObject
    provenance: CaptureProvenance
