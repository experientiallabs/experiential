"""Typed contracts for the asynchronous /v1/batches serving lane.

The batch lane is an explicit-request product: every JSONL line names a
batch-callable model, one provider serves one whole job, and results settle
per line through the host's accounting seam. These models are the frozen
boundary between the native routes, the batch engine, and the host's
persistence: they carry no secrets and no provider client state.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject

MAXIMUM_BATCH_LINES = 50_000
MAXIMUM_INPUT_FILE_BYTES = 100 * 1024 * 1024
COMPLETION_WINDOW = "24h"
COMPLETION_WINDOW_SECONDS = 24 * 60 * 60

BatchSurface = Literal["/v1/chat/completions", "/v1/responses", "/v1/messages"]

BATCH_SURFACES: tuple[BatchSurface, ...] = (
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages",
)


class BatchStatus(StrEnum):
    """Lifecycle states of one batch job, OpenAI Batch API compatible."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[BatchStatus] = frozenset(
    {
        BatchStatus.COMPLETED,
        BatchStatus.FAILED,
        BatchStatus.EXPIRED,
        BatchStatus.CANCELLED,
    }
)


class BatchLine(ContractModel):
    """One validated input line of a batch job.

    ``custom_id`` is the caller's per-line correlation key, unique inside one
    job. ``model`` is the catalog batch model the line explicitly requested,
    and ``provider_model`` is the provider wire id the job's provider serves
    it under. ``body`` is the surface-shaped request body passed through to
    the provider without dialect translation.
    """

    custom_id: str = Field(min_length=1, max_length=256)
    surface: BatchSurface
    model: str = Field(min_length=1, max_length=256)
    provider_model: str = Field(min_length=1, max_length=2_048)
    body: JsonObject
    estimated_input_tokens: int = Field(ge=0)
    maximum_output_tokens: int = Field(ge=0)
    reserved_micro_usd: int = Field(default=0, ge=0)


class BatchLineError(ContractModel):
    """One line rejected at submit validation, reported per line, never fatal."""

    line_number: int = Field(ge=1)
    custom_id: str | None = None
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2_048)


class BatchLineResult(ContractModel):
    """One settled output line, OpenAI batch output JSONL compatible.

    Exactly one of ``response`` and ``error`` is populated. ``usage`` carries
    the provider-reported token counts the host settles against.
    """

    custom_id: str
    status_code: int = Field(ge=100, le=599)
    response: JsonObject | None = None
    error: JsonObject | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    settled_micro_usd: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _require_exactly_one_payload(self) -> BatchLineResult:
        """Reject a result carrying both or neither of response and error."""
        if (self.response is None) == (self.error is None):
            raise ValueError("exactly one of response or error must be set")
        return self

    def output_jsonl_object(self, *, line_id: str) -> JsonObject:
        """Render the OpenAI batch output line for this result."""
        return {
            "id": line_id,
            "custom_id": self.custom_id,
            "response": (
                None
                if self.response is None
                else {"status_code": self.status_code, "body": self.response}
            ),
            "error": self.error,
        }


class BatchCounts(ContractModel):
    """Line counts mirrored into the public batch object."""

    total: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)


class BatchJob(ContractModel):
    """One batch job: identity, frozen submit-time facts, and live status.

    The job is content-free beyond the input/output file references: line
    bodies live in the host's file store, never in the job record.
    """

    batch_id: str = Field(min_length=1, max_length=128)
    organization_id: str = Field(min_length=1, max_length=128)
    identity_id: str = Field(min_length=1, max_length=128)
    surface: BatchSurface
    provider: str = Field(min_length=1, max_length=128)
    credential_reference: str = Field(min_length=1, max_length=512)
    dispatch_started: bool = False
    provider_batch_id: str | None = Field(default=None, max_length=256)
    input_file_id: str = Field(min_length=1, max_length=128)
    output_file_id: str | None = Field(default=None, max_length=128)
    error_file_id: str | None = Field(default=None, max_length=128)
    status: BatchStatus = BatchStatus.VALIDATING
    counts: BatchCounts = Field(default_factory=BatchCounts)
    lines: tuple[BatchLine, ...] = ()
    line_errors: tuple[BatchLineError, ...] = ()
    reserved_micro_usd: int = Field(default=0, ge=0)
    settled_micro_usd: int = Field(default=0, ge=0)
    failure_message: str | None = Field(default=None, max_length=2_048)
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: AwareDatetime
    expires_at: AwareDatetime
    finalized_at: AwareDatetime | None = None
    settled: bool = False

    def public_object(self) -> JsonObject:
        """Render the OpenAI Batch API object for this job.

        ``errors`` carries every reason the caller can act on: the submit-time
        per-line rejections, plus the job-level failure reason (a provider
        rejection, an interrupted dispatch, an elapsed window) under the
        terminal status as its code, the same code the error file stamps on
        each line that never ran. A terminal job also stamps the matching
        ``*_at`` timestamp, so a failed batch never reads as completed.
        """
        error_items: list[JsonObject] = [
            {
                "code": error.code,
                "message": error.message,
                "line": error.line_number,
                "custom_id": error.custom_id,
            }
            for error in self.line_errors
        ]
        if self.failure_message is not None:
            error_items.append(
                {
                    "code": self.status.value,
                    "message": self.failure_message,
                    "line": None,
                    "custom_id": None,
                }
            )
        errors: JsonObject | None = None
        if error_items:
            errors = {"object": "list", "data": error_items}
        finalized = None if self.finalized_at is None else int(self.finalized_at.timestamp())
        return {
            "id": self.batch_id,
            "object": "batch",
            "endpoint": self.surface,
            "errors": errors,
            "input_file_id": self.input_file_id,
            "completion_window": COMPLETION_WINDOW,
            "status": self.status.value,
            "output_file_id": self.output_file_id,
            "error_file_id": self.error_file_id,
            "created_at": int(self.created_at.timestamp()),
            "expires_at": int(self.expires_at.timestamp()),
            "completed_at": finalized if self.status is BatchStatus.COMPLETED else None,
            "failed_at": finalized if self.status is BatchStatus.FAILED else None,
            "expired_at": finalized if self.status is BatchStatus.EXPIRED else None,
            "cancelled_at": finalized if self.status is BatchStatus.CANCELLED else None,
            "request_counts": {
                "total": self.counts.total,
                "completed": self.counts.completed,
                "failed": self.counts.failed,
            },
            "metadata": self.metadata or None,
        }


class BatchFile(ContractModel):
    """Metadata for one stored batch input or output file."""

    file_id: str = Field(min_length=1, max_length=128)
    organization_id: str = Field(min_length=1, max_length=128)
    filename: str = Field(min_length=1, max_length=512)
    purpose: Literal["batch", "batch_output"] = "batch"
    size_bytes: int = Field(ge=0)
    created_at: AwareDatetime

    def public_object(self) -> JsonObject:
        """Render the OpenAI files object for this file."""
        return {
            "id": self.file_id,
            "object": "file",
            "bytes": self.size_bytes,
            "created_at": int(self.created_at.timestamp()),
            "filename": self.filename,
            "purpose": self.purpose,
        }


class BatchDeployment(ContractModel):
    """One batch-callable catalog model resolved by the host catalog seam.

    Prices are batch list prices in micro-USD per million tokens: the owner
    policy passes the provider batch discount through to the caller, so the
    host authors these rates on the batch catalog rows directly.
    """

    model: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=128)
    provider_model: str = Field(min_length=1, max_length=2_048)
    credential_reference: str = Field(min_length=1, max_length=512)
    surfaces: tuple[BatchSurface, ...] = Field(min_length=1)
    input_micro_usd_per_million_tokens: int = Field(ge=0)
    output_micro_usd_per_million_tokens: int = Field(ge=0)
    default_maximum_output_tokens: int = Field(default=4_096, gt=0)


class BatchSubmitError(Exception):
    """A whole-job submit rejection with an OpenAI-envelope error message."""

    def __init__(self, message: str, *, code: str = "invalid_request_error") -> None:
        """Bind the public message and stable error code."""
        super().__init__(message)
        self.code = code
        self.message = message


def parse_input_jsonl(payload: bytes) -> list[tuple[int, JsonObject]]:
    """Parse batch input JSONL into numbered raw line objects.

    Returns:
        One ``(line_number, object)`` pair per non-empty line, 1-indexed.

    Raises:
        BatchSubmitError: When the payload is not valid JSONL of objects or
            exceeds the size or line-count product limits.
    """
    if len(payload) > MAXIMUM_INPUT_FILE_BYTES:
        raise BatchSubmitError(
            f"batch input exceeds {MAXIMUM_INPUT_FILE_BYTES} bytes; split the job"
        )
    lines: list[tuple[int, JsonObject]] = []
    for line_number, raw in enumerate(payload.decode("utf-8", errors="strict").splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BatchSubmitError(f"line {line_number} is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise BatchSubmitError(f"line {line_number} must be a JSON object")
        lines.append((line_number, parsed))
    if not lines:
        raise BatchSubmitError("batch input carries no request lines")
    if len(lines) > MAXIMUM_BATCH_LINES:
        raise BatchSubmitError(
            f"batch input carries {len(lines)} lines; the limit is {MAXIMUM_BATCH_LINES}"
        )
    return lines
