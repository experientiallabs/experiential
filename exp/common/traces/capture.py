"""Capture response and measurement contracts shared by passive and gateway observers."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from exp.common.core.artifacts import ContractModel


class CaptureJsonResponse(ContractModel):
    """A complete JSON response body, before host-specific presentation decoration.

    Attributes:
        kind: JSON response discriminator, always json.
        status: Observed HTTP response status.
        body: Queryable response projection.
        source_json: Exact escaped source when normalization was needed, otherwise None.
    """

    kind: Literal["json"] = "json"
    status: int = Field(ge=100, le=599)
    body: JsonValue
    source_json: str | None = None


class CaptureSseResponse(ContractModel):
    """Ordered SSE data payloads with explicit loss and disconnect indicators.

    Attributes:
        kind: Streaming response discriminator, always sse.
        status: Observed HTTP response status.
        frames: Ordered complete SSE data payloads.
        truncated: Whether capture limits excluded part of the response.
        client_disconnected: Whether the consumer stopped before body completion.
        source_json: Exact escaped frames when normalization was needed, otherwise None.
    """

    kind: Literal["sse"] = "sse"
    status: int = Field(ge=100, le=599)
    frames: tuple[JsonValue, ...]
    truncated: bool
    client_disconnected: bool
    source_json: str | None = None


class CaptureUsage(ContractModel):
    """Provider reported counts, with every absent counter remaining unknown.

    Attributes:
        input_tokens: Total input tokens, including cache subsets when reported.
        output_tokens: Total output tokens, including the reasoning subset.
        cached_input_tokens: Observed cache-read subset, otherwise None.
        reasoning_tokens: Observed reasoning subset, otherwise None.
        cache_creation_input_tokens: Cache-write subset, otherwise None.
        cache_creation_1h_input_tokens: One-hour cache-write subset, otherwise None.
    """

    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    cached_input_tokens: int | None = Field(ge=0)
    reasoning_tokens: int | None = Field(ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_1h_input_tokens: int | None = Field(default=None, ge=0)


class CaptureMetrics(ContractModel):
    """Measurements for one observed exchange, never aggregate fallback billing.

    Attributes:
        started_at: Observed exchange start as Unix seconds.
        first_token_at: Observed first output-token time, otherwise unknown.
        terminal_at: Provider terminal time, absent for an interrupted stream.
        duration_ms: Observed exchange duration, absent before a provider terminal.
        usage: Observed normalized provider counts; unknown fields remain None.
        usage_complete: Whether a terminal and credible complete token totals were observed.
    """

    started_at: float = Field(ge=0, allow_inf_nan=False)
    first_token_at: float | None = Field(ge=0, allow_inf_nan=False)
    terminal_at: float | None = Field(ge=0, allow_inf_nan=False)
    duration_ms: float | None = Field(ge=0, allow_inf_nan=False)
    usage: CaptureUsage | None
    usage_complete: bool
