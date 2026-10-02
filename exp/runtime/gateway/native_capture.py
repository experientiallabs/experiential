"""Typed configuration and admission adapter for the shared Rust capture collector."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, JsonValue, model_validator
from pydantic_core import to_json

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.capture_context import capture_context_document
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest

if TYPE_CHECKING:
    from exp_gateway_native import CaptureCollector

_LOGGER = logging.getLogger(__name__)
Identifier = Annotated[str, Field(min_length=1, max_length=512)]


class CaptureDeliveryLimits(ContractModel):
    """Budgets include the record currently being written by the destination.

    Attributes:
        maximum_records: Queued and actively written records, defaulting to 256.
        maximum_bytes: Queue and prepared-payload budget, defaulting to 64 MiB.
            One worker reserves 5 times maximum_record_bytes plus 256 bytes for
            UTF-8 encoding and the widest Python string. Queued records use the remainder.
        maximum_record_bytes: Final encoded payload ceiling, defaulting to 8 MiB.
    """

    maximum_records: int = Field(default=256, strict=True, ge=1, le=4096)
    maximum_bytes: int = Field(default=64 * 1024 * 1024, strict=True, ge=1, le=256 * 1024 * 1024)
    maximum_record_bytes: int = Field(
        default=8 * 1024 * 1024, strict=True, ge=1, le=16 * 1024 * 1024
    )

    @model_validator(mode="after")
    def _validate_budget(self) -> CaptureDeliveryLimits:
        """Reserve one worker's preparation allocation outside queued content."""
        if self.maximum_bytes < 6 * self.maximum_record_bytes + 256:
            raise ValueError("capture byte budget must fit destination preparation and one record")
        return self


class CaptureConfiguration(ContractModel):
    """Bounded native collection, with hosted settlement eligibility enabled by default.

    Attributes:
        delivery: Destination queue limits, using CaptureDeliveryLimits defaults.
        maximum_pending_records: In-flight request ceiling through destination
            acknowledgement, defaulting to 2048.
        maximum_pending_bytes: Pending request and handoff memory ceiling,
            defaulting to 64 MiB. Handoffs keep their charge while waiting for
            destination capacity or acknowledgement.
        maximum_request_bytes: Encoded request ceiling, defaulting to 4 MiB.
        maximum_response_bytes: Response buffer ceiling, defaulting to 3,670,016 bytes.
        ttl_seconds: Unsettled request lifetime, defaulting to 1800 seconds.
        settlement_required: Require hosted retention permission, true by default.
        asynchronous_delivery: Explicit hosted opt-in to admission-owned handoff instead
            of waiting for delivery capacity or durable storage, false by default.
            The host owns draining. Admission and response-memory limits still apply.
        relay_metadata: Wait for an outer relay's caller-facing metadata, false by default.
        truncate_request: Preserve the hosted bounded-copy policy for oversized inputs.
    """

    delivery: CaptureDeliveryLimits = Field(default_factory=CaptureDeliveryLimits)
    maximum_pending_records: int = Field(default=2048, strict=True, ge=1, le=4096)
    maximum_pending_bytes: int = Field(
        default=64 * 1024 * 1024, strict=True, ge=1, le=256 * 1024 * 1024
    )
    maximum_request_bytes: int = Field(
        default=4 * 1024 * 1024, strict=True, ge=1, le=8 * 1024 * 1024
    )
    maximum_response_bytes: int = Field(default=3_670_016, strict=True, ge=1, le=4 * 1024 * 1024)
    ttl_seconds: int = Field(default=1800, strict=True, ge=1, le=3600)
    settlement_required: bool = True
    asynchronous_delivery: bool = False
    relay_metadata: bool = False
    truncate_request: bool = False

    @model_validator(mode="after")
    def _validate_pending_budget(self) -> CaptureConfiguration:
        """Require room for one complete input and one reserved response buffer."""
        if self.maximum_pending_bytes < max(
            self.maximum_request_bytes, self.maximum_response_bytes
        ):
            raise ValueError("capture pending budget must fit one request or response")
        return self


class CaptureScope(ContractModel):
    """Authenticated identity plus its explicitly configured application binding.

    Attributes:
        organization_id: Authenticated organization identifier.
        identity_id: Authenticated gateway identity identifier.
        application_id: Host-configured application identifier for this identity.
    """

    organization_id: Identifier
    identity_id: Identifier
    application_id: Identifier


class CaptureRequest(ContractModel):
    """Versioned effective context, independent of transport credentials.

    Attributes:
        request_id: Unique authenticated request identifier.
        scope: Authority-derived organization, identity and application binding.
        protocol: Public HTTP request surface.
        model_id: Selected model identifier, unknown before routing.
        context: Effective post-guardrail request and capture-only provider evidence.
    """

    request_id: Identifier
    scope: CaptureScope
    protocol: Literal["chat_completions", "responses", "messages"]
    model_id: Identifier | None
    context: JsonObject


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
    """Native reported counts, with every absent counter remaining unknown.

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
    """Winning attempt observations, not aggregate billing across fallback attempts.

    Attributes:
        started_at: Attempt start as Unix seconds.
        first_token_at: Existing native first output-token observation, if any.
        terminal_at: Provider terminal time, absent for an interrupted stream.
        duration_ms: Monotonic attempt duration, absent before a provider terminal.
        usage: Observed normalized provider counts; unknown fields remain None.
        usage_complete: Whether a terminal and credible complete token totals were observed.
    """

    started_at: float = Field(ge=0, allow_inf_nan=False)
    first_token_at: float | None = Field(ge=0, allow_inf_nan=False)
    terminal_at: float | None = Field(ge=0, allow_inf_nan=False)
    duration_ms: float | None = Field(ge=0, allow_inf_nan=False)
    usage: CaptureUsage | None
    usage_complete: bool


class CaptureRecord(ContractModel):
    """One idempotent update delivered to a local or hosted persistence adapter.

    Attributes:
        schema_version: Persisted contract version, always 1.
        request: Authenticated scope and effective request context.
        response: Eligible observed output, otherwise None.
        deployment_id: Selected provider deployment identifier, when available.
        provider_reasoning: Permitted exposed reasoning, defaulting to None.
        provider_reasoning_source_json: Exact exceptional reasoning, defaulting to None.
        provider_tool_calls_json: Exact completed tool-call records, defaulting to None.
        captured_at: Admission timestamp as Unix seconds.
        metrics: Winning-attempt observations only when response retention permits them.
        gemini_thought_parts: Ordered provider summary and signature evidence, not full CoT.
        gemini_thought_parts_source_json: Exact exceptional parts, otherwise None.
        transport: Optional outer-relay headers, timing and redacted wire input.
    """

    schema_version: Literal[1]
    request: CaptureRequest
    response: (
        Annotated[CaptureJsonResponse | CaptureSseResponse, Field(discriminator="kind")] | None
    )
    deployment_id: str | None
    provider_reasoning: str | None = None
    provider_reasoning_source_json: str | None = None
    provider_tool_calls_json: str | None = None
    captured_at: float = Field(ge=0, allow_inf_nan=False)
    metrics: CaptureMetrics | None
    gemini_thought_parts: tuple[JsonObject, ...]
    gemini_thought_parts_source_json: str | None
    transport: JsonObject | None = None


class CaptureController:
    """Prepare effective input; Rust owns buffering, output assembly and delivery."""

    def __init__(
        self,
        native: CaptureCollector,
        *,
        application_for: Callable[[AuthorizationSnapshot], str | None],
    ) -> None:
        """Bind the collector to a host-owned authenticated capture-policy decision.

        Args:
            native: Shared collector also passed to the native server.
            application_for: Return a configured application or None to decline capture.
        """
        self.native = native
        self._application_for = application_for

    def begin(
        self,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
        model_id: str | None,
        *,
        session_id: str | None = None,
    ) -> bool:
        """Prepare only an allowed identity's post-guardrail, expanded request."""
        application_id = self._application_for(authorization)
        if application_id is None or request.surface.value not in {
            "chat_completions",
            "responses",
            "messages",
        }:
            return True
        context = capture_context_document(request, session_id=session_id)
        # Authority and effective context are already typed. Serialize that
        # projection once; the native admission boundary validates the envelope.
        # Revalidating JsonObject here would copy every tool/schema container.
        record: JsonObject = {
            "request_id": authorization.request_id,
            "scope": {
                "organization_id": authorization.organization_id,
                "identity_id": authorization.identity_id,
                "application_id": application_id,
            },
            "protocol": request.surface.value,
            "model_id": model_id,
            "context": context,
        }
        return self.native.begin_bytes(to_json(record, inf_nan_mode="null"))


def begin_capture(
    controller: CaptureController | None,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    model_id: str | None = None,
    *,
    session_id: str | None = None,
) -> bool:
    """Reject unavailable required capture without exposing customer content or exceptions."""
    if controller is None:
        return True
    try:
        return controller.begin(authorization, request, model_id, session_id=session_id)
    except Exception:  # noqa: BLE001 - sanitize policy and collector failures at admission.
        _LOGGER.warning("capture.admission_failed request_id=%s", authorization.request_id)
        return False


def select_capture_model(
    controller: CaptureController | None, request_id: str, model_id: str
) -> None:
    """Freeze resolved model provenance without interfering with provider dispatch."""
    if controller is None:
        return
    try:
        controller.native.select_model(request_id, model_id)
    except Exception:  # noqa: BLE001 - capture is observational only.
        _LOGGER.warning("capture.model_dropped request_id=%s", request_id)
