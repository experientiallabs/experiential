"""Typed configuration and admission adapter for the shared Rust capture collector."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.capture_context import capture_request_context
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest

if TYPE_CHECKING:
    from exp_gateway_native import CaptureCollector

_LOGGER = logging.getLogger(__name__)
Identifier = Annotated[str, Field(min_length=1, max_length=512)]


class CaptureDeliveryLimits(ContractModel):
    """Budgets include the record currently being written by the destination."""

    maximum_records: int = Field(default=256, strict=True, ge=1, le=4096)
    maximum_bytes: int = Field(default=64 * 1024 * 1024, strict=True, ge=1, le=256 * 1024 * 1024)
    maximum_record_bytes: int = Field(
        default=8 * 1024 * 1024, strict=True, ge=1, le=8 * 1024 * 1024
    )

    @model_validator(mode="after")
    def _validate_budget(self) -> CaptureDeliveryLimits:
        """Require room for one maximum-size record."""
        if self.maximum_bytes < self.maximum_record_bytes:
            raise ValueError("capture byte budget must fit one record")
        return self


class CaptureConfiguration(ContractModel):
    """Bounded native collection, with hosted settlement eligibility enabled by default."""

    delivery: CaptureDeliveryLimits = Field(default_factory=CaptureDeliveryLimits)
    maximum_pending_records: int = Field(default=2048, strict=True, ge=1, le=4096)
    maximum_pending_bytes: int = Field(
        default=64 * 1024 * 1024, strict=True, ge=1, le=256 * 1024 * 1024
    )
    maximum_request_bytes: int = Field(default=1024 * 1024, strict=True, ge=1, le=1024 * 1024)
    maximum_response_bytes: int = Field(default=3_670_016, strict=True, ge=1, le=4 * 1024 * 1024)
    ttl_seconds: int = Field(default=1800, strict=True, ge=1, le=3600)
    settlement_required: bool = True

    @model_validator(mode="after")
    def _validate_pending_budget(self) -> CaptureConfiguration:
        """Require room for one maximum-size input."""
        if self.maximum_pending_bytes < self.maximum_request_bytes:
            raise ValueError("capture pending budget must fit one request")
        return self


class CaptureScope(ContractModel):
    """Authenticated identity plus its explicitly configured application binding."""

    organization_id: Identifier
    identity_id: Identifier
    application_id: Identifier


class CaptureRequest(ContractModel):
    """Versioned effective context, independent of transport credentials."""

    request_id: Identifier
    scope: CaptureScope
    protocol: Literal["chat_completions", "responses", "messages"]
    model_id: Identifier | None
    context: JsonObject


class CaptureJsonResponse(ContractModel):
    """A complete JSON response body, before host-specific presentation decoration."""

    kind: Literal["json"] = "json"
    status: int = Field(ge=100, le=599)
    body: JsonValue


class CaptureSseResponse(ContractModel):
    """Ordered SSE data payloads with explicit loss and disconnect indicators."""

    kind: Literal["sse"] = "sse"
    status: int = Field(ge=100, le=599)
    frames: tuple[JsonValue, ...]
    truncated: bool
    client_disconnected: bool


class CaptureRecord(ContractModel):
    """One idempotent update delivered to a local or hosted persistence adapter."""

    schema_version: Literal[1]
    request: CaptureRequest
    response: (
        Annotated[CaptureJsonResponse | CaptureSseResponse, Field(discriminator="kind")] | None
    )
    deployment_id: str | None
    captured_at: float = Field(ge=0, allow_inf_nan=False)


class CaptureController:
    """Prepare effective input; Rust owns buffering, output assembly and delivery."""

    def __init__(
        self,
        native: CaptureCollector,
        *,
        application_for: Callable[[AuthorizationSnapshot], str | None],
        maximum_request_bytes: int = 1_048_576,
    ) -> None:
        """Bind the collector to a host-owned authenticated capture-policy decision.

        Args:
            native: Shared collector also passed to the native server.
            application_for: Return a configured application or None to decline capture.
            maximum_request_bytes: Effective-context projection budget.
        """
        self.native = native
        self._application_for = application_for
        self._maximum_request_bytes = maximum_request_bytes

    def begin(
        self, authorization: AuthorizationSnapshot, request: GatewayRequest, model_id: str | None
    ) -> None:
        """Prepare only an allowed identity's post-guardrail, expanded request."""
        application_id = self._application_for(authorization)
        if application_id is None or request.surface.value not in {
            "chat_completions",
            "responses",
            "messages",
        }:
            return
        context = capture_request_context(request, maximum_bytes=self._maximum_request_bytes)
        if context is None:
            return
        record = CaptureRequest.model_validate(
            {
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
        )
        self.native.begin(record.model_dump_json())


def begin_capture(
    controller: CaptureController | None,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    model_id: str | None = None,
) -> None:
    """Contain optional capture failures without logging customer content or exceptions."""
    if controller is None:
        return
    try:
        controller.begin(authorization, request, model_id)
    except Exception:  # noqa: BLE001 - capture must never fail an otherwise valid admission.
        _LOGGER.warning("capture.admission_dropped request_id=%s", authorization.request_id)


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
