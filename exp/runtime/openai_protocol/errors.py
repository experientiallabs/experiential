"""Stable OpenAI-shaped public errors for the shared serving boundary."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass

THROTTLED_RETRY_AFTER_SECONDS = 5


class OpenAIErrorDetail(ContractModel):
    """One public OpenAI error without provider or credential details."""

    message: str = Field(min_length=1, max_length=2_048)
    type: Literal[
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "insufficient_quota",
        "api_error",
    ]
    param: str | None = Field(default=None, max_length=512)
    code: str = Field(min_length=1, max_length=128)


class OpenAIErrorEnvelope(ContractModel):
    """Top-level error shape parsed by official OpenAI clients."""

    error: OpenAIErrorDetail


class OpenAIProtocolError(ValueError):
    """Field-specific public protocol error carrying its HTTP representation."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        error_type: Literal[
            "invalid_request_error",
            "authentication_error",
            "permission_error",
            "insufficient_quota",
            "api_error",
        ] = "invalid_request_error",
        param: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        """Create one sanitized public protocol failure.

        Args:
            status_code: HTTP status associated with the error.
            code: Stable machine-readable gateway code.
            message: Display-safe explanation and remediation.
            error_type: OpenAI error category.
            param: Exact public request field responsible for the error.
            retry_after_seconds: Optional positive wait advertised as ``Retry-After``.
        """
        if retry_after_seconds is not None and retry_after_seconds <= 0:
            raise ValueError("retry_after_seconds must be positive")
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.detail = OpenAIErrorDetail(
            message=message,
            type=error_type,
            param=param,
            code=code,
        )

    def envelope(self) -> OpenAIErrorEnvelope:
        """Return the immutable OpenAI error envelope."""
        return OpenAIErrorEnvelope(error=self.detail)

    def json_body(self) -> JsonObject:
        """Return a JSON-compatible body suitable for an HTTP response."""
        return self.envelope().model_dump(mode="json")

    def headers(self) -> dict[str, str]:
        """Return transport headers implied by this error, such as ``Retry-After``."""
        if self.retry_after_seconds is None:
            return {}
        return {"Retry-After": str(self.retry_after_seconds)}


def invalid_field(param: str, message: str | None = None) -> OpenAIProtocolError:
    """Build one field-specific invalid-request error.

    Args:
        param: Public request field path.
        message: Optional safe explanation.

    Returns:
        Stable invalid-parameter error.
    """
    return OpenAIProtocolError(
        status_code=400,
        code="invalid_parameter",
        message=message or f"Invalid value for '{param}'.",
        param=param,
    )


def unsupported_field(param: str, *, capability: bool = False) -> OpenAIProtocolError:
    """Build one explicit unsupported field or capability error.

    Args:
        param: Public request field path.
        capability: Whether the field is conditionally supported by deployments.

    Returns:
        Stable pre-dispatch rejection.
    """
    code = "unsupported_capability" if capability else "unsupported_parameter"
    noun = "capability" if capability else "parameter"
    return OpenAIProtocolError(
        status_code=400,
        code=code,
        message=(
            f"The {noun} '{param}' is not supported by this gateway profile. "
            "Remove the field and resend the request."
        ),
        param=param,
    )


def public_failure_error(
    failure: GatewayFailure,
    *,
    param: str | None = None,
    now: datetime | None = None,
) -> OpenAIProtocolError:
    """Map one sanitized gateway failure to a stable public error.

    Quota exhaustion and provider throttling advertise a ``Retry-After`` wait. Gateway
    budgets are hard UTC-calendar-month allocations, so an exhausted quota reports the
    exact next month boundary as its reset time.

    Args:
        failure: Provider-neutral failure already stripped of sensitive details.
        param: Optional request field responsible for the failure.
        now: Injectable current UTC time used only to compute the quota reset boundary.

    Returns:
        OpenAI-shaped protocol error with no raw provider data.
    """
    mappings: dict[
        GatewayFailureClass,
        tuple[
            int,
            str,
            Literal[
                "invalid_request_error",
                "authentication_error",
                "permission_error",
                "insufficient_quota",
                "api_error",
            ],
        ],
    ] = {
        GatewayFailureClass.INVALID_REQUEST: (400, "invalid_request", "invalid_request_error"),
        GatewayFailureClass.UNSUPPORTED_CAPABILITY: (
            400,
            "unsupported_capability",
            "invalid_request_error",
        ),
        GatewayFailureClass.AUTHENTICATION: (401, "invalid_key", "authentication_error"),
        GatewayFailureClass.AUTHORIZATION: (403, "model_not_granted", "permission_error"),
        GatewayFailureClass.QUOTA_EXCEEDED: (429, "insufficient_quota", "insufficient_quota"),
        GatewayFailureClass.THROTTLED: (429, "unavailable_route", "api_error"),
        GatewayFailureClass.TIMEOUT: (504, "deadline_exceeded", "api_error"),
        GatewayFailureClass.CANCELLED: (499, "request_cancelled", "api_error"),
    }
    status, code, error_type = mappings.get(
        failure.failure_class,
        (502, "all_routes_failed", "api_error"),
    )
    message = failure.safe_message
    retry_after_seconds: int | None = None
    if failure.failure_class is GatewayFailureClass.THROTTLED:
        retry_after_seconds = THROTTLED_RETRY_AFTER_SECONDS
    elif failure.failure_class is GatewayFailureClass.QUOTA_EXCEEDED:
        moment = now if now is not None else datetime.now(UTC)
        reset = _next_utc_month_start(moment)
        retry_after_seconds = max(1, math.ceil((reset - moment).total_seconds()))
        boundary = reset.isoformat().replace("+00:00", "Z")
        message = (
            f"{message}. The allocation resets at {boundary}; retry after that time "
            "or ask the gateway operator to raise the monthly budget."
        )
    return OpenAIProtocolError(
        status_code=status,
        code=code,
        message=message,
        error_type=error_type,
        param=param,
        retry_after_seconds=retry_after_seconds,
    )


def _next_utc_month_start(moment: datetime) -> datetime:
    """Return the start of the UTC calendar month strictly after ``moment``.

    Args:
        moment: Timezone-aware current time.

    Returns:
        Midnight UTC on the first day of the following month.
    """
    anchored = moment.astimezone(UTC)
    if anchored.month == 12:
        return datetime(anchored.year + 1, 1, 1, tzinfo=UTC)
    return datetime(anchored.year, anchored.month + 1, 1, tzinfo=UTC)
