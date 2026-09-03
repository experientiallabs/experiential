"""Provider-neutral errors and shared wire validation for completed provider responses."""

from __future__ import annotations

import asyncio
from enum import StrEnum

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.models.providers.transport import ProviderTransportError


class ProviderResponseError(ValueError):
    """A provider returned a completed response that violates EXP's typed contract."""


class ProviderRetryableResponseError(ProviderResponseError):
    """A completed response decoded to no usable output and merits one bounded re-dispatch.

    Reasoning models can consume an entire output budget on hidden reasoning and return
    neither visible text nor a tool call. The response is well formed on the wire, so the
    request is safe to dispatch again under the client's bounded retry policy.
    """


class ProviderRefusalSignal(StrEnum):
    """Content-free refusal signals normalized from provider-specific wire values."""

    CONTENT_POLICY = "content_policy"
    SAFETY = "safety"
    COPYRIGHT = "copyright"
    SENSITIVE_INFORMATION = "sensitive_information"
    GUARDRAIL = "guardrail"
    PROVIDER_REFUSAL = "provider_refusal"


class ProviderRefusalError(ProviderResponseError):
    """A provider refused the request without retaining or exposing refusal content."""

    def __init__(self, *, provider: str, signal: ProviderRefusalSignal) -> None:
        """Record only the provider family and normalized content-free signal.

        Args:
            provider: Stable provider family name.
            signal: Sanitized refusal category derived from the wire response.
        """
        super().__init__(f"{provider} refused the request")
        self.provider = provider
        self.signal = signal


class ProviderCapabilityError(ValueError):
    """A request requires gateway behavior the deployment cannot preserve."""

    def __init__(self, *, capability: str) -> None:
        """Name the unsupported public capability without provider content.

        Args:
            capability: Stable capability field rejected before dispatch.
        """
        super().__init__(f"provider deployment does not support {capability}")
        self.capability = capability


class ProviderParameterError(ValueError):
    """A public parameter cannot be preserved by the resolved model route."""

    def __init__(self, *, message: str, param: str, code: str) -> None:
        """Create one sanitized, field-specific pre-dispatch rejection."""
        super().__init__(message)
        self.param = param
        self.code = code


class UnsupportedReasoningEffortError(ProviderParameterError):
    """An explicit reasoning effort cannot be preserved by one model route."""

    def __init__(
        self,
        *,
        effort: str,
        supported_efforts: tuple[str, ...],
        param: str,
    ) -> None:
        """Build one field-specific, provider-neutral pre-dispatch rejection.

        Args:
            effort: Exact caller-provided reasoning effort.
            supported_efforts: Ordered effort values accepted by every route deployment.
            param: Public request path carrying the unsupported value.
        """
        if supported_efforts:
            choices = ", ".join(repr(value) for value in supported_efforts)
            message = (
                f"Reasoning effort {effort!r} is not supported by this model route. "
                f"Supported values: {choices}."
            )
        else:
            message = (
                f"The parameter {param!r} is not supported by this model route. "
                "Remove the field or choose a different model."
            )
        super().__init__(message=message, param=param, code="unsupported_parameter")


def normalized_provider_failure(exception: BaseException) -> GatewayFailure:
    """Convert an execution error into one stable sanitized gateway failure.

    Args:
        exception: Provider, transport, deadline, cancellation, or preflight failure.

    Returns:
        A content-free failure safe for public errors, ledgers, and logs.
    """
    if isinstance(exception, asyncio.CancelledError):
        return GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message=(
                "provider request was cancelled; resend the request if cancellation "
                "was not intended"
            ),
        )
    if isinstance(exception, ProviderDeadlineExceeded):
        return GatewayFailure(
            failure_class=GatewayFailureClass.TIMEOUT,
            safe_message=(
                "provider request deadline exceeded; retry with a shorter prompt "
                "or a smaller max_tokens value"
            ),
            failover_eligible=True,
        )
    if isinstance(exception, ProviderRefusalError):
        return GatewayFailure(
            failure_class=GatewayFailureClass.REFUSAL,
            safe_message="provider refused the request; revise the request content and retry",
            safe_details={"signal": exception.signal.value},
        )
    if isinstance(exception, ProviderParameterError):
        return GatewayFailure(
            failure_class=GatewayFailureClass.INVALID_REQUEST,
            safe_message=str(exception),
            safe_details={
                "code": exception.code,
                "param": exception.param,
            },
        )
    if isinstance(exception, ProviderCapabilityError):
        # Keep the internal literal for operator diagnostics, but never put it in
        # the public-safe message. Protocol boundaries translate known literals
        # to real request fields when they have the API-surface context required
        # to do so.
        return GatewayFailure(
            failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
            safe_message=(
                "provider deployment cannot preserve a requested capability; "
                "remove the unsupported field or request a different model alias"
            ),
            safe_details={"capability": exception.capability},
        )
    if isinstance(exception, ProviderTransportError):
        return _transport_failure(exception.status_code)
    if isinstance(exception, ProviderResponseError):
        return GatewayFailure(
            failure_class=GatewayFailureClass.MALFORMED_RESPONSE,
            safe_message="provider returned a malformed response; retry the request",
            failover_eligible=True,
        )
    return GatewayFailure(
        failure_class=GatewayFailureClass.INTERNAL,
        safe_message="provider execution failed; retry the request",
    )


def _transport_failure(status_code: int | None) -> GatewayFailure:
    """Classify one sanitized HTTP or connection failure by status only.

    Remaining 4xx statuses are the caller's request being rejected, so they map to
    ``INVALID_REQUEST``: the caller gets a corrective 400 and the deployment health
    circuit never counts the attempt against the deployment.

    Args:
        status_code: Provider HTTP status, or ``None`` for a connection failure.

    Returns:
        Stable retry and failover policy without response content.
    """
    details: JsonObject = {}
    if status_code is not None:
        details["status_code"] = status_code
    if status_code in {401, 403}:
        return GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
            safe_message=(
                "provider authentication failed; ask the gateway operator to verify "
                "the provider connection credential"
            ),
            failover_eligible=True,
            safe_details=details,
        )
    if status_code == 404:
        return GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_NOT_FOUND,
            safe_message=(
                "provider deployment was not found; ask the gateway operator to verify "
                "the deployment model ID in the catalog"
            ),
            failover_eligible=True,
            safe_details=details,
        )
    if status_code == 429:
        return GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message=(
                "provider throttled the request; retry after the delay in the Retry-After header"
            ),
            failover_eligible=True,
            safe_details=details,
        )
    if status_code == 402:
        # The provider ACCOUNT's billing state (trial quota exhausted, postpaid
        # billing disabled), never the caller's request fields: operator-
        # actionable deadness that fails over in every failover mode instead of
        # surfacing a corrective 400 to the caller.
        return GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_QUOTA,
            safe_message=(
                "provider account quota or billing is exhausted; ask the gateway "
                "operator to fund or enable the provider account"
            ),
            failover_eligible=True,
            safe_details=details,
        )
    if status_code == 408:
        return GatewayFailure(
            failure_class=GatewayFailureClass.TIMEOUT,
            safe_message="provider request timed out; retry the request",
            retryable_same_deployment=True,
            failover_eligible=True,
            safe_details=details,
        )
    if status_code is not None and status_code >= 500:
        return GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
            safe_message="provider service failed; retry after a short delay",
            retryable_same_deployment=True,
            failover_eligible=True,
            safe_details=details,
        )
    if status_code in {409, 425}:
        return GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
            safe_message="provider reported a transient conflict; retry the request",
            retryable_same_deployment=True,
            failover_eligible=True,
            safe_details=details,
        )
    if status_code is not None and 400 <= status_code < 500:
        return GatewayFailure(
            failure_class=GatewayFailureClass.INVALID_REQUEST,
            safe_message=(
                "provider rejected the request; verify the request fields against "
                "the model alias capabilities"
            ),
            safe_details=details,
        )
    if status_code is not None:
        return GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
            safe_message="provider returned an unexpected status; retry the request",
            failover_eligible=True,
            safe_details=details,
        )
    return GatewayFailure(
        failure_class=GatewayFailureClass.TRANSPORT,
        safe_message="provider transport failed; retry the request",
        retryable_same_deployment=True,
        failover_eligible=True,
    )


def require_array(value: JsonValue | None, label: str) -> list[JsonValue]:
    """Return a response JSON array or raise a focused conversion error.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value unchanged when it is a JSON array.

    Raises:
        ProviderResponseError: The value is not a JSON array.
    """
    if not isinstance(value, list):
        raise ProviderResponseError(f"{label} must be an array")
    return value


def require_object(value: JsonValue | None, label: str) -> JsonObject:
    """Return a response JSON object or raise a focused conversion error.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value unchanged when it is a JSON object.

    Raises:
        ProviderResponseError: The value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise ProviderResponseError(f"{label} must be an object")
    return value


def require_string(value: JsonValue | None, label: str) -> str:
    """Return a required non-empty response string or raise a focused conversion error.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value unchanged when it is a non-empty string.

    Raises:
        ProviderResponseError: The value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise ProviderResponseError(f"{label} must be a non-empty string")
    return value


def require_integer(value: JsonValue | None, label: str) -> int:
    """Read an optional non-negative response integer, counting an absent field as zero.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value when it is a non-negative integer, or zero when absent, because providers
        omit usage fields whose count is zero.

    Raises:
        ProviderResponseError: The value is present but not a non-negative integer.
    """
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderResponseError(f"{label} must be a non-negative integer")
    return value
