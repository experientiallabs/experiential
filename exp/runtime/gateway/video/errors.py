"""Sanitized failures for video admission and durable job storage."""

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def unsupported_failure() -> GatewayFailure:
    """The terminal failure for an alias that does not generate video."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
        safe_message="the model alias does not generate video",
    )


def storage_failure() -> GatewayFailure:
    """The failure for a rung whose host upload destination could not be minted."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.INTERNAL,
        safe_message="the video output storage is unavailable",
        failover_eligible=True,
    )


def unavailable_error() -> OpenAIProtocolError:
    """The 503 for a video whose serving provider or storage is unreachable now."""
    return OpenAIProtocolError(
        status_code=503,
        code="unavailable_route",
        message="The provider serving this video is not available; retry shortly.",
        error_type="api_error",
    )


def internal_failure() -> GatewayFailure:
    """The terminal failure for a job row that could not be persisted."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.INTERNAL,
        safe_message="the video job could not be recorded",
    )
