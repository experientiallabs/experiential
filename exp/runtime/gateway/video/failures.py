"""Public errors and terminal failures of the video surface."""

from __future__ import annotations

from exp.runtime.gateway.stream_contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def not_a_video_model_error(alias: str) -> OpenAIProtocolError:
    """Build the public 400 for an alias none of whose rungs generate video."""
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            f"The model {alias!r} does not generate video. Choose a video model from "
            "GET /v1/models and resend the request to /v1/videos."
        ),
        param="model",
    )


def video_not_found_error(video_id: str) -> OpenAIProtocolError:
    """Build the public 404 shared by absent, deleted, and foreign video ids."""
    return OpenAIProtocolError(
        status_code=404,
        code="not_found",
        message=f"No video found with id {video_id!r}.",
        param="video_id",
    )


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
