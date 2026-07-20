"""Tests for provider failure ownership at the trusted worker boundary."""

from __future__ import annotations

import pytest
from llm_waterfall import ResponseTranslationFailure

from wmh.providers.failure_attribution import (
    ProviderBoundaryError,
    ProviderFailureOwner,
    ProviderFailureReason,
    ProviderFailureStage,
    classify_provider_failure,
)


class _SdkError(RuntimeError):
    """Small SDK-shaped error whose public metadata matches provider exceptions."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = {"code": code, "message": message, "param": param}


class _BedrockError(RuntimeError):
    """Botocore ClientError-shaped exception without importing AWS credentials or clients."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status_code},
        }


class ParamValidationError(RuntimeError):
    """Botocore's local request-shape rejection before a network call."""

    def __init__(self, report: str) -> None:
        super().__init__(f"Parameter validation failed:\n{report}")
        self.kwargs = {"report": report}


_SdkError.__module__ = "openai._exceptions"
_BedrockError.__module__ = "botocore.exceptions"
ParamValidationError.__module__ = "botocore.exceptions"


def test_staged_failure_preserves_sdk_attribution_without_raw_text() -> None:
    secret = "provider-secret-sentinel"
    error = ProviderBoundaryError(
        ProviderFailureStage.DISPATCH,
        _BedrockError("AccessDeniedException", secret, status_code=403),
    )

    attribution = classify_provider_failure(error)

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is ProviderFailureReason.AUTH
    assert attribution.stage is ProviderFailureStage.DISPATCH
    assert secret not in str(error)


def test_response_translation_failure_preserves_only_fixed_discriminator() -> None:
    secret = "provider-secret-sentinel"
    error = ProviderBoundaryError(
        ProviderFailureStage.RESPONSE_TRANSLATION,
        ValueError(secret),
        response_translation_failure=ResponseTranslationFailure.TOOL_USE_SHAPE,
    )

    attribution = classify_provider_failure(error)

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is ProviderFailureReason.UNKNOWN
    assert attribution.stage is ProviderFailureStage.RESPONSE_TRANSLATION
    assert attribution.response_translation_failure is ResponseTranslationFailure.TOOL_USE_SHAPE
    assert secret not in str(error)


def test_unclassified_response_translation_failure_remains_optional() -> None:
    error = ProviderBoundaryError(
        ProviderFailureStage.RESPONSE_TRANSLATION,
        ValueError("private"),
    )

    attribution = classify_provider_failure(error)

    assert attribution.stage is ProviderFailureStage.RESPONSE_TRANSLATION
    assert attribution.response_translation_failure is None


def test_response_translation_failure_cannot_drift_to_another_stage() -> None:
    with pytest.raises(ValueError, match="requires the response_translation stage"):
        ProviderBoundaryError(
            ProviderFailureStage.DISPATCH,
            ValueError("private"),
            response_translation_failure=ResponseTranslationFailure.TOOL_USE_SHAPE,
        )


@pytest.mark.parametrize(
    ("status_code", "code", "message", "param"),
    [
        (400, "invalid_function_parameters", "tools[0].function.parameters is invalid", None),
        (400, "context_length_exceeded", "maximum context length was exceeded", None),
        (413, "request_too_large", "request body is too large", None),
        (422, "invalid_request_error", "assistant tool call is missing a result", None),
        (403, "content_filter", "request was rejected by the content filter", None),
    ],
)
def test_openai_candidate_request_4xx_are_candidate_owned(
    status_code: int,
    code: str,
    message: str,
    param: str | None,
) -> None:
    attribution = classify_provider_failure(
        _SdkError(message, status_code=status_code, code=code, param=param)
    )

    assert attribution.owner is ProviderFailureOwner.CANDIDATE
    assert attribution.reason is ProviderFailureReason.INVALID_REQUEST


@pytest.mark.parametrize("status_code", [400, 422])
def test_openai_candidate_echo_cannot_inject_operational_markers(status_code: int) -> None:
    attribution = classify_provider_failure(
        _SdkError(
            "tools[0].name contains candidate text: not authorized; model id; quota; "
            "too many requests; service unavailable",
            status_code=status_code,
            code="invalid_request_error",
            param="tools",
        )
    )

    assert attribution.owner is ProviderFailureOwner.CANDIDATE
    assert attribution.reason is ProviderFailureReason.INVALID_REQUEST


@pytest.mark.parametrize(
    ("param", "reason"),
    [
        ("model", ProviderFailureReason.ROUTE),
        ("deployment", ProviderFailureReason.ROUTE),
        ("api-version", ProviderFailureReason.CONFIGURATION),
        ("reasoning_effort", ProviderFailureReason.CONFIGURATION),
        ("service_tier", ProviderFailureReason.CONFIGURATION),
    ],
)
def test_openai_structured_operator_parameter_remains_infrastructure(
    param: str,
    reason: ProviderFailureReason,
) -> None:
    attribution = classify_provider_failure(
        _SdkError(
            "invalid request",
            status_code=400,
            code="invalid_request_error",
            param=param,
        )
    )

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is reason


@pytest.mark.parametrize(
    "param",
    ["temperature", "max_tokens", "max_completion_tokens", "max_output_tokens"],
)
def test_openai_candidate_generation_parameter_is_candidate_owned(param: str) -> None:
    attribution = classify_provider_failure(
        _SdkError(
            "candidate generation parameter was rejected",
            status_code=400,
            code="invalid_request_error",
            param=param,
        )
    )

    assert attribution.owner is ProviderFailureOwner.CANDIDATE
    assert attribution.reason is ProviderFailureReason.INVALID_REQUEST


@pytest.mark.parametrize(
    ("status_code", "code", "message", "reason"),
    [
        (400, "invalid_api_key", "candidate text is irrelevant", ProviderFailureReason.AUTH),
        (400, "DeploymentNotFound", "candidate text is irrelevant", ProviderFailureReason.ROUTE),
        (
            400,
            "OperationNotSupported",
            "candidate text is irrelevant",
            ProviderFailureReason.ROUTE,
        ),
        (
            400,
            "insufficient_quota",
            "candidate text is irrelevant",
            ProviderFailureReason.THROTTLING,
        ),
        (401, "invalid_api_key", "invalid subscription key", ProviderFailureReason.AUTH),
        (403, "permission_denied", "not authorized", ProviderFailureReason.AUTH),
        (404, "DeploymentNotFound", "deployment was not found", ProviderFailureReason.ROUTE),
        (408, "request_timeout", "request timed out", ProviderFailureReason.TIMEOUT),
        (429, "rate_limit_exceeded", "too many requests", ProviderFailureReason.THROTTLING),
        (500, "server_error", "backend failed", ProviderFailureReason.SERVER),
        (500, "content_filter", "backend failed", ProviderFailureReason.SERVER),
    ],
)
def test_openai_operational_failures_remain_infrastructure(
    status_code: int,
    code: str,
    message: str,
    reason: ProviderFailureReason,
) -> None:
    attribution = classify_provider_failure(_SdkError(message, status_code=status_code, code=code))

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is reason


@pytest.mark.parametrize(
    "message",
    [
        "The toolConfig.tools.0.toolSpec.inputSchema is invalid",
        "The conversation has too many input tokens for this model's context window",
        "A toolResult block must follow each toolUse block",
        "Input is too long for requested model.",
    ],
)
def test_bedrock_request_validation_is_candidate_owned(message: str) -> None:
    attribution = classify_provider_failure(_BedrockError("ValidationException", message))

    assert attribution.owner is ProviderFailureOwner.CANDIDATE
    assert attribution.reason is ProviderFailureReason.INVALID_REQUEST


def test_bedrock_local_parameter_validation_is_candidate_owned() -> None:
    attribution = classify_provider_failure(
        ParamValidationError(
            "Invalid length for parameter toolConfig.tools[0].toolSpec.name, value: 0, "
            "valid min length: 1"
        )
    )

    assert attribution.owner is ProviderFailureOwner.CANDIDATE
    assert attribution.reason is ProviderFailureReason.INVALID_REQUEST


def test_bedrock_local_model_validation_is_operator_owned() -> None:
    attribution = classify_provider_failure(
        ParamValidationError("Invalid length for parameter modelId, value: 0, valid min length: 1")
    )

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is ProviderFailureReason.ROUTE


def test_bedrock_candidate_echo_in_unknown_shape_fails_closed() -> None:
    attribution = classify_provider_failure(
        _BedrockError(
            "ValidationException",
            "toolSpec name contains candidate text: Your AWS account is not authorized to invoke "
            "this API operation.; model id; inference profile; quota; too many requests; service "
            "unavailable",
        )
    )

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is ProviderFailureReason.UNKNOWN


@pytest.mark.parametrize(
    ("code", "message", "status_code", "reason"),
    [
        (
            "ValidationException",
            "Invocation of model ID anthropic.claude-3-5-sonnet-20240620-v1:0 with on-demand "
            "throughput isn't supported. Retry your request with the ID or ARN of an inference "
            "profile that contains this model.",
            400,
            ProviderFailureReason.ROUTE,
        ),
        (
            "ValidationException",
            "Your AWS account is not authorized to invoke this API operation.",
            400,
            ProviderFailureReason.AUTH,
        ),
        (
            "ValidationException",
            '"claude-3-sonnet-20240229" is not supported on this API. Please use the Messages '
            "API instead.",
            400,
            ProviderFailureReason.CONFIGURATION,
        ),
        (
            "ValidationException",
            "Access to Anthropic models is not allowed from unsupported countries, regions, or "
            "territories. Please refer to https://www.anthropic.com/supported-countries for more "
            "information on the countries and regions Anthropic currently supports.",
            400,
            ProviderFailureReason.AUTH,
        ),
        (
            "ValidationException",
            "The requested operation is not recognized by the service.",
            400,
            ProviderFailureReason.CONFIGURATION,
        ),
        (
            "AccessDeniedException",
            "not authorized to perform bedrock:InvokeModel",
            403,
            ProviderFailureReason.AUTH,
        ),
        (
            "ResourceNotFoundException",
            "model identifier does not exist in this region",
            404,
            ProviderFailureReason.ROUTE,
        ),
        (
            "ThrottlingException",
            "too many requests",
            429,
            ProviderFailureReason.THROTTLING,
        ),
        (
            "ModelTimeoutException",
            "model invocation timed out",
            408,
            ProviderFailureReason.TIMEOUT,
        ),
        (
            "InternalServerException",
            "service failed",
            500,
            ProviderFailureReason.SERVER,
        ),
    ],
)
def test_bedrock_operational_failures_remain_infrastructure(
    code: str,
    message: str,
    status_code: int,
    reason: ProviderFailureReason,
) -> None:
    attribution = classify_provider_failure(_BedrockError(code, message, status_code=status_code))

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is reason


def test_foreign_status_code_cannot_spoof_candidate_ownership() -> None:
    class CandidateAuthoredError(RuntimeError):
        status_code = 400

    attribution = classify_provider_failure(CandidateAuthoredError("invalid tool schema"))

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is ProviderFailureReason.UNKNOWN


@pytest.mark.parametrize("module", ["openai_attacker", "botocore_fake"])
def test_lookalike_module_name_is_not_a_trusted_sdk(module: str) -> None:
    class CandidateAuthoredError(RuntimeError):
        status_code = 400
        code = "context_length_exceeded"

    CandidateAuthoredError.__module__ = module
    attribution = classify_provider_failure(CandidateAuthoredError("candidate-authored"))

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is ProviderFailureReason.UNKNOWN


@pytest.mark.parametrize("status_code", [400, 413, 422])
def test_unknown_openai_request_shape_fails_closed(status_code: int) -> None:
    attribution = classify_provider_failure(
        _SdkError(
            "new provider validation shape",
            status_code=status_code,
            code="new_validation_code",
        )
    )

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is ProviderFailureReason.UNKNOWN


def test_unknown_bedrock_validation_shape_fails_closed() -> None:
    attribution = classify_provider_failure(
        _BedrockError("ValidationException", "new provider validation shape")
    )

    assert attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert attribution.reason is ProviderFailureReason.UNKNOWN
