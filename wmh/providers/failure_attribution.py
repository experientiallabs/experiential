"""Classify provider exceptions by ownership without exposing provider error text.

Only trusted SDK exception shapes are interpreted. Candidate-owned failures are limited to
requests whose content came from the harness, such as message history, context size, tool schema,
or tool-use ordering. Credentials, routes, model availability, throttling, timeouts, transport,
and server failures remain evaluator infrastructure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from llm_waterfall import ResponseTranslationFailure


class ProviderFailureOwner(StrEnum):
    """Which side of the evaluation boundary owns a failed provider call."""

    CANDIDATE = "candidate"
    INFRASTRUCTURE = "infrastructure"


class ProviderFailureStage(StrEnum):
    """Bounded provider-neutral boundary at which one request failed."""

    CLIENT_INIT = "client_init"
    REQUEST_TRANSLATION = "request_translation"
    DISPATCH = "dispatch"
    RESPONSE_TRANSLATION = "response_translation"
    RECEIPT = "receipt"
    BUDGET = "budget"
    UNKNOWN = "unknown"


class ProviderFailureReason(StrEnum):
    """Sanitized provider failure class retained for control flow and audit evidence."""

    INVALID_REQUEST = "invalid_request"
    AUTH = "auth"
    ROUTE = "route"
    CONFIGURATION = "configuration"
    THROTTLING = "throttling"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    SERVER = "server"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderFailureAttribution:
    """Stable ownership and reason for one provider exception."""

    owner: ProviderFailureOwner
    reason: ProviderFailureReason
    stage: ProviderFailureStage = ProviderFailureStage.UNKNOWN
    response_translation_failure: ResponseTranslationFailure | None = None

    def __post_init__(self) -> None:
        if (
            self.stage is not ProviderFailureStage.RESPONSE_TRANSLATION
            and self.response_translation_failure is not None
        ):
            raise ValueError("response translation failure requires the response_translation stage")


class ProviderBoundaryError(RuntimeError):
    """Attach a bounded stage while retaining raw provider detail only in-process."""

    def __init__(
        self,
        stage: ProviderFailureStage,
        cause: Exception,
        *,
        response_translation_failure: ResponseTranslationFailure | None = None,
    ) -> None:
        if not isinstance(stage, ProviderFailureStage):
            raise TypeError("provider failure stage must be a ProviderFailureStage")
        if (
            stage is not ProviderFailureStage.RESPONSE_TRANSLATION
            and response_translation_failure is not None
        ):
            raise ValueError("response translation failure requires the response_translation stage")
        super().__init__(f"provider request failed during {stage.value}")
        self.stage = stage
        self.cause = cause
        self.response_translation_failure = response_translation_failure


_INFRASTRUCTURE_UNKNOWN = ProviderFailureAttribution(
    ProviderFailureOwner.INFRASTRUCTURE,
    ProviderFailureReason.UNKNOWN,
)
_CANDIDATE_REQUEST = ProviderFailureAttribution(
    ProviderFailureOwner.CANDIDATE,
    ProviderFailureReason.INVALID_REQUEST,
)

_OPENAI_MODULE_ROOTS = ("openai", "azure.ai.openai")
_BEDROCK_MODULE_ROOTS = ("botocore", "boto3")

_OPENAI_CANDIDATE_CODES = frozenset(
    {
        "content_filter",
        "content_policy_violation",
        "context_length_exceeded",
        "invalid_function_parameters",
        "request_too_large",
        "responsibleaipolicyviolation",
    }
)
_OPENAI_AUTH_CODES = frozenset(
    {
        "authentication_error",
        "forbidden",
        "invalid_api_key",
        "permission_denied",
        "unauthorized",
    }
)
_OPENAI_ROUTE_CODES = frozenset(
    {
        "deployment_not_found",
        "deploymentnotfound",
        "invalid_api_version",
        "invalid_model",
        "model_not_found",
        "operation_not_supported",
        "operationnotsupported",
        "unsupported_model",
    }
)
_OPENAI_THROTTLING_CODES = frozenset(
    {
        "insufficient_quota",
        "quota_exceeded",
        "rate_limit_error",
        "rate_limit_exceeded",
    }
)
_OPENAI_TIMEOUT_CODES = frozenset({"request_timeout", "timeout"})
_OPENAI_SERVER_CODES = frozenset(
    {"bad_gateway", "internal_server_error", "server_error", "service_unavailable"}
)
_OPENAI_TRANSPORT_TYPES = frozenset(
    {"apiconnectionclosederror", "apiconnectionerror", "connectionerror"}
)
_OPENAI_TIMEOUT_TYPES = frozenset({"apitimeouterror", "timeouterror"})
_OPENAI_ROUTE_PARAMETERS = frozenset({"deployment", "model"})
_OPENAI_CONFIGURATION_PARAMETERS = frozenset(
    {
        "api_version",
        "reasoning_effort",
        "service_tier",
    }
)
_OPENAI_CANDIDATE_PARAMETERS = frozenset(
    {
        "functions",
        "input",
        "instructions",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "messages",
        "parallel_tool_calls",
        "response_format",
        "stop",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
    }
)

_BEDROCK_AUTH_CODES = frozenset(
    {
        "accessdeniedexception",
        "expiredtokenexception",
        "invalidsignatureexception",
        "missingauthenticationtokenexception",
        "signaturedoesnotmatchexception",
        "unrecognizedclientexception",
    }
)
_BEDROCK_ROUTE_CODES = frozenset(
    {
        "modelnotfoundexception",
        "resourcenotfoundexception",
        "unsupportedoperationexception",
    }
)
_BEDROCK_THROTTLING_CODES = frozenset(
    {
        "limitexceededexception",
        "modelnotreadyexception",
        "servicequotaexceededexception",
        "throttlingexception",
        "toomanyrequestsexception",
    }
)
_BEDROCK_TIMEOUT_CODES = frozenset({"modeltimeoutexception", "requesttimeoutexception"})
_BEDROCK_SERVER_CODES = frozenset(
    {
        "internalfailure",
        "internalserverexception",
        "serviceunavailableexception",
    }
)
_BEDROCK_TRANSPORT_TYPES = frozenset(
    {"connectionclosederror", "endpointconnectionerror", "protocolerror"}
)
_BEDROCK_TIMEOUT_TYPES = frozenset({"connecttimeouterror", "readtimeouterror", "timeouterror"})
_BEDROCK_PARAMETER_VALIDATION_TYPES = frozenset({"paramvalidationerror"})
_BEDROCK_CANDIDATE_PARAMETER_ROOTS = frozenset(
    {
        "additionalModelRequestFields",
        "additionalModelResponseFieldPaths",
        "guardrailConfig",
        "inferenceConfig",
        "messages",
        "performanceConfig",
        "promptVariables",
        "requestMetadata",
        "system",
        "toolConfig",
    }
)
_BEDROCK_PARAMETER_PATH_TEMPLATES = (
    re.compile(
        r"Invalid (?:length|range|type|value) for parameter "
        r"(?P<path>[A-Za-z][A-Za-z0-9]*(?:\[\d+\]|\.[A-Za-z][A-Za-z0-9]*)*)(?:,.*)?"
    ),
    re.compile(
        r'Missing required parameter in input: "'
        r'(?P<path>[A-Za-z][A-Za-z0-9]*(?:\[\d+\]|\.[A-Za-z][A-Za-z0-9]*)*)"'
    ),
)

# Bedrock overloads ValidationException for both candidate-controlled request bodies and
# operator-controlled account/model configuration. Only known whole provider templates cross that
# boundary. Candidate values echoed inside a validation message cannot change ownership by merely
# containing an operational phrase.
_BEDROCK_VALIDATION_OPERATIONAL_TEMPLATES = (
    (
        re.compile(
            r"Invocation of model ID [A-Za-z0-9._:/-]{1,512} with on-demand throughput "
            r"(?:is not|isn't) supported(?:; use an inference profile|\. Retry your request with "
            r"the ID or ARN of an inference profile that contains this model\.)"
        ),
        ProviderFailureReason.ROUTE,
    ),
    (re.compile(r"The provided model identifier is invalid\."), ProviderFailureReason.ROUTE),
    (
        re.compile(
            r"(?:Your AWS account|Your account) is not authorized to invoke this API operation\."
        ),
        ProviderFailureReason.AUTH,
    ),
    (
        re.compile(r"You don't have access to the model with the specified model ID\."),
        ProviderFailureReason.AUTH,
    ),
    (
        re.compile(
            r'"[A-Za-z0-9._:/-]{1,512}" is not supported on this API\. ?'
            r"Please use the Messages API instead\."
        ),
        ProviderFailureReason.CONFIGURATION,
    ),
    (
        re.compile(
            r"Access to Anthropic models is not allowed from unsupported countries, regions, "
            r"or territories\. Please refer to https://www\.anthropic\.com/supported-countries "
            r"for more information(?: on the countries and regions Anthropic currently "
            r"supports)?\."
        ),
        ProviderFailureReason.AUTH,
    ),
    (
        re.compile(r"The requested operation is not recognized by the service\."),
        ProviderFailureReason.CONFIGURATION,
    ),
)
_BEDROCK_VALIDATION_CANDIDATE_TEMPLATES = (
    re.compile(r"The toolConfig\.tools\.\d+\.toolSpec\.inputSchema is invalid"),
    re.compile(r"The conversation has too many input tokens for this model's context window"),
    re.compile(r"A toolResult block must follow each toolUse block"),
    re.compile(r"Input is too long for requested model\."),
)


def classify_provider_failure(error: Exception) -> ProviderFailureAttribution:
    """Classify an SDK exception at the credential-bearing worker boundary.

    Unknown exceptions fail closed as infrastructure. The exception text is used only inside this
    classifier and is never copied into the returned value or candidate-visible evidence.
    """
    stage = ProviderFailureStage.UNKNOWN
    response_translation_failure: ResponseTranslationFailure | None = None
    if isinstance(error, ProviderBoundaryError):
        stage = error.stage
        response_translation_failure = error.response_translation_failure
        error = error.cause
    module = type(error).__module__.lower()
    if _module_has_root(module, _BEDROCK_MODULE_ROOTS):
        attribution = _classify_bedrock(error)
    elif _module_has_root(module, _OPENAI_MODULE_ROOTS):
        attribution = _classify_openai(error)
    else:
        attribution = _INFRASTRUCTURE_UNKNOWN
    return ProviderFailureAttribution(
        owner=attribution.owner,
        reason=attribution.reason,
        stage=stage,
        response_translation_failure=response_translation_failure,
    )


def _classify_openai(error: Exception) -> ProviderFailureAttribution:
    status = _status_code(error)
    code = _error_code(error).casefold()
    error_type = type(error).__name__.casefold()

    status_failure = _operational_from_status(status)
    content_policy_rejection = status == 403 and code in _OPENAI_CANDIDATE_CODES
    if status_failure is not None and not content_policy_rejection:
        return status_failure
    operational = _operational_from_code(
        code,
        auth=_OPENAI_AUTH_CODES,
        route=_OPENAI_ROUTE_CODES,
        throttling=_OPENAI_THROTTLING_CODES,
        timeout=_OPENAI_TIMEOUT_CODES,
        server=_OPENAI_SERVER_CODES,
    )
    if operational is not None:
        return operational
    parameter = _error_parameter(error)
    if parameter in _OPENAI_ROUTE_PARAMETERS:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.ROUTE,
        )
    if parameter in _OPENAI_CONFIGURATION_PARAMETERS:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.CONFIGURATION,
        )
    if (
        code in _OPENAI_CANDIDATE_CODES
        or status == 422
        and code == "invalid_request_error"
        or parameter in _OPENAI_CANDIDATE_PARAMETERS
    ):
        return _CANDIDATE_REQUEST
    operational = _operational_from_type(
        error_type,
        transport=_OPENAI_TRANSPORT_TYPES,
        timeout=_OPENAI_TIMEOUT_TYPES,
    )
    if operational is not None:
        return operational
    return _INFRASTRUCTURE_UNKNOWN


def _classify_bedrock(error: Exception) -> ProviderFailureAttribution:
    code = _error_code(error).casefold()
    status = _status_code(error)
    error_type = type(error).__name__.casefold()

    operational = _operational_from_status(status)
    if operational is not None:
        return operational
    if error_type in _BEDROCK_PARAMETER_VALIDATION_TYPES:
        return _classify_bedrock_parameter_validation(error)
    operational = _operational_from_code(
        code,
        auth=_BEDROCK_AUTH_CODES,
        route=_BEDROCK_ROUTE_CODES,
        throttling=_BEDROCK_THROTTLING_CODES,
        timeout=_BEDROCK_TIMEOUT_CODES,
        server=_BEDROCK_SERVER_CODES,
    )
    if operational is not None:
        return operational
    if code == "validationexception":
        message = _provider_message(error)
        reason = _bedrock_validation_operational_reason(message)
        if reason is not None:
            return ProviderFailureAttribution(
                ProviderFailureOwner.INFRASTRUCTURE,
                reason,
            )
        if _is_bedrock_candidate_validation(message):
            return _CANDIDATE_REQUEST
        return _INFRASTRUCTURE_UNKNOWN
    operational = _operational_from_type(
        error_type,
        transport=_BEDROCK_TRANSPORT_TYPES,
        timeout=_BEDROCK_TIMEOUT_TYPES,
    )
    if operational is not None:
        return operational
    return _INFRASTRUCTURE_UNKNOWN


def _operational_from_code(
    code: str,
    *,
    auth: frozenset[str],
    route: frozenset[str],
    throttling: frozenset[str],
    timeout: frozenset[str],
    server: frozenset[str],
) -> ProviderFailureAttribution | None:
    """Classify only an exact SDK-owned provider code, never raw error text."""
    if code in auth:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.AUTH,
        )
    if code in route:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.ROUTE,
        )
    if code in throttling:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.THROTTLING,
        )
    if code in timeout:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.TIMEOUT,
        )
    if code in server:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.SERVER,
        )
    return None


def _operational_from_type(
    error_type: str,
    *,
    transport: frozenset[str],
    timeout: frozenset[str],
) -> ProviderFailureAttribution | None:
    """Classify exact SDK exception types that do not carry HTTP metadata."""
    if error_type in transport:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.TRANSPORT,
        )
    if error_type in timeout:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.TIMEOUT,
        )
    return None


def _operational_from_status(status: int | None) -> ProviderFailureAttribution | None:
    """Classify unambiguous HTTP outcomes before candidate-owned request statuses."""
    reason = {
        401: ProviderFailureReason.AUTH,
        403: ProviderFailureReason.AUTH,
        404: ProviderFailureReason.ROUTE,
        408: ProviderFailureReason.TIMEOUT,
        429: ProviderFailureReason.THROTTLING,
    }.get(status)
    if reason is not None:
        return ProviderFailureAttribution(ProviderFailureOwner.INFRASTRUCTURE, reason)
    if status is not None and 500 <= status <= 599:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.SERVER,
        )
    return None


def _bedrock_validation_operational_reason(message: str) -> ProviderFailureReason | None:
    """Return the reason for a whole provider-authored operational validation template."""
    for pattern, reason in _BEDROCK_VALIDATION_OPERATIONAL_TEMPLATES:
        if pattern.fullmatch(message) is not None:
            return reason
    return None


def _classify_bedrock_parameter_validation(error: Exception) -> ProviderFailureAttribution:
    """Classify a botocore local validation report from its SDK-authored parameter paths."""
    kwargs = getattr(error, "kwargs", None)
    report = kwargs.get("report") if isinstance(kwargs, dict) else None
    if not isinstance(report, str) or not report or len(report) > 4_000:
        return _INFRASTRUCTURE_UNKNOWN
    roots: list[str] = []
    for line in report.splitlines():
        path = next(
            (
                match.group("path")
                for pattern in _BEDROCK_PARAMETER_PATH_TEMPLATES
                if (match := pattern.fullmatch(line)) is not None
            ),
            None,
        )
        if path is None:
            return _INFRASTRUCTURE_UNKNOWN
        roots.append(path.split("[", 1)[0].split(".", 1)[0])
    if "modelId" in roots:
        return ProviderFailureAttribution(
            ProviderFailureOwner.INFRASTRUCTURE,
            ProviderFailureReason.ROUTE,
        )
    if roots and all(root in _BEDROCK_CANDIDATE_PARAMETER_ROOTS for root in roots):
        return _CANDIDATE_REQUEST
    return _INFRASTRUCTURE_UNKNOWN


def _is_bedrock_candidate_validation(message: str) -> bool:
    """Match a whole provider-authored candidate request validation template."""
    return any(
        pattern.fullmatch(message) is not None
        for pattern in _BEDROCK_VALIDATION_CANDIDATE_TEMPLATES
    )


def _module_has_root(module: str, roots: tuple[str, ...]) -> bool:
    """Trust only an exact SDK module root or one of its dotted children."""
    return any(module == root or module.startswith(f"{root}.") for root in roots)


def _status_code(error: Exception) -> int | None:
    """Extract an HTTP status from OpenAI or botocore's documented public shapes."""
    direct = getattr(error, "status_code", None)
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, dict):
            value = metadata.get("HTTPStatusCode")
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        value = response.get("status_code")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and not isinstance(response_status, bool):
        return response_status
    return None


def _error_code(error: Exception) -> str:
    """Extract a stable provider error code from OpenAI or botocore shapes."""
    direct = getattr(error, "code", None)
    if isinstance(direct, str):
        return direct
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        provider_error = response.get("Error")
        if isinstance(provider_error, dict) and isinstance(provider_error.get("Code"), str):
            return cast("str", provider_error["Code"])
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body_code = body.get("code")
        if isinstance(body_code, str):
            return body_code
        body_error = body.get("error")
        if isinstance(body_error, dict) and isinstance(body_error.get("code"), str):
            return cast("str", body_error["code"])
    return ""


def _error_parameter(error: Exception) -> str:
    """Extract the top-level OpenAI request parameter from structured SDK metadata."""
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return ""
    raw = body.get("param")
    if not isinstance(raw, str):
        nested = body.get("error")
        raw = nested.get("param") if isinstance(nested, dict) else None
    if not isinstance(raw, str):
        return ""
    normalized = raw.casefold().replace("-", "_")
    return normalized.split("[", 1)[0].split(".", 1)[0]


def _provider_message(error: Exception) -> str:
    """Extract only a bounded structured SDK message for anchored Bedrock templates."""
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return ""
    provider_error = response.get("Error")
    if not isinstance(provider_error, dict):
        return ""
    message = provider_error.get("Message")
    return message[:4_000] if isinstance(message, str) else ""
