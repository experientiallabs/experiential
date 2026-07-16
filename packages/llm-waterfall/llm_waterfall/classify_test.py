"""Tests for capacity-vs-client error classification (the package's core contract)."""

from __future__ import annotations

import pytest

from llm_waterfall.classify import is_capacity_error, outcome_for


class _BotocoreShaped(Exception):
    """Duck-types botocore.exceptions.ClientError: carries `.response["Error"]["Code"]`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


def _sdk_exception(
    name: str, module: str, message: str = "", status_code: int | None = None
) -> Exception:
    """Build a fake SDK exception with a chosen class name, module, and optional status_code."""
    attrs: dict[str, int] = {}
    cls = type(name, (Exception,), attrs)
    cls.__module__ = module
    exc = cls(message)
    if status_code is not None:
        exc.status_code = status_code
    return exc


# --- Tier 1: botocore structured codes ---


@pytest.mark.parametrize(
    "code",
    [
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ServiceQuotaExceededException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "InternalServerException",
    ],
)
def test_capacity_codes_are_capacity(code: str) -> None:
    assert is_capacity_error(_BotocoreShaped(code, "boom"))


def test_validation_exception_with_timeout_message_is_client_error() -> None:
    # Regression: a structured non-capacity code must win over message substrings. The prototype
    # once classified this as capacity because the message contains "timeout".
    exc = _BotocoreShaped("ValidationException", "request timeout too large for this model")
    assert not is_capacity_error(exc)
    assert outcome_for(exc) == "client_error"


def test_access_denied_is_client_error() -> None:
    assert outcome_for(_BotocoreShaped("AccessDeniedException", "not authorized")) == "client_error"


# --- Tier 2: SDK exception types, gated on module ---


@pytest.mark.parametrize(
    "name",
    [
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "OverloadedError",
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
        "PoolTimeout",
    ],
)
@pytest.mark.parametrize("module", ["openai", "anthropic._exceptions", "httpx", "httpcore"])
def test_sdk_capacity_types_are_capacity(name: str, module: str) -> None:
    assert is_capacity_error(_sdk_exception(name, module))


def test_sdk_type_name_from_foreign_module_is_client_error() -> None:
    # Same class name, wrong module: the gate must reject lookalikes from arbitrary code.
    assert outcome_for(_sdk_exception("RateLimitError", "myapp.errors")) == "client_error"


def test_sdk_bad_request_type_is_client_error() -> None:
    assert outcome_for(_sdk_exception("BadRequestError", "openai")) == "client_error"


# --- Tier 3: status codes on SDK errors ---


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
def test_capacity_status_codes_are_capacity(status: int) -> None:
    assert is_capacity_error(_sdk_exception("APIStatusError", "anthropic", status_code=status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_status_codes_are_client_error(status: int) -> None:
    exc = _sdk_exception("APIStatusError", "openai", status_code=status)
    assert outcome_for(exc) == "client_error"


def test_status_code_from_foreign_module_is_ignored() -> None:
    # A random exception with a status_code attribute must not be treated as an SDK signal...
    exc = _sdk_exception("MyError", "myapp", status_code=429)
    assert outcome_for(exc) == "client_error"


# --- Tier 4: conservative transport substrings ---


@pytest.mark.parametrize(
    "message",
    [
        "Read timeout on endpoint URL",
        "Connect timeout on endpoint URL",
        "Connection reset by peer",
        "Connection aborted.",
        "request timed out",
        "throttling: too fast",
        "Service Unavailable",
        "model not ready yet",
        # botocore EndpointConnectionError — found live: an unreachable endpoint must spill.
        'Could not connect to the endpoint URL: "https://bedrock-runtime.x.amazonaws.com/"',
        # botocore ConnectionClosedError - found live (killed a GEPA sweep mid-scoring): a
        # connection that was ESTABLISHED and then dropped mid-response is transient by
        # construction, unlike a bad request which never gets that far.
        "Connection was closed before we received a valid response from endpoint URL",
    ],
)
def test_transport_messages_are_capacity(message: str) -> None:
    assert is_capacity_error(ConnectionError(message))


@pytest.mark.parametrize(
    "message",
    [
        "got 429 in payload",  # never match raw "429"
        "field must be < 503",  # never match raw "503"
        "service capacity exceeded",  # never match raw "capacity"
        "invalid request body",
    ],
)
def test_generic_tokens_never_match(message: str) -> None:
    assert outcome_for(ValueError(message)) == "client_error"


# --- Structural edge cases found in review ---


def test_non_dict_error_value_never_crashes_classification() -> None:
    # Regression: a duck-typed .response with a non-dict "Error" must not raise from classify.
    class Weird(Exception):
        def __init__(self) -> None:
            super().__init__("throttled")
            self.response = {"Error": "overloaded"}

    assert is_capacity_error(Weird())  # falls through to the "throttl" transport marker


def test_status_code_read_from_response_object() -> None:
    # Regression: httpx.HTTPStatusError carries the status on .response.status_code.
    class _Resp:
        status_code = 529

    resp = _Resp()
    exc = _sdk_exception("HTTPStatusError", "httpx", "Server error '529 ' for url 'https://x'")
    setattr(exc, "response", resp)  # noqa: B010 - dynamic attr on a synthesized exception
    assert is_capacity_error(exc)
    resp.status_code = 404
    assert outcome_for(exc) == "client_error"


def test_waterfall_exhausted_is_capacity() -> None:
    # Regression: a nested chain's exhaustion is capacity by definition — outer retry loops
    # and outer waterfalls must treat it as transient, not as a client error.
    from llm_waterfall.types import WaterfallExhausted

    assert is_capacity_error(WaterfallExhausted("all rungs throttled", []))


def test_internal_failure_code_is_capacity() -> None:
    # botocore's generic 5xx code — a live run died on it before this classified.
    exc = _BotocoreShaped("InternalFailure", "An error occurred (InternalFailure)")
    assert outcome_for(exc) == "capacity_error"


def test_botocore_connection_family_matches_by_mro_without_imports() -> None:
    # Any member of botocore's ConnectionError/HTTPClientError family classifies by MRO
    # (module, name), so never-seen subclasses with novel messages still fail over. Message
    # matching alone lost this twice (ConnectionClosedError, EndpointConnectionError).
    class ConnectionError(Exception):  # noqa: A001 - mirrors botocore's own name
        pass

    ConnectionError.__module__ = "botocore.exceptions"

    class NewlyInventedSubclass(ConnectionError):
        pass

    NewlyInventedSubclass.__module__ = "botocore.exceptions"
    assert outcome_for(NewlyInventedSubclass("some phrasing we have never seen")) == (
        "capacity_error"
    )

    # An identically named class from application code proves nothing.
    class AppConnectionError(Exception):
        pass

    assert outcome_for(AppConnectionError("connection something")) == "client_error"
