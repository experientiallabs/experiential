"""Capacity-vs-client error classification — the contract that decides failover vs propagate.

Capacity errors (throttling, transient 5xx, model-not-ready, transport timeouts) mean "this backend
can't serve right now; the next one might" — the waterfall spills. Client errors (bad request,
auth, validation) mean "this request is wrong" — failing over would just mask a real bug behind a
different model's answer, so they propagate immediately.

Classification is pure duck-typing with zero SDK imports, so the package works with any subset of
provider SDKs installed and the logic is testable with fake exceptions. Signals are checked in
fidelity order: a structured error code always wins over anything derived from the message.
"""

from __future__ import annotations

from typing import Literal

from llm_waterfall.types import WaterfallExhausted

Outcome = Literal["capacity_error", "client_error"]

# Botocore error CODES that mean "this model is capacity-constrained right now" — the reliable
# signal (from `exc.response["Error"]["Code"]`), preferred over everything else.
_CAPACITY_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ServiceQuotaExceededException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "InternalServerException",  # transient 5xx, safe to fail over
    }
)

# Exception class NAMES from the OpenAI/Anthropic SDKs (and their httpx/httpcore transport) that
# mean capacity/transport failure. Matched by name + module gate rather than isinstance so the
# SDKs never need to be importable.
_SDK_CAPACITY_TYPE_NAMES = frozenset(
    {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "OverloadedError",  # anthropic 529
        # httpx/httpcore transient transport failures (no status code on any of them):
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectError",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",  # server disconnected mid-response
    }
)

# Top-level modules whose exceptions we trust for name/status-code classification. An exception
# named `RateLimitError` from arbitrary application code proves nothing.
_TRUSTED_SDK_MODULES = frozenset({"openai", "anthropic", "httpx", "httpcore"})

# HTTP statuses on a trusted SDK error that mean capacity/transient failure. 529 is Anthropic's
# "overloaded". Everything else (400/401/403/404/422/...) is a client error.
_CAPACITY_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 529})

# Last-resort substrings for structureless transport errors (e.g. botocore's Read/ConnectTimeout,
# raw ConnectionError), which carry no code, no trusted type, no status. Kept conservative: only
# phrases that unambiguously mean capacity/transport failure, NOT generic tokens like "429"/"503"/
# "capacity" that can appear inside a bad-request message and turn a real error into a silent
# failover. (That was a real bug once: `ValidationException: request timeout too large`.)
_TRANSPORT_MARKERS = (
    "throttl",
    "read timeout",
    "connect timeout",
    "connection reset",
    "connection aborted",
    "could not connect to the endpoint",  # botocore EndpointConnectionError (DNS/unreachable)
    "timed out",
    "service unavailable",
    "model not ready",
)


def _from_trusted_sdk(exc: Exception) -> bool:
    """Whether `exc` was defined by a provider SDK or its HTTP transport."""
    module = type(exc).__module__ or ""
    return module.split(".")[0] in _TRUSTED_SDK_MODULES


def outcome_for(exc: Exception) -> Outcome:
    """Classify an exception raised by a backend attempt.

    Checks, in fidelity order:
      1. Our own `WaterfallExhausted` — a nested chain that exhausted was capacity-constrained
         by definition (outer waterfalls/retry loops must treat it as transient).
      2. Structured botocore error code — authoritative; a non-capacity code stops here so a
         message substring can never overrule it.
      3. SDK exception type name, gated on the defining module.
      4. HTTP status, from a `status_code` attribute on the exception or on its `.response`
         (httpx.HTTPStatusError carries it there), same module gate.
      5. Conservative transport-phrase substrings for structureless errors.
    """
    if isinstance(exc, WaterfallExhausted):
        return "capacity_error"

    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        # Guard the shape: a duck-typed `.response` dict may carry a non-dict "Error" value;
        # classification must never raise over the original exception.
        code = error.get("Code") if isinstance(error, dict) else None
        if code is not None:
            return "capacity_error" if code in _CAPACITY_ERROR_CODES else "client_error"

    if _from_trusted_sdk(exc):
        if type(exc).__name__ in _SDK_CAPACITY_TYPE_NAMES:
            return "capacity_error"
        status = getattr(exc, "status_code", None)
        if not isinstance(status, int):
            # httpx.HTTPStatusError keeps the status on the response object instead.
            status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return "capacity_error" if status in _CAPACITY_STATUS_CODES else "client_error"

    message = str(exc).lower()
    if any(marker in message for marker in _TRANSPORT_MARKERS):
        return "capacity_error"
    return "client_error"


def is_capacity_error(exc: Exception) -> bool:
    """True when the waterfall should spill to the next backend instead of raising."""
    return outcome_for(exc) == "capacity_error"
