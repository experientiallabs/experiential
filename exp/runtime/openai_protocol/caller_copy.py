"""Caller-facing wording for the gateway's own upstream failure sentences.

A failure's ``safe_message`` is the durable, operator-facing sentence the
ledger, alert rules and dashboards key on ("provider service failed", "all
exact-model deployments are unavailable"), so it never changes here. What a
caller reads is derived at the public boundary instead: each known sentence
maps to one that says what happened in the caller's terms (the model's
provider, never a rung, lane, deployment or alias) and what to do next. Any
sentence not listed passes through unchanged, so a specific message (a
parameter refusal, a capability refusal) keeps its own words.

Mirrors the native plane's ``caller_copy.rs``; the taxonomy parity test drives
both through :data:`CALLER_COPY_EXAMPLES`.
"""

from __future__ import annotations

import re

from exp.runtime.gateway.contracts import GatewayFailureClass

_C = GatewayFailureClass

# (class, durable sentence prefix, caller sentence). A prefix matches the whole
# message or a leading clause ended by ";" or ":".
_FIXED: tuple[tuple[GatewayFailureClass, str, str], ...] = (
    (
        _C.PROVIDER_INTERNAL,
        "provider service failed",
        "The model's provider had a server error and no other provider for this model "
        "could take the request. Retry after a short delay; if it keeps failing, choose "
        "another model.",
    ),
    (
        _C.PROVIDER_INTERNAL,
        "all exact-model deployments are unavailable",
        "No provider for this model can take the request right now. Retry after a short "
        "delay; if it keeps failing, choose another model.",
    ),
    (
        _C.PROVIDER_INTERNAL,
        "provider returned an unexpected status",
        "The model's provider returned an unexpected response. Retry the request; if it "
        "keeps failing, choose another model.",
    ),
    (
        _C.PROVIDER_INTERNAL,
        "provider reported a transient conflict",
        "The model's provider reported a temporary conflict. Retry the request.",
    ),
    (
        _C.TRANSPORT,
        "provider transport failed",
        "The gateway could not reach the model's provider. Retry after a short delay; if "
        "it keeps failing, choose another model.",
    ),
    (
        _C.MALFORMED_RESPONSE,
        "provider returned a malformed response",
        "The model's provider sent an incomplete or invalid response, so the gateway "
        "discarded it. Retry the request; if it keeps happening, choose another model.",
    ),
    (
        _C.TIMEOUT,
        "provider did not send response headers in time",
        "The model's provider did not start responding in time and no other provider for "
        "this model was available. Retry after a short delay; if it keeps happening, "
        "choose another model.",
    ),
    (
        _C.TIMEOUT,
        "provider did not send the first token in time",
        "The model's provider accepted the request but sent no output in time. Retry the "
        "request; a very long prompt takes longer to start, so a shorter prompt can help.",
    ),
    (
        _C.TIMEOUT,
        "provider stopped making progress",
        "The model's provider stopped sending output partway through the response. Retry "
        "the request.",
    ),
    (
        _C.TIMEOUT,
        "provider request timed out",
        "The model's provider timed out before answering. Retry the request.",
    ),
    (
        _C.THROTTLED,
        "provider throttled the request",
        "The model's provider is rate limiting requests right now. Retry after a short "
        "delay (the Retry-After header gives the wait when the response carries one).",
    ),
    (
        _C.PROVIDER_NOT_FOUND,
        "provider deployment was not found",
        "The model's provider no longer serves this model at its configured address. "
        "Retry later or choose another model; if it persists, ask the gateway operator "
        "to update the model's route.",
    ),
    (
        _C.INTERNAL,
        "the native engine cannot serve the authorized route",
        "The gateway cannot serve this model right now because of a problem on its side. "
        "Retry later or choose another model; if it persists, contact the gateway "
        "operator.",
    ),
)
_THROTTLE_WINDOW = "all exact-model deployments are inside a provider throttle window"
_REFUSAL = "provider refused the request"
_FIELDLESS_REJECTION = (
    "provider rejected the request; verify the request fields against the model alias capabilities"
)
_RETRY_IN = re.compile(r"retry in ([0-9]+)s")

# Durable sentences with their caller wording, for the native parity test.
CALLER_COPY_EXAMPLES: tuple[tuple[GatewayFailureClass, str], ...] = (
    *((failure_class, prefix) for failure_class, prefix, _ in _FIXED),
    (_C.PROVIDER_INTERNAL, "provider service failed; retry after a short delay"),
    (_C.THROTTLED, f"{_THROTTLE_WINDOW}; retry in 27s"),
    (_C.REFUSAL, _REFUSAL),
    (_C.REFUSAL, f"{_REFUSAL}: content policy"),
    (_C.REFUSAL, f"{_REFUSAL}; revise the request content and retry"),
    (_C.THROTTLED, _THROTTLE_WINDOW),
    (_C.INVALID_REQUEST, _FIELDLESS_REJECTION),
)
# Specific sentences that keep their own words on both planes.
CALLER_COPY_PASSTHROUGH: tuple[tuple[GatewayFailureClass, str], ...] = (
    (_C.INVALID_REQUEST, "The parameter 'n' is not supported."),
    (_C.PROVIDER_INTERNAL, "provider service failedness"),
    (_C.TIMEOUT, "provider service failed"),
)


def _after(message: str, prefix: str) -> str | None:
    """The clause after ``prefix``, or None when ``prefix`` is not a leading clause."""
    if not message.startswith(prefix):
        return None
    rest = message[len(prefix) :]
    if not rest:
        return ""
    if rest[0] not in ";:":
        return None
    return rest[1:].strip()


def caller_copy(failure_class: GatewayFailureClass, safe_message: str) -> str:
    """Return the caller sentence for ``safe_message`` of ``failure_class``.

    Args:
        failure_class: The failure's stable class.
        safe_message: The durable, operator-facing sentence.

    Returns:
        The caller wording, or ``safe_message`` itself when none is defined.
    """
    for known_class, prefix, text in _FIXED:
        if known_class is failure_class and _after(safe_message, prefix) is not None:
            return text
    if failure_class is _C.THROTTLED:
        rest = _after(safe_message, _THROTTLE_WINDOW)
        if rest is not None:
            match = _RETRY_IN.fullmatch(rest)
            wait = (
                f"Retry in {match.group(1)} seconds."
                if match is not None
                else "Retry after a short delay (the Retry-After header gives the wait "
                "when the response carries one)."
            )
            return f"Every provider serving this model is rate limiting requests right now. {wait}"
    if failure_class is _C.REFUSAL:
        rest = _after(safe_message, _REFUSAL)
        # Only a ":" clause names a refusal category; a ";" clause is remediation.
        if rest is not None and not safe_message.startswith(f"{_REFUSAL}:"):
            rest = ""
        if rest == "":
            return (
                "The model's provider declined to answer this request. Rephrase the request "
                "or choose another model."
            )
        if rest is not None:
            return (
                f"The model's provider declined this request ({rest}). Rephrase the request "
                "or choose another model."
            )
    if failure_class is _C.INVALID_REQUEST and safe_message == _FIELDLESS_REJECTION:
        return (
            "The model's provider rejected the request without giving a reason. Check the "
            "request's parameters against what this model supports "
            "(GET /v1/models/{model}), or choose another model."
        )
    return safe_message
