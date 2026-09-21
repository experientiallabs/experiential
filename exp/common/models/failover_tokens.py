"""The closed vocabulary a rung's ``failover_only_on`` set is authored in.

A deployment may restrict itself to failover duty: it is never chosen for a
request's first dial and is dialed only as a successor to a failure whose
token is in its set (the customer's own OpenAI key enrolled in a trusted-access
program, taking over exactly the requests the house rung refused under its
cyber policy). Every token names one failure the waterfall can fail over from:
a failover-eligible failure class by its wire name, ``refusal`` for any
provider refusal, or ``refusal:<reason>`` for one bounded refusal category.
Classes that never advance the ladder (a caller's ``invalid_request``, the
gateway's own ``quota_exceeded``) are not tokens, so a rule cannot promise a
failover the waterfall would never perform.

The names mirror the native engine's ``FailureClass`` and ``RefusalReason``
wire names exactly; ``failover_tokens_test`` pins the mirror against the Rust
source and the python enums.
"""

from __future__ import annotations

from typing import Literal, get_args

FailoverToken = Literal[
    "throttled",
    "timeout",
    "transport",
    "provider_internal",
    "provider_quota",
    "provider_authentication",
    "provider_not_found",
    "unavailable",
    "empty_completion",
    "malformed_response",
    "guardrail",
    "refusal",
    "refusal:cyber_policy",
    "refusal:cbrn",
    "refusal:content_policy",
    "refusal:recitation",
    "refusal:data_inspection",
    "refusal:unspecified",
]
"""One authored failover token; the type the catalog field validates against."""

FAILOVER_TOKENS: tuple[str, ...] = get_args(FailoverToken)
"""The canonical ordered vocabulary, importable by the platform's authoring surfaces."""

REFUSAL_TOKEN = "refusal"
"""The bare refusal token: matches a provider refusal of any bounded reason."""

REFUSAL_TOKEN_PREFIX = "refusal:"
"""Prefix of the per-reason refusal tokens (``refusal:cyber_policy``)."""

FAILOVER_CLASS_TOKENS: tuple[str, ...] = tuple(
    token for token in FAILOVER_TOKENS if not token.startswith(REFUSAL_TOKEN)
)
"""The failure-class tokens alone: every class the waterfall may fail over from."""

REFUSAL_REASON_TOKENS: tuple[str, ...] = tuple(
    token for token in FAILOVER_TOKENS if token.startswith(REFUSAL_TOKEN_PREFIX)
)
"""The per-reason refusal tokens alone, one per bounded refusal category."""
