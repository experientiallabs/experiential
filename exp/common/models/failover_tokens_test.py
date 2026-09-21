"""The failover-token vocabulary is one closed list mirrored on both sides of the bridge."""

from __future__ import annotations

import re
from pathlib import Path

from exp.common.models.failover_tokens import (
    FAILOVER_CLASS_TOKENS,
    FAILOVER_TOKENS,
    REFUSAL_REASON_TOKENS,
    REFUSAL_TOKEN,
    REFUSAL_TOKEN_PREFIX,
)
from exp.runtime.gateway.stream_contracts import GatewayFailureClass, GatewayRefusalReason

_ERRORS_RS = Path(__file__).resolve().parents[2] / "runtime" / "gateway" / "native" / "src"

# The classes a rung may name: every class the waterfall can fail over from.
# A caller's own error, the gateway's own budget refusal, and the terminal
# bookkeeping classes never advance the ladder, so they are not tokens.
_NEVER_FAILOVER_CLASSES = frozenset(
    {
        GatewayFailureClass.INVALID_REQUEST,
        GatewayFailureClass.UNSUPPORTED_CAPABILITY,
        GatewayFailureClass.AUTHENTICATION,
        GatewayFailureClass.AUTHORIZATION,
        GatewayFailureClass.QUOTA_EXCEEDED,
        GatewayFailureClass.REFUSAL,  # Spelled as `refusal` / `refusal:<reason>` instead.
        GatewayFailureClass.CANCELLED,
        GatewayFailureClass.INTERNAL,
    }
)


def _rust_wire_names(enum_name: str) -> tuple[str, ...]:
    """Read one Rust enum's ``as_str`` wire names from ``errors.rs`` in source order."""
    source = (_ERRORS_RS / "errors.rs").read_text(encoding="utf-8")
    body = re.search(rf"impl {enum_name} \{{.*?fn as_str.*?match self \{{(.*?)\}}", source, re.S)
    assert body is not None, f"{enum_name}::as_str not found"
    return tuple(re.findall(rf'{enum_name}::\w+ => "([a-z_]+)"', body.group(1)))


def test_class_tokens_are_exactly_the_failover_eligible_python_classes() -> None:
    """Every class token is a python failure class, and every failover-capable class is a token."""
    expected = tuple(
        member.value for member in GatewayFailureClass if member not in _NEVER_FAILOVER_CLASSES
    )
    assert set(FAILOVER_CLASS_TOKENS) == set(expected)
    assert len(FAILOVER_CLASS_TOKENS) == len(set(FAILOVER_CLASS_TOKENS))


def test_class_tokens_are_rust_failure_class_wire_names() -> None:
    """The Rust ``FailureClass::as_str`` names cover every class token exactly."""
    rust = _rust_wire_names("FailureClass")
    assert set(FAILOVER_CLASS_TOKENS) <= set(rust)
    assert "refusal" in rust


def test_refusal_tokens_mirror_both_refusal_reason_enums() -> None:
    """One ``refusal:<reason>`` token per Rust and python refusal reason, plus the bare token."""
    rust = _rust_wire_names("RefusalReason")
    python = tuple(member.value for member in GatewayRefusalReason)
    assert rust == python
    assert REFUSAL_REASON_TOKENS == tuple(f"{REFUSAL_TOKEN_PREFIX}{reason}" for reason in rust)
    assert REFUSAL_TOKEN in FAILOVER_TOKENS
    assert set(FAILOVER_TOKENS) == set(FAILOVER_CLASS_TOKENS) | {REFUSAL_TOKEN} | set(
        REFUSAL_REASON_TOKENS
    )


def test_vocabulary_matches_the_rust_fallback_rules_pin() -> None:
    """The Rust child module carries the same list verbatim, so neither side drifts alone."""
    source = (_ERRORS_RS / "waterfall" / "fallback_rules.rs").read_text(encoding="utf-8")
    body = re.search(r"pub\(crate\) const FAILOVER_TOKENS: &\[&str\] = &\[(.*?)\];", source, re.S)
    assert body is not None, "FAILOVER_TOKENS not found in fallback_rules.rs"
    assert tuple(re.findall(r'"([a-z_:]+)"', body.group(1))) == FAILOVER_TOKENS
