"""Tests for the rungs a context-window refusal may advance to."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_context_window import wider_windows


def _refusal(limit: int | None) -> GatewayFailure:
    """Build one provider context-window 400 as the data plane classifies it."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.INVALID_REQUEST,
        safe_message="This model's maximum context length is 16384 tokens.",
        failover_eligible=True,
        context_window_rejected=True,
        context_window_limit_tokens=limit,
    )


@pytest.mark.parametrize(
    ("limit", "windows", "expected"),
    [
        # A lane serving 16,384 of an advertised 262,144: every wider rung fits.
        (16_384, (262_144, 262_144, 16_384, None), (1, 3)),
        # No stated window: the failed rung's advertised window is the bar.
        (None, (32_768, 32_768, 131_072, None), (2, 3)),
        # Neither stated nor advertised: no rung can be ruled out.
        (None, (None, 8_192, 8_192, None), (1, 2, 3)),
    ],
)
def test_only_a_rung_wider_than_the_refusing_window_remains(
    limit: int | None, windows: tuple[int | None, ...], expected: tuple[int, ...]
) -> None:
    """A rung no wider than the refusing window would refuse the same request."""
    assert wider_windows((1, 2, 3), _refusal(limit), 0, windows) == expected


def test_a_route_without_window_facts_keeps_every_depth() -> None:
    """With no window facts the untried depths are returned unchanged."""
    assert wider_windows((1, 2), _refusal(16_384), 0, ()) == (1, 2)
