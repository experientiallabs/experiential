"""Tests for shared trace-extension readers; broader coverage lives in the source tests."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.simulation.ingest.trace_extensions import consistent_text


class _FormatError(ValueError):
    """Source-specific validation error used by these tests."""


def test_consistent_text_rejects_a_declared_null_value() -> None:
    """An explicitly declared null excludes the trace instead of reading as undeclared."""
    with pytest.raises(_FormatError, match="exp.outcome.status must be non-empty text"):
        consistent_text(
            ({"exp.outcome.status": None},),
            ("exp.outcome.status",),
            error_type=_FormatError,
        )


def test_consistent_text_lenient_skips_invalid_values_and_falls_back() -> None:
    """Lenient reads skip non-text and blank values and fall through to the next key."""
    attributes_by_span: tuple[JsonObject, ...] = (
        {"exp.conversation.id": " ", "gen_ai.conversation.id": "conv-1"},
        {"exp.conversation.id": 42},
    )

    value = consistent_text(
        attributes_by_span,
        ("exp.conversation.id", "gen_ai.conversation.id"),
        error_type=_FormatError,
        lenient=True,
    )

    assert value == "conv-1"
