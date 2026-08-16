"""Tests for the shared signed token hashing embedder."""

from __future__ import annotations

import math

import pytest

from wmo.common.core.hashing import signed_token_embedding


def test_signed_token_embedding_is_deterministic_and_unit_length() -> None:
    """The same text and width always produce one finite unit vector."""
    first = signed_token_embedding("Reset the password for acct-9", 16)
    second = signed_token_embedding("Reset the password for acct-9", 16)

    assert first == second
    assert len(first) == 16
    assert all(math.isfinite(value) for value in first)
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0, abs_tol=1e-9)


def test_signed_token_embedding_rejects_narrow_vectors() -> None:
    """Widths below eight dimensions are unusable and fail closed."""
    with pytest.raises(ValueError, match="at least 8 dimensions"):
        signed_token_embedding("hello", 7)
