"""Tests for the shared signed token hashing embedder."""

from __future__ import annotations

import math

import pytest

from exp.common.core.hashing import signed_token_embedding


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


@pytest.mark.parametrize(
    ("text", "dimensions"),
    [("fix index", 64), ("show user", 256)],
)
def test_signed_token_embedding_handles_cancelled_tokens(text: str, dimensions: int) -> None:
    """Opposite signed hashes still produce a deterministic unit vector."""
    vector = signed_token_embedding(text, dimensions)

    assert vector == signed_token_embedding(text, dimensions)
    assert len(vector) == dimensions
    assert all(math.isfinite(value) for value in vector)
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)


def test_cancelled_texts_keep_distinct_fallback_vectors() -> None:
    """Different cancelled token sequences must not share one generic sentinel."""
    assert signed_token_embedding("fix index", 64) != signed_token_embedding("bug data", 64)
