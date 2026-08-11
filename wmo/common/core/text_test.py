"""Tests for durable-text canonicalization (NULs and UTF-16 surrogates)."""

from __future__ import annotations

import json

import pytest

from wmo.common.core.text import normalize_durable_text, validate_durable_text

_REPLACEMENT = "\N{REPLACEMENT CHARACTER}"


def test_normalize_leaves_ordinary_text_untouched() -> None:
    value = "ok: caf\u00e9 \U0001f600 \n\ttab"

    assert normalize_durable_text(value) == value


def test_normalize_replaces_embedded_nuls() -> None:
    assert normalize_durable_text("a\x00b") == f"a{_REPLACEMENT}b"


def test_normalize_replaces_lone_surrogates_of_either_half() -> None:
    assert normalize_durable_text("a\ud800b") == f"a{_REPLACEMENT}b"
    assert normalize_durable_text("a\udc00b") == f"a{_REPLACEMENT}b"
    assert normalize_durable_text("\ud800") == _REPLACEMENT


def test_normalize_folds_a_valid_surrogate_pair_into_its_scalar() -> None:
    # A pair that arrived as two code points (a lone-surrogate JSON decode) is the same character
    # as the scalar, so it survives canonicalization instead of degrading to two replacements.
    assert normalize_durable_text("x\ud83d\ude00y") == "x\U0001f600y"


def test_normalized_text_survives_a_utf8_round_trip() -> None:
    # The point of the pass: every WMO persistence boundary (files, HTTP, JSONB) is UTF-8, and
    # raw surrogates or NULs are exactly what those encoders reject.
    hostile = "a\x00b\ud800c\ud83d\ude00"

    normalized = normalize_durable_text(hostile)

    assert normalized.encode("utf-8").decode("utf-8") == normalized
    assert json.loads(json.dumps(normalized)) == normalized


def test_validate_accepts_text_canonicalization_would_not_change() -> None:
    validate_durable_text("plain \U0001f600 text", field="observation")  # does not raise


def test_validate_rejects_a_nul_and_names_the_field() -> None:
    with pytest.raises(ValueError, match="observation contains an embedded NUL"):
        validate_durable_text("a\x00b", field="observation")


def test_validate_rejects_surrogates_including_a_pair() -> None:
    # Content-addressed text must be rejected, not silently folded: a caller that hashed the
    # pre-normalization bytes would otherwise store a digest for text nobody can reproduce.
    with pytest.raises(ValueError, match="unpaired UTF-16 surrogate"):
        validate_durable_text("a\ud800b", field="state")
    with pytest.raises(ValueError, match="unpaired UTF-16 surrogate"):
        validate_durable_text("a\ud83d\ude00b", field="state")
