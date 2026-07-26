"""Tests for the segment-aware compression seam (protocol, identity, truncate, registry)."""

from __future__ import annotations

import pytest

from wmh.optimize.compression import (
    CompressionConfig,
    Compressor,
    IdentityCompressor,
    TruncateCompressor,
    estimate_tokens,
    get_compressor,
)


def test_identity_returns_segments_bit_for_bit() -> None:
    segments = ["  leading space", "tabs\tand\nnewlines ", ""]
    result = IdentityCompressor().compress(
        segments, CompressionConfig(compressor_id="identity", aggressiveness=1.0)
    )
    # Bit-for-bit even at max aggressiveness: identity is the compression-off contract.
    assert result.segments == segments
    assert result.tokens_in_compressed == result.tokens_in_raw
    assert result.cost_usd == 0.0


def test_truncate_drops_the_trailing_fraction() -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    result = TruncateCompressor().compress(["one two three four", "a b"], config)
    assert result.segments == ["one two", "a"]
    assert result.tokens_in_compressed < result.tokens_in_raw


def test_truncate_at_zero_aggressiveness_is_a_no_op() -> None:
    segments = ["exact bytes  preserved\twhen nothing is removed"]
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.0)
    result = TruncateCompressor().compress(segments, config)
    # keep >= len(words) short-circuits to the original string, whitespace intact.
    assert result.segments == segments
    assert result.tokens_in_compressed == result.tokens_in_raw


def test_truncate_is_deterministic_per_segment() -> None:
    # Append-stability: an unchanged segment compresses to the same bytes on every call,
    # regardless of what other segments accompany it.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.3)
    compressor = TruncateCompressor()
    alone = compressor.compress(["stable segment with several words here"], config)
    with_neighbor = compressor.compress(
        ["stable segment with several words here", "another later segment"], config
    )
    assert alone.segments[0] == with_neighbor.segments[0]


def test_compressors_preserve_segment_count_and_order() -> None:
    segments = ["first has words", "second", "", "fourth trailing"]
    for compressor_id in ("identity", "truncate"):
        config = CompressionConfig(compressor_id=compressor_id, aggressiveness=0.5)
        result = get_compressor(compressor_id).compress(segments, config)
        assert len(result.segments) == len(segments), compressor_id


def test_registry_resolves_known_ids_and_satisfies_the_protocol() -> None:
    for compressor_id in ("identity", "truncate"):
        compressor = get_compressor(compressor_id)
        assert isinstance(compressor, Compressor)
        assert compressor.id == compressor_id


def test_registry_rejects_unknown_id_with_guidance() -> None:
    with pytest.raises(ValueError, match="unknown compressor 'llmzip'.*identity, truncate"):
        get_compressor("llmzip")


def test_aggressiveness_is_bounded() -> None:
    with pytest.raises(ValueError):
        CompressionConfig(compressor_id="truncate", aggressiveness=1.5)
    with pytest.raises(ValueError):
        CompressionConfig(compressor_id="truncate", aggressiveness=-0.1)


def test_estimate_tokens_is_ceil_of_quarter_chars() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
