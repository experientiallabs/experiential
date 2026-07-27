"""Tests for the segment-aware compression seam (protocol, identity, truncate, registry)."""

from __future__ import annotations

from typing import cast

import pytest

from wmo.optimize.compression import (
    CompressingEmbedder,
    CompressionConfig,
    CompressionResult,
    Compressor,
    IdentityCompressor,
    TruncateCompressor,
    estimate_tokens,
    get_compressor,
    register_compressor,
    same_compression,
    servable_compressor,
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


class _CountingEmbedder:
    """Records exactly which texts it was asked to embed."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.append(list(texts))
        return [[float(len(text))] for text in texts]


class _Churny:
    """A compressor that admits it rewrites already-emitted bytes (C1's percentile family)."""

    id = "churny-compression-test"
    version = "1"
    append_stable = False

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        del config
        raw = sum(estimate_tokens(segment) for segment in segments)
        return CompressionResult(
            segments=list(segments), tokens_in_raw=raw, tokens_in_compressed=raw, latency_s=0.0
        )


_CHURNY = _Churny()


def test_the_reference_compressors_attest_append_stability() -> None:
    # Both are servable in v1: identity changes nothing, and truncate is head-absolute per
    # segment (C1 round 0 measured churn 0.000 on all five corpora for head-keep truncation).
    assert IdentityCompressor.append_stable is True
    assert TruncateCompressor.append_stable is True
    assert servable_compressor(CompressionConfig(compressor_id="truncate")) is not None
    assert servable_compressor(None) is None


def test_a_churny_compressor_is_not_servable() -> None:
    register_compressor(_CHURNY)
    with pytest.raises(ValueError, match="not attested append-stable"):
        servable_compressor(CompressionConfig(compressor_id=_CHURNY.id))


def test_register_compressor_refuses_to_rebind_an_id() -> None:
    register_compressor(_CHURNY)
    register_compressor(_CHURNY)  # idempotent for the same object
    assert get_compressor(_CHURNY.id) is _CHURNY

    class _Impostor:
        id = _CHURNY.id
        version = "2"
        append_stable = True

        def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        register_compressor(cast("Compressor", _Impostor()))


def test_same_compression_compares_the_whole_triple() -> None:
    base = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    assert same_compression(base, base.model_copy())
    assert same_compression(None, None)
    assert not same_compression(base, None)
    assert not same_compression(base, base.model_copy(update={"aggressiveness": 0.25}))
    # A version bump changes the emitted bytes exactly as a different id would.
    assert not same_compression(base, base.model_copy(update={"compressor_version": "2"}))


def test_compressing_embedder_embeds_the_compressed_text() -> None:
    # The fit-side half of representation consistency: the bank rows must be the geometry of
    # what serving will send, not of the raw task text.
    inner = _CountingEmbedder()
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    CompressingEmbedder(inner, config).embed(["one two three four", "alpha beta"])
    assert inner.seen == [["one two", "alpha"]]


def test_the_dial_invariants_hold_for_every_reference_compressor() -> None:
    # `aggressiveness` is a compressor-defined dial, not a removal fraction, so these two
    # invariants are the whole contract an implementation has to honor: 0.0 is a strict
    # bit-for-bit no-op, and higher never removes less.
    segments = ["one two three four five six seven eight", "alpha beta gamma", ""]
    for compressor_id in ("identity", "truncate"):
        compressor = get_compressor(compressor_id)
        removed = []
        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            config = CompressionConfig(compressor_id=compressor_id, aggressiveness=level)
            result = compressor.compress(segments, config)
            if level == 0.0:
                assert result.segments == segments, compressor_id
            removed.append(result.tokens_in_raw - result.tokens_in_compressed)
        assert removed == sorted(removed), f"{compressor_id} dial is not monotone: {removed}"


def test_one_call_carries_every_segment() -> None:
    # The batching contract a network-backed compressor depends on: the caller hands over all
    # of a request's mutable segments at once, so the implementation pays one round trip and
    # can batch internally.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    result = get_compressor("truncate").compress(["one two", "three four", "five six"], config)
    assert len(result.segments) == 3


class _V2Compressor:
    """Stands in for a compressor whose implementation was version-bumped under a stable id."""

    id = "versioned-compression-test"
    version = "2"
    append_stable = True

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        del config
        out = [segment.upper() for segment in segments]
        return CompressionResult(
            segments=out,
            tokens_in_raw=sum(estimate_tokens(segment) for segment in segments),
            tokens_in_compressed=sum(estimate_tokens(segment) for segment in out),
            latency_s=0.0,
        )


_V2 = _V2Compressor()


def test_a_config_version_the_build_cannot_produce_is_refused() -> None:
    # The id alone does not identify the bytes. An artifact stamped against a version this
    # build does not run was fitted in a DIFFERENT implementation's geometry, which is the same
    # failure as serving a different compressor and is invisible without this check.
    with pytest.raises(ValueError, match="version 1 in this build.*fitted against version 99"):
        servable_compressor(CompressionConfig(compressor_id="truncate", compressor_version="99"))


def test_a_version_bumped_implementation_refuses_the_old_stamp() -> None:
    # The other direction: the build moved forward and the artifact did not.
    register_compressor(_V2)
    assert (
        servable_compressor(CompressionConfig(compressor_id=_V2.id, compressor_version="2"))
        is not None
    )
    with pytest.raises(ValueError, match="version 2 in this build.*fitted against version 1"):
        servable_compressor(CompressionConfig(compressor_id=_V2.id, compressor_version="1"))
