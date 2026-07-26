"""Tests for the research-track compressors behind the D-COMPRESS seam."""

from __future__ import annotations

import pytest

from wmh.optimize.compression import (
    CompressionConfig,
    Compressor,
    get_compressor,
    register_compressor,
)
from wmh.research.compression import split_words
from wmh.research.compressors import (
    DedupKeepFirstCompressor,
    RandomRemovalCompressor,
    ScoredWordCompressor,
    TruncateProtectTaskCompressor,
)

SEGMENTS = [
    "Task: report cash flow from operations for FY2020.",
    "error: connection refused at port 8080\nerror: connection refused at port 8080",
    "The filing shows operating cash flow of 3,676.0 million dollars for the year.",
]


def _cfg(aggressiveness: float) -> CompressionConfig:
    return CompressionConfig(compressor_id="x", aggressiveness=aggressiveness)


def _len_scores(segment: str) -> list[float]:
    return [float(len(w.strip())) for w in split_words(segment)]


ALL = [
    RandomRemovalCompressor(),
    DedupKeepFirstCompressor(),
    TruncateProtectTaskCompressor(),
    ScoredWordCompressor("scored-len", _len_scores, threshold=5.0),
]


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_protocol_conformance(compressor) -> None:  # noqa: ANN001
    assert isinstance(compressor, Compressor)


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_zero_aggressiveness_is_noop(compressor) -> None:  # noqa: ANN001
    result = compressor.compress(SEGMENTS, _cfg(0.0))
    assert result.segments == SEGMENTS
    assert result.tokens_in_compressed == result.tokens_in_raw


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_deterministic_and_segment_preserving(compressor) -> None:  # noqa: ANN001
    a = compressor.compress(SEGMENTS, _cfg(0.5))
    b = compressor.compress(SEGMENTS, _cfg(0.5))
    assert a.segments == b.segments
    assert len(a.segments) == len(SEGMENTS)


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_append_stability_over_segments(compressor) -> None:  # noqa: ANN001
    """Adding a segment never changes the segments already emitted (kill bar 1)."""
    short = compressor.compress(SEGMENTS[:2], _cfg(0.5)).segments
    long = compressor.compress(SEGMENTS, _cfg(0.5)).segments
    assert long[:2] == short


def test_random_removal_matches_ratio() -> None:
    seg = " ".join(f"w{i}" for i in range(200))
    out = RandomRemovalCompressor().compress([seg], _cfg(0.4)).segments[0]
    assert abs(len(split_words(out)) - 120) <= 2


def test_dedup_drops_repeats_across_segments() -> None:
    out = DedupKeepFirstCompressor().compress(SEGMENTS, _cfg(1.0)).segments
    assert out[1].count("connection refused") == 1


def test_truncate_protect_task_keeps_segment_zero() -> None:
    out = TruncateProtectTaskCompressor().compress(SEGMENTS, _cfg(0.9)).segments
    assert out[0] == SEGMENTS[0]
    assert len(split_words(out[2])) < len(split_words(SEGMENTS[2]))


def test_scored_word_compressor_validates_scorer() -> None:
    bad = ScoredWordCompressor("scored-bad", lambda s: [1.0], threshold=0.5)
    with pytest.raises(ValueError, match="scores for"):
        bad.compress(SEGMENTS, _cfg(0.5))


def test_scored_word_compressor_amortizes_cost() -> None:
    comp = ScoredWordCompressor("scored-cost", _len_scores, threshold=5.0, usd_per_10k=0.001)
    result = comp.compress(SEGMENTS, _cfg(0.5))
    assert result.cost_usd == pytest.approx(result.tokens_in_raw / 10_000 * 0.001)


def test_register_compressor_roundtrip_and_collision() -> None:
    from wmh.optimize import compression as compression_module

    comp = ScoredWordCompressor("scored-registry-test", _len_scores, threshold=5.0)
    try:
        register_compressor(comp)
        assert get_compressor("scored-registry-test") is comp
        register_compressor(comp)  # same object: idempotent
        clone = ScoredWordCompressor("scored-registry-test", _len_scores, threshold=5.0)
        with pytest.raises(ValueError, match="already registered"):
            register_compressor(clone)
    finally:
        # The registry is module-global process state; leaving the test id behind would
        # leak into other tests' exact known-ids assertions.
        compression_module._COMPRESSORS.pop("scored-registry-test", None)
