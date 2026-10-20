"""Tests for the research-track control compressors behind the D-COMPRESS seam."""

from __future__ import annotations

import pytest

from wmo.optimize import compression as compression_module
from wmo.optimize.compression import (
    CompressionConfig,
    Compressor,
    compress_segments,
    get_compressor,
    register_compressor,
    servable_compressor,
)
from wmo.research.compressors import (
    RandomRemovalCompressor,
    TruncateProtectTaskCompressor,
    split_words,
)

SEGMENTS = [
    "Task: report cash flow from operations for FY2020.",
    "error: connection refused at port 8080\nerror: connection refused at port 8080",
    "The filing shows operating cash flow of 3,676.0 million dollars for the year.",
]

ALL: list[Compressor] = [RandomRemovalCompressor(), TruncateProtectTaskCompressor()]


def _cfg(aggressiveness: float) -> CompressionConfig:
    return CompressionConfig(compressor_id="x", aggressiveness=aggressiveness)


def test_split_words_is_lossless() -> None:
    for segment in SEGMENTS:
        assert "".join(split_words(segment)) == segment


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_protocol_conformance(compressor: Compressor) -> None:
    assert isinstance(compressor, Compressor)
    assert compressor.append_stable is True


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_zero_aggressiveness_is_noop(compressor: Compressor) -> None:
    result = compressor.compress(SEGMENTS, _cfg(0.0))
    assert result.segments == SEGMENTS
    assert result.tokens_in_compressed == result.tokens_in_raw


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_deterministic_and_segment_preserving(compressor: Compressor) -> None:
    first = compressor.compress(SEGMENTS, _cfg(0.5))
    second = compressor.compress(SEGMENTS, _cfg(0.5))
    assert first.segments == second.segments
    assert len(first.segments) == len(SEGMENTS)


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_append_stability_over_segments(compressor: Compressor) -> None:
    """Adding a segment never changes the segments already emitted (round 0's kill bar 1)."""
    short = compressor.compress(SEGMENTS[:2], _cfg(0.5)).segments
    long = compressor.compress(SEGMENTS, _cfg(0.5)).segments
    assert long[:2] == short


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_dial_is_monotone_in_removal(compressor: Compressor) -> None:
    """A higher aggressiveness never removes less than a lower one (the seam's dial invariant)."""
    gentle = compressor.compress(SEGMENTS, _cfg(0.2))
    harsh = compressor.compress(SEGMENTS, _cfg(0.6))
    assert harsh.tokens_in_compressed <= gentle.tokens_in_compressed <= harsh.tokens_in_raw


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_seam_chunking_entry_point_agrees(compressor: Compressor) -> None:
    """`compress_segments` (what the fit-side embedder calls) matches a direct call."""
    direct = compressor.compress(SEGMENTS, _cfg(0.5)).segments
    assert compress_segments(compressor, SEGMENTS, _cfg(0.5)).segments == direct


def test_random_removal_matches_ratio() -> None:
    segment = " ".join(f"w{i}" for i in range(200))
    out = RandomRemovalCompressor().compress([segment], _cfg(0.4)).segments[0]
    assert abs(len(split_words(out)) - 120) <= 2


def test_truncate_protect_task_keeps_segment_zero() -> None:
    out = TruncateProtectTaskCompressor().compress(SEGMENTS, _cfg(0.9)).segments
    assert out[0] == SEGMENTS[0]
    assert len(split_words(out[2])) < len(split_words(SEGMENTS[2]))


def test_truncate_protect_task_keeps_a_word_of_every_segment() -> None:
    """A segment truncated to nothing would read as an empty turn, not a shortened one."""
    out = TruncateProtectTaskCompressor().compress(SEGMENTS, _cfg(1.0)).segments
    assert all(segment.strip() for segment in out)


@pytest.mark.parametrize("compressor", ALL, ids=lambda c: c.id)
def test_registers_and_is_servable(compressor: Compressor) -> None:
    """The grid registers these and stamps the running version, exactly as `wmo optimize` does."""
    try:
        register_compressor(compressor)
        assert get_compressor(compressor.id) is compressor
        register_compressor(compressor)  # same object: idempotent
        config = CompressionConfig(
            compressor_id=compressor.id,
            compressor_version=compressor.version,
            aggressiveness=0.4,
        )
        assert servable_compressor(config) is compressor
    finally:
        # The registry is module-global process state; leaving research ids behind would leak
        # into other tests' assertions about what is registered.
        compression_module._COMPRESSORS.pop(compressor.id, None)
