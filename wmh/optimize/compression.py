"""Segment-aware context compression behind the optimizer interface (D-COMPRESS seam).

A compressor removes low-information tokens from the model's input before the provider call.
The seam is SEGMENT-aware, not string-aware: the caller splits the conversation into
(cached-prefix, turn-local) segments and hands the compressor ONLY the segments it is allowed
to change. The provider prompt cache is prefix-matched, so recompressing an incumbent
conversation's cached prefix would forfeit ~0.9x discounts to save ~0.1x of tokens; keeping
that prefix out of the compressor's hands makes cache safety a property of the construction,
not a convention each compressor must remember.

This module ships the protocol plus the two reference implementations the seam is proven
with: `identity` (today's behavior, bit-for-bit) and `truncate` (the trivial ratio-matched
control every learned compressor must beat). Real compressors are chosen by the research
track and register here; nothing else in the harness hardcodes one.

Every compressor must be deterministic: the same segments and config always produce the same
output (append-stability of the serving path depends on it, and so does replaying a request
log). Token counts reported here are a deterministic chars/4 PROXY for accounting the
compressor's own effect; billable truth stays with the provider-reported `TokenUsage`.
"""

from __future__ import annotations

import math
import time
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

# The proxy tokenizer: ~4 chars per token, the industry rule of thumb. Deterministic and
# provider-agnostic; used only for the compressor's own raw-vs-compressed accounting.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Deterministic proxy token count of `text` (ceil of chars/4; 0 only for empty text)."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


class CompressionConfig(BaseModel):
    """Per-cluster compression choice carried on the policy artifact (D-COMPRESS shape).

    `aggressiveness` is the fraction of content the compressor may remove, in [0, 1];
    risk-tiered per cluster and learned within bounds by the research track. 0.0 must be a
    no-op for every compressor. `compressor_version` records the version the config was
    fitted against (provenance); serving logs the version that actually ran.
    """

    compressor_id: str = Field(min_length=1)
    compressor_version: str = "1"
    aggressiveness: float = Field(default=0.0, ge=0.0, le=1.0)


class CompressionResult(BaseModel):
    """What one compress() call did: the output segments plus its own accounting.

    `segments` has the same length and order as the input (a compressor rewrites segments,
    it never merges, splits, or reorders them; the caller owns the conversation structure).
    Token counts use `estimate_tokens`; `cost_usd` and `latency_s` are the compressor's OWN
    inference cost and wall-clock, which sit inside effective cost per the track's rules.
    """

    segments: list[str]
    tokens_in_raw: int
    tokens_in_compressed: int
    latency_s: float
    cost_usd: float = 0.0


@runtime_checkable
class Compressor(Protocol):
    """The pluggable compressor seam. Implementations must be deterministic and 0.0-safe."""

    id: str
    version: str

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        """Compress the mutable segments; never sees the cached prefix by construction."""
        ...


class IdentityCompressor:
    """Compression off: returns every segment bit-for-bit. The seam's do-no-harm proof."""

    id = "identity"
    version = "1"

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        del config
        start = time.monotonic()
        raw = sum(estimate_tokens(segment) for segment in segments)
        return CompressionResult(
            segments=list(segments),
            tokens_in_raw=raw,
            tokens_in_compressed=raw,
            latency_s=time.monotonic() - start,
        )


class TruncateCompressor:
    """The trivial ratio-matched control: drop the trailing `aggressiveness` fraction.

    Keeps the leading ceil((1 - aggressiveness) * n) whitespace-delimited tokens of each
    segment. Deterministic per segment, so an unchanged segment always compresses to the same
    bytes (append-stability). This is a CONTROL, not a method: a learned compressor that does
    not beat it at equal ratio has learned nothing (the track's mandatory baseline).
    """

    id = "truncate"
    version = "1"

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        start = time.monotonic()
        out: list[str] = []
        for segment in segments:
            words = segment.split()
            keep = math.ceil(len(words) * (1.0 - config.aggressiveness))
            out.append(segment if keep >= len(words) else " ".join(words[:keep]))
        return CompressionResult(
            segments=out,
            tokens_in_raw=sum(estimate_tokens(segment) for segment in segments),
            tokens_in_compressed=sum(estimate_tokens(segment) for segment in out),
            latency_s=time.monotonic() - start,
        )


_COMPRESSORS: dict[str, Compressor] = {
    IdentityCompressor.id: IdentityCompressor(),
    TruncateCompressor.id: TruncateCompressor(),
}


def get_compressor(compressor_id: str) -> Compressor:
    """Resolve a compressor by id, or raise naming the known ids (fail at mount, not mid-call)."""
    compressor = _COMPRESSORS.get(compressor_id)
    if compressor is None:
        known = ", ".join(sorted(_COMPRESSORS))
        raise ValueError(
            f"unknown compressor '{compressor_id}'; known compressors: {known}. "
            "Register new compressors in wmh.optimize.compression before referencing them "
            "in a policy's compression config."
        )
    return compressor
