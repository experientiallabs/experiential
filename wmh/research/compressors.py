"""Research-track compressor implementations behind the D-COMPRESS seam (track C1).

These are the round 0 survivors adapted to the production `Compressor` protocol
(wmh.optimize.compression), plus the mandatory random-removal control. They register at
runtime via `register_compressor` from the acceptance-benchmark runner; nothing in the
serving path references them until a method survives the accuracy bar and is promoted.

Protocol obligations honored by every implementation here: deterministic (same segments
+ config -> same bytes), aggressiveness 0.0 is a bit-for-bit no-op, segments are
rewritten in place (never merged, split, or reordered), and decisions are local to a
segment or to the segments-so-far scan order, which is what keeps an episode's growing
transcript append-stable (round 0's kill bar 1).

The scored compressors take an injected word-scoring callable so this module stays
torch-free: the acceptance runner constructs the scorer (LLMLingua-2 keep-probabilities
or GPT-2 self-information) in its own environment and passes it in.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable

from wmh.optimize.compression import CompressionConfig, CompressionResult, estimate_tokens
from wmh.research.compression import _shingles, split_units, split_words


def _result(raw_segments: list[str], out_segments: list[str], started: float) -> CompressionResult:
    return CompressionResult(
        segments=out_segments,
        tokens_in_raw=sum(estimate_tokens(s) for s in raw_segments),
        tokens_in_compressed=sum(estimate_tokens(s) for s in out_segments),
        latency_s=time.monotonic() - started,
    )


class RandomRemovalCompressor:
    """Matched-ratio random word removal, the control every method must beat.

    Removes `aggressiveness` of each segment's words, rng seeded from the segment text
    (deterministic per segment; unchanged segments always produce identical bytes).
    """

    id = "random-removal"
    version = "1"

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        started = time.monotonic()
        if config.aggressiveness == 0.0:
            return _result(segments, list(segments), started)
        out: list[str] = []
        for segment in segments:
            words = split_words(segment)
            seed = int.from_bytes(hashlib.sha256(segment.encode()).digest()[:8], "big")
            rng = random.Random(seed)
            n_remove = round(len(words) * config.aggressiveness)
            drop = set(rng.sample(range(len(words)), min(n_remove, len(words))))
            out.append("".join(w for i, w in enumerate(words) if i not in drop))
        return _result(segments, out, started)


class DedupKeepFirstCompressor:
    """Cross-segment line dedup, keep-first, fixed near-dup threshold (round 0 survivor).

    Scan order over segments makes it append-stable: a later segment can never change an
    earlier segment's output. `aggressiveness` gates on/off only (the method has no
    ratio knob; achieved ratio is whatever the redundancy allows and is recorded by the
    caller).
    """

    id = "dedup-keep-first"
    version = "1"
    jaccard = 0.9

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        started = time.monotonic()
        if config.aggressiveness == 0.0:
            return _result(segments, list(segments), started)
        seen_exact: set[str] = set()
        seen_shingles: list[set[str]] = []
        out: list[str] = []
        for segment in segments:
            kept: list[str] = []
            for unit in split_units(segment):
                body = unit.strip()
                if not body:
                    kept.append(unit)
                    continue
                if body in seen_exact:
                    continue
                sh = _shingles(body)
                if any(
                    len(sh | prior) and len(sh & prior) / len(sh | prior) >= self.jaccard
                    for prior in seen_shingles
                ):
                    continue
                kept.append(unit)
                seen_exact.add(body)
                seen_shingles.append(sh)
            out.append("".join(kept))
        return _result(segments, out, started)


class TruncateProtectTaskCompressor:
    """Head-keep ratio truncation of every segment EXCEPT the first (round 0's symbolic arm).

    Segment 0 carries the task and is never touched; later segments keep the leading
    (1 - aggressiveness) fraction of words, decided per segment (byte-stable for
    unchanged segments). Differs from the `truncate` control exactly by the protection,
    so the pair measures what protecting the instruction is worth.
    """

    id = "truncate-protect-task"
    version = "1"

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        started = time.monotonic()
        out: list[str] = []
        for i, segment in enumerate(segments):
            if i == 0 or config.aggressiveness == 0.0:
                out.append(segment)
                continue
            words = split_words(segment)
            keep = max(1, round(len(words) * (1.0 - config.aggressiveness)))
            out.append(segment if keep >= len(words) else "".join(words[:keep]).rstrip())
        return _result(segments, out, started)


WordScoreFn = Callable[[str], list[float]]
"""Scores one segment's words in isolation (same order/count as split_words(segment))."""


class ScoredWordCompressor:
    """Fixed-threshold word filter over an injected scorer (round 0's surviving learned shape).

    Keeps words whose score clears `threshold`; the per-input percentile rule is
    deliberately NOT offered (round 0 killed it). The scorer sees one segment at a time,
    so decisions are local and the growing transcript stays append-stable. `usd_per_10k`
    amortizes the measured hosting cost into `cost_usd` (H100 latency leg).
    """

    version = "1"

    def __init__(
        self,
        compressor_id: str,
        score_fn: WordScoreFn,
        threshold: float,
        usd_per_10k: float = 0.0,
    ) -> None:
        self.id = compressor_id
        self._score_fn = score_fn
        self._threshold = threshold
        self._usd_per_10k = usd_per_10k

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        started = time.monotonic()
        if config.aggressiveness == 0.0:
            return _result(segments, list(segments), started)
        out: list[str] = []
        for segment in segments:
            words = split_words(segment)
            scores = self._score_fn(segment)
            if len(scores) != len(words):
                raise ValueError(
                    f"scorer returned {len(scores)} scores for {len(words)} words "
                    f"(compressor '{self.id}')"
                )
            out.append(
                "".join(w for w, s in zip(words, scores, strict=True) if s >= self._threshold)
            )
        result = _result(segments, out, started)
        result.cost_usd = result.tokens_in_raw / 10_000 * self._usd_per_10k
        return result
