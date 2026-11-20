"""Research-track compressors behind the D-COMPRESS seam: the controls the grid measures against.

The compression track's accuracy grid needs three non-learned arms beside the learned one, and
the seam already ships two of them (`identity` = off, `truncate` = the head-keep control). This
module holds the other two, adapted from the track's round 0 slate to the production `Compressor`
protocol (`wmo.optimize.compression`):

- `random-removal`: matched-ratio random word removal, the control every learned method must beat.
- `truncate-protect-task`: head-keep truncation that never touches the first segment, so pairing
  it with the built-in `truncate` control measures what protecting the instruction is worth.

Neither is a serving candidate. They register at runtime from the acceptance-benchmark runner
(`.agents/scripts/run_acceptance_bench.py`), and nothing in the serving path references them
until a method survives the accuracy bar and is promoted. Registering nothing at import keeps
`import wmo` free of research ids.

Protocol obligations honored by both: deterministic (same segments and config, same bytes out),
aggressiveness 0.0 is a bit-for-bit no-op, the dial is monotone in how much it removes, segments
are rewritten in place (never merged, split, dropped, or reordered), and every decision is local
to its own segment, which is what makes the attested `append_stable` true.

Torch stays out of this module and out of this track's local code: the learned arm is the H100
endpoint (`wmo.optimize.compression_endpoint`), not a model loaded in-process.
"""

from __future__ import annotations

import hashlib
import random
import re
import time

from wmo.optimize.compression import CompressionConfig, CompressionResult, estimate_tokens

WORD_RE = re.compile(r"\S+\s*")
"""A whitespace-delimited unit WITH its trailing whitespace, so joining the parts is lossless."""


def split_words(text: str) -> list[str]:
    """Split `text` into whitespace-delimited words, trailing whitespace attached.

    Terminology pin carried over from the track: "word" means a run of non-whitespace plus the
    whitespace that follows it, so `"".join(split_words(text)) == text` for any text. Removal
    rules can then drop elements without inventing or deleting separators.
    """
    return WORD_RE.findall(text)


def _result(raw_segments: list[str], out_segments: list[str], started: float) -> CompressionResult:
    """The seam's result shape for one call: output segments plus proxy-token accounting."""
    return CompressionResult(
        segments=out_segments,
        tokens_in_raw=sum(estimate_tokens(segment) for segment in raw_segments),
        tokens_in_compressed=sum(estimate_tokens(segment) for segment in out_segments),
        latency_s=time.monotonic() - started,
    )


class RandomRemovalCompressor:
    """Matched-ratio random word removal: the control every learned method must beat.

    Removes `aggressiveness` of each segment's words, with the RNG seeded from that segment's own
    text. Seeding on content rather than on a call counter is what makes this usable as a control
    in a served-shaped grid: an unchanged segment always yields identical bytes, so the method is
    deterministic and repeated calls over a growing transcript never rewrite what they already
    emitted.

    The dial is monotone in the AMOUNT removed (a higher value drops more words), which is the
    seam's invariant. The removed SET is not nested across dial values, because `random.sample`
    draws a fresh combination per k; that is a property of the control as round 0 defined it and
    it matters to nothing the seam relies on.
    """

    id = "random-removal"
    version = "1"
    # Selection is per segment and depends only on that segment's own bytes, so appending a
    # segment cannot change the output of any segment already emitted.
    append_stable = True

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


class TruncateProtectTaskCompressor:
    """Head-keep truncation of every segment EXCEPT the first (round 0's symbolic arm).

    Segment 0 carries the task and is never touched; later segments keep the leading
    (1 - aggressiveness) fraction of their words. Running this beside the seam's built-in
    `truncate` at one matched achieved ratio is how the track measures what protecting the
    instruction is worth.

    Two SECONDARY differences from that built-in, both inherited from round 0's implementation
    and neither about the protection: this keeps each word's original trailing whitespace
    (`split_words` is lossless, where `truncate` rejoins on single spaces and so flattens
    newlines), and it rounds the keep count where `truncate` takes the ceiling. Both are small at
    the ratios this track runs, and both are why the pair's achieved ratios differ slightly at
    the same dial value, but a reading of the protection's effect should not treat the pair as
    differing in one variable only.

    Keeping at least one word of a truncated segment is deliberate: a segment compressed to
    nothing reads to the model as an empty turn rather than as a shortened one, which is a
    different intervention from truncation.
    """

    id = "truncate-protect-task"
    version = "1"
    # Head-keep is head-absolute per segment: the kept prefix of a segment depends on nothing
    # outside that segment, and segment 0's protection is positional, so appending a segment
    # never rewrites an emitted one. (C1's correction to the lit review applies: the
    # "ratio budgets are never append-only" rule is about percentile SELECTION, not head-keep.)
    append_stable = True

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        started = time.monotonic()
        out: list[str] = []
        for index, segment in enumerate(segments):
            if index == 0 or config.aggressiveness == 0.0:
                out.append(segment)
                continue
            words = split_words(segment)
            keep = max(1, round(len(words) * (1.0 - config.aggressiveness)))
            out.append(segment if keep >= len(words) else "".join(words[:keep]).rstrip())
        return _result(segments, out, started)
