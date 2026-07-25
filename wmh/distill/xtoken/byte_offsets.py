"""Exact per-token byte offsets for a token sequence, with no re-encoding.

Cross-tokenizer chunk alignment needs to know which BYTES of the decoded text
each token covers, on both the student side (the exact sampled ids, which
TITO forbids re-encoding) and the teacher side. Re-encoding is not an option
for the student: sampling routinely emits a NON-canonical BPE segmentation of
its own text (measured on the headline run's real sinks, `'S' + 'olver'` where
canonical BPE gives `'Solver'`), so `encode(decode(ids)) != ids` for 14% of
spans covering 42% of sampled tokens. Gating on that round trip would discard
those tokens as "unstable" when nothing about them is unstable.

Byte-level BPE vocabularies (the GPT-2 lineage, which both Qwen3.6 and
GLM-5.2 use) make the exact answer available directly: every ordinary token's
surface form is a reversible byte-level encoding of the bytes it covers, so
per-token byte lengths come from the vocabulary alone. `span_byte_ends`
reconstructs the span's bytes that way and VERIFIES the reconstruction against
the tokenizer's own decode before returning, so a vocabulary that does not
work this way (a SentencePiece tokenizer that normalizes text, say) is
reported as unusable instead of silently producing wrong offsets.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class ByteOffsetTokenizer(Protocol):
    """The tokenizer slice byte-offset reconstruction needs.

    HuggingFace fast tokenizers satisfy this structurally, as do the small
    deterministic fakes in the tests (extra defaulted parameters do not
    matter).
    """

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str | None]:
        """The surface form of each token id; None for ids outside the vocab."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text."""
        ...


def _byte_decoder() -> dict[str, int]:
    """The GPT-2 byte-level BPE surface-character to raw-byte map.

    Byte-level BPE renders each of the 256 possible bytes as exactly one
    printable character so a BPE vocabulary can be built over text. The map is
    a fixed property of the scheme, not of any model, so it is derived here
    rather than read from a tokenizer.
    """
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    surface = list(printable)
    spare = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            surface.append(256 + spare)
            spare += 1
    return {chr(code): value for value, code in zip(printable, surface, strict=True)}


BYTE_DECODER = _byte_decoder()
"""Surface character to raw byte, for byte-level BPE token surface forms."""


def token_bytes(surface: str) -> bytes | None:
    """The raw bytes one byte-level BPE token surface form stands for.

    Args:
        surface: A token's surface form from `convert_ids_to_tokens`.

    Returns:
        The bytes the token covers, or None when the surface form contains a
        character outside the byte-level alphabet. That is the normal case for
        special and added tokens (`<|im_end|>`), whose surface form is their
        literal text; callers fall back to encoding that text directly.
    """
    try:
        return bytes(BYTE_DECODER[char] for char in surface)
    except KeyError:
        return None


def span_byte_ends(
    tokenizer: ByteOffsetTokenizer, token_ids: list[int]
) -> tuple[list[int], bytes] | None:
    """Cumulative byte-end offset per token, plus the span's decoded bytes.

    Entry i is the byte offset just past token i, so token i covers
    `bytes[ends[i - 1] : ends[i]]` (with `ends[-1] == 0` implied for i = 0).
    The result is exact for the tokenizer's ACTUAL ids: nothing is re-encoded,
    so a non-canonical sampled segmentation is handled like any other.

    Args:
        tokenizer: The tokenizer that produced `token_ids`.
        token_ids: The span's token ids, in order.

    Returns:
        `(byte_ends, span_bytes)` when reconstruction agrees with the
        tokenizer's own decode, else None. None means this tokenizer does not
        expose reversible per-token bytes (it normalizes text, or a token id
        is outside the vocabulary), so the caller must treat the span as
        unscoreable rather than guess at offsets.
    """
    if not token_ids:
        return [], b""
    surfaces = tokenizer.convert_ids_to_tokens(list(token_ids))
    if len(surfaces) != len(token_ids):
        logger.warning(
            "tokenizer returned %d surface form(s) for %d token id(s); cannot "
            "reconstruct byte offsets for this span",
            len(surfaces),
            len(token_ids),
        )
        return None
    pieces: list[bytes] = []
    for index, surface in enumerate(surfaces):
        if surface is None:
            logger.warning(
                "token id %d at position %d has no surface form (outside the "
                "vocabulary); cannot reconstruct byte offsets for this span",
                token_ids[index],
                index,
            )
            return None
        raw = token_bytes(surface)
        pieces.append(surface.encode("utf-8") if raw is None else raw)
    span_bytes = b"".join(pieces)
    # The reconstruction is only trustworthy if it reproduces what the
    # tokenizer itself decodes. A normalizing tokenizer (SentencePiece
    # whitespace rewriting, NFKC) fails here, which is exactly the signal the
    # caller needs: alignment by byte offset is invalid for that vocabulary.
    decoded = tokenizer.decode(list(token_ids))
    if span_bytes != decoded.encode("utf-8"):
        logger.warning(
            "per-token byte reconstruction (%d bytes) does not match the "
            "tokenizer's decode (%d bytes); this vocabulary does not expose "
            "reversible per-token bytes, so the span cannot be byte-aligned",
            len(span_bytes),
            len(decoded.encode("utf-8")),
        )
        return None
    ends: list[int] = []
    total = 0
    for piece in pieces:
        total += len(piece)
        ends.append(total)
    return ends, span_bytes
