# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""The packaged BPE tokenizer behind every gateway input-token reservation.

The o200k vocabulary ships inside the package (gzip-compressed, digest
pinned), and the encoder is built directly from those ranks. Nothing here
consults tiktoken's registry, plugin discovery, on-disk cache, or the public
download URL, so a cold serving host with no egress loads the same tokenizer
as every other host, and a corrupt or missing table fails loud at bind time
with a message naming the file.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
from functools import cache
from importlib import resources

import tiktoken

RESERVATION_ENCODING = "o200k_base"
"""BPE used to count prompt text for every route.

It is the tokenizer of the OpenAI models the gateway serves most, and the
other served families (Anthropic, DeepSeek, Gemini) tokenize the same text
within the reservation headroom on production traffic, so one encoding keeps
the estimate deployment-independent: a ladder walk counts once and prices
each candidate from the same number.
"""

PACKAGED_RANKS_RESOURCE = "o200k_base.tiktoken.gz"
"""Package-data file holding the published ``o200k_base.tiktoken`` rank table,
gzip-compressed. One ``base64-token rank`` pair per line, as OpenAI publishes it."""

PACKAGED_RANKS_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
"""SHA-256 of the decompressed rank table: the digest tiktoken itself pins for
``o200k_base``, so the packaged copy is provably the published vocabulary."""

# The published o200k pre-tokenization pattern (tiktoken's openai_public plugin).
_O200K_PATTERN = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",  # noqa: E501
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",  # noqa: E501
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
_O200K_SPECIAL_TOKENS = {"<|endoftext|>": 199_999, "<|endofprompt|>": 200_018}


class ReservationTokenizerError(RuntimeError):
    """The packaged reservation vocabulary is missing or does not match its pinned digest."""


def packaged_ranks() -> dict[bytes, int]:
    """Decode the packaged rank table after verifying its pinned digest.

    Returns:
        Mergeable byte sequences mapped to their BPE ranks.

    Raises:
        ReservationTokenizerError: The resource is absent, not gzip, or its
            decompressed digest differs from :data:`PACKAGED_RANKS_SHA256`.
    """
    resource = resources.files(__package__).joinpath(PACKAGED_RANKS_RESOURCE)
    try:
        table = gzip.decompress(resource.read_bytes())
    except (OSError, EOFError) as exc:
        raise ReservationTokenizerError(
            f"gateway reservation tokenizer: packaged vocabulary {PACKAGED_RANKS_RESOURCE} "
            f"is missing or unreadable ({exc}); reinstall the experiential distribution"
        ) from exc
    digest = hashlib.sha256(table).hexdigest()
    if digest != PACKAGED_RANKS_SHA256:
        raise ReservationTokenizerError(
            f"gateway reservation tokenizer: packaged vocabulary {PACKAGED_RANKS_RESOURCE} "
            f"has digest {digest}, expected {PACKAGED_RANKS_SHA256}; reinstall the "
            "experiential distribution"
        )
    ranks: dict[bytes, int] = {}
    for line in table.splitlines():
        if not line:
            continue
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return ranks


@cache
def reservation_encoder() -> tiktoken.Encoding:
    """Return the process-wide reservation tokenizer, built once from package data.

    Building the encoder decodes two hundred thousand ranks (well under a
    second), so a serving process warms it at bind time instead of on its
    first request; the result is cached for the life of the process.

    Raises:
        ReservationTokenizerError: The packaged vocabulary failed verification.
    """
    return tiktoken.Encoding(
        name=RESERVATION_ENCODING,
        pat_str=_O200K_PATTERN,
        mergeable_ranks=packaged_ranks(),
        special_tokens=_O200K_SPECIAL_TOKENS,
    )


EXACT_COUNT_MAX_CHARACTERS = 1 << 20
"""Prompt characters the reservation counts with the BPE on a bridge thread.

About a quarter million tokens of prose or code, which covers every ordinary
coding-agent turn; the remainder of a longer prompt is bounded by its UTF-8
length (see :func:`reservation_token_bound`), so one oversized request holds a
bridge thread for a fixed tokenizer slice, never for its whole prompt."""


def count_reservation_tokens(text: str) -> int:
    """Count ``text`` with the reservation BPE exactly as ``encode_ordinary`` does.

    Encodes into tiktoken's native token buffer instead of a Python list, so the
    whole encode runs with the GIL released and no per-token Python integer is
    ever allocated. Special-token strings count as ordinary text, and lone
    surrogates (legal in a decoded JSON string) are replaced the way
    ``encode_ordinary`` replaces them.
    """
    if not text:
        return 0
    encoder = reservation_encoder()
    try:
        return len(encoder.encode_to_numpy(text, disallowed_special=()))
    except UnicodeEncodeError:
        repaired = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        return len(encoder.encode_to_numpy(repaired, disallowed_special=()))


def reservation_token_bound(
    text: str, *, exact_characters: int = EXACT_COUNT_MAX_CHARACTERS
) -> int:
    """Return a bound at or above the reservation BPE count of ``text``.

    Text within ``exact_characters`` is counted exactly. Longer text is split at
    a pre-tokenization reset (a newline directly followed by a letter, where no
    o200k pre-token can span the boundary, so the BPE count of the whole is the
    sum of the counts of the halves); the prefix is counted exactly and the
    suffix is bounded by its UTF-8 length, since every byte-level BPE token
    covers at least one byte. Without a reset inside the window the whole text
    is bounded by its UTF-8 length.
    """
    if len(text) <= exact_characters:
        return count_reservation_tokens(text)
    split = _pretokenizer_reset(text, exact_characters)
    return count_reservation_tokens(text[:split]) + _utf8_length(text[split:])


_RESET_SEARCH_NEWLINES = 4_096
"""Newlines inspected backwards from the window edge for a reset point."""


def _pretokenizer_reset(text: str, limit: int) -> int:
    """Return the largest reset index at or below ``limit``, or 0 when none is near.

    A reset is an index ``i`` with ``text[i - 1] == "\\n"`` and ``text[i]`` a
    letter: no o200k pattern alternative matches a newline followed by a letter
    (letter runs exclude ``\\r\\n`` from their optional prefix, punctuation runs
    stop after trailing newlines, whitespace runs stop before non-space), and
    the pattern has no look-behind, so matching restarts identically there.
    """
    end = limit
    for _ in range(_RESET_SEARCH_NEWLINES):
        newline = text.rfind("\n", 0, end)
        if newline < 0:
            return 0
        if text[newline + 1].isalpha():
            return newline + 1
        end = newline
    return 0


def _utf8_length(text: str) -> int:
    """UTF-8 length of ``text``, counting a lone surrogate as its three encoded bytes."""
    return len(text.encode("utf-8", "surrogatepass"))
