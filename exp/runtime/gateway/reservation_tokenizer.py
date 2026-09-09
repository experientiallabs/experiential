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
