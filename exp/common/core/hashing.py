"""Deterministic signed token hashing used by local embedders."""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def signed_token_embedding(text: str, dimensions: int) -> tuple[float, ...]:
    """Hash normalized content tokens into one fixed-width unit vector.

    Args:
        text: Source text whose tokens are hashed.
        dimensions: Fixed vector width. Must be at least 8.

    Returns:
        A unit-normalized signed hashing vector.

    Raises:
        ValueError: The requested width is smaller than 8.
    """
    if dimensions < 8:
        raise ValueError("signed token embedding needs at least 8 dimensions")
    vector = [0.0] * dimensions
    tokens = _TOKEN_PATTERN.findall(text.casefold()) or ["empty"]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # Preserve nonzero embeddings and give cancelling token sequences distinct vectors.
        digest = hashlib.blake2b("\0".join(tokens).encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] = 1.0 if digest[4] % 2 == 0 else -1.0
        norm = 1.0
    return tuple(value / norm for value in vector)
