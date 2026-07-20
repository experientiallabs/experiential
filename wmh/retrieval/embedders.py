"""Embedders for retrieval (phi), and the factory that picks one from config.

Two flavors of phi:

* `HashingEmbedder` — the offline, zero-config default. A deterministic hashed-bag-of-character-
  trigrams vector (the "hashing trick"), L2-normalized. Lexical, not semantic, but needs no creds or
  network, so the whole build/serve loop runs on completions alone.
* A real provider's embeddings API (Bedrock Titan / OpenAI / Azure OpenAI) — semantic phi. Selected
  by setting `embed_provider` (an `EmbedderKind`) + the backend's credentials.

Both satisfy the `wmh.providers.base.Embedder` protocol, so `EmbeddingRetriever` and the world model
consume either interchangeably. `get_embedder` is the single place this choice is resolved.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from wmh.core.types import JsonObject
from wmh.providers.base import EmbedderKind

if TYPE_CHECKING:
    from wmh.config import HarnessConfig
    from wmh.providers.base import Embedder

DEFAULT_DIM = 512
_NGRAM = 3


class HashingEmbedder:
    """Deterministic offline embedder: hashed character-trigram bag, L2-normalized.

    Not semantic, but stable and zero-dependency: identical text always maps to the identical
    vector, and lexically similar (state, action) pairs land near each other under cosine — which is
    what the retriever ranks on.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"embedding dim must be positive, got {dim}")
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def evaluation_context(self) -> JsonObject:
        """Return the exact deterministic embedder configuration."""
        return {"kind": "hashing_character_trigrams", "dimension": self._dim, "ngram": _NGRAM}

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float64)
        normalized = text.lower()
        if len(normalized) < _NGRAM:
            normalized = normalized.ljust(_NGRAM)
        for i in range(len(normalized) - _NGRAM + 1):
            gram = normalized[i : i + _NGRAM]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self._dim
            # Sign from one extra bit so colliding grams can cancel rather than only accumulate.
            sign = 1.0 if digest[0] & 1 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


def get_embedder(config: HarnessConfig) -> Embedder:
    """Resolve the configured phi embedder from a `HarnessConfig`.

    `embed_provider == HASHING` (the default) returns the offline `HashingEmbedder` sized to
    `config.embed_dim` — no credentials, no network. Any other kind constructs the matching backend
    provider (via the registry) with `embed_dim` threaded through, so the provider requests vectors
    of exactly the persisted dimension and the index/query vectors line up.

    The registry import is deferred to keep `wmh.retrieval` free of a hard dependency on the
    provider backends (retrieval only needs the `Embedder` protocol).
    """
    if config.embed_provider is EmbedderKind.HASHING:
        return HashingEmbedder(dim=config.embed_dim)

    from wmh.providers import get_provider

    # `embed_provider_config` resolves the backing provider and stamps `embed_dim` on it.
    return get_provider(config.embed_provider_config())
