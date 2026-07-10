"""Unified LLM provider layer.

One interface (`Provider`), multiple backends, one entry point (`get_provider` — or
`provider_or_chain`, which upgrades to the local `.wmh/fallback.toml` failover chain when present).
All can be verified on startup with a cheap ping. Built fresh for this repo; no external client
framework.
"""

from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    EmbedderKind,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
)
from wmh.providers.registry import get_provider, verify_all, verify_embedder
from wmh.providers.waterfall import WaterfallProvider, provider_or_chain

__all__ = [
    "Provider",
    "ProviderConfig",
    "ProviderKind",
    "EmbedderKind",
    "DEFAULT_MAX_TOKENS",
    "Completion",
    "Message",
    "VerifyResult",
    "get_provider",
    "provider_or_chain",
    "WaterfallProvider",
    "verify_all",
    "verify_embedder",
]
