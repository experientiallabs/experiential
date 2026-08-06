"""Unified LLM provider layer.

One interface (`Provider`), multiple backends, one entry point (`get_provider` — or
`provider_or_chain`, which upgrades to the local `.wmo/fallback.toml` failover chain when present).
All can be verified on startup with a cheap ping. Built fresh for this repo; no external client
framework.

Waterfall exports load on first access so a caller that only needs `get_provider` never pays for
the failover chain (or the Bedrock helper it uses for region resolution).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wmo.common.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    EmbedderKind,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
)
from wmo.common.providers.models import (
    ProviderModel,
    model_types_for_provider,
    resolve_provider_model,
)
from wmo.common.providers.registry import get_provider, verify_all, verify_embedder

if TYPE_CHECKING:
    from wmo.common.providers.waterfall import WaterfallProvider as WaterfallProvider
    from wmo.common.providers.waterfall import provider_or_chain as provider_or_chain

__all__ = [
    "Provider",
    "ProviderConfig",
    "ProviderKind",
    "ProviderModel",
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
    "model_types_for_provider",
    "resolve_provider_model",
]


def __getattr__(name: str) -> object:
    if name in ("WaterfallProvider", "provider_or_chain"):
        from wmo.common.providers.waterfall import WaterfallProvider, provider_or_chain

        globals()["WaterfallProvider"] = WaterfallProvider
        globals()["provider_or_chain"] = provider_or_chain
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
