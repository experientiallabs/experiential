"""Centralized provider entry point.

`get_provider` is the single constructor the rest of the harness uses; nothing imports a concrete
backend directly. `verify_all` powers `wmo providers verify`.

Backends are imported only when their `ProviderKind` is requested, so a CLI that never talks to
Tinker (or Anthropic, …) never pays for those modules.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from wmo.providers.base import Provider, ProviderConfig, ProviderKind, VerifyResult

if TYPE_CHECKING:
    from collections.abc import Callable

# kind -> "module.path:ClassName". Resolved on first get_provider call for that kind only.
_BACKEND_PATHS: dict[ProviderKind, str] = {
    ProviderKind.ANTHROPIC: "wmo.providers.anthropic:AnthropicProvider",
    ProviderKind.BEDROCK: "wmo.providers.bedrock:BedrockProvider",
    ProviderKind.AZURE_OPENAI: "wmo.providers.azure_openai:AzureOpenAIProvider",
    ProviderKind.OPENAI: "wmo.providers.openai:OpenAIProvider",
    ProviderKind.OPENAI_RESPONSES: "wmo.providers.openai_responses:OpenAIResponsesProvider",
    ProviderKind.OPENROUTER: "wmo.providers.openrouter:OpenRouterProvider",
    ProviderKind.TINKER: "wmo.providers.tinker:TinkerChatProvider",
}

_backend_cache: dict[ProviderKind, Callable[..., Provider]] = {}


def _load_backend(kind: ProviderKind) -> Callable[..., Provider]:
    """Import and cache the constructor for `kind`."""
    cached = _backend_cache.get(kind)
    if cached is not None:
        return cached
    try:
        path = _BACKEND_PATHS[kind]
    except KeyError:  # pragma: no cover - exhaustive over the enum
        raise ValueError(f"unknown provider kind: {kind}") from None
    module_path, _, class_name = path.partition(":")
    module = importlib.import_module(module_path)
    backend = getattr(module, class_name)
    _backend_cache[kind] = backend
    return backend


def get_provider(config: ProviderConfig, *, api_key: str | None = None) -> Provider:
    """Construct the provider for `config.kind`. The one place backends are wired in.

    `api_key` is the trusted explicit-credential channel: only operator-owned call sites (the
    model pool, tests) can pass it, so untrusted bundle config can never choose a credential.
    When set, the backend authenticates with exactly this key instead of its default env vars.
    """
    backend = _load_backend(config.kind)
    if api_key is None:
        return backend(config)
    return backend(config, api_key=api_key)


def verify_all(configs: list[ProviderConfig]) -> list[VerifyResult]:
    """Ping every configured provider; never raises (failures come back as ok=False)."""
    results: list[VerifyResult] = []
    for cfg in configs:
        try:
            results.append(get_provider(cfg).verify())
        except Exception as exc:  # noqa: BLE001 - verification must not crash startup
            results.append(VerifyResult(ok=False, kind=cfg.kind, model=cfg.model, detail=str(exc)))
    return results


def verify_embedder(config: ProviderConfig) -> VerifyResult:
    """Embed one tiny string to confirm the embeddings path (creds + model) works.

    Mirrors `verify_via_ping` for the embed half: never raises — a failure (missing creds, no
    embeddings API, wrong model) comes back as `ok=False` with the detail. The reported `model` is
    the embeddings model (`embed_model`), falling back to the completion model when unset.
    """
    embed_model = config.embed_model or config.model
    try:
        vectors = get_provider(config).embed(["ping"])
    except Exception as exc:  # noqa: BLE001 - verification must not crash startup
        return VerifyResult(ok=False, kind=config.kind, model=embed_model, detail=str(exc))
    # A successful call must return one non-empty vector; an empty result or a zero-width vector
    # means the embed path didn't actually produce usable phi — report that as a failure, not ok.
    dim = len(vectors[0]) if vectors else 0
    if dim == 0:
        return VerifyResult(
            ok=False, kind=config.kind, model=embed_model, detail="embed returned no vector"
        )
    return VerifyResult(ok=True, kind=config.kind, model=embed_model, detail=f"dim={dim}")
