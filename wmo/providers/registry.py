"""Centralized provider entry point.

`get_provider` is the single constructor the rest of the harness uses; nothing imports a concrete
backend directly. `verify_all` powers `wmo providers verify`.
"""

from __future__ import annotations

from wmo.providers.anthropic import AnthropicProvider
from wmo.providers.azure_openai import AzureOpenAIProvider
from wmo.providers.base import Provider, ProviderConfig, ProviderKind, VerifyResult
from wmo.providers.bedrock import BedrockProvider
from wmo.providers.openai import OpenAIProvider
from wmo.providers.openai_responses import OpenAIResponsesProvider
from wmo.providers.tinker import TinkerChatProvider

_BACKENDS = {
    ProviderKind.ANTHROPIC: AnthropicProvider,
    ProviderKind.BEDROCK: BedrockProvider,
    ProviderKind.AZURE_OPENAI: AzureOpenAIProvider,
    ProviderKind.OPENAI: OpenAIProvider,
    ProviderKind.OPENAI_RESPONSES: OpenAIResponsesProvider,
    ProviderKind.TINKER: TinkerChatProvider,
}


def get_provider(config: ProviderConfig, *, api_key: str | None = None) -> Provider:
    """Construct the provider for `config.kind`. The one place backends are wired in.

    `api_key` is the trusted explicit-credential channel: only operator-owned call sites (the
    model pool, tests) can pass it, so untrusted bundle config can never choose a credential.
    When set, the backend authenticates with exactly this key instead of its default env vars.
    """
    try:
        backend = _BACKENDS[config.kind]
    except KeyError:  # pragma: no cover - exhaustive over the enum
        raise ValueError(f"unknown provider kind: {config.kind}") from None
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
