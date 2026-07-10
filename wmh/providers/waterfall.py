"""A Provider backed by llm-waterfall: fail over across a chain of backends on capacity errors.

Wraps `llm_waterfall.Waterfall` (github.com/experientiallabs/llm-waterfall) behind the wmh
`Provider` protocol so long GEPA/eval runs degrade gracefully to the next backend instead of
aborting when the preferred model throttles. Capacity errors (throttling / transient 5xx /
timeouts) spill down the chain; real errors (bad request, auth) propagate immediately.

`config` reports the *primary* config (the model we intend to use); per-call metering is still
attributed to the model that actually served, via `Completion.model`. The full attempt trail and
`provider_used` stay on the underlying package result — use `llm_waterfall.Waterfall` directly
when a caller needs failover observability beyond cost attribution.

Note on `embed`: the Provider protocol returns bare vectors, so embed usage/attribution is not
carried through. Failover also assumes the chain shares one embedding space — keep `embed_model`
consistent across rungs (see the llm-waterfall README).
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from llm_waterfall import Backend, CompletionResult, EmbeddingResult, RetryPolicy, Waterfall
from llm_waterfall import Message as WfMessage
from llm_waterfall import VerifyResult as WfVerifyResult
from pydantic import BaseModel, ConfigDict, ValidationError

from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmh.providers.registry import get_provider

# ProviderKinds with a REAL llm-waterfall adapter, mapped to the package's provider names
# (wmh's azure value is "azure"; the package spells it "azure_openai"). OPENAI_RESPONSES has no
# equivalent — the package speaks chat-completions; keep wmh's native provider for it.
_KIND_TO_PROVIDER = {
    ProviderKind.ANTHROPIC: "anthropic",
    ProviderKind.BEDROCK: "bedrock",
    ProviderKind.OPENAI: "openai",
    ProviderKind.AZURE_OPENAI: "azure_openai",
}
_SUPPORTED_KINDS = frozenset(_KIND_TO_PROVIDER)


def to_backend(config: ProviderConfig, *, profile: str | None = None) -> Backend:
    """Map a wmh ProviderConfig onto an llm-waterfall Backend.

    `profile` selects a named AWS profile (Bedrock), letting one chain span multiple accounts —
    wmh configs don't model that, so it's a separate argument (see `WaterfallProvider(profiles=)`).
    """
    provider = _KIND_TO_PROVIDER.get(config.kind)
    if provider is None:
        raise ValueError(
            f"provider kind {config.kind.value!r} has no llm-waterfall backend; supported: "
            f"{', '.join(sorted(k.value for k in _SUPPORTED_KINDS))}"
        )
    return Backend(
        provider,
        config.model,
        profile=profile,
        region=config.region,
        endpoint=config.endpoint,
        deployment=config.deployment,
        api_version=config.api_version,
        embed_model=config.embed_model,
        embed_dim=config.embed_dim,
    )


class WaterfallLike(Protocol):
    """The slice of `llm_waterfall.Waterfall` this provider uses (injectable in tests)."""

    def complete(
        self,
        system: str = "",
        messages: Sequence[WfMessage | Mapping[str, str]] = (),
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult: ...

    def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...

    def verify(self) -> list[WfVerifyResult]: ...


class WaterfallProvider:
    """Try a chain of backends in order per call; fail over only on capacity errors.

    `profiles`, when given, is zipped with `configs` to pin each Bedrock rung to a named AWS
    profile — one chain spanning several accounts sidesteps per-account throttling.
    """

    def __init__(
        self,
        configs: Sequence[ProviderConfig],
        *,
        profiles: Sequence[str | None] | None = None,
        retry: RetryPolicy | None = None,
        waterfall: WaterfallLike | None = None,
    ) -> None:
        if not configs:
            raise ValueError("WaterfallProvider needs at least one ProviderConfig")
        if profiles is not None and len(profiles) != len(configs):
            raise ValueError(
                f"profiles ({len(profiles)}) must match configs ({len(configs)}) one-to-one"
            )
        rung_profiles = profiles if profiles is not None else [None] * len(configs)
        self._waterfall = waterfall or Waterfall(
            [to_backend(c, profile=p) for c, p in zip(configs, rung_profiles, strict=True)],
            retry=retry if retry is not None else RetryPolicy(),
        )
        self.config: ProviderConfig = configs[0]

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        # Temperature is intentionally not forwarded — matches every other wmh provider
        # (current reasoning models reject non-default sampling params).
        del temperature
        result = self._waterfall.complete(
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
        )
        return Completion(
            text=result.text,
            usage=TokenUsage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            ),
            model=result.model_used,  # true attribution even when a fallback served
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._waterfall.embed(texts).vectors

    def verify(self) -> VerifyResult:
        """Ping every rung individually; ok only when the whole chain is healthy.

        A single ping through the chain would let a fallback silently answer for a dead
        primary — and never check the fallbacks' creds at all. Failing rungs are named in
        `detail` so `wmh providers verify` surfaces exactly which account/model is broken.
        """
        results = self._waterfall.verify()
        failing = [r for r in results if not r.ok]
        detail = "; ".join(f"{r.provider}/{r.model}: {r.detail}" for r in failing)
        return VerifyResult(
            ok=not failing,
            kind=self.config.kind,
            model=self.config.model,
            detail=detail or f"all {len(results)} backends verified",
        )


# Failover chains live in a gitignored file (`.wmh/` is ignored wholesale): profile names
# identify AWS accounts and the file may carry API keys, none of which belong in git. Format —
# one or more named chains, each an array of rungs, plus an optional default:
#
#     default = "main"
#
#     [[chain.main]]
#     kind = "bedrock"                 # bedrock | openai | anthropic
#     model = "us.anthropic.claude-opus-4-6-v1"
#     profile = "endflow"              # optional: named AWS profile (bedrock)
#     region = "us-west-2"             # optional
#     # api_key = "sk-..."             # optional: openai/anthropic key, seeded into the env
#     # embed_model / embed_dim        # optional: embeddings attribution
#
#     [[chain.opus-48]]
#     ...
FALLBACK_CONFIG_PATH = Path(".wmh/fallback.toml")

# The env var each kind's adapter reads; `api_key` in the file only seeds it (env wins), so the
# gitignored config is self-contained.
_API_KEY_ENV = {
    ProviderKind.OPENAI: "OPENAI_API_KEY",
    ProviderKind.ANTHROPIC: "ANTHROPIC_API_KEY",
    ProviderKind.AZURE_OPENAI: "AZURE_OPENAI_API_KEY",
}

Chain = tuple[list[ProviderConfig], list[str | None]]


class _Rung(BaseModel):
    """One `[[chain.<name>]]` entry; `extra="forbid"` turns typos into loud errors."""

    model_config = ConfigDict(extra="forbid")

    kind: ProviderKind
    model: str
    profile: str | None = None
    region: str | None = None
    endpoint: str | None = None  # azure resource URL / custom OpenAI base URL
    deployment: str | None = None  # azure deployment name (defaults to model)
    api_version: str | None = None  # azure api version
    api_key: str | None = None
    embed_model: str | None = None
    embed_dim: int | None = None


def _parse_rungs(path: Path, name: str, entries: list[dict[str, object]]) -> Chain:
    """Parse one chain's rung entries into (configs, profiles), validating loudly."""
    configs: list[ProviderConfig] = []
    profiles: list[str | None] = []
    for index, entry in enumerate(entries):
        where = f"{path}: chain {name!r} rung #{index + 1}"
        try:
            rung = _Rung.model_validate(entry)
        except ValidationError as exc:
            raise ValueError(f"{where} is invalid (unknown key or bad value): {exc}") from None
        if rung.kind not in _SUPPORTED_KINDS:
            raise ValueError(
                f"{where}: kind {rung.kind.value!r} has no llm-waterfall backend; "
                f"supported: {sorted(k.value for k in _SUPPORTED_KINDS)}"
            )
        if rung.api_key is not None:
            env_var = _API_KEY_ENV.get(rung.kind)
            if env_var is None:
                raise ValueError(
                    f"{where}: api_key only applies to kind='openai'/'anthropic'/'azure' "
                    "(bedrock uses AWS profiles)"
                )
            os.environ.setdefault(env_var, rung.api_key)
        configs.append(
            ProviderConfig(
                kind=rung.kind,
                model=rung.model,
                region=rung.region,
                endpoint=rung.endpoint,
                deployment=rung.deployment,
                api_version=rung.api_version,
                embed_model=rung.embed_model,
                embed_dim=rung.embed_dim,
            )
        )
        profiles.append(rung.profile)
    return configs, profiles


def _parse_fallback_config(path: Path) -> tuple[dict[str, Chain], str | None]:
    """Parse the named chains and the optional `default` selector."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown_top = set(data) - {"chain", "default"}
    if unknown_top:
        raise ValueError(
            f"{path}: unknown top-level key(s) {sorted(unknown_top)}; expected "
            '`default = "<name>"` and [[chain.<name>]] rung entries'
        )
    chains_raw = data.get("chain")
    if not isinstance(chains_raw, dict) or not chains_raw:
        raise ValueError(
            f"{path}: no chains found; define rungs as [[chain.<name>]] entries "
            "(kind/model/profile/region/api_key/embed_model/embed_dim per rung)"
        )
    chains = {name: _parse_rungs(path, name, entries) for name, entries in chains_raw.items()}
    default = data.get("default")
    if default is not None and default not in chains:
        raise ValueError(
            f"{path}: default = {default!r} names no chain; available: {sorted(chains)}"
        )
    return chains, default


def provider_or_chain(
    config: ProviderConfig, *, chain: str | None = None, path: Path | None = None
) -> Provider:
    """The default provider-construction seam: single backend, or a local failover chain.

    When `.wmh/fallback.toml` exists, the requested provider is served by the selected chain —
    `chain` names one of the file's chains (falling back to its `default`, or its only chain);
    the requested (kind, model) leads as the primary unless it already heads the chain. Without
    the file this is exactly `get_provider(config)`.
    """
    chain_path = path if path is not None else FALLBACK_CONFIG_PATH
    if not chain_path.exists():
        if chain is not None:
            raise ValueError(f"chain {chain!r} requested but {chain_path} does not exist")
        return get_provider(config)
    chains, default = _parse_fallback_config(chain_path)
    name = chain if chain is not None else default
    if name is None:
        if len(chains) > 1:
            raise ValueError(
                f"{chain_path} defines {sorted(chains)} but no `default`; pass a chain name "
                'or set `default = "<name>"` in the file'
            )
        name = next(iter(chains))
    if name not in chains:
        raise ValueError(f"chain {name!r} not in {chain_path}; available: {sorted(chains)}")
    configs, profiles = chains[name]
    heads_chain = configs[0].kind is config.kind and configs[0].model == config.model
    if not heads_chain:
        keep = [
            (c, p)
            for c, p in zip(configs, profiles, strict=True)
            if not (c.kind is config.kind and c.model == config.model and p is None)
        ]
        configs = [config, *(c for c, _ in keep)]
        profiles = [None, *(p for _, p in keep)]
    return WaterfallProvider(configs, profiles=profiles)
