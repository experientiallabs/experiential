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

# ProviderKinds with a REAL llm-waterfall adapter. AZURE_OPENAI is excluded until the package
# implements it (its adapter is a construction-time stub); OPENAI_RESPONSES has no equivalent —
# the package speaks chat-completions. Keep using wmh's native providers for both.
_SUPPORTED_KINDS = frozenset({ProviderKind.ANTHROPIC, ProviderKind.BEDROCK, ProviderKind.OPENAI})


def to_backend(config: ProviderConfig, *, profile: str | None = None) -> Backend:
    """Map a wmh ProviderConfig onto an llm-waterfall Backend.

    `profile` selects a named AWS profile (Bedrock), letting one chain span multiple accounts —
    wmh configs don't model that, so it's a separate argument (see `WaterfallProvider(profiles=)`).
    """
    if config.kind not in _SUPPORTED_KINDS:
        raise ValueError(
            f"provider kind {config.kind.value!r} has no llm-waterfall backend; supported: "
            f"{', '.join(sorted(k.value for k in _SUPPORTED_KINDS))}"
        )
    return Backend(
        config.kind.value,
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


# The default failover chain lives in a gitignored file (`.wmh/` is ignored wholesale): profile
# names identify AWS accounts and the file may carry an OpenAI key, none of which belong in git.
FALLBACK_CONFIG_PATH = Path(".wmh/fallback.toml")

_ALLOWED_KEYS = frozenset(
    {"kind", "model", "profile", "region", "api_key", "embed_model", "embed_dim"}
)


def _parse_fallback_config(path: Path) -> tuple[list[ProviderConfig], list[str | None]]:
    """Parse `[[backend]]` entries into (configs, profiles), validating loudly."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = data.get("backend")
    if not entries:
        raise ValueError(
            f"{path}: no [[backend]] entries; each rung needs at least kind + model "
            "(kind/model/profile/region/api_key/embed_model/embed_dim)"
        )
    configs: list[ProviderConfig] = []
    profiles: list[str | None] = []
    for index, entry in enumerate(entries):
        unknown = set(entry) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"{path}: backend #{index + 1} has unknown key(s) {sorted(unknown)}; "
                f"allowed: {sorted(_ALLOWED_KEYS)}"
            )
        try:
            kind = ProviderKind(entry["kind"])
        except (KeyError, ValueError):
            raise ValueError(
                f"{path}: backend #{index + 1} needs kind ∈ "
                f"{sorted(k.value for k in _SUPPORTED_KINDS)} (got {entry.get('kind')!r})"
            ) from None
        if kind not in _SUPPORTED_KINDS:
            raise ValueError(
                f"{path}: backend #{index + 1} kind {kind.value!r} has no llm-waterfall "
                f"backend; supported: {sorted(k.value for k in _SUPPORTED_KINDS)}"
            )
        if "model" not in entry:
            raise ValueError(f"{path}: backend #{index + 1} is missing required key 'model'")
        api_key = entry.get("api_key")
        if api_key is not None:
            if kind is not ProviderKind.OPENAI:
                raise ValueError(
                    f"{path}: backend #{index + 1}: api_key only applies to kind='openai' "
                    "(bedrock uses AWS profiles; anthropic reads ANTHROPIC_API_KEY)"
                )
            # The env var is the adapter's credential channel; the file only seeds it so the
            # gitignored config is self-contained. A real env var always wins.
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        configs.append(
            ProviderConfig(
                kind=kind,
                model=entry["model"],
                region=entry.get("region"),
                embed_model=entry.get("embed_model"),
                embed_dim=entry.get("embed_dim"),
            )
        )
        profiles.append(entry.get("profile"))
    return configs, profiles


def provider_or_chain(config: ProviderConfig, *, path: Path | None = None) -> Provider:
    """The default provider-construction seam: single backend, or the local failover chain.

    When `.wmh/fallback.toml` exists, the requested provider is served by the whole chain —
    the requested (kind, model) leads as the primary unless it already heads the chain, and the
    file's rungs back it up. Without the file this is exactly `get_provider(config)`.
    """
    chain_path = path if path is not None else FALLBACK_CONFIG_PATH
    if not chain_path.exists():
        return get_provider(config)
    configs, profiles = _parse_fallback_config(chain_path)
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
