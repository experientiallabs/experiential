"""Candidate model pool: the editable roster the routing optimizer selects over.

The pool is one operator-owned TOML file (default `.wmo/pool.toml`, like `.wmo/fallback.toml`),
one `[[model]]` table per candidate. Swapping the roster is editing that file; nothing else in
the harness hardcodes candidate models.

Trust note: the pool file is trusted local config, unlike a model bundle's `config.toml`. That is
why entries may name `api_key_env` (which environment variable holds that account's API key) and
`pool_provider` resolves it and hands the key to `get_provider` as an explicit argument: the
explicit-argument channel is unreachable from bundle-controlled config, so a bundle can never
choose which env var gets read or where its value gets sent.

Pricing: entries for models in the built-in `wmo.tracking.pricing` table need no price fields;
anything else must declare `input_per_mtok`/`output_per_mtok` so downstream cost numbers stay
honest (an unpriced candidate would silently report $0). `cached_input_per_mtok` is the provider
cache-READ price, carried for cache-aware routing costs.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmo.providers.base import Provider, ProviderConfig, ProviderKind, TokenUsage
from wmo.providers.registry import get_provider
from wmo.tracking.pricing import ModelPrice, price_for

DEFAULT_POOL_PATH = Path(".wmo/pool.toml")

# D-REPORT ModelRef vocabulary: "frontier" anchors the improvement report's comparison; "open"
# models carry the run-10x-more-for-the-same-budget story.
Tier = Literal["frontier", "open"]


class PoolEntry(BaseModel):
    """One candidate model. `name` is the stable handle policy artifacts and request logs key on.

    `extra="forbid"`: a typo like `api_key_evn` must fail at load, not surface as a 401 at
    request time with no hint (same policy as `.wmo/fallback.toml`'s rungs).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: ProviderKind
    model: str = Field(min_length=1)  # provider runtime id (on Azure: the base model id)
    endpoint: str | None = None
    deployment: str | None = None  # Azure deployment name
    api_version: str | None = None  # Azure api-version
    region: str | None = None  # AWS Bedrock region (bedrock entries only)
    api_key_env: str | None = None  # env var holding this entry's API key (multi-account pools)
    tier: Tier = "frontier"
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cached_input_per_mtok: float | None = None  # provider cache-read price, USD per 1M tokens

    @model_validator(mode="after")
    def _validate_price(self) -> PoolEntry:
        if (self.input_per_mtok is None) != (self.output_per_mtok is None):
            raise ValueError(
                f"pool model '{self.name}': set both input_per_mtok and output_per_mtok, or neither"
            )
        if self.input_per_mtok is None and price_for(self.model) is None:
            raise ValueError(
                f"pool model '{self.name}': '{self.model}' has no built-in price; add "
                "input_per_mtok and output_per_mtok (USD per 1M tokens) to its pool entry"
            )
        if self.kind is ProviderKind.AZURE_OPENAI and self.deployment is None:
            # Without this the entry loads fine and the first request routed to it 500s
            # from AzureOpenAIProvider._deployment(); load is the validation boundary.
            raise ValueError(
                f"pool model '{self.name}': azure entries need `deployment` (the Azure "
                "deployment name to call)"
            )
        return self

    def price(self) -> ModelPrice:
        """This entry's price row: the explicit override, else the built-in pricing table."""
        if self.input_per_mtok is not None and self.output_per_mtok is not None:
            return ModelPrice(
                input_per_mtok=self.input_per_mtok, output_per_mtok=self.output_per_mtok
            )
        price = price_for(self.model)
        if price is None:  # unreachable after validation; keep the failure loud, not $0
            raise ValueError(f"pool model '{self.name}': no price available for '{self.model}'")
        return price

    def cost_usd(self, usage: TokenUsage) -> float:
        """Effective USD cost of `usage` priced by THIS entry's row (overrides included).

        Cache-adjusted: cached prompt tokens (`usage.cached_input_tokens`) bill at
        `cached_input_per_mtok` when the entry carries a cache-read price, and at the full
        input rate otherwise (never silently free). The global `wmo.tracking.pricing.cost_usd`
        only knows the built-in table; pool entries with explicit prices must be costed here
        or they would silently read $0.
        """
        price = self.price()
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        cached_rate = (
            self.cached_input_per_mtok
            if self.cached_input_per_mtok is not None
            else price.input_per_mtok
        )
        return (
            (usage.input_tokens - cached) * price.input_per_mtok
            + cached * cached_rate
            + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000

    def provider_config(self) -> ProviderConfig:
        return ProviderConfig(
            kind=self.kind,
            model=self.model,
            endpoint=self.endpoint,
            deployment=self.deployment,
            api_version=self.api_version,
            region=self.region,
        )


class ModelPool(BaseModel):
    """The full candidate roster, as loaded from one pool TOML."""

    models: list[PoolEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> ModelPool:
        seen: set[str] = set()
        for entry in self.models:
            if entry.name in seen:
                raise ValueError(f"pool model '{entry.name}' is declared twice; names are handles")
            seen.add(entry.name)
        return self

    def entry(self, name: str) -> PoolEntry:
        for candidate in self.models:
            if candidate.name == name:
                return candidate
        available = ", ".join(m.name for m in self.models)
        raise KeyError(f"no pool model named '{name}'; available: {available}")


def load_pool(path: Path = DEFAULT_POOL_PATH) -> ModelPool:
    """Load and validate the pool file at `path`."""
    if not path.is_file():
        raise FileNotFoundError(
            f"no model pool file at {path}; create it with one [[model]] table per candidate "
            "(fields: name, kind, model, and for non-built-in models input_per_mtok/"
            "output_per_mtok; endpoint/deployment/api_version/api_key_env as the backend needs)"
        )
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return ModelPool.model_validate({"models": data.get("model", [])})


def pool_provider(entry: PoolEntry) -> Provider:
    """Construct the provider for one pool entry, resolving its per-account API key."""
    api_key: str | None = None
    if entry.api_key_env:
        api_key = os.environ.get(entry.api_key_env)
        if not api_key:
            raise ValueError(
                f"pool model '{entry.name}': environment variable {entry.api_key_env} is unset "
                "or empty; export that account's API key or drop api_key_env to use the "
                "backend's default credentials"
            )
    return get_provider(entry.provider_config(), api_key=api_key)
