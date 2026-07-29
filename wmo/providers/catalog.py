"""What models each provider kind can offer, so nothing has to ship a second hardcoded roster.

`wmo providers set` registers routing candidates, which means answering "which models can I pick
from this provider". Three sources answer it, and `ProviderCatalog.source` records which one did
so the CLI can say where its list came from:

- `PUBLISHED`: the provider's own catalog. OpenRouter is the one backend that publishes a
  complete and PRICED list (`GET /api/v1/models`), and `wmo.providers.openrouter_pricing`
  already fetches, normalizes, and caches it, so this reuses that seam rather than fetching
  again.
- `BUILT_IN`: WMO's canonical `ProviderModel` registry (`wmo.providers.models`), the same table
  the worker picker and every capability lookup already resolve against.
- `NONE`: there is nothing to enumerate (Tinker names weights, not a catalog).

Only OpenRouter is `PUBLISHED` on purpose. The other vendors' list endpoints answer a different
question than this one asks: OpenAI's `GET /v1/models` returns embeddings, speech, and moderation
models beside the chat ones, so filtering it back down to routable candidates would need exactly
the hardcoded model-name knowledge this module exists to avoid, and neither it nor Anthropic's
publishes prices, so every enumerated row would still stop at a manual price prompt. The built-in
registry is better data for those kinds, and a typed id covers everything either list would add.

A SELF-HOSTED OpenAI-compatible server is the exception to that reasoning, which is why
`endpoint_catalog` exists beside `list_provider_models`: its `GET {endpoint}/models` lists
exactly what that one server serves (Ollama lists the pulled models, vLLM the model it was
launched with), there is no embeddings/speech noise to filter, and no price is expected because
a self-hosted candidate's price is whatever the operator declares (default 0).

A catalog is a set of SUGGESTIONS, never a whitelist: every kind accepts a typed id, because a
vendor's lineup moves faster than any release of this package.
"""

from __future__ import annotations

from enum import StrEnum

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmo.providers.base import ProviderKind
from wmo.providers.models import model_types_for_provider, resolve_provider_model
from wmo.providers.openrouter import OPENROUTER_MODELS_URL
from wmo.providers.openrouter_pricing import price_table
from wmo.tracking.pricing import ModelPrice, price_for


class CatalogSource(StrEnum):
    """Where a `ProviderCatalog`'s rows came from (see the module docstring)."""

    PUBLISHED = "published"
    BUILT_IN = "built-in"
    NONE = "none"


class CatalogModel(BaseModel):
    """One offerable model: what a pool entry would put in `model`, plus what we know about it."""

    model_config = ConfigDict(extra="forbid")

    # The provider runtime id, i.e. exactly what `PoolEntry.model` should carry.
    id: str = Field(min_length=1)
    # Canonical identity when `id` does not carry it (Bedrock's `us.anthropic.claude-opus-4-8`
    # is the runtime id of the `claude-opus-4-8` model type). None when the two are the same.
    model_type: str | None = None
    # A price we already know, from the published catalog or the built-in table. None means the
    # registry has to ask for one, because an unpriced candidate silently reports $0.
    price: ModelPrice | None = None

    def label(self) -> str:
        """This row as one line of picker text: the id, and its price when one is known."""
        if self.price is None:
            return self.id
        return (
            f"{self.id}  ${self.price.input_per_mtok:g} / ${self.price.output_per_mtok:g} per Mtok"
        )


class ProviderCatalog(BaseModel):
    """Everything one kind can be asked for, plus where the list came from."""

    model_config = ConfigDict(extra="forbid")

    kind: ProviderKind
    source: CatalogSource
    models: list[CatalogModel] = Field(default_factory=list)
    # Why the list is empty or short, written to be shown to the user as-is. Empty when the
    # catalog is exactly what it should be.
    detail: str = ""

    def find(self, model_id: str) -> CatalogModel | None:
        """The row whose id or model type is `model_id` (case-insensitive), else None."""
        wanted = model_id.strip().lower()
        for model in self.models:
            if wanted in (model.id.lower(), (model.model_type or "").lower()):
                return model
        return None

    def search(self, term: str) -> list[CatalogModel]:
        """Every row whose id or model type contains `term`, case-insensitively.

        Substring rather than prefix: OpenRouter ids are `vendor/model`, so "sonnet" has to reach
        `anthropic/claude-sonnet-4.5` for the picker to be usable over 338 rows at all.
        """
        needle = term.strip().lower()
        if not needle:
            return list(self.models)
        return [
            model
            for model in self.models
            if needle in model.id.lower() or needle in (model.model_type or "").lower()
        ]


def list_provider_models(kind: ProviderKind) -> ProviderCatalog:
    """Everything `kind` can offer as a routing candidate, from the best source it has.

    Never raises and never blocks on more than the one bounded catalog fetch
    `wmo.providers.openrouter_pricing` already owns: an offline machine gets an empty catalog
    whose `detail` says so, and the caller falls back to a typed id.

    Args:
        kind: The provider backend to enumerate.

    Returns:
        The catalog, with `source` naming where its rows came from.
    """
    match kind:
        case ProviderKind.OPENROUTER:
            return _openrouter_catalog()
        case ProviderKind.TINKER:
            return ProviderCatalog(
                kind=kind,
                source=CatalogSource.NONE,
                detail=(
                    "tinker candidates name a BASE model (the renderer and tokenizer resolve "
                    "from it), not a catalog entry, so type the base model id"
                ),
            )
        case (
            ProviderKind.OPENAI
            | ProviderKind.OPENAI_RESPONSES
            | ProviderKind.ANTHROPIC
            | ProviderKind.BEDROCK
            | ProviderKind.AZURE_OPENAI
        ):
            return _built_in_catalog(kind)


def _openrouter_catalog() -> ProviderCatalog:
    """OpenRouter's published catalog, from the same cached table that prices pool entries."""
    table, detail = price_table()
    if table is None:
        return ProviderCatalog(
            kind=ProviderKind.OPENROUTER, source=CatalogSource.PUBLISHED, detail=detail
        )
    models = [
        CatalogModel(id=model_id, price=price) for model_id, price in sorted(table.prices.items())
    ]
    return ProviderCatalog(
        kind=ProviderKind.OPENROUTER,
        source=CatalogSource.PUBLISHED,
        models=models,
        detail=f"{len(models)} models published at {OPENROUTER_MODELS_URL}",
    )


def _built_in_catalog(kind: ProviderKind) -> ProviderCatalog:
    """`kind`'s rows in WMO's canonical model registry, priced from the built-in table."""
    models: list[CatalogModel] = []
    for model_type in model_types_for_provider(kind):
        spec = resolve_provider_model(kind, model_type)
        models.append(
            CatalogModel(
                id=spec.model_id,
                model_type=None if spec.model_id == spec.model_type else spec.model_type,
                price=price_for(spec.model_id) or price_for(spec.model_type),
            )
        )
    return ProviderCatalog(
        kind=kind,
        source=CatalogSource.BUILT_IN,
        models=models,
        detail=(
            f"{len(models)} models from WMO's built-in registry; "
            f"{kind.value} publishes no priced catalog, so type any id it serves"
        ),
    )


ENDPOINT_CATALOG_TIMEOUT_S = 5.0
"""Bound on the one `GET {endpoint}/models` probe: a wrong URL must fail as a prompt answer,
not hang the registration flow."""


class _EndpointModelRow(BaseModel):
    """One row of an OpenAI-compatible `GET /models` response; extras ignored on purpose."""

    id: str = Field(min_length=1)


class _EndpointModelList(BaseModel):
    """The `{"object": "list", "data": [...]}` body every compatible server answers with."""

    data: list[_EndpointModelRow]


def endpoint_catalog(endpoint: str) -> ProviderCatalog:
    """What one OpenAI-compatible server serves, from its own `GET {endpoint}/models`.

    The self-hosted counterpart of `list_provider_models`, keyed on the URL instead of the kind
    (the entries it feeds are plain `ProviderKind.OPENAI` rows with `endpoint` set). Rows carry
    no price: a self-hosted candidate is priced by the operator, default 0. Like every other
    catalog it never raises and is only ever a set of suggestions; an unreachable or
    non-conforming server comes back as an empty catalog whose `detail` says what happened, and
    the caller falls back to a typed id.

    Args:
        endpoint: The server's base URL as a pool entry would carry it (".../v1").

    Returns:
        The catalog, with `detail` naming the URL it asked.
    """
    url = f"{endpoint.rstrip('/')}/models"
    try:
        response = httpx.get(url, timeout=ENDPOINT_CATALOG_TIMEOUT_S)
        response.raise_for_status()
        listing = _EndpointModelList.model_validate(response.json())
    except (httpx.HTTPError, ValidationError, ValueError) as exc:
        return ProviderCatalog(
            kind=ProviderKind.OPENAI,
            source=CatalogSource.NONE,
            detail=f"could not list models from {url} ({exc}); type the model id the server serves",
        )
    models = [CatalogModel(id=row.id) for row in listing.data]
    return ProviderCatalog(
        kind=ProviderKind.OPENAI,
        source=CatalogSource.PUBLISHED,
        models=models,
        detail=f"{len(models)} models served at {url}",
    )
