"""Prices for OpenRouter models, resolved from its published catalog and cached on disk.

The launch promise is "pass in traces and an OpenRouter key". Hand-writing
`input_per_mtok`/`output_per_mtok` on every `[[model]]` table is what breaks that promise, so a
`kind = "openrouter"` pool entry that declares no price gets one from OpenRouter's public
catalog (`GET /api/v1/models`, which publishes per-token prices for every model it fronts).

Three properties matter more than the fetch itself:

- **Lazy.** Nothing here runs at import. The only caller is `PoolEntry._validate_price`, and it
  calls only for an OpenRouter entry that supplied no price, so validating any other provider's
  config never touches the network or the disk.
- **Bounded and degrading.** One HTTP request per process at most: a failure is remembered in
  `_FETCH_ERROR`, so a pool of twenty entries on an offline machine pays one timeout, not
  twenty. `resolve_price` never raises. When there is no price it returns the reason, and the
  caller folds that into its existing "supply the prices explicitly" error.
- **Recorded once.** The resolved numbers are stamped onto the `PoolEntry`, and the pool entry
  is what `RoutingPolicy.pool` and `OutcomeMatrix.pool` serialize. A fitted policy therefore
  carries the prices it was fitted under and never re-resolves them at serve time.

The cache is OUR normalized table, not OpenRouter's payload: `$WMO_OPENROUTER_CATALOG`, else
`openrouter-prices.json` under `$WMO_HOME` (`~/.wmo`). Pointing that variable at a checked-in
file is also the air-gapped path, and it is how the test suite stays offline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from wmo.core.types import JsonValue
from wmo.platform.credentials import wmo_home
from wmo.providers.openrouter import OPENROUTER_MODELS_URL
from wmo.tracking.pricing import ModelPrice

logger = logging.getLogger(__name__)

CATALOG_PATH_ENV = "WMO_OPENROUTER_CATALOG"
CATALOG_FILENAME = "openrouter-prices.json"
CATALOG_TTL_SECONDS = 24 * 60 * 60
"""How long a cached table is used without refetching. Prices move on vendor announcements,
not hourly, and every routing artifact records the numbers it used, so a day is plenty."""

FETCH_TIMEOUT_SECONDS = 10.0
"""A price lookup is a validation step, not a request. It fails fast rather than blocking."""

# Set to the reason the last fetch failed, so the rest of the process degrades immediately
# instead of paying the timeout once per unpriced entry. Deliberately unlocked: the worst a race
# can do is a second redundant fetch, never a wrong price. Reset per test by the autouse fixture
# in `wmo/conftest.py`.
_FETCH_ERROR: str | None = None


class _CatalogPricing(BaseModel):
    """The `pricing` object of one catalog row: USD per TOKEN, as decimal strings.

    `extra="ignore"`: this mirrors a third-party payload that carries tiers we do not price
    (`request`, `image`, `web_search`, ...) and gains new ones without notice.
    """

    model_config = ConfigDict(extra="ignore")

    prompt: str | None = None
    completion: str | None = None
    input_cache_read: str | None = None
    input_cache_write: str | None = None


class _CatalogRow(BaseModel):
    """One model in OpenRouter's catalog (only the fields we consume)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    pricing: _CatalogPricing = Field(default_factory=_CatalogPricing)


class _CatalogResponse(BaseModel):
    """The `GET /api/v1/models` envelope."""

    model_config = ConfigDict(extra="ignore")

    data: list[_CatalogRow] = Field(default_factory=list)


class PriceCatalog(BaseModel):
    """The normalized price table we persist: model id -> price row, plus when it was fetched."""

    model_config = ConfigDict(extra="forbid")

    fetched_at: float  # unix seconds
    source: str  # the URL it came from (provenance for a hand-seeded file)
    prices: dict[str, ModelPrice]

    def is_stale(self, *, now: float | None = None) -> bool:
        """Whether this table is older than `CATALOG_TTL_SECONDS`."""
        return (now if now is not None else time.time()) - self.fetched_at > CATALOG_TTL_SECONDS


class PriceResolution(BaseModel):
    """One lookup's outcome: the price row, or `detail` explaining why there is none.

    `detail` is written as a mid-sentence clause, because its job is to be spliced into the
    caller's own "declare the prices explicitly" error rather than stand alone.
    """

    model_config = ConfigDict(extra="forbid")

    price: ModelPrice | None = None
    detail: str = ""


def catalog_path() -> Path:
    """Where the normalized price table is cached."""
    override = os.environ.get(CATALOG_PATH_ENV)
    return Path(override) if override else wmo_home() / CATALOG_FILENAME


def _per_mtok(value: str | None) -> float | None:
    """OpenRouter's USD-per-token decimal string -> USD per 1M tokens, or None if unusable.

    A negative price ("-1") marks variable pricing (the auto router), which cannot be recorded
    as a number; zero is a real price (free models) and is kept as one. The scaling goes
    through `Decimal` so a published "0.0000001" lands on 0.1 rather than 0.09999999999999999
    in every artifact that records it.
    """
    if value is None:
        return None
    try:
        per_token = Decimal(value)
    except InvalidOperation:
        return None
    if per_token < 0:
        return None
    return float(per_token * 1_000_000)


def _row_price(row: _CatalogRow) -> ModelPrice | None:
    """A catalog row's price, or None when it does not publish both completion tiers."""
    input_per_mtok = _per_mtok(row.pricing.prompt)
    output_per_mtok = _per_mtok(row.pricing.completion)
    if input_per_mtok is None or output_per_mtok is None:
        return None
    return ModelPrice(
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        cache_read_per_mtok=_per_mtok(row.pricing.input_cache_read),
        cache_write_per_mtok=_per_mtok(row.pricing.input_cache_write),
    )


def parse_catalog(payload: JsonValue, *, source: str = OPENROUTER_MODELS_URL) -> PriceCatalog:
    """Normalize a `GET /api/v1/models` payload into a price table.

    Args:
        payload: The decoded JSON body.
        source: Where it came from, recorded on the table for provenance.

    Returns:
        The normalized table, stamped with the current time.

    Raises:
        ValueError: If the payload has no usable priced model, which means the shape changed
            and silently caching an empty table would look exactly like "model not listed".
    """
    parsed = _CatalogResponse.model_validate(payload)
    prices = {row.id: price for row in parsed.data if (price := _row_price(row)) is not None}
    if not prices:
        raise ValueError(f"{source} returned no priced models ({len(parsed.data)} rows)")
    return PriceCatalog(fetched_at=time.time(), source=source, prices=prices)


def _fetch_catalog() -> PriceCatalog:
    """Fetch the live catalog. The suite replaces this attribute; nothing else calls it.

    Sent unauthenticated on purpose: the catalog is public, and a price lookup is no reason to
    put the user's key on the wire.
    """
    response = httpx.get(OPENROUTER_MODELS_URL, timeout=FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse_catalog(response.json())


def _read_cache(path: Path) -> PriceCatalog | None:
    """Read the cached table, or None when it is absent or unreadable."""
    if not path.is_file():
        return None
    try:
        return PriceCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("ignoring unreadable OpenRouter price cache at %s: %s", path, exc)
        return None


def _write_cache(path: Path, catalog: PriceCatalog) -> None:
    """Persist the table atomically. A cache that cannot be written is a warning, not an error."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f"{path.name}.partial")
        staging.write_text(json.dumps(catalog.model_dump(mode="json")), encoding="utf-8")
        staging.replace(path)
    except OSError as exc:
        logger.warning("could not cache the OpenRouter price table at %s: %s", path, exc)


def _catalog() -> tuple[PriceCatalog | None, str]:
    """The price table to answer lookups from, or None and the reason there is none.

    Fresh cache wins outright. A stale or missing cache triggers at most one fetch per process;
    if that fails, a stale table still beats no prices at all (loudly, and the caller records
    whatever it uses), and only a failure with nothing cached returns None.
    """
    global _FETCH_ERROR  # noqa: PLW0603 - one process-wide "already failed" latch, see above
    path = catalog_path()
    cached = _read_cache(path)
    if cached is not None and not cached.is_stale():
        return cached, ""
    if _FETCH_ERROR is None:
        try:
            fresh = _fetch_catalog()
        except Exception as exc:  # noqa: BLE001 - degrading is the contract; nothing may raise
            _FETCH_ERROR = f"{type(exc).__name__}: {exc}"
            logger.warning("could not fetch the OpenRouter model catalog: %s", _FETCH_ERROR)
        else:
            _write_cache(path, fresh)
            return fresh, ""
    if cached is not None:
        logger.warning(
            "serving OpenRouter prices from the stale cache at %s (refresh failed: %s)",
            path,
            _FETCH_ERROR,
        )
        return cached, ""
    return None, (
        f"the OpenRouter price catalog at {OPENROUTER_MODELS_URL} is unreachable "
        f"({_FETCH_ERROR}) and nothing is cached at {path}"
    )


def resolve_price(model: str) -> PriceResolution:
    """Look up `model`'s published price. Never raises, never blocks past one fetch.

    Args:
        model: The OpenRouter catalog id, e.g. `anthropic/claude-sonnet-4`.

    Returns:
        The price row, or an empty resolution whose `detail` says whether the catalog was
        unreachable or the model simply is not listed.
    """
    catalog, detail = _catalog()
    if catalog is None:
        return PriceResolution(detail=detail)
    price = catalog.prices.get(model)
    if price is None:
        return PriceResolution(
            detail=(
                f"it is not in the OpenRouter model catalog ({len(catalog.prices)} models, "
                f"cached at {catalog_path()}), so check the exact id at "
                "https://openrouter.ai/models"
            )
        )
    return PriceResolution(price=price)
