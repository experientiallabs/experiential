"""Tests for OpenRouter price resolution: parsing, the disk cache, and offline degradation.

Never touches the network. `wmo/conftest.py` already points `WMO_OPENROUTER_CATALOG` at a
per-test temp path and replaces `_fetch_catalog` with a refusal; tests that need a live-looking
fetch install their own stub on top.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from wmo.common.core.types import JsonObject
from wmo.common.observability.pricing import ModelPrice
from wmo.common.providers import openrouter_pricing
from wmo.common.providers.openrouter_pricing import (
    CATALOG_PATH_ENV,
    CATALOG_TTL_SECONDS,
    PriceCatalog,
    catalog_path,
    parse_catalog,
    resolve_price,
)

_SONNET = "anthropic/claude-sonnet-4"


def _payload() -> JsonObject:
    """A trimmed `GET /api/v1/models` body: prices are USD per TOKEN, as decimal strings."""
    return {
        "data": [
            {
                "id": _SONNET,
                "name": "Anthropic: Claude Sonnet 4",
                "context_length": 200000,
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                    "request": "0",
                    "image": "0.0048",
                    "input_cache_read": "0.0000003",
                    "input_cache_write": "0.00000375",
                },
            },
            {
                "id": "z-ai/glm-4.6:free",
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "openrouter/auto",
                "pricing": {"prompt": "-1", "completion": "-1"},
            },
            {
                "id": "vendor/unparsable",
                "pricing": {"prompt": "cheap", "completion": "0.00001"},
            },
            {
                "id": "vendor/no-completion-price",
                "pricing": {"prompt": "0.00001"},
            },
        ]
    }


def _cache(tmp_path: Path, prices: dict[str, ModelPrice], *, age_seconds: float = 0.0) -> Path:
    """Write a cache file and point the resolver at it."""
    path = tmp_path / "prices.json"
    catalog = PriceCatalog(
        fetched_at=time.time() - age_seconds, source="test fixture", prices=prices
    )
    path.write_text(catalog.model_dump_json(), encoding="utf-8")
    return path


def test_parse_catalog_converts_per_token_strings_to_per_mtok() -> None:
    # Exact, not approximate: the scaling runs through Decimal so a published "0.0000003" is
    # recorded as 0.3, and every artifact that persists the number reads cleanly.
    price = parse_catalog(_payload()).prices[_SONNET]
    assert price.input_per_mtok == 3.0
    assert price.output_per_mtok == 15.0
    assert price.cache_read_per_mtok == 0.3
    assert price.cache_write_per_mtok == 3.75


def test_parse_catalog_keeps_free_models_priced_at_zero() -> None:
    # $0 is a real published price, unlike "unknown"; recording it keeps a free candidate
    # comparable in the cost columns instead of dropping it from the pool entirely.
    free = parse_catalog(_payload()).prices["z-ai/glm-4.6:free"]
    assert free.input_per_mtok == 0.0
    assert free.output_per_mtok == 0.0


def test_parse_catalog_drops_rows_it_cannot_price() -> None:
    # "-1" marks the variable-priced auto router; the other two are malformed or partial. None
    # can be recorded as a number, and a wrong number is worse than no number.
    prices = parse_catalog(_payload()).prices
    assert "openrouter/auto" not in prices
    assert "vendor/unparsable" not in prices
    assert "vendor/no-completion-price" not in prices


def test_parse_catalog_rejects_a_payload_with_no_priced_model() -> None:
    # A shape change must not cache an empty table, which would read as "model not listed".
    with pytest.raises(ValueError, match="no priced models"):
        parse_catalog({"data": [{"id": "openrouter/auto", "pricing": {"prompt": "-1"}}]})


def test_resolve_price_reads_the_cache_without_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The conftest fetch stub raises, so reaching the network here would fail the test.
    monkeypatch.setenv(
        CATALOG_PATH_ENV,
        str(_cache(tmp_path, {_SONNET: ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)})),
    )
    resolution = resolve_price(_SONNET)
    assert resolution.price is not None
    assert resolution.price.input_per_mtok == 3.0
    assert resolution.detail == ""


def test_resolve_price_fetches_once_and_caches_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "prices.json"
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))
    calls: list[int] = []

    def _fetch() -> PriceCatalog:
        calls.append(1)
        return parse_catalog(_payload())

    monkeypatch.setattr(openrouter_pricing, "_fetch_catalog", _fetch)

    first = resolve_price(_SONNET)
    second = resolve_price("z-ai/glm-4.6:free")

    assert first.price is not None
    assert second.price is not None
    assert len(calls) == 1  # the second lookup read the cache the first one wrote
    assert path.is_file()
    assert PriceCatalog.model_validate_json(path.read_text(encoding="utf-8")).prices[_SONNET]


def test_resolve_price_offline_says_so_instead_of_raising() -> None:
    # The conftest fixture is the offline machine: no cache file, and a fetch that refuses.
    resolution = resolve_price(_SONNET)
    assert resolution.price is None
    assert "unreachable" in resolution.detail
    assert "nothing is cached" in resolution.detail


def test_one_failed_fetch_is_remembered_for_the_whole_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A twenty-entry pool on an offline machine must pay one timeout, not twenty.
    attempts: list[int] = []

    def _fetch() -> PriceCatalog:
        attempts.append(1)
        raise TimeoutError("connect timeout")

    monkeypatch.setattr(openrouter_pricing, "_fetch_catalog", _fetch)

    assert resolve_price(_SONNET).price is None
    assert resolve_price("z-ai/glm-4.6:free").price is None
    assert len(attempts) == 1


def test_a_stale_cache_is_refreshed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        CATALOG_PATH_ENV,
        str(
            _cache(
                tmp_path,
                {_SONNET: ModelPrice(input_per_mtok=99.0, output_per_mtok=99.0)},
                age_seconds=CATALOG_TTL_SECONDS + 60,
            )
        ),
    )
    monkeypatch.setattr(openrouter_pricing, "_fetch_catalog", lambda: parse_catalog(_payload()))

    resolution = resolve_price(_SONNET)

    assert resolution.price is not None
    assert resolution.price.input_per_mtok == pytest.approx(3.0)


def test_a_stale_cache_still_answers_when_the_refresh_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Yesterday's published price beats no price at all, and whatever is used gets recorded on
    # the pool entry anyway, so the artifact still says exactly what it was priced with.
    monkeypatch.setenv(
        CATALOG_PATH_ENV,
        str(
            _cache(
                tmp_path,
                {_SONNET: ModelPrice(input_per_mtok=2.5, output_per_mtok=12.5)},
                age_seconds=CATALOG_TTL_SECONDS + 60,
            )
        ),
    )
    resolution = resolve_price(_SONNET)
    assert resolution.price is not None
    assert resolution.price.input_per_mtok == 2.5


def test_an_unlisted_model_is_reported_as_unlisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        CATALOG_PATH_ENV,
        str(_cache(tmp_path, {_SONNET: ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)})),
    )
    resolution = resolve_price("vendor/typo")
    assert resolution.price is None
    assert "not in the OpenRouter model catalog (1 models" in resolution.detail


def test_a_corrupt_cache_is_ignored_rather_than_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "prices.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))
    monkeypatch.setattr(openrouter_pricing, "_fetch_catalog", lambda: parse_catalog(_payload()))

    resolution = resolve_price(_SONNET)

    assert resolution.price is not None


def test_catalog_path_defaults_under_wmo_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    monkeypatch.setenv("WMO_HOME", "/tmp/wmo-home-fixture")
    assert catalog_path() == Path("/tmp/wmo-home-fixture/openrouter-prices.json")
