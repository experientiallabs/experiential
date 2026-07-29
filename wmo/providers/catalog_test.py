"""Tests for per-kind model enumeration.

Never touches the network: `wmo/conftest.py` already cuts every test off from the live OpenRouter
catalog, and the tests that need one seed a cache file at `WMO_OPENROUTER_CATALOG`.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

from wmo.providers.base import ProviderKind
from wmo.providers.catalog import (
    CatalogModel,
    CatalogSource,
    ProviderCatalog,
    endpoint_catalog,
    list_provider_models,
)
from wmo.providers.openrouter_pricing import CATALOG_PATH_ENV, PriceCatalog
from wmo.tracking.pricing import ModelPrice

_FAKE_PRICES = {
    "anthropic/claude-sonnet-4.5": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0),
    "deepseek/deepseek-v3.2": ModelPrice(input_per_mtok=0.27, output_per_mtok=0.4),
    "qwen/qwen3-coder": ModelPrice(input_per_mtok=0.3, output_per_mtok=1.2),
}


@pytest.fixture
def openrouter_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Seed a fresh OpenRouter price cache so enumeration reads it instead of the network."""
    path = tmp_path / "openrouter-prices.json"
    catalog = PriceCatalog(fetched_at=time.time(), source="test fixture", prices=_FAKE_PRICES)
    path.write_text(catalog.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))
    return path


def test_openrouter_enumerates_its_published_catalog(openrouter_cache: Path) -> None:
    # The launch promise is "pass in an OpenRouter key": the picker must offer everything
    # OpenRouter fronts, priced, without a second fetch or a shipped list of names.
    catalog = list_provider_models(ProviderKind.OPENROUTER)
    assert catalog.source is CatalogSource.PUBLISHED
    assert [model.id for model in catalog.models] == sorted(_FAKE_PRICES)
    sonnet = catalog.find("anthropic/claude-sonnet-4.5")
    assert sonnet is not None
    assert sonnet.price == _FAKE_PRICES["anthropic/claude-sonnet-4.5"]


def test_openrouter_degrades_to_an_empty_catalog_with_the_reason() -> None:
    # Offline with nothing cached, enumeration must not raise: the picker falls back to a typed
    # id and shows why there is no list. (conftest already points the cache at a missing path.)
    catalog = list_provider_models(ProviderKind.OPENROUTER)
    assert catalog.models == []
    assert "unreachable" in catalog.detail


def test_bedrock_rows_carry_the_runtime_id_and_the_canonical_type() -> None:
    # A pool entry's `model` is the wire id; capability and price lookups need the model type
    # too, and only Bedrock's ids differ from it.
    catalog = list_provider_models(ProviderKind.BEDROCK)
    assert catalog.source is CatalogSource.BUILT_IN
    opus = catalog.find("claude-opus-4-8")
    assert opus is not None
    assert opus.id == "us.anthropic.claude-opus-4-8"
    assert opus.model_type == "claude-opus-4-8"
    assert opus.price is not None


def test_azure_rows_do_not_repeat_the_id_as_a_model_type() -> None:
    # model_type is only carried when it adds something; repeating the id would write a
    # redundant field into every Azure pool entry.
    catalog = list_provider_models(ProviderKind.AZURE_OPENAI)
    assert all(model.model_type is None for model in catalog.models)


def test_tinker_has_nothing_to_enumerate_and_says_what_to_type() -> None:
    # A tinker candidate names a BASE model so the renderer and tokenizer resolve; there is no
    # catalog of those, and an empty list with no explanation would read as a bug.
    catalog = list_provider_models(ProviderKind.TINKER)
    assert catalog.source is CatalogSource.NONE
    assert catalog.models == []
    assert "base model" in catalog.detail


@pytest.mark.parametrize("kind", list(ProviderKind))
def test_every_kind_enumerates_without_raising(kind: ProviderKind) -> None:
    # The picker calls this for whichever backend the user chose; a kind added later must not
    # fall off the end of the match and return None.
    catalog = list_provider_models(kind)
    assert catalog.kind is kind


def test_search_matches_a_substring_of_a_vendor_prefixed_id(openrouter_cache: Path) -> None:
    # 338 models are not a scrollable list, and every id is `vendor/model`, so a prefix match
    # would make "sonnet" find nothing.
    catalog = list_provider_models(ProviderKind.OPENROUTER)
    assert [model.id for model in catalog.search("SONNET")] == ["anthropic/claude-sonnet-4.5"]
    assert len(catalog.search("")) == len(catalog.models)
    assert catalog.search("no-such-model") == []


def test_search_also_matches_the_canonical_model_type() -> None:
    # Bedrock's ids carry routing prefixes, so searching for what the docs call the model has
    # to reach `us.anthropic.claude-haiku-4-5-...`.
    catalog = list_provider_models(ProviderKind.BEDROCK)
    assert any(model.model_type == "claude-haiku-4-5" for model in catalog.search("haiku"))


def test_label_shows_a_price_only_when_one_is_known() -> None:
    priced = CatalogModel(id="x/y", price=ModelPrice(input_per_mtok=1.5, output_per_mtok=6.0))
    assert priced.label() == "x/y  $1.5 / $6 per Mtok"
    assert CatalogModel(id="x/y").label() == "x/y"


def test_find_is_case_insensitive_on_both_identities() -> None:
    # Azure deployments are routinely created with vendor casing, and a user retyping an id
    # from a dashboard must not get "no match" for a model that is right there.
    catalog = ProviderCatalog(
        kind=ProviderKind.BEDROCK,
        source=CatalogSource.BUILT_IN,
        models=[CatalogModel(id="us.anthropic.Claude-Opus", model_type="Claude-Opus")],
    )
    assert catalog.find("us.anthropic.claude-opus") is not None
    assert catalog.find("CLAUDE-OPUS") is not None
    assert catalog.find("opus") is None


class _FakeResponse:
    """The two calls `endpoint_catalog` makes on an httpx response, canned."""

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def test_endpoint_catalog_lists_what_the_server_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_get(url: str, timeout: float) -> _FakeResponse:
        seen.append(url)
        return _FakeResponse(
            {"object": "list", "data": [{"id": "qwen3:4b", "owned_by": "library"}]}
        )

    monkeypatch.setattr("wmo.providers.catalog.httpx.get", fake_get)
    catalog = endpoint_catalog("http://localhost:11434/v1/")

    assert seen == ["http://localhost:11434/v1/models"]
    assert catalog.kind is ProviderKind.OPENAI
    assert catalog.source is CatalogSource.PUBLISHED
    assert [model.id for model in catalog.models] == ["qwen3:4b"]
    # No price on purpose: a self-hosted candidate is priced by the operator (default 0).
    assert catalog.models[0].price is None


def test_endpoint_catalog_answers_an_unreachable_server_as_an_empty_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, timeout: float) -> _FakeResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("wmo.providers.catalog.httpx.get", fake_get)
    catalog = endpoint_catalog("http://localhost:9")

    assert catalog.source is CatalogSource.NONE
    assert catalog.models == []
    assert "http://localhost:9/models" in catalog.detail
    assert "type the model id" in catalog.detail


def test_endpoint_catalog_answers_a_nonconforming_body_as_an_empty_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "wmo.providers.catalog.httpx.get",
        lambda url, timeout: _FakeResponse({"unexpected": "shape"}),
    )
    catalog = endpoint_catalog("http://localhost:11434/v1")

    assert catalog.source is CatalogSource.NONE
    assert catalog.models == []
