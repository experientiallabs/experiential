"""Tests for the candidate model pool (schema, loading, pricing, provider construction)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.providers.azure_openai import AzureOpenAIProvider
from wmo.providers.base import ProviderConfig, ProviderKind, TokenUsage
from wmo.providers.openrouter import OPENROUTER_API_KEY_ENV, OpenRouterProvider
from wmo.providers.openrouter_pricing import CATALOG_PATH_ENV, PriceCatalog
from wmo.providers.pool import DEFAULT_POOL_PATH, PoolEntry, load_pool, pool_provider
from wmo.providers.registry import get_provider
from wmo.tracking.pricing import ModelPrice

_POOL_TOML = """
[[model]]
name = "deepseek-v4-pro"
kind = "azure"
model = "DeepSeek-V4-Pro"
deployment = "DeepSeek-V4-Pro"
endpoint = "https://silen-resource.services.ai.azure.com"
api_version = "2024-10-21"
api_key_env = "AZURE_SILEN_RESOURCE_API_KEY"
tier = "open"
input_per_mtok = 1.2
output_per_mtok = 4.8
cached_input_per_mtok = 0.12
cache_write_per_mtok = 1.5

[[model]]
name = "gpt-5.5"
kind = "azure"
model = "gpt-5.5"
deployment = "gpt-5.5"
endpoint = "https://google-sheets.openai.azure.com"
api_version = "2024-10-21"
api_key_env = "AZURE_GOOGLE_SHEETS_API_KEY"

[[model]]
name = "fable"
kind = "anthropic"
model = "claude-fable-5"
"""


def _write_pool(tmp_path: Path, text: str = _POOL_TOML) -> Path:
    path = tmp_path / "pool.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_pool_parses_entries(tmp_path: Path) -> None:
    pool = load_pool(_write_pool(tmp_path))
    assert [m.name for m in pool.models] == ["deepseek-v4-pro", "gpt-5.5", "fable"]
    deepseek = pool.entry("deepseek-v4-pro")
    assert deepseek.kind is ProviderKind.AZURE_OPENAI
    assert deepseek.tier == "open"
    assert deepseek.api_key_env == "AZURE_SILEN_RESOURCE_API_KEY"
    assert deepseek.cached_input_per_mtok == 0.12
    assert deepseek.cache_write_per_mtok == 1.5
    # Entries default to the frontier tier (the D-REPORT ModelRef vocabulary).
    assert pool.entry("fable").tier == "frontier"


def test_load_pool_missing_file_says_what_to_create(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    with pytest.raises(FileNotFoundError, match=r"\[\[model\]\]"):
        load_pool(missing)
    assert DEFAULT_POOL_PATH == Path(".wmo/pool.toml")


def test_duplicate_names_rejected(tmp_path: Path) -> None:
    extra = '\n[[model]]\nname = "fable"\nkind = "anthropic"\nmodel = "claude-fable-5"\n'
    dupe = _POOL_TOML + extra
    with pytest.raises(ValueError, match="fable"):
        load_pool(_write_pool(tmp_path, dupe))


def test_empty_pool_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_pool(_write_pool(tmp_path, "# no models\n"))


def test_unknown_model_requires_explicit_price() -> None:
    with pytest.raises(ValueError, match="input_per_mtok"):
        PoolEntry(name="glm", kind=ProviderKind.OPENAI, model="FW-GLM-5.2")


def test_price_must_be_set_as_a_pair() -> None:
    with pytest.raises(ValueError, match="both"):
        PoolEntry(name="glm", kind=ProviderKind.OPENAI, model="FW-GLM-5.2", input_per_mtok=1.0)


def test_price_falls_back_to_builtin_table() -> None:
    entry = PoolEntry(name="fable", kind=ProviderKind.ANTHROPIC, model="claude-fable-5")
    price = entry.price()
    assert price.input_per_mtok == 10.0
    assert price.output_per_mtok == 50.0


def test_price_override_wins_over_table() -> None:
    entry = PoolEntry(
        name="fable-discount",
        kind=ProviderKind.ANTHROPIC,
        model="claude-fable-5",
        input_per_mtok=1.0,
        output_per_mtok=2.0,
    )
    assert entry.price().input_per_mtok == 1.0


def test_provider_config_maps_backend_knobs() -> None:
    entry = PoolEntry(
        name="gpt",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        endpoint="https://google-sheets.openai.azure.com",
        api_version="2024-10-21",
    )
    config = entry.provider_config()
    assert config.kind is ProviderKind.AZURE_OPENAI
    assert config.model == "gpt-5.5"
    assert config.deployment == "gpt-5.5"
    assert config.endpoint == "https://google-sheets.openai.azure.com"
    assert config.api_version == "2024-10-21"


def test_pool_provider_requires_named_env_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AZURE_SILEN_RESOURCE_API_KEY", raising=False)
    pool = load_pool(_write_pool(tmp_path))
    with pytest.raises(ValueError, match="AZURE_SILEN_RESOURCE_API_KEY"):
        pool_provider(pool.entry("deepseek-v4-pro"))


def test_pool_provider_passes_explicit_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_SILEN_RESOURCE_API_KEY", "sk-pool-test")
    pool = load_pool(_write_pool(tmp_path))
    provider = pool_provider(pool.entry("deepseek-v4-pro"))
    assert isinstance(provider, AzureOpenAIProvider)
    # The explicit key is the trusted channel: the client authenticates with the entry's own
    # account key even though the endpoint differs from AZURE_OPENAI_ENDPOINT (which would
    # otherwise downgrade auth to the WMO_ENDPOINT_API_KEY placeholder).
    client = provider._get_client()  # noqa: SLF001 - asserting the wired credential
    assert client.api_key == "sk-pool-test"


def test_pool_entry_unknown_name_lists_available(tmp_path: Path) -> None:
    pool = load_pool(_write_pool(tmp_path))
    with pytest.raises(KeyError, match="deepseek-v4-pro"):
        pool.entry("not-a-model")


def test_get_provider_rejects_api_key_for_bedrock() -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8")
    with pytest.raises(ValueError, match="[Bb]edrock"):
        get_provider(config, api_key="sk-nope")


def test_cost_usd_is_cache_adjusted() -> None:
    entry = PoolEntry(
        name="cached",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        input_per_mtok=10.0,
        output_per_mtok=20.0,
        cached_input_per_mtok=1.0,
    )
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=400_000)
    # 600k fresh @ $10/M + 400k cached @ $1/M = $6.40 - never the $10 list price.
    assert entry.cost_usd(usage) == pytest.approx(6.4)


def test_cost_usd_without_cache_price_bills_cached_tokens_at_full_rate() -> None:
    entry = PoolEntry(
        name="no-cache-price",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        input_per_mtok=10.0,
        output_per_mtok=20.0,
    )
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=400_000)
    assert entry.cost_usd(usage) == pytest.approx(10.0)  # honest fallback, never free


def test_cost_usd_bills_cache_writes_at_entry_override() -> None:
    entry = PoolEntry(
        name="write-priced",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        input_per_mtok=10.0,
        output_per_mtok=20.0,
        cache_write_per_mtok=12.5,
    )
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cache_write_input_tokens=400_000)
    # 600k fresh @ $10/M + 400k written @ $12.5/M = 6.0 + 5.0 = $11.00.
    assert entry.cost_usd(usage) == pytest.approx(11.0)


def test_cost_usd_falls_back_to_builtin_cache_tiers() -> None:
    # An entry with NO explicit prices uses the built-in table, including its cache tiers
    # (fable: reads 0.1x -> $1/M, writes 1.25x -> $12.5/M on a $10/M input rate).
    entry = PoolEntry(name="fable", kind=ProviderKind.ANTHROPIC, model="claude-fable-5")
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=300_000,
        cache_write_input_tokens=200_000,
    )
    # 500k fresh @ $10 + 300k read @ $1 + 200k write @ $12.5 = 5.0 + 0.3 + 2.5 = $7.80.
    assert entry.cost_usd(usage) == pytest.approx(7.8)


def test_bedrock_entry_pins_region() -> None:
    entry = PoolEntry(
        name="opus-4-8",
        kind=ProviderKind.BEDROCK,
        model="us.anthropic.claude-opus-4-8",
        region="us-east-1",
    )
    assert entry.provider_config().region == "us-east-1"


def test_unknown_pool_keys_fail_at_load() -> None:
    # A typo like api_key_evn must fail at load, not surface as a 401 at request time.
    with pytest.raises(ValidationError, match="api_key_evn"):
        PoolEntry.model_validate(
            {
                "name": "typo",
                "kind": "anthropic",
                "model": "claude-haiku-4-5",
                "api_key_evn": "SOME_KEY",
            }
        )


def test_azure_entry_requires_deployment() -> None:
    with pytest.raises(ValidationError, match="deployment"):
        PoolEntry(
            name="no-deploy",
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
        )


# --- OpenRouter entries: priced from the published catalog, not by hand -----------------------

_OPENROUTER_POOL = """
[[model]]
name = "or-sonnet"
kind = "openrouter"
model = "anthropic/claude-sonnet-4"

[[model]]
name = "or-glm-free"
kind = "openrouter"
model = "z-ai/glm-4.6:free"
tier = "open"
"""


_SONNET_PRICE = ModelPrice(
    input_per_mtok=3.0,
    output_per_mtok=15.0,
    cache_read_per_mtok=0.3,
    cache_write_per_mtok=3.75,
)
_FREE_PRICE = ModelPrice(input_per_mtok=0.0, output_per_mtok=0.0)


def _catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prices: dict[str, ModelPrice] | None = None,
) -> Path:
    """Point price resolution at a fixture catalog on disk (the suite never fetches)."""
    path = tmp_path / "openrouter-prices.json"
    catalog = PriceCatalog(
        fetched_at=time.time(),
        source="test fixture",
        prices=prices
        if prices is not None
        else {"anthropic/claude-sonnet-4": _SONNET_PRICE, "z-ai/glm-4.6:free": _FREE_PRICE},
    )
    path.write_text(catalog.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv(CATALOG_PATH_ENV, str(path))
    return path


def test_openrouter_entry_needs_only_a_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The launch promise: a pool entry is a name, a kind, and a model id. Both tiers, including
    # the cache rates, come from OpenRouter's published catalog.
    _catalog(tmp_path, monkeypatch)
    pool = load_pool(_write_pool(tmp_path, _OPENROUTER_POOL))

    entry = pool.entry("or-sonnet")
    assert entry.price().input_per_mtok == 3.0
    assert entry.price().output_per_mtok == 15.0
    assert entry.cached_input_per_mtok == pytest.approx(0.3)
    assert entry.cache_write_per_mtok == pytest.approx(3.75)
    # 500k fresh @ $3 + 500k cached @ $0.30 = 1.5 + 0.15 = $1.65.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=500_000)
    assert entry.cost_usd(usage) == pytest.approx(1.65)
    assert pool.entry("or-glm-free").price().input_per_mtok == 0.0


def test_openrouter_entry_offline_falls_back_to_the_explicit_price_error(tmp_path: Path) -> None:
    # No cache and no network (the conftest fetch stub refuses): the entry must fail with the
    # ordinary "declare the prices" instruction, and say WHY the automatic route did not apply.
    with pytest.raises(ValidationError) as excinfo:
        load_pool(_write_pool(tmp_path, _OPENROUTER_POOL))
    message = str(excinfo.value)
    assert "OpenRouter price catalog" in message
    assert "unreachable" in message
    assert "add input_per_mtok and output_per_mtok" in message


def test_openrouter_entry_with_explicit_prices_keeps_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A negotiated rate is the operator's, and the catalog is never consulted for it (the
    # fixture below prices the same model very differently, and must lose).
    _catalog(tmp_path, monkeypatch)
    entry = PoolEntry(
        name="or-sonnet",
        kind=ProviderKind.OPENROUTER,
        model="anthropic/claude-sonnet-4",
        input_per_mtok=1.0,
        output_per_mtok=2.0,
    )
    assert entry.price().input_per_mtok == 1.0
    assert entry.cached_input_per_mtok is None


def test_a_priced_entry_is_never_repriced_by_a_later_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The persistence property behind a fitted policy: the resolved numbers live ON the entry,
    # and RoutingPolicy/OutcomeMatrix serialize entries verbatim. Re-validating a persisted
    # entry against a catalog that has since doubled its price must not move it.
    _catalog(tmp_path, monkeypatch)
    fitted = load_pool(_write_pool(tmp_path, _OPENROUTER_POOL)).entry("or-sonnet")
    snapshot = fitted.model_dump_json()

    _catalog(
        tmp_path,
        monkeypatch,
        {"anthropic/claude-sonnet-4": ModelPrice(input_per_mtok=6.0, output_per_mtok=30.0)},
    )
    reloaded = PoolEntry.model_validate_json(snapshot)

    assert reloaded.input_per_mtok == 3.0
    assert reloaded.price().output_per_mtok == 15.0


def test_openrouter_pool_entry_resolves_to_the_openrouter_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENROUTER_ACCOUNT_B_KEY", "sk-or-account-b")
    pool = load_pool(
        _write_pool(tmp_path, _OPENROUTER_POOL + '\napi_key_env = "OPENROUTER_ACCOUNT_B_KEY"\n')
    )

    provider = pool_provider(pool.entry("or-glm-free"))

    assert isinstance(provider, OpenRouterProvider)
    assert provider.config.kind is ProviderKind.OPENROUTER
    assert provider.config.model == "z-ai/glm-4.6:free"
    assert provider._get_client().api_key == "sk-or-account-b"  # noqa: SLF001 - asserting wiring


def test_openrouter_provider_falls_back_to_the_shared_account_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No api_key_env on the entry: the single-key launch path, straight from the environment.
    _catalog(tmp_path, monkeypatch)
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-or-shared")
    pool = load_pool(_write_pool(tmp_path, _OPENROUTER_POOL))

    provider = pool_provider(pool.entry("or-sonnet"))

    assert isinstance(provider, OpenRouterProvider)
    assert provider._get_client().api_key == "sk-or-shared"  # noqa: SLF001 - asserting wiring
