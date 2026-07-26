"""Tests for the candidate model pool (schema, loading, pricing, provider construction)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.providers.azure_openai import AzureOpenAIProvider
from wmo.providers.base import ProviderConfig, ProviderKind, TokenUsage
from wmo.providers.pool import DEFAULT_POOL_PATH, PoolEntry, load_pool, pool_provider
from wmo.providers.registry import get_provider

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
