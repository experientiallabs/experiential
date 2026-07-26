"""Tests for the candidate model pool (schema, loading, pricing, provider construction)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.providers.azure_openai import AzureOpenAIProvider
from wmo.providers.base import ProviderConfig, ProviderKind, TokenUsage
from wmo.providers.pool import (
    DEFAULT_POOL_PATH,
    PoolEntry,
    load_pool,
    pool_api_key,
    pool_provider,
    prepare_pool_provider,
    static_requirements,
)
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


def test_pool_provider_names_the_entry_when_a_backend_refuses_its_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A backend that refuses to be built says nothing about WHICH candidate did it, and callers
    # loop over a whole pool (`wmo optimize route sweep` constructs all of them as a pre-flight),
    # so the entry name and kind have to survive into the message an operator reads.
    monkeypatch.setenv("WMO_POOL_TEST_KEY", "sk-present")
    entry = PoolEntry(
        name="student",
        kind=ProviderKind.TINKER,
        model="Qwen/Qwen3-8B",
        api_key_env="WMO_POOL_TEST_KEY",  # set: this is a backend refusal, not a missing key
        input_per_mtok=0.1,
        output_per_mtok=0.2,
    )
    with pytest.raises(ValueError, match=r"pool model 'student' \(kind=tinker\)") as failure:
        pool_provider(entry)
    # The backend's own advice is preserved, not replaced by the identification.
    assert "TINKER_API_KEY" in str(failure.value)


def test_static_requirements_pass_a_complete_entry() -> None:
    # Nothing is required of the kinds whose only prerequisite (a credential, a region) lives in
    # the environment rather than the entry, and a complete azure entry is complete.
    assert (
        static_requirements(
            PoolEntry(name="fable", kind=ProviderKind.ANTHROPIC, model="claude-fable-5")
        )
        == []
    )
    complete_azure = PoolEntry(
        name="gpt",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        api_version="2024-10-21",
    )
    assert static_requirements(complete_azure) == []


def test_static_requirements_name_the_azure_api_version() -> None:
    # `AzureOpenAIProvider._get_client` refuses without an api-version, and that check runs inside
    # the FIRST call: a swept candidate would abort mid-run. Knowable from the entry alone.
    entry = PoolEntry(
        name="gpt",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
    )
    assert [problem for problem in static_requirements(entry) if "api_version" in problem]


def test_static_requirements_reject_a_tinker_weights_path() -> None:
    # A `tinker://` path can never render a prompt from a pool entry: the renderer and tokenizer
    # resolve from `ProviderConfig.model_type`, and a pool entry has no field that fills it.
    entry = PoolEntry(
        name="student",
        kind=ProviderKind.TINKER,
        model="tinker://abc/sampler_weights/42",
        input_per_mtok=0.1,
        output_per_mtok=0.2,
    )
    problems = static_requirements(entry)
    assert len(problems) == 1
    assert "model_type" not in problems[0]  # worded for the pool file, not the provider config
    assert "base model" in problems[0]


def test_prepare_pool_provider_forces_the_lazy_client(monkeypatch: pytest.MonkeyPatch) -> None:
    # The point of the seam: an azure entry with no endpoint (and no AZURE_OPENAI_ENDPOINT)
    # CONSTRUCTS fine, because `__init__` only stores the config, and fails only when the client is
    # built. `prepare_pool_provider` builds it, without a request, and names the entry.
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    entry = PoolEntry(
        name="gpt-azure",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        api_version="2024-10-21",
    )
    assert isinstance(pool_provider(entry), AzureOpenAIProvider)  # construction alone says nothing
    with pytest.raises(ValueError, match=r"pool model 'gpt-azure' \(kind=azure\)") as failure:
        prepare_pool_provider(entry)
    assert "AZURE_OPENAI_ENDPOINT" in str(failure.value)  # the backend's own advice survives


def test_prepare_pool_provider_returns_a_usable_entrys_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WMO_POOL_TEST_KEY", "sk-present")
    entry = PoolEntry(
        name="gpt-azure",
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="gpt-5.5",
        endpoint="https://example.openai.azure.com",
        api_version="2024-10-21",
        api_key_env="WMO_POOL_TEST_KEY",
    )
    provider = prepare_pool_provider(entry)
    assert isinstance(provider, AzureOpenAIProvider)


def test_pool_api_key_checks_credentials_without_building_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The seam a caller about to spend on a whole pool uses to check every candidate up front:
    # same verdict as `pool_provider`, no provider constructed and no network client touched.
    pool = load_pool(_write_pool(tmp_path))
    monkeypatch.delenv("AZURE_SILEN_RESOURCE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AZURE_SILEN_RESOURCE_API_KEY"):
        pool_api_key(pool.entry("deepseek-v4-pro"))
    monkeypatch.setenv("AZURE_SILEN_RESOURCE_API_KEY", "sk-pool-test")
    assert pool_api_key(pool.entry("deepseek-v4-pro")) == "sk-pool-test"
    # An entry with no api_key_env uses the backend's default credentials, and says so with None.
    assert pool_api_key(pool.entry("fable")) is None


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


def test_bedrock_entry_rejects_api_key_env_at_load() -> None:
    # `BedrockProvider.__init__` refuses an explicit key, and providers are built lazily per
    # eval cell: caught at load this is a config typo, caught at the first cell it aborts a
    # paid-for sweep. Same boundary as the azure `deployment` rule above.
    with pytest.raises(ValidationError, match="api_key_env"):
        PoolEntry(
            name="claude-bedrock",
            kind=ProviderKind.BEDROCK,
            model="us.anthropic.claude-opus-4-8",
            api_key_env="AWS_SOMETHING",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
        )
