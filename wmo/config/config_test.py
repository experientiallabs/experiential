"""Tests for config persistence and the artifact layout."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.config.config import (
    _DEFAULT_TRAIN_SPLIT,
    PROVIDER_ENV_VARS,
    ArtifactPaths,
    HarnessConfig,
    load_config,
    save_config,
)
from wmo.providers.base import EmbedderKind, ProviderConfig, ProviderKind
from wmo.simulation.model.build import DEFAULT_TRAIN_SPLIT


def test_gepa_budget_default_is_modest() -> None:
    # 10 iterations ~= 700 metric calls with the capped valset; 50 was ~3.4k calls (hours).
    assert HarnessConfig().gepa_budget == 10


@pytest.mark.parametrize("kind", list(ProviderKind))
def test_every_provider_kind_declares_its_credential_env_vars(kind: ProviderKind) -> None:
    # A missing kind is silent: the CLI's credential prompt, the picker's "creds set"
    # annotation, and the `providers verify` failure hint all degrade to "no credentials
    # known" rather than erroring, so only this exhaustiveness check catches the gap.
    assert PROVIDER_ENV_VARS.get(kind), f"{kind.value} has no PROVIDER_ENV_VARS entry"


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    config = HarnessConfig(
        providers=[
            ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8"),
            ProviderConfig(
                kind=ProviderKind.AZURE_OPENAI,
                model="gpt-5.5",
                embed_model="text-embedding-3-large",
                embed_dim=1024,
                endpoint="https://example.openai.azure.com",
                deployment="gpt-55",
                api_version="2024-02-01",
            ),
        ],
        serve_provider=ProviderKind.ANTHROPIC,
        embed_provider=EmbedderKind.AZURE_OPENAI,
        embed_dim=1024,
        top_k=8,
        train_split=0.7,
        gepa_budget=120,
        trace_adapter="otel-genai",
    )

    save_config(config, root=tmp_path / ".wmo")
    loaded = load_config(root=tmp_path / ".wmo")

    assert loaded == config


def test_save_creates_artifact_dir(tmp_path: Path) -> None:
    root = tmp_path / ".wmo"
    assert not root.exists()
    save_config(HarnessConfig(), root=root)
    assert (root / "config.toml").is_file()


def test_load_without_artifact_dir_raises_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="wmo build"):
        load_config(root=tmp_path / ".wmo")


def test_load_with_dir_but_no_config_raises_friendly_error(tmp_path: Path) -> None:
    root = tmp_path / ".wmo"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="config"):
        load_config(root=root)


def test_defaults_round_trip(tmp_path: Path) -> None:
    save_config(HarnessConfig(), root=tmp_path / ".wmo")
    assert load_config(root=tmp_path / ".wmo") == HarnessConfig()


def test_agentic_fields_default_off_so_old_configs_load(tmp_path: Path) -> None:
    # A config.toml written before the agentic-mode fields existed must still validate, with the
    # new behavior OFF (existing model artifacts serve unchanged).
    config = HarnessConfig()
    assert config.knowledge is False
    assert config.reasoning is False
    assert config.grounder == "none"
    assert config.confidence is False
    assert config.confidence_why is False
    root = tmp_path / ".wmo"
    save_config(config, root=root)
    raw = (root / "config.toml").read_text(encoding="utf-8")
    stripped = "\n".join(
        line
        for line in raw.splitlines()
        if not line.startswith(("knowledge", "reasoning", "grounder", "confidence"))
    )
    (root / "config.toml").write_text(stripped, encoding="utf-8")
    assert load_config(root=root) == config


def test_artifact_paths_knowledge_dir(tmp_path: Path) -> None:
    paths = ArtifactPaths(tmp_path / ".wmo")
    assert paths.knowledge == tmp_path / ".wmo" / "knowledge"


def test_load_corrupt_toml_raises_friendly_error(tmp_path: Path) -> None:
    root = tmp_path / ".wmo"
    root.mkdir()
    (root / "config.toml").write_text("this is = = not valid", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_config(root=root)


def test_load_schema_invalid_toml_raises_friendly_error(tmp_path: Path) -> None:
    root = tmp_path / ".wmo"
    root.mkdir()
    (root / "config.toml").write_text('top_k = "not-an-int"', encoding="utf-8")
    with pytest.raises(ValueError, match="config schema"):
        load_config(root=root)


def test_load_unreadable_config_raises_friendly_error(tmp_path: Path) -> None:
    # A root-owned or chmod-000 artifact used to escape as a bare PermissionError with no path
    # and no next step; it is the same user problem as a corrupt one.
    root = tmp_path / ".wmo"
    root.mkdir()
    config = root / "config.toml"
    config.write_text("top_k = 3", encoding="utf-8")
    config.chmod(0o000)
    try:
        with pytest.raises(ValueError, match="could not be read"):
            load_config(root=root)
    finally:
        config.chmod(0o644)


def test_load_non_utf8_config_raises_friendly_error(tmp_path: Path) -> None:
    # `tomllib.load` decodes before it parses, and UnicodeDecodeError is a ValueError rather than
    # a TOMLDecodeError, so a non-UTF-8 artifact used to surface a bare codec error naming no file.
    root = tmp_path / ".wmo"
    root.mkdir()
    (root / "config.toml").write_bytes(b'system_prompt = "\x93hi\x94"\n')
    with pytest.raises(ValueError) as excinfo:
        load_config(root=root)
    message = str(excinfo.value)
    assert "not valid TOML" in message
    assert str(root / "config.toml") in message


def test_save_does_not_leave_temp_file(tmp_path: Path) -> None:
    root = tmp_path / ".wmo"
    save_config(HarnessConfig(), root=root)
    assert list(root.glob("*.tmp")) == []


def test_serve_provider_config_resolves_by_kind() -> None:
    config = HarnessConfig(
        providers=[
            ProviderConfig(kind=ProviderKind.BEDROCK, model="opus"),
            ProviderConfig(kind=ProviderKind.OPENAI, model="gpt"),
        ],
        serve_provider=ProviderKind.BEDROCK,
    )
    assert config.serve_provider_config().model == "opus"
    assert config.provider_config(ProviderKind.OPENAI).model == "gpt"


def test_provider_config_missing_kind_raises() -> None:
    config = HarnessConfig(providers=[], serve_provider=ProviderKind.BEDROCK)
    with pytest.raises(ValueError, match="no provider config for bedrock"):
        config.serve_provider_config()


def test_embed_provider_config_stamps_embed_dim() -> None:
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.OPENAI, model="gpt", embed_model="te3")],
        embed_provider=EmbedderKind.OPENAI,
        embed_dim=256,
    )
    embed_cfg = config.embed_provider_config()
    assert embed_cfg.embed_model == "te3"
    assert embed_cfg.embed_dim == 256  # stamped from HarnessConfig.embed_dim


def test_for_build_reuses_serve_config_when_embed_backend_matches() -> None:
    # Bedrock serves AND embeds -> one provider config carrying both model + embed_model.
    config = HarnessConfig.for_build(
        serve_provider=ProviderKind.BEDROCK,
        serve_model="us.anthropic.claude-opus-4-8",
        region="us-east-1",
        embed_provider=EmbedderKind.BEDROCK,
        embed_model="amazon.titan-embed-text-v2:0",
        embed_dim=256,
        gepa_budget=10,
    )
    assert len(config.providers) == 1
    only = config.providers[0]
    assert only.model == "us.anthropic.claude-opus-4-8"
    assert only.embed_model == "amazon.titan-embed-text-v2:0"
    assert config.embed_dim == 256


def test_for_build_adds_separate_config_for_cross_backend_embedder() -> None:
    # Bedrock serves, OpenAI embeds -> two provider configs.
    config = HarnessConfig.for_build(
        serve_provider=ProviderKind.BEDROCK,
        serve_model="opus",
        region=None,
        embed_provider=EmbedderKind.OPENAI,
        embed_model="text-embedding-3-small",
        embed_dim=512,
        gepa_budget=10,
    )
    kinds = {pc.kind for pc in config.providers}
    assert kinds == {ProviderKind.BEDROCK, ProviderKind.OPENAI}
    openai_cfg = config.provider_config(ProviderKind.OPENAI)
    assert openai_cfg.embed_model == "text-embedding-3-small"


def test_for_build_hashing_embedder_needs_no_embed_provider_config() -> None:
    config = HarnessConfig.for_build(
        serve_provider=ProviderKind.BEDROCK,
        serve_model="opus",
        region=None,
        embed_provider=EmbedderKind.HASHING,
        embed_model=None,
        embed_dim=512,
        gepa_budget=10,
    )
    assert [pc.kind for pc in config.providers] == [ProviderKind.BEDROCK]
    assert config.embed_provider is EmbedderKind.HASHING


def test_config_train_split_default_mirrors_the_canonical_constant() -> None:
    # `_DEFAULT_TRAIN_SPLIT` is a mirror: the canonical value lives next to the split functions in
    # `wmo.simulation.model.build`, which this module cannot import because build.py imports it.
    # A build config that cuts the hash line somewhere other than where `wmo eval` cuts it
    # hands GEPA's training traces back as scored "held-out" steps, so pin the mirror here.
    assert _DEFAULT_TRAIN_SPLIT == DEFAULT_TRAIN_SPLIT
    assert HarnessConfig().train_split == DEFAULT_TRAIN_SPLIT


def test_for_build_threads_train_split() -> None:
    # train_split defaults to DEFAULT_TRAIN_SPLIT but is overridable (so `wmo build --train-split`
    # reaches GEPA).
    default = HarnessConfig.for_build(
        serve_provider=ProviderKind.BEDROCK,
        serve_model="opus",
        region=None,
        embed_provider=EmbedderKind.HASHING,
        embed_model=None,
        embed_dim=512,
        gepa_budget=10,
    )
    assert default.train_split == 0.8
    custom = HarnessConfig.for_build(
        serve_provider=ProviderKind.BEDROCK,
        serve_model="opus",
        region=None,
        embed_provider=EmbedderKind.HASHING,
        embed_model=None,
        embed_dim=512,
        gepa_budget=10,
        train_split=0.5,
    )
    assert custom.train_split == 0.5


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_train_split_must_be_a_proper_fraction(bad: float) -> None:
    # A degenerate ratio empties one side of the split; reject it up front rather than letting GEPA
    # "succeed" on a leaked/empty valset.
    with pytest.raises(ValidationError):
        HarnessConfig(train_split=bad)
