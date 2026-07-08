"""Tests for the named-world-model store (resolution, listing, summaries)."""

from __future__ import annotations

import json

import pytest

from wmh.config import ArtifactPaths, HarnessConfig, save_config
from wmh.config.store import DEFAULT_MODEL_NAME, WorldModelStore, normalize_name, validate_name
from wmh.providers.base import ProviderConfig, ProviderKind


def _build_fake_model(store: WorldModelStore, name: str, accuracy: float = 0.5) -> None:
    """Write a minimal but valid artifact (config + metrics + frontier) for `name`."""
    root = store.model_dir(name)
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="opus")],
        serve_provider=ProviderKind.BEDROCK,
    )
    save_config(config, root)
    paths = ArtifactPaths(root)
    paths.metrics.write_text(
        json.dumps({"held_out_accuracy": accuracy, "rollouts_used": 7}), encoding="utf-8"
    )
    paths.frontier.parent.mkdir(parents=True, exist_ok=True)
    paths.frontier.write_text(json.dumps(["a", "b"]), encoding="utf-8")


def test_validate_name_accepts_safe_names_and_rejects_traversal() -> None:
    assert validate_name("tau2-airline") == "tau2-airline"
    assert validate_name("retail.v2") == "retail.v2"
    for bad in ["../escape", "a/b", ".", "", ".hidden", "with space"]:
        with pytest.raises(ValueError, match="invalid world model name"):
            validate_name(bad)


def test_normalize_name_dash_joins_whitespace() -> None:
    assert normalize_name("tau bench") == "tau-bench"
    assert normalize_name("  tau   bench  ") == "tau-bench"
    assert normalize_name("tau-bench") == "tau-bench"  # already safe: unchanged
    with pytest.raises(ValueError, match="invalid world model name"):
        validate_name(normalize_name("tau/bench"))  # normalization never rescues separators


def test_list_names_and_info(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    store = WorldModelStore(tmp_path / ".wmh")
    assert store.list_names() == []
    _build_fake_model(store, "beta", accuracy=0.4)
    _build_fake_model(store, "alpha", accuracy=0.9)

    assert store.list_names() == ["alpha", "beta"]  # sorted
    info = store.info("alpha")
    assert info.serve_provider == "bedrock"
    assert info.serve_model == "opus"
    assert info.held_out_accuracy == 0.9
    assert info.rollouts_used == 7
    assert info.frontier_size == 2


def test_resolve_explicit_and_singleton_and_ambiguous(tmp_path) -> None:  # noqa: ANN001
    store = WorldModelStore(tmp_path / ".wmh")

    # No models yet: resolve(None) errors helpfully.
    with pytest.raises(FileNotFoundError, match="no world models built"):
        store.resolve(None)

    _build_fake_model(store, DEFAULT_MODEL_NAME)
    # Exactly one model: resolve(None) picks it.
    assert store.resolve(None) == store.model_dir(DEFAULT_MODEL_NAME)
    # Explicit name resolves to its dir.
    assert store.resolve(DEFAULT_MODEL_NAME) == store.model_dir(DEFAULT_MODEL_NAME)

    _build_fake_model(store, "second")
    # Two models: resolve(None) is ambiguous.
    with pytest.raises(ValueError, match="multiple world models"):
        store.resolve(None)


def test_resolve_unknown_name_lists_available(tmp_path) -> None:  # noqa: ANN001
    store = WorldModelStore(tmp_path / ".wmh")
    _build_fake_model(store, "alpha")
    with pytest.raises(FileNotFoundError, match="alpha"):
        store.resolve("nope")
