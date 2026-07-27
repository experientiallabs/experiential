"""Tests for the named-world-model store (resolution, listing, summaries)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmo.config import ArtifactPaths, HarnessConfig, save_config
from wmo.config.store import DEFAULT_MODEL_NAME, WorldModelStore, normalize_name, validate_name
from wmo.providers.base import ProviderConfig, ProviderKind


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
    store = WorldModelStore(tmp_path / ".wmo")
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
    store = WorldModelStore(tmp_path / ".wmo")

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
    store = WorldModelStore(tmp_path / ".wmo")
    _build_fake_model(store, "alpha")
    with pytest.raises(FileNotFoundError, match="alpha"):
        store.resolve("nope")


def test_one_unreadable_artifact_does_not_hide_the_healthy_ones(tmp_path) -> None:  # noqa: ANN001
    # A model dir arrives from `wmo pull` or a hand copy, so a config.toml this CLI cannot parse
    # is an ordinary state. It must cost its own row, not the whole listing.
    store = WorldModelStore(tmp_path / ".wmo")
    _build_fake_model(store, "alpha-healthy")
    for name, payload in (("zz-bad-toml", "this is not toml ="), ("zz-bad-schema", 'top_k = "x"')):
        bad = store.model_dir(name)
        bad.mkdir(parents=True)
        (bad / "config.toml").write_text(payload, encoding="utf-8")

    infos = {info.name: info for info in store.list_info()}

    assert infos["alpha-healthy"].error is None
    assert infos["alpha-healthy"].serve_provider == "bedrock"
    # Each bad row names its own file and the way out.
    assert "zz-bad-toml/config.toml is not valid TOML" in str(infos["zz-bad-toml"].error)
    assert "re-run `wmo build`" in str(infos["zz-bad-toml"].error)
    assert "does not match the current config schema" in str(infos["zz-bad-schema"].error)


def test_unreadable_metrics_names_the_file_it_could_not_read(tmp_path) -> None:  # noqa: ANN001
    # metrics.json is written after config.toml, so a half-written artifact fails here; the row
    # has to say which file rather than a bare json decode position.
    store = WorldModelStore(tmp_path / ".wmo")
    _build_fake_model(store, "alpha")
    ArtifactPaths(store.model_dir("alpha")).metrics.write_text("{not json", encoding="utf-8")

    (info,) = store.list_info()
    assert "metrics.json could not be read" in str(info.error)


def test_non_utf8_artifact_json_names_the_file(tmp_path) -> None:  # noqa: ANN001
    # UnicodeDecodeError is a ValueError, not an OSError, so it slips past a read/parse guard
    # that only names those two and arrives as a bare codec error with no path.
    store = WorldModelStore(tmp_path / ".wmo")
    _build_fake_model(store, "alpha")
    ArtifactPaths(store.model_dir("alpha")).metrics.write_bytes(b'{"held_out_accuracy": "\x80"}')

    (info,) = store.list_info()
    assert "metrics.json could not be read" in str(info.error)
    assert "codec can't decode" in str(info.error)


@pytest.mark.parametrize(
    ("filename", "payload", "expected"),
    [
        ("metrics", "[0.5, 7]", "metrics.json is not a JSON object"),
        ("frontier", '{"a": 1}', "frontier.json is not a JSON array"),
    ],
    ids=["metrics-array", "frontier-object"],
)
def test_a_wrongly_shaped_summary_file_is_an_unreadable_row(
    tmp_path: Path, filename: str, payload: str, expected: str
) -> None:
    # Valid JSON of the wrong top-level type is still a corrupt artifact. Skipping past it would
    # print a row identical to a healthy model that was simply never evaluated.
    store = WorldModelStore(tmp_path / ".wmo")
    _build_fake_model(store, "alpha")
    path = getattr(ArtifactPaths(store.model_dir("alpha")), filename)
    path.write_text(payload, encoding="utf-8")

    (info,) = store.list_info()
    assert expected in str(info.error)
    assert "re-run `wmo build`" in str(info.error)
