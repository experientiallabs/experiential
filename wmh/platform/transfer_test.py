"""Tests for model-bundle packing and unpacking."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from wmh.platform.transfer import (
    BundleFormatError,
    extract_push_meta,
    pack_model_dir,
    sha256_file,
    unpack_model_bundle,
)


def _model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "tau-bench"
    (directory / "prompts").mkdir(parents=True)
    (directory / "index").mkdir()
    (directory / "runs").mkdir()
    (directory / "traces").mkdir()
    (directory / "config.toml").write_text("embed_dim = 64\n", encoding="utf-8")
    (directory / "metrics.json").write_text('{"held_out_accuracy": 0.7}', encoding="utf-8")
    (directory / "prompts" / "base.txt").write_text("you are the environment", encoding="utf-8")
    (directory / "index" / "steps.jsonl").write_text('{"action": "a"}\n', encoding="utf-8")
    (directory / "runs" / "run-1.json").write_text("{}", encoding="utf-8")
    (directory / "traces" / "corpus.jsonl").write_text('{"private": true}\n', encoding="utf-8")
    return directory


def test_pack_includes_model_files_and_excludes_runs_and_traces(tmp_path: Path) -> None:
    bundle = pack_model_dir(_model_dir(tmp_path), tmp_path / "out.tar.gz")

    with tarfile.open(bundle.path, mode="r:gz") as tar:
        names = set(tar.getnames())

    assert "config.toml" in names
    assert "metrics.json" in names
    assert "prompts/base.txt" in names
    assert "index/steps.jsonl" in names
    assert not any(name.startswith(("runs", "traces")) for name in names)
    assert bundle.byte_size == bundle.path.stat().st_size
    assert bundle.sha256 == sha256_file(bundle.path)


def test_pack_requires_config_toml(tmp_path: Path) -> None:
    directory = tmp_path / "not-a-model"
    directory.mkdir()

    with pytest.raises(BundleFormatError, match="config.toml"):
        pack_model_dir(directory, tmp_path / "out.tar.gz")
    with pytest.raises(BundleFormatError, match="does not exist"):
        pack_model_dir(tmp_path / "absent", tmp_path / "out.tar.gz")


def test_round_trip_and_force_semantics(tmp_path: Path) -> None:
    bundle = pack_model_dir(_model_dir(tmp_path), tmp_path / "out.tar.gz")
    dest = tmp_path / "pulled" / "tau-bench"

    unpack_model_bundle(bundle.path, dest)
    assert (dest / "config.toml").read_text(encoding="utf-8") == "embed_dim = 64\n"
    assert (dest / "prompts" / "base.txt").is_file()

    with pytest.raises(FileExistsError, match="--force"):
        unpack_model_bundle(bundle.path, dest)
    unpack_model_bundle(bundle.path, dest, force=True)
    assert (dest / "config.toml").is_file()


def test_unpack_rejects_garbage_bytes(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.tar.gz"
    garbage.write_bytes(b"not a tarball")
    with pytest.raises(BundleFormatError, match="unpacked"):
        unpack_model_bundle(garbage, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


def test_extract_push_meta_reads_typed_config_and_metrics(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path)

    meta = extract_push_meta(directory)

    assert meta["serve_provider"] == "anthropic"  # HarnessConfig default
    assert meta["embed_dim"] == 64
    assert meta["metrics"] == {"held_out_accuracy": 0.7}
    # No provider block is configured, so no serve model is claimed.
    assert "serve_model" not in meta


def test_extract_push_meta_includes_serve_model_when_configured(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path)
    (directory / "config.toml").write_text(
        '[[providers]]\nkind = "anthropic"\nmodel = "claude-sonnet-4-5"\n',
        encoding="utf-8",
    )

    meta = extract_push_meta(directory)

    assert meta["serve_model"] == "claude-sonnet-4-5"
