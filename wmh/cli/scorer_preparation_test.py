"""Tests for durable scorer preparation artifact lifecycle helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.cli.scorer_preparation import (
    preparation_staging_path,
    publish_preparation_artifact,
    remove_preparation_staging,
)


def test_preparation_artifact_moves_from_sibling_to_claimed_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    staging = preparation_staging_path(run_dir)
    assert staging == tmp_path / ".run.scorer-preparation.bin"
    staging.write_bytes(b"receipt\n")
    run_dir.mkdir()

    published = publish_preparation_artifact(run_dir, staging.read_bytes())
    remove_preparation_staging(staging)

    assert published == run_dir / "scorer-preparation.bin"
    assert published.read_bytes() == b"receipt\n"
    assert not staging.exists()


def test_preparation_artifact_rejects_nonregular_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"receipt\n")
    staging = preparation_staging_path(run_dir)
    staging.symlink_to(target)

    with pytest.raises(RuntimeError, match="not a regular file"):
        remove_preparation_staging(staging)

    destination = run_dir / "scorer-preparation.bin"
    destination.symlink_to(target)
    with pytest.raises(RuntimeError, match="already exists"):
        publish_preparation_artifact(run_dir, b"other\n")
