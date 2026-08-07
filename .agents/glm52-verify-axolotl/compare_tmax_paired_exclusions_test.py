"""Tests for manifest-scoped paired TMax comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compare_tmax_paired_exclusions import compare


def write_result(root: Path, task_id: str, reward: float) -> None:
    """Write one minimal scored result."""
    path = root / task_id / "episode_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"task_id": task_id, "status": "scored", "reward": reward}),
        encoding="utf-8",
    )


def test_manifest_selects_base_superset(tmp_path: Path) -> None:
    """A manifest may select an exact adapter subset from a base superset."""
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    write_result(base, "a", 0.0)
    write_result(base, "b", 1.0)
    write_result(base, "extra", 1.0)
    write_result(adapter, "a", 1.0)
    write_result(adapter, "b", 1.0)
    manifest = tmp_path / "tasks.txt"
    manifest.write_text("a\nb\n", encoding="utf-8")

    result = compare(
        base_root=base,
        adapter_root=adapter,
        bootstrap_samples=100,
        bootstrap_seed=1,
        task_ids_path=manifest,
    )

    assert result["attempted_task_count"] == 2
    assert result["paired"]["mean_reward_delta"] == 0.5


def test_manifest_rejects_unrequested_adapter_attempt(tmp_path: Path) -> None:
    """The adapter arm must contain exactly the manifested task set."""
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    for task_id in ("a", "extra"):
        write_result(base, task_id, 1.0)
        write_result(adapter, task_id, 1.0)
    manifest = tmp_path / "tasks.txt"
    manifest.write_text("a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extra_adapter"):
        compare(
            base_root=base,
            adapter_root=adapter,
            bootstrap_samples=10,
            bootstrap_seed=1,
            task_ids_path=manifest,
        )
