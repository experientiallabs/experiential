"""Tests for score-independent grouped benchmark partition manifests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from wmh.evals.partition import BenchmarkPartitionManifest, PartitionTask


def _task(task_id: str, stratum: str, group: str) -> PartitionTask:
    return PartitionTask(
        task_id=task_id,
        stratum=stratum,
        group_id=group,
        content_digest="sha256:" + task_id.encode().hex().ljust(64, "0")[:64],
    )


def _tasks() -> tuple[PartitionTask, ...]:
    return (
        _task("easy-a", "easy", "easy-a"),
        _task("easy-b", "easy", "mixed-family"),
        _task("medium-a", "medium", "mixed-family"),
        _task("medium-b", "medium", "medium-b"),
        _task("medium-c", "medium", "medium-c"),
        _task("hard-a", "hard", "hard-family"),
        _task("hard-b", "hard", "hard-family"),
        _task("hard-c", "hard", "hard-c"),
    )


def _manifest(tasks: tuple[PartitionTask, ...] | None = None) -> BenchmarkPartitionManifest:
    return BenchmarkPartitionManifest.create(
        tasks=tasks or _tasks(),
        discovery_counts={"easy": 1, "medium": 2, "hard": 2},
        selection_seed="split-v1",
        seal_nonce="private-random-nonce-v1",
    )


def test_partition_is_canonical_exact_and_never_splits_a_group() -> None:
    manifest = _manifest()
    reordered = _manifest(tuple(reversed(_tasks())))

    assert manifest == reordered
    assert manifest.digest == reordered.digest
    assert manifest.discovery_counts == {"easy": 1, "hard": 2, "medium": 2}
    assert manifest.confirmation_counts == {"easy": 1, "hard": 1, "medium": 1}
    assert set(manifest.discovery_task_ids).isdisjoint(manifest.confirmation_task_ids)
    assert set(manifest.discovery_task_ids) | set(manifest.confirmation_task_ids) == {
        task.task_id for task in _tasks()
    }
    by_group: dict[str, set[str]] = {}
    for task in manifest.tasks:
        partition = "discovery" if task.task_id in manifest.discovery_task_ids else "confirmation"
        by_group.setdefault(task.group_id, set()).add(partition)
    assert all(len(partitions) == 1 for partitions in by_group.values())


def test_impossible_grouped_quota_fails_before_any_score_exists() -> None:
    with pytest.raises(ValueError, match="cannot satisfy"):
        BenchmarkPartitionManifest.create(
            tasks=_tasks(),
            discovery_counts={"easy": 2, "medium": 0, "hard": 0},
            selection_seed="split-v1",
            seal_nonce="private-random-nonce-v1",
        )


def test_duplicate_task_identity_and_invalid_digest_are_rejected() -> None:
    duplicate = (_tasks()[0], _tasks()[0])
    with pytest.raises(ValueError, match="duplicate task_id"):
        BenchmarkPartitionManifest.create(
            tasks=duplicate,
            discovery_counts={"easy": 1},
            selection_seed="split-v1",
            seal_nonce="private-random-nonce-v1",
        )
    with pytest.raises(ValidationError, match="content_digest"):
        PartitionTask(
            task_id="task",
            stratum="easy",
            group_id="task",
            content_digest="mutable-tag",
        )


def test_discovery_view_does_not_serialize_control_plane_secrets_or_heldout_ids() -> None:
    manifest = _manifest()
    discovery = manifest.discovery_view()
    wire = discovery.model_dump_json()

    assert discovery.partition_manifest_digest == manifest.digest
    assert set(discovery.tasks) == {
        task for task in manifest.tasks if task.task_id in manifest.discovery_task_ids
    }
    assert discovery.confirmation_counts == manifest.confirmation_counts
    assert "selection_seed" not in wire
    assert "seal_nonce" not in wire
    assert "private-random-nonce-v1" not in wire
    assert all(task_id not in wire for task_id in manifest.confirmation_task_ids)


def test_confirmation_opening_binds_the_already_frozen_candidate() -> None:
    manifest = _manifest()
    candidate_hash = "sha256:" + "a" * 64
    opened = manifest.open_confirmation(candidate_hash=candidate_hash)

    assert opened.partition_manifest_digest == manifest.digest
    assert opened.candidate_hash == candidate_hash
    assert tuple(task.task_id for task in opened.tasks) == manifest.confirmation_task_ids
    assert opened.confirmation_commitment == manifest.confirmation_commitment
    assert json.loads(opened.model_dump_json())["candidate_hash"] == candidate_hash

    with pytest.raises(ValidationError, match="candidate_hash"):
        manifest.open_confirmation(candidate_hash="mutable-candidate-name")


def test_content_change_preserves_score_independent_membership_but_changes_identity() -> None:
    first = _manifest()
    changed_tasks = list(_tasks())
    changed_tasks[0] = changed_tasks[0].model_copy(update={"content_digest": "sha256:" + "f" * 64})
    second = _manifest(tuple(changed_tasks))

    assert first.discovery_task_ids == second.discovery_task_ids
    assert first.confirmation_task_ids == second.confirmation_task_ids
    assert first.digest != second.digest
    assert first.confirmation_commitment != second.confirmation_commitment
