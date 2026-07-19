"""Tests for score-independent grouped benchmark partition manifests."""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from wmh.evals.partition import (
    BenchmarkPartitionManifest,
    PartitionControlScope,
    PartitionControlStore,
    PartitionTask,
    _build_partition_space,
    _count_feasible_subsets,
    freeze_confirmation_candidate,
    initialize_partition_genesis,
    open_confirmation_once,
)


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


def _store(tmp_path: Path, *, name: str = "partition-control") -> PartitionControlStore:
    directory = tmp_path / name
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    return PartitionControlStore(directory)


def _scope(
    *,
    experiment_id: str = "optimizer-study",
    protocol_id: str = "grouped-holdout-v1",
) -> PartitionControlScope:
    return PartitionControlScope(
        experiment_id=experiment_id,
        protocol_id=protocol_id,
    )


def _protocol_digest(value: str = "c") -> str:
    return "sha256:" + value * 64


def _manifest(
    store: PartitionControlStore,
    tasks: tuple[PartitionTask, ...] | None = None,
    *,
    scope: PartitionControlScope | None = None,
) -> BenchmarkPartitionManifest:
    selected_tasks = tasks or _tasks()
    counts = {"easy": 1, "medium": 2, "hard": 2}
    genesis = initialize_partition_genesis(
        store,
        scope=scope or _scope(),
        tasks=selected_tasks,
        discovery_counts=counts,
    )
    return BenchmarkPartitionManifest.create(
        tasks=selected_tasks,
        discovery_counts=counts,
        genesis=genesis,
    )


def test_partition_is_canonical_exact_and_never_splits_a_group(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = _manifest(store)
    reordered = _manifest(store, tuple(reversed(_tasks())))

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


def test_uniform_partition_space_exhaustively_unranks_each_feasible_subset_once() -> None:
    tasks = (
        _task("easy-a", "easy", "easy-a"),
        _task("easy-b", "easy", "easy-b"),
        _task("medium-a", "medium", "medium-a"),
        _task("medium-b", "medium", "medium-b"),
        _task("mixed-easy", "easy", "mixed"),
        _task("mixed-medium", "medium", "mixed"),
    )
    space = _build_partition_space(
        tasks,
        {"easy": 1, "medium": 1},
    )

    selected = tuple(space.discovery_groups_for_rank(rank) for rank in range(space.feasible_count))

    assert space.feasible_count == 5
    assert len(selected) == len(set(selected))
    assert set(selected) == {
        ("easy-a", "medium-a"),
        ("easy-a", "medium-b"),
        ("easy-b", "medium-a"),
        ("easy-b", "medium-b"),
        ("mixed",),
    }
    observed = Counter(group for subset in selected for group in subset)
    assert space.discovery_inclusion_probability("mixed") == Fraction(1, 5)
    for group_id in ("easy-a", "easy-b", "medium-a", "medium-b"):
        assert observed[group_id] == 2
        assert space.discovery_inclusion_probability(group_id) == Fraction(2, 5)


def test_manifest_persists_uniform_selection_evidence_and_exact_probabilities(
    tmp_path: Path,
) -> None:
    manifest = _manifest(_store(tmp_path))

    assert manifest.partition_version == "2"
    assert manifest.selection_algorithm == "uniform-feasible-subsets-v1"
    assert manifest.feasible_subset_count > 1
    assert manifest.selection_rank_commitment.startswith("sha256:")
    assert tuple(item.group_id for item in manifest.group_inclusion_probabilities) == tuple(
        sorted({task.group_id for task in manifest.tasks})
    )
    for item in manifest.group_inclusion_probabilities:
        probability = Fraction(item.discovery_numerator, item.denominator)
        assert 0 <= probability <= 1
    assert any(
        0 < item.discovery_probability < 1 for item in manifest.group_inclusion_probabilities
    )


def test_large_grouped_roster_feasible_count_regression() -> None:
    group_vectors = (
        *((0, 0, 1),) * 44,
        *((0, 0, 2),) * 3,
        *((0, 1, 0),) * 19,
        *((0, 1, 1),) * 3,
        *((0, 2, 0),) * 3,
        (0, 2, 1),
        *((1, 0, 0),) * 3,
        (1, 0, 1),
    )

    assert _count_feasible_subsets(group_vectors, target=(1, 10, 19)) == (
        20_533_886_319_672_671_229
    )


def test_impossible_grouped_quota_fails_before_any_score_exists(tmp_path: Path) -> None:
    store = _store(tmp_path)
    counts = {"easy": 2, "medium": 0, "hard": 0}
    genesis = initialize_partition_genesis(
        store,
        scope=_scope(protocol_id="impossible-quota-v1"),
        tasks=_tasks(),
        discovery_counts=counts,
    )
    with pytest.raises(ValueError, match="cannot satisfy"):
        BenchmarkPartitionManifest.create(
            tasks=_tasks(),
            discovery_counts=counts,
            genesis=genesis,
        )


def test_duplicate_task_identity_and_invalid_digest_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    duplicate = (_tasks()[0], _tasks()[0])
    with pytest.raises(ValueError, match="duplicate task_id"):
        initialize_partition_genesis(
            store,
            scope=_scope(),
            tasks=duplicate,
            discovery_counts={"easy": 1},
        )
    with pytest.raises(ValidationError, match="content_digest"):
        PartitionTask(
            task_id="task",
            stratum="easy",
            group_id="task",
            content_digest="mutable-tag",
        )


def test_discovery_view_does_not_serialize_control_plane_secrets_or_heldout_ids(
    tmp_path: Path,
) -> None:
    manifest = _manifest(_store(tmp_path))
    discovery = manifest.discovery_view()
    wire = discovery.model_dump_json()
    payload = json.loads(wire)

    assert discovery.partition_manifest_digest == manifest.digest
    assert {task.task_id for task in discovery.tasks} == set(manifest.discovery_task_ids)
    assert discovery.confirmation_counts == manifest.confirmation_counts
    assert "selection_seed" not in wire
    assert "seal_nonce" not in wire
    assert all(set(task) == {"task_id", "content_digest"} for task in payload["tasks"])
    assert manifest.selection_seed not in wire
    assert manifest.seal_nonce not in wire
    assert all(task_id not in wire for task_id in manifest.confirmation_task_ids)


def test_confirmation_opening_binds_the_already_frozen_candidate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = _manifest(store)
    candidate_execution_digest = "sha256:" + "a" * 64
    confirmation_protocol_digest = _protocol_digest()
    freeze = freeze_confirmation_candidate(
        store,
        manifest=manifest,
        candidate_execution_digest=candidate_execution_digest,
        confirmation_protocol_digest=confirmation_protocol_digest,
    )

    with pytest.raises(ValueError, match="manifest and protocol"):
        open_confirmation_once(
            store,
            manifest=manifest,
            confirmation_protocol_digest=_protocol_digest("d"),
        )
    assert not tuple(store.directory.glob("confirmation-opening-*.json"))

    opened = open_confirmation_once(
        store,
        manifest=manifest,
        confirmation_protocol_digest=confirmation_protocol_digest,
    )

    assert opened.partition_manifest_digest == manifest.digest
    assert opened.candidate_execution_digest == candidate_execution_digest
    assert opened.confirmation_protocol_digest == confirmation_protocol_digest
    assert freeze.confirmation_protocol_digest == confirmation_protocol_digest
    assert tuple(task.task_id for task in opened.tasks) == manifest.confirmation_task_ids
    assert opened.confirmation_commitment == manifest.confirmation_commitment
    assert opened.candidate_freeze_digest == freeze.digest
    assert opened.opening_record_digest.startswith("sha256:")
    assert (
        json.loads(opened.model_dump_json())["candidate_execution_digest"]
        == candidate_execution_digest
    )
    assert (
        open_confirmation_once(
            store,
            manifest=manifest,
            confirmation_protocol_digest=confirmation_protocol_digest,
        )
        == opened
    )

    with pytest.raises(ValidationError, match="candidate_execution_digest"):
        freeze_confirmation_candidate(
            store,
            manifest=manifest,
            candidate_execution_digest="mutable-candidate-name",
            confirmation_protocol_digest=confirmation_protocol_digest,
        )
    with pytest.raises(ValidationError, match="confirmation_protocol_digest"):
        freeze_confirmation_candidate(
            store,
            manifest=manifest,
            candidate_execution_digest=candidate_execution_digest,
            confirmation_protocol_digest="mutable-protocol-name",
        )
    with pytest.raises(ValueError, match="already frozen"):
        freeze_confirmation_candidate(
            store,
            manifest=manifest,
            candidate_execution_digest="sha256:" + "b" * 64,
            confirmation_protocol_digest=confirmation_protocol_digest,
        )
    with pytest.raises(ValueError, match="already frozen"):
        freeze_confirmation_candidate(
            store,
            manifest=manifest,
            candidate_execution_digest=candidate_execution_digest,
            confirmation_protocol_digest=_protocol_digest("d"),
        )
    assert len(tuple(store.directory.glob("candidate-freeze-*.json"))) == 1
    assert len(tuple(store.directory.glob("confirmation-opening-*.json"))) == 1


def test_content_change_changes_the_frozen_partition_identity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _manifest(store)
    changed_tasks = list(_tasks())
    changed_tasks[0] = changed_tasks[0].model_copy(update={"content_digest": "sha256:" + "f" * 64})
    second = _manifest(
        store,
        tuple(changed_tasks),
    )

    assert first.digest != second.digest
    assert first.confirmation_commitment != second.confirmation_commitment


def test_partition_genesis_is_one_shot_for_its_canonical_scope_and_inputs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = initialize_partition_genesis(
        store,
        scope=_scope(),
        tasks=_tasks(),
        discovery_counts={"easy": 1, "medium": 2, "hard": 2},
    )
    second = initialize_partition_genesis(
        store,
        scope=_scope(),
        tasks=tuple(reversed(_tasks())),
        discovery_counts={"hard": 2, "medium": 2, "easy": 1},
    )

    assert first == second
    assert len(first.selection_seed) == 64
    assert len(first.seal_nonce) == 64
    assert first.selection_seed != first.seal_nonce
    records = tuple(store.directory.glob("partition-genesis-*.json"))
    assert len(records) == 1
    assert records[0].stat().st_mode & 0o777 == 0o600

    changed_inputs = initialize_partition_genesis(
        store,
        scope=_scope(),
        tasks=_tasks(),
        discovery_counts={"easy": 1, "medium": 1, "hard": 2},
    )
    assert changed_inputs != first
    assert changed_inputs.tasks_digest == first.tasks_digest
    assert changed_inputs.discovery_strata != first.discovery_strata
    assert len(tuple(store.directory.glob("partition-genesis-*.json"))) == 2


def test_freeze_is_canonical_across_alternate_path_spellings_and_genesis_keys(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    manifest = _manifest(store)
    first = freeze_confirmation_candidate(
        store,
        manifest=manifest,
        candidate_execution_digest="sha256:" + "a" * 64,
        confirmation_protocol_digest=_protocol_digest(),
    )
    initialize_partition_genesis(
        store,
        scope=_scope(experiment_id="alternate", protocol_id="alternate"),
        tasks=_tasks(),
        discovery_counts={"easy": 1, "medium": 2, "hard": 2},
    )
    alternate_spelling = PartitionControlStore(store.directory / ".." / store.directory.name)

    assert (
        freeze_confirmation_candidate(
            alternate_spelling,
            manifest=manifest,
            candidate_execution_digest=first.candidate_execution_digest,
            confirmation_protocol_digest=first.confirmation_protocol_digest,
        )
        == first
    )
    with pytest.raises(ValueError, match="already frozen"):
        freeze_confirmation_candidate(
            alternate_spelling,
            manifest=manifest,
            candidate_execution_digest="sha256:" + "b" * 64,
            confirmation_protocol_digest=first.confirmation_protocol_digest,
        )
    assert len(tuple(store.directory.glob("candidate-freeze-*.json"))) == 1

    alternate_store = _store(tmp_path, name="alternate-control-store")
    with pytest.raises(ValueError, match="does not contain this manifest's genesis"):
        freeze_confirmation_candidate(
            alternate_store,
            manifest=manifest,
            candidate_execution_digest="sha256:" + "b" * 64,
            confirmation_protocol_digest=first.confirmation_protocol_digest,
        )


def test_freeze_and_open_revalidate_copied_manifest_instances(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = _manifest(store)
    freeze_confirmation_candidate(
        store,
        manifest=manifest,
        candidate_execution_digest="sha256:" + "a" * 64,
        confirmation_protocol_digest=_protocol_digest(),
    )

    changed_commitment = manifest.model_copy(
        update={"confirmation_commitment": "sha256:" + "c" * 64}
    )
    with pytest.raises(ValidationError, match="confirmation commitment"):
        freeze_confirmation_candidate(
            store,
            manifest=changed_commitment,
            candidate_execution_digest="sha256:" + "b" * 64,
            confirmation_protocol_digest=_protocol_digest(),
        )

    changed_membership = manifest.model_copy(
        update={"confirmation_task_ids": manifest.discovery_task_ids}
    )
    with pytest.raises(ValidationError, match="partition membership"):
        open_confirmation_once(
            store,
            manifest=changed_membership,
            confirmation_protocol_digest=_protocol_digest(),
        )

    changed_selection_evidence = manifest.model_copy(
        update={"feasible_subset_count": manifest.feasible_subset_count + 1}
    )
    with pytest.raises(ValidationError, match="selection evidence"):
        open_confirmation_once(
            store,
            manifest=changed_selection_evidence,
            confirmation_protocol_digest=_protocol_digest(),
        )

    assert len(tuple(store.directory.glob("candidate-freeze-*.json"))) == 1
    assert not tuple(store.directory.glob("confirmation-opening-*.json"))


def test_control_store_rejects_unsafe_or_symlinked_parent(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o750)
    with pytest.raises(ValueError, match="mode 0700"):
        PartitionControlStore(unsafe)

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        PartitionControlStore(linked)


def test_control_store_rejects_parent_inode_replacement(tmp_path: Path) -> None:
    store = _store(tmp_path)
    moved = tmp_path / "moved-control"
    store.directory.rename(moved)
    store.directory.mkdir(mode=0o700)
    store.directory.chmod(0o700)

    with pytest.raises(ValueError, match="changed after it was opened"):
        initialize_partition_genesis(
            store,
            scope=_scope(),
            tasks=_tasks(),
            discovery_counts={"easy": 1, "medium": 2, "hard": 2},
        )


@pytest.mark.parametrize("attack", ["symlink", "hard-link", "mode"])
def test_control_store_rejects_unsafe_partition_records(
    tmp_path: Path,
    attack: str,
) -> None:
    store = _store(tmp_path)
    initialize_partition_genesis(
        store,
        scope=_scope(),
        tasks=_tasks(),
        discovery_counts={"easy": 1, "medium": 2, "hard": 2},
    )
    (record,) = tuple(store.directory.glob("partition-genesis-*.json"))

    if attack == "symlink":
        payload = record.read_bytes()
        target = tmp_path / "record-target.json"
        target.write_bytes(payload)
        record.unlink()
        record.symlink_to(target)
        expected = "symbolic link"
    elif attack == "hard-link":
        os.link(record, store.directory / "unexpected-alias.json")
        expected = "exactly one link"
    else:
        record.chmod(0o640)
        expected = "mode 0600"

    with pytest.raises(ValueError, match=expected):
        initialize_partition_genesis(
            store,
            scope=_scope(),
            tasks=_tasks(),
            discovery_counts={"easy": 1, "medium": 2, "hard": 2},
        )


def test_candidate_freeze_race_publishes_exactly_one_candidate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = _manifest(store)
    candidates = tuple("sha256:" + value * 64 for value in ("a", "b"))

    def freeze(candidate_execution_digest: str) -> str | None:
        try:
            return freeze_confirmation_candidate(
                store,
                manifest=manifest,
                candidate_execution_digest=candidate_execution_digest,
                confirmation_protocol_digest=_protocol_digest(),
            ).candidate_execution_digest
        except ValueError as error:
            assert "already frozen" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(freeze, candidates * 16))

    winners = {result for result in results if result is not None}
    assert len(winners) == 1
    assert len(tuple(store.directory.glob("candidate-freeze-*.json"))) == 1


def test_orphaned_temporary_record_does_not_change_crash_retry_result(tmp_path: Path) -> None:
    store = _store(tmp_path)
    orphan = store.directory / (".partition-record-" + "f" * 64 + ".tmp")
    orphan.write_text('{"partial":', encoding="utf-8")
    orphan.chmod(0o600)

    first = initialize_partition_genesis(
        store,
        scope=_scope(),
        tasks=_tasks(),
        discovery_counts={"easy": 1, "medium": 2, "hard": 2},
    )
    second = initialize_partition_genesis(
        store,
        scope=_scope(),
        tasks=_tasks(),
        discovery_counts={"easy": 1, "medium": 2, "hard": 2},
    )

    assert first == second
    assert len(tuple(store.directory.glob("partition-genesis-*.json"))) == 1
