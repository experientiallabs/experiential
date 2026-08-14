"""Automatic runtime SFT composition tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest

import wmo.optimize.model.sft.automatic as automatic_module
from wmo.common.core.artifacts import canonical_json_bytes
from wmo.common.models import (
    AssistantAction,
    ConnectionConfig,
    ModelCatalog,
    ModelMessage,
    ModelRecord,
    write_model_catalog,
)
from wmo.common.project import ProjectStore, artifact_input
from wmo.optimize.model.sft.automatic import (
    AutomaticSFTPreparationError,
    InitialSFTModelOptimizationSettings,
    prepare_runtime_sft_model_optimization,
)
from wmo.optimize.model.sft.composition import (
    SFTModelOptimizationConfig,
    create_sft_model_optimization_config,
    preflight_sft_model_optimization,
    write_sft_model_optimization_config,
)
from wmo.optimize.model.sft.contracts import SFTBuildSpec
from wmo.optimize.model.sft.runtime_source_test import _accept, _complete, _request
from wmo.optimize.model.sft.selection import (
    SFTModelOptimizationSelectionError,
    latest_sft_model_optimization_path,
    load_latest_sft_model_optimization,
    write_latest_sft_model_optimization,
)
from wmo.optimize.model.sft.training_test import _TIME, _FakeBackend, _persisted_dataset, _spec
from wmo.runtime.router import RuntimeInteractionJournal


@dataclass(frozen=True)
class _Bootstrap:
    """Project with explicit bounded Tinker settings but no automatic runtime dataset."""

    store: ProjectStore
    config: SFTModelOptimizationConfig


def _bootstrap(tmp_path: Path) -> _Bootstrap:
    """Persist one accepted seed W12 and user-selected bounded Tinker configuration.

    Args:
        tmp_path: Pytest-owned state directory.

    Returns:
        Initialized project ready to accumulate runtime interactions.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"tinker": ConnectionConfig(provider="tinker")},
            models={"base": ModelRecord(connection="tinker", model="test-base-model")},
        ),
    )
    config = create_sft_model_optimization_config(
        fixture.store,
        dataset_id=fixture.artifact.dataset.dataset_id,
        model_alias="trained",
        tinker_connection="tinker",
        base_model_alias="base",
        training=_spec(maximum_cost_usd=1.0),
        created_at=_TIME,
        code_revision="automatic-sft-test",
    )
    write_sft_model_optimization_config(fixture.store, config)
    return _Bootstrap(store=fixture.store, config=config)


def _append_completed(
    store: ProjectStore,
    *,
    key: str,
    minute: int,
) -> None:
    """Append one completed routed interaction to the project's durable journal.

    Args:
        store: Project receiving the interaction.
        key: Unique idempotency identity and visible fixture tag.
        minute: Timestamp offset that preserves global journal order.
    """
    _complete(
        RuntimeInteractionJournal(store.paths),
        key=key,
        conversation="automatic-sft-conversation",
        request=_request(ModelMessage(role="user", content=f"Request {key}")),
        output=AssistantAction(content=f"Response {key}"),
        now=_TIME + timedelta(minutes=minute),
    )


def test_first_runtime_prefix_needs_no_bootstrap_dataset_or_config(tmp_path: Path) -> None:
    """Create the first W12 and config directly from confirmed settings and runtime data.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"tinker": ConnectionConfig(provider="tinker")},
            models={"base": ModelRecord(connection="tinker", model="test-base-model")},
        ),
    )
    _append_completed(fixture.store, key="first", minute=1)

    prepared = prepare_runtime_sft_model_optimization(
        fixture.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
        initial_settings=InitialSFTModelOptimizationSettings(
            model_alias_prefix="support-project-sft",
            tinker_connection="tinker",
            base_model_alias="base",
            training=_spec(maximum_cost_usd=25.0),
        ),
    )
    latest = load_latest_sft_model_optimization(fixture.store)

    assert prepared.created is True
    assert prepared.dataset.build_spec == SFTBuildSpec(held_out_fraction=0.0)
    assert len(prepared.dataset.rows) == 1
    assert fixture.store.load_project().model_optimization_config is None
    assert latest is not None
    assert latest.config.artifact_id == prepared.config.config_id


def test_automatic_runtime_sft_schedules_every_lineage_and_retains_duplicates(
    tmp_path: Path,
) -> None:
    """Train every completed target while leakage-linking duplicate cross-lineage rows.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"tinker": ConnectionConfig(provider="tinker")},
            models={"base": ModelRecord(connection="tinker", model="test-base-model")},
        ),
    )
    journal = RuntimeInteractionJournal(fixture.store.paths)
    request = _request(ModelMessage(role="user", content="Retain this duplicate"))
    target = AssistantAction(content="Retained target")
    _complete(
        journal,
        key="first-lineage",
        conversation="conversation-one",
        request=request,
        output=target,
        now=_TIME + timedelta(minutes=1),
    )
    _complete(
        journal,
        key="second-lineage",
        conversation="conversation-two",
        request=request,
        output=target,
        now=_TIME + timedelta(minutes=2),
    )
    prepared = prepare_runtime_sft_model_optimization(
        fixture.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-all-target-test",
        initial_settings=InitialSFTModelOptimizationSettings(
            model_alias_prefix="support-project-sft",
            tinker_connection="tinker",
            base_model_alias="base",
            training=_spec(maximum_cost_usd=25.0),
        ),
    )
    preflight = preflight_sft_model_optimization(
        fixture.store,
        prepared.config.config_id,
        _FakeBackend(conservative_cost_per_batch=0.1),
        code_revision="automatic-all-target-test",
    )

    assert len(prepared.dataset.rows) == 2
    assert {row.partition for row in prepared.dataset.rows} == {"train"}
    assert len({row.example.example_id for row in prepared.dataset.rows}) == 2
    assert len({row.fingerprint for row in prepared.dataset.rows}) == 1
    lineage_ids = {row.example.leakage_group_id for row in prepared.dataset.rows}
    assert len(lineage_ids) == 2
    assert len(prepared.dataset.partitions) == 1
    assert set(prepared.dataset.partitions[0].leakage_group_ids) == lineage_ids
    assert sum(preflight.planned_batch_counts) == 2


def test_unchanged_prefix_reuses_snapshot_dataset_config_and_pointer_bytes(
    tmp_path: Path,
) -> None:
    """Replay an unchanged journal without creating or rewriting any selected state.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    first = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    artifact_ids = bootstrap.store.artifacts.list_ids()
    pointer_path = latest_sft_model_optimization_path(bootstrap.store)
    pointer_bytes = pointer_path.read_bytes()

    replay = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(days=1),
        code_revision="different-replay-revision",
    )

    assert first.created is True
    assert replay.created is False
    assert replay.snapshot == first.snapshot
    assert replay.dataset == first.dataset
    assert replay.config == first.config
    assert bootstrap.store.artifacts.list_ids() == artifact_ids
    assert pointer_path.read_bytes() == pointer_bytes


def test_appended_interaction_creates_new_graph_without_mutating_old_artifacts(
    tmp_path: Path,
) -> None:
    """Advance to a full appended prefix while retaining every prior immutable artifact.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    first = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    first_pointer = load_latest_sft_model_optimization(bootstrap.store)
    assert first_pointer is not None
    old_config_bytes = bootstrap.store.artifacts.read_bytes(first.config.config_id, "config.json")
    old_dataset_bytes = bootstrap.store.artifacts.read_bytes(
        first.dataset.dataset.dataset_id, "examples.jsonl"
    )
    _append_completed(bootstrap.store, key="second", minute=2)

    appended = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=2),
        code_revision="automatic-sft-test",
    )
    latest = load_latest_sft_model_optimization(bootstrap.store)

    assert appended.created is True
    assert appended.snapshot.snapshot_id != first.snapshot.snapshot_id
    assert appended.dataset.dataset.dataset_id != first.dataset.dataset.dataset_id
    assert appended.config.config_id != first.config.config_id
    assert len(appended.dataset.rows) == 2
    assert {row.example.target.content for row in appended.dataset.rows} == {
        "Response first",
        "Response second",
    }
    assert (
        bootstrap.store.artifacts.read_bytes(first.config.config_id, "config.json")
        == old_config_bytes
    )
    assert (
        bootstrap.store.artifacts.read_bytes(first.dataset.dataset.dataset_id, "examples.jsonl")
        == old_dataset_bytes
    )
    assert latest is not None
    assert latest.config.artifact_id == appended.config.config_id
    assert latest.dataset.artifact_id == appended.dataset.dataset.dataset_id
    assert latest.runtime_snapshot.artifact_id == appended.snapshot.snapshot_id
    bootstrap_input = artifact_input(
        bootstrap.store.artifacts.read(bootstrap.config.config_id).manifest
    )
    with pytest.raises(SFTModelOptimizationSelectionError, match="changed before commit"):
        write_latest_sft_model_optimization(
            bootstrap.store,
            first_pointer.model_copy(update={"updated_at": _TIME + timedelta(hours=3)}),
            expected_current=bootstrap_input,
        )


def test_empty_or_incomplete_journal_fails_without_materializing_runtime_artifacts(
    tmp_path: Path,
) -> None:
    """Reject missing completed targets before creating snapshot, dataset, or config artifacts.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    bootstrap = _bootstrap(tmp_path)
    original_ids = bootstrap.store.artifacts.list_ids()

    with pytest.raises(AutomaticSFTPreparationError, match="no interactions"):
        prepare_runtime_sft_model_optimization(
            bootstrap.store,
            created_at=_TIME + timedelta(hours=1),
            code_revision="automatic-sft-test",
        )

    assert bootstrap.store.artifacts.list_ids() == original_ids
    assert not latest_sft_model_optimization_path(bootstrap.store).exists()

    _accept(
        RuntimeInteractionJournal(bootstrap.store.paths),
        key="disconnected",
        conversation="automatic-sft-conversation",
        request=_request(ModelMessage(role="user", content="Disconnected request")),
        now=_TIME + timedelta(minutes=1),
    )
    with pytest.raises(AutomaticSFTPreparationError, match="no completed routed interactions"):
        prepare_runtime_sft_model_optimization(
            bootstrap.store,
            created_at=_TIME + timedelta(hours=1),
            code_revision="automatic-sft-test",
        )

    assert bootstrap.store.artifacts.list_ids() == original_ids
    assert not latest_sft_model_optimization_path(bootstrap.store).exists()


def test_corrupt_latest_pointer_fails_closed_before_new_materialization(tmp_path: Path) -> None:
    """Reject a malformed coordination pointer without replacing its selected artifacts.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    original_ids = bootstrap.store.artifacts.list_ids()
    latest_sft_model_optimization_path(bootstrap.store).write_text("{not-json", encoding="utf-8")

    with pytest.raises(AutomaticSFTPreparationError, match="pointer is invalid"):
        prepare_runtime_sft_model_optimization(
            bootstrap.store,
            created_at=_TIME + timedelta(hours=2),
            code_revision="automatic-sft-test",
        )

    assert bootstrap.store.artifacts.list_ids() == original_ids


def test_crash_after_immutable_graph_reuses_artifacts_before_pointer_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover a first-run crash after config persistence without duplicating artifacts.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped failure injected at the mutable pointer commit.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"tinker": ConnectionConfig(provider="tinker")},
            models={"base": ModelRecord(connection="tinker", model="test-base-model")},
        ),
    )
    _append_completed(fixture.store, key="first", minute=1)
    settings = InitialSFTModelOptimizationSettings(
        model_alias_prefix="support-project-sft",
        tinker_connection="tinker",
        base_model_alias="base",
        training=_spec(maximum_cost_usd=25.0),
    )

    def crash_before_pointer(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        """Fail after immutable artifacts exist but before mutable selection commits.

        Args:
            *args: Positional pointer-write inputs, deliberately unused.
            **kwargs: Keyword pointer-write inputs, deliberately unused.

        Raises:
            SFTModelOptimizationSelectionError: Always, to simulate the crash boundary.
        """
        del args, kwargs
        raise SFTModelOptimizationSelectionError("injected pointer crash")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            automatic_module,
            "write_latest_sft_model_optimization",
            crash_before_pointer,
        )
        with pytest.raises(AutomaticSFTPreparationError, match="injected pointer crash"):
            prepare_runtime_sft_model_optimization(
                fixture.store,
                created_at=_TIME + timedelta(hours=1),
                code_revision="automatic-sft-test",
                initial_settings=settings,
            )
    artifact_ids_after_crash = fixture.store.artifacts.list_ids()
    assert load_latest_sft_model_optimization(fixture.store) is None

    recovered = prepare_runtime_sft_model_optimization(
        fixture.store,
        created_at=_TIME + timedelta(hours=2),
        code_revision="automatic-sft-test",
        initial_settings=settings,
    )

    assert recovered.created is True
    assert fixture.store.artifacts.list_ids() == artifact_ids_after_crash
    assert load_latest_sft_model_optimization(fixture.store) is not None


def test_two_first_run_pointer_proposals_allow_only_one_compare_and_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow only one of two valid concurrent first-run graphs to become latest.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped capture of independently materialized pointer proposals.
    """
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"tinker": ConnectionConfig(provider="tinker")},
            models={
                "base-a": ModelRecord(connection="tinker", model="test-base-model"),
                "base-b": ModelRecord(connection="tinker", model="test-base-model"),
            },
        ),
    )
    _append_completed(fixture.store, key="first", minute=1)
    proposals = []

    def capture_pointer(store, pointer, *, expected_current):  # noqa: ANN001, ANN202
        """Capture one complete graph proposal and interrupt before pointer mutation.

        Args:
            store: Project store that would receive the pointer.
            pointer: Fully verified proposed latest graph.
            expected_current: Compare-and-swap predecessor observed by the attempt.

        Raises:
            SFTModelOptimizationSelectionError: Always, after recording the proposal.
        """
        del store
        proposals.append((pointer, expected_current))
        raise SFTModelOptimizationSelectionError("captured proposal")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            automatic_module,
            "write_latest_sft_model_optimization",
            capture_pointer,
        )
        for base_alias in ("base-a", "base-b"):
            with pytest.raises(AutomaticSFTPreparationError, match="captured proposal"):
                prepare_runtime_sft_model_optimization(
                    fixture.store,
                    created_at=_TIME + timedelta(hours=1),
                    code_revision="automatic-sft-test",
                    initial_settings=InitialSFTModelOptimizationSettings(
                        model_alias_prefix=f"{base_alias}-trained",
                        tinker_connection="tinker",
                        base_model_alias=base_alias,
                        training=_spec(maximum_cost_usd=25.0),
                    ),
                )
    assert len(proposals) == 2
    first_pointer, first_expected = proposals[0]
    second_pointer, second_expected = proposals[1]
    write_latest_sft_model_optimization(
        fixture.store, first_pointer, expected_current=first_expected
    )
    with pytest.raises(SFTModelOptimizationSelectionError, match="changed before commit"):
        write_latest_sft_model_optimization(
            fixture.store, second_pointer, expected_current=second_expected
        )


def test_latest_pointer_rejects_symlink_and_wrong_project(tmp_path: Path) -> None:
    """Fail closed when latest coordination is redirected or claims another project.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    path = latest_sft_model_optimization_path(bootstrap.store)
    pointer = load_latest_sft_model_optimization(bootstrap.store)
    assert pointer is not None
    target = tmp_path / "redirected-latest.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(SFTModelOptimizationSelectionError, match="not a safe file"):
        load_latest_sft_model_optimization(bootstrap.store)

    path.unlink()
    path.write_bytes(
        canonical_json_bytes(pointer.model_copy(update={"project_id": "another-project"}))
    )
    with pytest.raises(SFTModelOptimizationSelectionError, match="another project"):
        load_latest_sft_model_optimization(bootstrap.store)


def test_latest_pointer_rejects_symlinked_coordination_directory_for_read_and_write(
    tmp_path: Path,
) -> None:
    """Reject a parent-directory redirect before pointer read, lock, or write.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    path = latest_sft_model_optimization_path(bootstrap.store)
    pointer = load_latest_sft_model_optimization(bootstrap.store)
    assert pointer is not None
    coordination_directory = path.parent
    external_directory = tmp_path / "external-model-optimization"
    coordination_directory.rename(external_directory)
    coordination_directory.symlink_to(external_directory, target_is_directory=True)
    external_pointer_bytes = (external_directory / path.name).read_bytes()

    with pytest.raises(SFTModelOptimizationSelectionError, match="directory is not safe"):
        load_latest_sft_model_optimization(bootstrap.store)
    with pytest.raises(SFTModelOptimizationSelectionError, match="directory is not safe"):
        write_latest_sft_model_optimization(
            bootstrap.store,
            pointer,
            expected_current=pointer.config,
        )
    _append_completed(bootstrap.store, key="second", minute=2)
    with pytest.raises(AutomaticSFTPreparationError, match="directory is not safe"):
        prepare_runtime_sft_model_optimization(
            bootstrap.store,
            created_at=_TIME + timedelta(hours=2),
            code_revision="automatic-sft-test",
        )
    assert (external_directory / path.name).read_bytes() == external_pointer_bytes


def test_latest_pointer_rejects_model_alias_prefix_rewrite(tmp_path: Path) -> None:
    """Bind the mutable alias prefix to the selected immutable config and dataset.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    path = latest_sft_model_optimization_path(bootstrap.store)
    pointer = load_latest_sft_model_optimization(bootstrap.store)
    assert pointer is not None
    artifact_ids = bootstrap.store.artifacts.list_ids()
    path.write_bytes(
        canonical_json_bytes(pointer.model_copy(update={"model_alias_prefix": "attacker-prefix"}))
    )

    with pytest.raises(SFTModelOptimizationSelectionError, match="does not derive"):
        load_latest_sft_model_optimization(bootstrap.store)
    _append_completed(bootstrap.store, key="second", minute=2)
    with pytest.raises(AutomaticSFTPreparationError, match="does not derive"):
        prepare_runtime_sft_model_optimization(
            bootstrap.store,
            created_at=_TIME + timedelta(hours=2),
            code_revision="automatic-sft-test",
        )
    assert bootstrap.store.artifacts.list_ids() == artifact_ids


@pytest.mark.parametrize("schema_version", [0, 2])
def test_latest_pointer_rejects_unsupported_canonical_schema_versions(
    tmp_path: Path, schema_version: int
) -> None:
    """Reject canonically encoded pointer versions outside the exact supported contract.

    Args:
        tmp_path: Pytest-owned state directory.
        schema_version: Unsupported lower or higher coordination schema version.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    path = latest_sft_model_optimization_path(bootstrap.store)
    pointer = load_latest_sft_model_optimization(bootstrap.store)
    assert pointer is not None
    payload = pointer.model_dump(mode="json", exclude_none=False)
    payload["schema_version"] = schema_version
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(SFTModelOptimizationSelectionError, match="pointer is invalid"):
        load_latest_sft_model_optimization(bootstrap.store)
