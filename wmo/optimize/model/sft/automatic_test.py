"""Automatic runtime SFT composition tests."""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Never

import pytest

import wmo.optimize.model.sft.automatic as automatic_module
import wmo.optimize.model.sft.run_manifest as run_manifest_module
from wmo.common.core.artifacts import canonical_json_bytes, sha256_json
from wmo.common.models import (
    AssistantAction,
    BillingSource,
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
    accept_runtime_sft_model_optimization,
    prepare_runtime_sft_model_optimization,
)
from wmo.optimize.model.sft.composition import (
    SFTModelOptimizationConfig,
    create_sft_model_optimization_config,
    preflight_sft_model_optimization,
    sft_model_optimization_output_dir,
    write_sft_model_optimization_config,
)
from wmo.optimize.model.sft.contracts import SFTBuildSpec
from wmo.optimize.model.sft.runtime_source_test import _accept, _complete, _request
from wmo.optimize.model.sft.selection import (
    SFTModelOptimizationSelectionError,
    latest_sft_model_optimization_path,
    load_latest_sft_model_optimization,
    versioned_sft_model_alias,
    write_latest_sft_model_optimization,
)
from wmo.optimize.model.sft.training_contracts import TinkerSFTResumeError
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
            models={
                "base": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="tinker",
                    model="test-base-model",
                )
            },
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
            models={
                "base": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="tinker",
                    model="test-base-model",
                )
            },
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
            models={
                "base": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="tinker",
                    model="test-base-model",
                )
            },
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


def test_run_acceptance_rejects_an_advanced_journal_before_manifest_write(
    tmp_path: Path,
) -> None:
    """Require a new graph and consent when a completion precedes W13 acceptance.

    Args:
        tmp_path: Pytest-owned project and artifact directory.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepared = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    output_dir = sft_model_optimization_output_dir(bootstrap.store, prepared.config.config_id)
    _append_completed(bootstrap.store, key="second", minute=2)

    with pytest.raises(AutomaticSFTPreparationError, match="rerun.*every durable completion"):
        accept_runtime_sft_model_optimization(
            bootstrap.store,
            prepared,
            created_at=_TIME + timedelta(hours=1),
            code_revision="automatic-sft-test",
        )

    assert not output_dir.exists()
    refreshed = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=2),
        code_revision="automatic-sft-test",
    )
    manifest = accept_runtime_sft_model_optimization(
        bootstrap.store,
        refreshed,
        created_at=_TIME + timedelta(hours=2),
        code_revision="automatic-sft-test",
    )

    assert len(refreshed.dataset.rows) == 2
    assert manifest.dataset_id == refreshed.dataset.dataset.dataset_id
    assert (
        sft_model_optimization_output_dir(bootstrap.store, refreshed.config.config_id)
        / "manifest.json"
    ).is_file()


def test_run_acceptance_holds_journal_lock_through_w13_manifest_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serialize concurrent appends until the config-bound W13 manifest is durable.

    Args:
        tmp_path: Pytest-owned project and artifact directory.
        monkeypatch: Scoped W13 initializer wrapper used to expose the lock boundary.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepared = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    output_dir = sft_model_optimization_output_dir(bootstrap.store, prepared.config.config_id)
    original_initialize = automatic_module.initialize_tinker_sft_run
    writer_started = threading.Event()
    writer_finished = threading.Event()
    manifest_committed_before_writer: list[bool] = []

    def append_concurrently() -> None:
        """Attempt one append while the acceptance callback owns the journal lock."""
        writer_started.set()
        _append_completed(bootstrap.store, key="second", minute=2)
        writer_finished.set()

    def initialize_while_writer_waits(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        """Expose that W13 initialization runs before a queued append can finish.

        Args:
            args: Positional W13 initializer inputs forwarded unchanged.
            kwargs: Keyword W13 initializer inputs forwarded unchanged.

        Returns:
            The original initializer's durable run manifest.
        """
        writer = threading.Thread(target=append_concurrently)
        writer.start()
        assert writer_started.wait(timeout=1.0)
        assert not writer_finished.wait(timeout=0.05)
        manifest = original_initialize(*args, **kwargs)
        manifest_committed_before_writer.append((output_dir / "manifest.json").is_file())
        assert not writer_finished.is_set()
        return manifest

    monkeypatch.setattr(
        automatic_module,
        "initialize_tinker_sft_run",
        initialize_while_writer_waits,
    )

    accepted = accept_runtime_sft_model_optimization(
        bootstrap.store,
        prepared,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )

    assert writer_finished.wait(timeout=1.0)
    assert manifest_committed_before_writer == [True]
    assert accepted.dataset_id == prepared.dataset.dataset.dataset_id
    refreshed = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=2),
        code_revision="automatic-sft-test",
    )
    assert refreshed.config == prepared.config
    assert refreshed.accepted is True
    assert len(refreshed.dataset.rows) == 1


def test_crash_before_acceptance_pointer_leaves_orphan_inert_and_reconsents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treat a crash after the immutable receipt but before its CAS pointer as unaccepted.

    Args:
        tmp_path: Pytest-owned project and artifact directory.
        monkeypatch: Scoped acceptance-pointer crash replacement.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepared = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    original_select = automatic_module.write_automatic_sft_acceptance_selection_unlocked

    def crash_before_pointer(*args: object, **kwargs: object) -> Never:
        """Interrupt after immutable acceptance persistence but before pointer selection.

        Args:
            args: Unexpected selection positional inputs.
            kwargs: Unexpected selection keyword inputs.

        Raises:
            RuntimeError: Always, at the selected crash boundary.
        """
        del args, kwargs
        raise RuntimeError("simulated crash before acceptance pointer")

    monkeypatch.setattr(
        automatic_module,
        "write_automatic_sft_acceptance_selection_unlocked",
        crash_before_pointer,
    )
    with pytest.raises(RuntimeError, match="before acceptance pointer"):
        accept_runtime_sft_model_optimization(
            bootstrap.store,
            prepared,
            created_at=_TIME + timedelta(hours=1),
            code_revision="automatic-sft-test",
        )

    assert run_manifest_module.load_automatic_sft_acceptance_selection(bootstrap.store) is None
    retry = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=2),
        code_revision="automatic-sft-test",
    )
    assert retry.config == prepared.config
    assert retry.accepted is False

    monkeypatch.setattr(
        automatic_module,
        "write_automatic_sft_acceptance_selection_unlocked",
        original_select,
    )
    accept_runtime_sft_model_optimization(
        bootstrap.store,
        retry,
        created_at=_TIME + timedelta(hours=2),
        code_revision="automatic-sft-test",
    )
    selected = run_manifest_module.load_automatic_sft_acceptance_selection(bootstrap.store)
    assert selected is not None
    resumed = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=3),
        code_revision="automatic-sft-test",
    )
    assert resumed.accepted is True


def test_copied_w13_acceptance_cannot_authorize_another_selected_config(
    tmp_path: Path,
) -> None:
    """Reject a manifest and acceptance copied into a different config namespace.

    Args:
        tmp_path: Pytest-owned project and artifact directory.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepared = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    accept_runtime_sft_model_optimization(
        bootstrap.store,
        prepared,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    latest_a = load_latest_sft_model_optimization(bootstrap.store)
    assert latest_a is not None
    selected_a = run_manifest_module.load_automatic_sft_acceptance_selection(bootstrap.store)
    assert selected_a is not None
    with pytest.raises(TinkerSFTResumeError, match="changed before consent commit"):
        run_manifest_module.write_automatic_sft_acceptance_selection_unlocked(
            bootstrap.store,
            selected_a.model_copy(
                update={
                    "acceptance": selected_a.acceptance.model_copy(
                        update={"artifact_id": "forged-automatic-acceptance"}
                    ),
                    "updated_at": _TIME + timedelta(hours=2),
                }
            ),
            expected_current=None,
        )
    pointer_path = run_manifest_module.automatic_sft_acceptance_path(bootstrap.store)
    pointer_bytes = pointer_path.read_bytes()
    bootstrap_config_input = artifact_input(
        bootstrap.store.artifacts.read(bootstrap.config.config_id).manifest
    )
    pointer_path.write_bytes(
        canonical_json_bytes(selected_a.model_copy(update={"config": bootstrap_config_input}))
    )
    with pytest.raises(AutomaticSFTPreparationError, match="pointer differs"):
        prepare_runtime_sft_model_optimization(
            bootstrap.store,
            created_at=_TIME + timedelta(hours=2),
            code_revision="automatic-sft-test",
        )
    pointer_path.write_bytes(pointer_bytes)
    alias_prefix_b = "alternate-trained"
    config_b = create_sft_model_optimization_config(
        bootstrap.store,
        dataset_id=prepared.dataset.dataset.dataset_id,
        model_alias=versioned_sft_model_alias(alias_prefix_b, prepared.dataset.dataset.dataset_id),
        tinker_connection=prepared.config.tinker_connection,
        base_model_alias=prepared.config.base_model_alias,
        training=prepared.config.training,
        created_at=_TIME + timedelta(hours=2),
        code_revision="automatic-sft-test",
    )
    write_sft_model_optimization_config(bootstrap.store, config_b, bind_project=False)
    config_b_input = artifact_input(bootstrap.store.artifacts.read(config_b.config_id).manifest)
    output_a = sft_model_optimization_output_dir(bootstrap.store, prepared.config.config_id)
    latest_b = latest_a.model_copy(
        update={
            "config": config_b_input,
            "model_alias_prefix": alias_prefix_b,
            "updated_at": _TIME + timedelta(hours=2),
        }
    )
    write_latest_sft_model_optimization(
        bootstrap.store,
        latest_b,
        expected_current=latest_a.config,
    )
    output_b = sft_model_optimization_output_dir(bootstrap.store, config_b.config_id)
    output_b.mkdir(parents=True)
    shutil.copyfile(output_a / "manifest.json", output_b / "manifest.json")

    with pytest.raises(AutomaticSFTPreparationError, match="incomplete"):
        prepare_runtime_sft_model_optimization(
            bootstrap.store,
            created_at=_TIME + timedelta(hours=3),
            code_revision="automatic-sft-test",
        )

    manifest_b = run_manifest_module.load_tinker_sft_run(
        bootstrap.store,
        prepared.dataset.dataset.dataset_id,
        config_b.training,
        output_b,
        code_revision="automatic-sft-test",
    )
    assert manifest_b is not None
    _receipt_b, acceptance_b_input = run_manifest_module.initialize_automatic_sft_acceptance(
        bootstrap.store,
        manifest=manifest_b,
        previous_acceptance=selected_a.acceptance,
        config=config_b_input,
        dataset=latest_b.dataset,
        runtime_snapshot=latest_b.runtime_snapshot,
        model_alias=config_b.model_alias,
        tinker_connection=config_b.tinker_connection,
        base_model=config_b.base_model,
        connection_config_sha256=config_b.connection_config_sha256,
        training_spec_sha256=sha256_json(config_b.training),
        runtime_last_ordinal=prepared.snapshot.last_ordinal,
        runtime_prefix_sha256=prepared.snapshot.prefix_sha256,
        created_at=_TIME + timedelta(hours=3),
        code_revision="automatic-sft-test",
    )
    run_manifest_module.write_automatic_sft_acceptance_selection_unlocked(
        bootstrap.store,
        run_manifest_module.AutomaticSFTRunAcceptanceSelection(
            schema_version=1,
            project_id=bootstrap.store.paths.project_id,
            acceptance=acceptance_b_input,
            previous_acceptance=selected_a.acceptance,
            config=config_b_input,
            updated_at=_TIME + timedelta(hours=3),
        ),
        expected_current=selected_a.acceptance,
    )

    with pytest.raises(AutomaticSFTPreparationError, match="incomplete"):
        prepare_runtime_sft_model_optimization(
            bootstrap.store,
            created_at=_TIME + timedelta(hours=4),
            code_revision="automatic-sft-test",
        )


@pytest.mark.parametrize("target_exists", [False, True])
def test_automatic_acceptance_pointer_rejects_broken_and_live_symlinks_before_write(
    tmp_path: Path,
    *,
    target_exists: bool,
) -> None:
    """Reject a redirected acceptance pointer without creating or changing its target.

    Args:
        tmp_path: Pytest-owned project and external target directory.
        target_exists: Whether the redirect names an existing external file.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepared = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    pointer_path = run_manifest_module.automatic_sft_acceptance_path(bootstrap.store)
    external_target = tmp_path / "external-automatic-acceptance.json"
    original_external_bytes = b"external pointer must remain unchanged"
    if target_exists:
        external_target.write_bytes(original_external_bytes)
    pointer_path.symlink_to(external_target)

    with pytest.raises(TinkerSFTResumeError, match="not a safe file"):
        run_manifest_module.load_automatic_sft_acceptance_selection(bootstrap.store)
    with pytest.raises(AutomaticSFTPreparationError, match="not a safe file"):
        accept_runtime_sft_model_optimization(
            bootstrap.store,
            prepared,
            created_at=_TIME + timedelta(hours=1),
            code_revision="automatic-sft-test",
        )

    if target_exists:
        assert external_target.read_bytes() == original_external_bytes
    else:
        assert not external_target.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_automatic_acceptance_pointer_replaces_a_symlink_swapped_at_write_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_exists: bool,
) -> None:
    """Replace a raced pointer link itself without following or changing its target.

    Args:
        tmp_path: Pytest-owned project and external target directory.
        monkeypatch: Pytest mutation helper used at the exact atomic-replace boundary.
        target_exists: Whether the raced redirect names an existing external file.
    """
    bootstrap = _bootstrap(tmp_path)
    _append_completed(bootstrap.store, key="first", minute=1)
    prepared = prepare_runtime_sft_model_optimization(
        bootstrap.store,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )
    pointer_path = run_manifest_module.automatic_sft_acceptance_path(bootstrap.store)
    external_target = tmp_path / "raced-external-automatic-acceptance.json"
    original_external_bytes = b"raced external pointer must remain unchanged"
    if target_exists:
        external_target.write_bytes(original_external_bytes)
    real_replace = os.replace

    def swap_pointer_before_replace(source: str | Path, destination: str | Path) -> None:
        """Install the adversarial link immediately before the production replace.

        Args:
            source: Exclusive sibling staging path.
            destination: Canonical acceptance pointer path.
        """
        if Path(destination) == pointer_path:
            pointer_path.symlink_to(external_target)
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", swap_pointer_before_replace)
    accept_runtime_sft_model_optimization(
        bootstrap.store,
        prepared,
        created_at=_TIME + timedelta(hours=1),
        code_revision="automatic-sft-test",
    )

    assert pointer_path.is_file()
    assert not pointer_path.is_symlink()
    assert run_manifest_module.load_automatic_sft_acceptance_selection(bootstrap.store) is not None
    if target_exists:
        assert external_target.read_bytes() == original_external_bytes
    else:
        assert not external_target.exists()


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
            models={
                "base": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="tinker",
                    model="test-base-model",
                )
            },
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
                "base-a": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="tinker",
                    model="test-base-model",
                ),
                "base-b": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="tinker",
                    model="test-base-model",
                ),
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
