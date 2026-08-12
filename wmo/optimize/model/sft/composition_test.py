"""Deterministic W14M composition tests with W12 and W13 persisted local fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from wmo.common.models import (
    ConnectionConfig,
    ModelCatalog,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.project import ProjectStore
from wmo.optimize.model.sft.composition import (
    SFTModelOptimizationConfig,
    SFTModelOptimizationError,
    SFTModelOptimizationPreflightError,
    create_sft_model_optimization_config,
    preflight_sft_model_optimization,
    run_sft_model_optimization,
    sft_model_optimization_output_dir,
    write_sft_model_optimization_config,
)
from wmo.optimize.model.sft.training import TinkerSFTOptimizer
from wmo.optimize.model.sft.training_test import (
    _TIME,
    _FakeBackend,
    _persisted_dataset,
    _spec,
)


@dataclass(frozen=True)
class _Prepared:
    """One persisted W12 source, its local store, and immutable W14M config."""

    store: ProjectStore
    dataset_id: str
    config: SFTModelOptimizationConfig


def _prepared(tmp_path: Path) -> _Prepared:
    """Create one project with a persisted accepted W12 dataset and Tinker catalog connection."""
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"tinker": ConnectionConfig(provider="tinker")},
            models={"base": ModelRecord(connection="tinker", model="base-model")},
        ),
    )
    config = create_sft_model_optimization_config(
        fixture.store,
        dataset_id=fixture.artifact.dataset.dataset_id,
        model_alias="trained",
        tinker_connection="tinker",
        training=_spec(),
        created_at=_TIME,
        code_revision="w14m-test",
    )
    write_sft_model_optimization_config(fixture.store, config)
    return _Prepared(
        store=fixture.store,
        dataset_id=fixture.artifact.dataset.dataset_id,
        config=config,
    )


def test_composition_trains_only_the_persisted_dataset_and_registers_after_verification(
    tmp_path: Path,
) -> None:
    """W14M runs W13 from W12 and adds one alias only after terminal verification."""
    prepared = _prepared(tmp_path)
    store = prepared.store
    config = prepared.config
    backend = _FakeBackend()

    preflight = preflight_sft_model_optimization(
        store,
        config,
        backend,
        code_revision="w14m-test",
    )
    completed = run_sft_model_optimization(
        store,
        config,
        backend,
        created_at=_TIME,
        code_revision="w14m-test",
        preflight=preflight,
    )

    catalog = load_model_catalog(store.model_catalog_path)
    assert store.load_project().model_optimization_config_id == config.config_id
    assert completed.catalog_updated is True
    assert catalog.models["trained"].connection == "tinker"
    assert catalog.models["trained"].model == completed.model.sampling_handle
    assert backend.open_resume_paths == [None]
    assert (sft_model_optimization_output_dir(store, config) / "result.json").is_file()


def test_completed_run_is_verified_then_idempotently_reused_without_opening_backend(
    tmp_path: Path,
) -> None:
    """A second composition call revalidates W13 and never dispatches a new fake session."""
    prepared = _prepared(tmp_path)
    store = prepared.store
    config = prepared.config
    first_backend = _FakeBackend()
    first_preflight = preflight_sft_model_optimization(
        store,
        config,
        first_backend,
        code_revision="w14m-test",
    )
    run_sft_model_optimization(
        store,
        config,
        first_backend,
        created_at=_TIME,
        code_revision="w14m-test",
        preflight=first_preflight,
    )
    second_backend = _FakeBackend()

    second_preflight = preflight_sft_model_optimization(
        store,
        config,
        second_backend,
        code_revision="w14m-test",
    )
    completed = run_sft_model_optimization(
        store,
        config,
        second_backend,
        created_at=_TIME,
        code_revision="w14m-test",
        preflight=second_preflight,
    )

    assert second_preflight.completed_result is not None
    assert second_preflight.completed_model is not None
    assert completed.catalog_updated is False
    assert second_backend.open_resume_paths == []


def test_resume_uses_only_a_durable_checkpoint_before_catalog_registration(tmp_path: Path) -> None:
    """A safe pre-terminal W13 state resumes from its checkpoint and registers once complete."""
    prepared = _prepared(tmp_path)
    store = prepared.store
    config = prepared.config
    output_dir = sft_model_optimization_output_dir(store, config)
    initial_backend = _FakeBackend()
    TinkerSFTOptimizer(initial_backend).optimize(
        store=store,
        dataset_id=config.dataset.artifact_id,
        spec=config.training,
        output_dir=output_dir,
        created_at=_TIME,
        code_revision="w14m-test",
    )
    for name in ("model.json", "model-intent.json", "result.json"):
        (output_dir / name).unlink()
    resumed_backend = _FakeBackend()

    preflight = preflight_sft_model_optimization(
        store,
        config,
        resumed_backend,
        code_revision="w14m-test",
    )
    completed = run_sft_model_optimization(
        store,
        config,
        resumed_backend,
        created_at=_TIME,
        code_revision="w14m-test",
        preflight=preflight,
    )

    assert completed.catalog_updated is True
    assert resumed_backend.open_resume_paths == [
        f"fake://state/{initial_backend.saved_state_names[-1]}"
    ]
    assert load_model_catalog(store.model_catalog_path).models["trained"].model == (
        completed.model.sampling_handle
    )


def test_budgeted_tinker_run_fails_before_backend_open_when_no_estimate_exists(
    tmp_path: Path,
) -> None:
    """A maximum-cost setting cannot use Tinker when it has no supported conservative estimate."""
    prepared = _prepared(tmp_path)
    store = prepared.store
    config = prepared.config
    budgeted = create_sft_model_optimization_config(
        store,
        dataset_id=config.dataset.artifact_id,
        model_alias=config.model_alias,
        tinker_connection=config.tinker_connection,
        training=_spec(maximum_cost_usd=1.0),
        created_at=_TIME,
        code_revision="w14m-test",
    )
    backend = _FakeBackend(conservative_cost_per_batch=None)

    with pytest.raises(SFTModelOptimizationPreflightError, match="no supported conservative cost"):
        preflight_sft_model_optimization(
            store,
            budgeted,
            backend,
            code_revision="w14m-test",
        )

    assert backend.open_resume_paths == []


def test_ambiguous_w13_run_never_enters_models_toml(tmp_path: Path) -> None:
    """A stale pre-dispatch intent blocks recovery and leaves the trained alias absent."""
    prepared = _prepared(tmp_path)
    store = prepared.store
    config = prepared.config
    output_dir = sft_model_optimization_output_dir(store, config)
    with pytest.raises(RuntimeError, match="injected failure after optimizer dispatch"):
        TinkerSFTOptimizer(_FakeBackend(fail_after_train_call=1)).optimize(
            store=store,
            dataset_id=config.dataset.artifact_id,
            spec=config.training,
            output_dir=output_dir,
            created_at=_TIME,
            code_revision="w14m-test",
        )
    backend = _FakeBackend()

    with pytest.raises(SFTModelOptimizationError, match="did not complete safely"):
        run_sft_model_optimization(
            store,
            config,
            backend,
            created_at=_TIME,
            code_revision="w14m-test",
        )

    assert "trained" not in load_model_catalog(store.model_catalog_path).models
