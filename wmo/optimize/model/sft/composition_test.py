"""Deterministic W14M composition tests with W12 and W13 persisted local fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from wmo.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.project import ProjectStore
from wmo.optimize.model.sft.composition import (
    SFTModelOptimizationConfig,
    SFTModelOptimizationError,
    SFTModelOptimizationPreflight,
    SFTModelOptimizationPreflightError,
    create_sft_model_optimization_config,
    preflight_sft_model_optimization,
    run_sft_model_optimization,
    sft_model_optimization_output_dir,
    write_sft_model_optimization_config,
)
from wmo.optimize.model.sft.training import TinkerSFTOptimizer
from wmo.optimize.model.sft.training_contracts import TinkerSFTSpec
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


def _prepared(
    tmp_path: Path,
    training: TinkerSFTSpec | None = None,
) -> _Prepared:
    """Create one project with a persisted accepted W12 dataset and Tinker catalog connection."""
    fixture = _persisted_dataset(tmp_path)
    write_model_catalog(
        fixture.store.model_catalog_path,
        ModelCatalog(
            connections={"tinker": ConnectionConfig(provider="tinker")},
            models={"base": ModelRecord(connection="tinker", model="test-base-model")},
        ),
    )
    resolved_training = _spec(maximum_cost_usd=1.0) if training is None else training
    config = create_sft_model_optimization_config(
        fixture.store,
        dataset_id=fixture.artifact.dataset.dataset_id,
        model_alias="trained",
        tinker_connection="tinker",
        base_model_alias="base",
        training=resolved_training,
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
        config.config_id,
        backend,
        code_revision="w14m-test",
    )
    completed = run_sft_model_optimization(
        store,
        config.config_id,
        backend,
        created_at=_TIME,
        code_revision="w14m-test",
        preflight=preflight,
    )

    catalog = load_model_catalog(store.model_catalog_path)
    project_binding = store.load_project().model_optimization_config
    assert project_binding is not None
    assert project_binding.artifact_id == config.config_id
    assert completed.catalog_updated is True
    assert catalog.models["trained"].connection == "tinker"
    assert catalog.models["trained"].model == completed.model.sampling_handle
    assert backend.open_resume_paths == [None]
    assert (sft_model_optimization_output_dir(store, config.config_id) / "result.json").is_file()


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
        config.config_id,
        first_backend,
        code_revision="w14m-test",
    )
    run_sft_model_optimization(
        store,
        config.config_id,
        first_backend,
        created_at=_TIME,
        code_revision="w14m-test",
        preflight=first_preflight,
    )
    second_backend = _FakeBackend()

    second_preflight = preflight_sft_model_optimization(
        store,
        config.config_id,
        second_backend,
        code_revision="w14m-test",
    )
    completed = run_sft_model_optimization(
        store,
        config.config_id,
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
    output_dir = sft_model_optimization_output_dir(store, config.config_id)
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
        config.config_id,
        resumed_backend,
        code_revision="w14m-test",
    )
    completed = run_sft_model_optimization(
        store,
        config.config_id,
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
    prepared = _prepared(tmp_path, _spec(maximum_cost_usd=1.0))
    store = prepared.store
    config = prepared.config
    backend = _FakeBackend(conservative_cost_per_batch=None)

    with pytest.raises(SFTModelOptimizationPreflightError, match="no supported conservative cost"):
        preflight_sft_model_optimization(
            store,
            config.config_id,
            backend,
            code_revision="w14m-test",
        )

    assert backend.open_resume_paths == []
    assert backend.train_calls == 0


def test_config_requires_an_explicit_finite_maximum_cost_usd(tmp_path: Path) -> None:
    """W14M never creates a config without a finite cap even if W13 permits one in isolation."""
    with pytest.raises(SFTModelOptimizationError, match="require a finite maximum_cost_usd"):
        _prepared(tmp_path, _spec(maximum_cost_usd=None))


def test_config_rejects_a_nonfinite_maximum_cost_usd(tmp_path: Path) -> None:
    """W14M also rejects a nonfinite cap from an otherwise valid model instance."""
    nonfinite_training = _spec(maximum_cost_usd=1.0).model_copy(
        update={"maximum_cost_usd": float("inf")}
    )

    with pytest.raises(SFTModelOptimizationError, match="require a finite maximum_cost_usd"):
        _prepared(tmp_path, nonfinite_training)


def test_full_schedule_budget_preflight_blocks_before_any_backend_dispatch(tmp_path: Path) -> None:
    """Every exact W13 batch is priced before a total cap can permit execution."""
    prepared = _prepared(
        tmp_path,
        _spec(epochs=2, maximum_cost_usd=0.15),
    )
    backend = _FakeBackend(conservative_cost_per_batch=0.10)

    with pytest.raises(SFTModelOptimizationPreflightError, match="full Tinker SFT schedule"):
        preflight_sft_model_optimization(
            prepared.store,
            prepared.config.config_id,
            backend,
            code_revision="w14m-test",
        )

    assert backend.cost_calls == 4
    assert backend.open_resume_paths == []
    assert backend.train_calls == 0


def test_supplied_preflight_cannot_bypass_the_full_schedule_budget_gate(tmp_path: Path) -> None:
    """Run recomputes the authoritative schedule even when a caller supplies a forged preflight."""
    prepared = _prepared(tmp_path, _spec(epochs=2, maximum_cost_usd=0.15))
    backend = _FakeBackend(conservative_cost_per_batch=0.10)
    config_input = prepared.store.load_project().model_optimization_config
    assert config_input is not None
    forged = SFTModelOptimizationPreflight(
        config=prepared.config,
        config_input=config_input,
        output_dir=sft_model_optimization_output_dir(prepared.store, prepared.config.config_id),
        completed_result=None,
        completed_model=None,
        planned_batch_counts=(),
        conservative_schedule_cost_usd=None,
    )

    with pytest.raises(SFTModelOptimizationPreflightError, match="full Tinker SFT schedule"):
        run_sft_model_optimization(
            prepared.store,
            prepared.config.config_id,
            backend,
            created_at=_TIME,
            code_revision="w14m-test",
            preflight=forged,
        )

    assert backend.cost_calls == 4
    assert backend.open_resume_paths == []
    assert backend.train_calls == 0


def test_model_only_completed_w13_recovers_before_budget_estimation(tmp_path: Path) -> None:
    """A verified W13 model regenerates its missing result locally with no estimator or dispatch."""
    prepared = _prepared(tmp_path, _spec(maximum_cost_usd=1.0))
    output_dir = sft_model_optimization_output_dir(prepared.store, prepared.config.config_id)
    TinkerSFTOptimizer(_FakeBackend()).optimize(
        store=prepared.store,
        dataset_id=prepared.config.dataset.artifact_id,
        spec=prepared.config.training,
        output_dir=output_dir,
        created_at=_TIME,
        code_revision="w14m-test",
    )
    (output_dir / "result.json").unlink()
    verifier_probe = _FakeBackend(conservative_cost_per_batch=None)

    preflight = preflight_sft_model_optimization(
        prepared.store,
        prepared.config.config_id,
        verifier_probe,
        code_revision="w14m-test",
    )

    assert preflight.completed_result is not None
    assert (output_dir / "result.json").is_file()
    assert verifier_probe.cost_calls == 0
    assert verifier_probe.open_resume_paths == []
    assert verifier_probe.train_calls == 0


def test_connection_key_reference_drift_blocks_dispatch_after_preflight(tmp_path: Path) -> None:
    """The launch re-resolves the exact frozen base connection, including key reference digest."""
    prepared = _prepared(tmp_path)
    backend = _FakeBackend()
    preflight = preflight_sft_model_optimization(
        prepared.store,
        prepared.config.config_id,
        backend,
        code_revision="w14m-test",
    )
    catalog = load_model_catalog(prepared.store.model_catalog_path)
    write_model_catalog(
        prepared.store.model_catalog_path,
        catalog.model_copy(
            update={
                "connections": {
                    "tinker": ConnectionConfig(provider="tinker", api_key_env="TINKER_KEY_NEXT")
                }
            }
        ),
    )

    with pytest.raises(SFTModelOptimizationPreflightError, match="connection metadata drifted"):
        run_sft_model_optimization(
            prepared.store,
            prepared.config.config_id,
            backend,
            created_at=_TIME,
            code_revision="w14m-test",
            preflight=preflight,
        )

    assert backend.open_resume_paths == []
    assert backend.train_calls == 0


def test_base_model_capability_drift_blocks_dispatch_after_preflight(tmp_path: Path) -> None:
    """The launch re-resolves the frozen base model capability digest before any provider call."""
    prepared = _prepared(tmp_path)
    backend = _FakeBackend()
    preflight = preflight_sft_model_optimization(
        prepared.store,
        prepared.config.config_id,
        backend,
        code_revision="w14m-test",
    )
    catalog = load_model_catalog(prepared.store.model_catalog_path)
    write_model_catalog(
        prepared.store.model_catalog_path,
        catalog.model_copy(
            update={
                "models": {
                    "base": catalog.models["base"].model_copy(
                        update={"capabilities": ModelCapabilities(supports_tools=True)}
                    )
                }
            }
        ),
    )

    with pytest.raises(SFTModelOptimizationPreflightError, match="base model snapshot drifted"):
        run_sft_model_optimization(
            prepared.store,
            prepared.config.config_id,
            backend,
            created_at=_TIME,
            code_revision="w14m-test",
            preflight=preflight,
        )

    assert backend.open_resume_paths == []
    assert backend.train_calls == 0


def test_catalog_record_hash_binds_w12_w13_base_and_sampling_provenance(tmp_path: Path) -> None:
    """The registered alias carries all immutable inputs and result identities needed for reuse."""
    prepared = _prepared(tmp_path)
    completed = run_sft_model_optimization(
        prepared.store,
        prepared.config.config_id,
        _FakeBackend(),
        created_at=_TIME,
        code_revision="w14m-test",
    )
    provenance = (
        load_model_catalog(prepared.store.model_catalog_path)
        .models[prepared.config.model_alias]
        .sft_provenance
    )

    assert provenance is not None
    assert provenance.source_dataset == prepared.config.dataset
    assert provenance.optimization_config == prepared.store.load_project().model_optimization_config
    assert provenance.run_id == completed.training_result.run_id
    assert provenance.model_id == completed.model.model_id
    assert provenance.result_id == completed.training_result.result_id
    assert provenance.base_model == prepared.config.base_model


def test_unbound_config_id_cannot_train_even_when_its_object_was_just_created(
    tmp_path: Path,
) -> None:
    """Public execution accepts only a project-bound artifact ID, never an in-memory config."""
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
        code_revision="w14m-test",
    )
    backend = _FakeBackend()

    with pytest.raises(SFTModelOptimizationError, match="not a verified artifact"):
        run_sft_model_optimization(
            fixture.store,
            config.config_id,
            backend,
            created_at=_TIME,
            code_revision="w14m-test",
        )

    assert backend.open_resume_paths == []
    assert backend.train_calls == 0


def test_ambiguous_w13_run_never_enters_models_toml(tmp_path: Path) -> None:
    """A stale pre-dispatch intent blocks recovery and leaves the trained alias absent."""
    prepared = _prepared(tmp_path)
    store = prepared.store
    config = prepared.config
    output_dir = sft_model_optimization_output_dir(store, config.config_id)
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
            config.config_id,
            backend,
            created_at=_TIME,
            code_revision="w14m-test",
        )

    assert "trained" not in load_model_catalog(store.model_catalog_path).models
