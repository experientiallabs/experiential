"""End-to-end CLI coverage for persisted-dataset W14M Tinker SFT composition."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.models import ConnectionConfig, ModelCatalog, ModelRecord, write_model_catalog
from wmo.common.project import ProjectStore
from wmo.optimize.model.sft import (
    SFTModelOptimizationConfig,
    TinkerSFTSpec,
    create_sft_model_optimization_config,
    write_sft_model_optimization_config,
)
from wmo.optimize.model.sft.training_test import _TIME, _FakeBackend, _persisted_dataset, _spec


@dataclass(frozen=True)
class _ConfiguredProject:
    """One local W12 source and write-once W14M project configuration for CLI tests."""

    store: ProjectStore
    config: SFTModelOptimizationConfig


def _configured_project(tmp_path: Path, training: TinkerSFTSpec) -> _ConfiguredProject:
    """Persist W12 evidence, a native Tinker connection, and one selected W14M config."""
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
        training=training,
        created_at=_TIME,
        code_revision="w14m-test",
    )
    write_sft_model_optimization_config(fixture.store, config)
    return _ConfiguredProject(store=fixture.store, config=config)


def test_optimize_help_exposes_persisted_sft_and_not_the_deleted_distill_branch() -> None:
    """The W14M command replaces the old training CLI branch without a compatibility alias."""
    result = CliRunner().invoke(app, ["optimize", "--help"])

    assert result.exit_code == 0, result.output
    assert "model" in result.output
    assert "distill" not in result.output


def test_cli_runs_fake_w13_then_idempotently_resumes_without_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The customer path trains once with --yes and later verifies a completed run without spend."""
    configured = _configured_project(tmp_path, _spec())
    command = importlib.import_module("wmo.cli.model_optimize")
    first_backend = _FakeBackend()
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda: first_backend)
    monkeypatch.setattr(command, "_current_revision", lambda: "w14m-test")
    runner = CliRunner()

    first = runner.invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
            "--yes",
        ],
    )

    assert first.exit_code == 0, first.output
    assert "registered model alias" in first.output
    assert first_backend.open_resume_paths == [None]
    second_backend = _FakeBackend()
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda: second_backend)

    second = runner.invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
        ],
    )

    assert second.exit_code == 0, second.output
    assert "already registered" in second.output
    assert second_backend.open_resume_paths == []


def test_cli_yes_does_not_bypass_the_unsupported_budget_estimate_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing --yes cannot turn an unpriceable maximum-cost Tinker plan into a dispatch."""
    configured = _configured_project(tmp_path, _spec(maximum_cost_usd=1.0))
    command = importlib.import_module("wmo.cli.model_optimize")
    backend = _FakeBackend(conservative_cost_per_batch=None)
    monkeypatch.setattr(command, "_compose_tinker_backend", lambda: backend)
    monkeypatch.setattr(command, "_current_revision", lambda: "w14m-test")

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "model",
            configured.store.paths.project_id,
            "--root",
            str(configured.store.paths.root),
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "nosupportedconservativecostestimate" in "".join(
        character
        for character in result.output
        if not character.isspace() and character not in "│┃"
    )
    assert backend.open_resume_paths == []
