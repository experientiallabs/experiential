"""Tests for project-local settings under .wmh/settings.toml."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.config.settings import (
    ModelRole,
    ModelsSettings,
    ProjectSettings,
    ensure_telemetry_anonymous_id,
    load_settings,
    save_settings,
    set_telemetry_enabled,
    settings_path,
)


def test_missing_settings_defaults_to_telemetry_enabled(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / ".wmh")
    assert settings.telemetry.enabled is True
    assert settings.telemetry.anonymous_id is None


def test_save_then_load_settings_round_trips(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    save_settings(ProjectSettings(), root)
    loaded = load_settings(root)
    assert loaded.telemetry.enabled is True


def test_set_telemetry_enabled_writes_project_settings(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    set_telemetry_enabled(False, root)
    assert load_settings(root).telemetry.enabled is False
    assert "enabled = false" in settings_path(root).read_text(encoding="utf-8")


def test_ensure_telemetry_anonymous_id_persists_value(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    first = ensure_telemetry_anonymous_id(root)
    second = ensure_telemetry_anonymous_id(root)
    assert first == second
    assert len(first) == 32


def test_load_corrupt_settings_raises_friendly_error(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    root.mkdir()
    settings_path(root).write_text("not = = toml", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_settings(root)


def test_model_roles_round_trip_through_toml(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    settings = ProjectSettings(
        models=ModelsSettings(
            worker=ModelRole(provider="azure", model="gpt-5.4", endpoint="https://x.example/v1"),
            judge=ModelRole(
                provider="bedrock", model="us.anthropic.claude-opus-4-8", region="us-east-2"
            ),
        )
    )
    save_settings(settings, root)
    loaded = load_settings(root)
    assert loaded.models.worker is not None
    assert loaded.models.worker.model == "gpt-5.4"
    assert loaded.models.judge is not None
    assert loaded.models.judge.region == "us-east-2"
    assert "[models.worker]" in settings_path(root).read_text(encoding="utf-8")


def test_resolve_falls_back_to_worker_for_unset_roles() -> None:
    worker = ModelRole(provider="azure", model="gpt-5.4")
    models = ModelsSettings(worker=worker)
    assert models.resolve("judge") is worker
    assert models.resolve("summary") is worker
    assert models.resolve("worker") is worker
    assert ModelsSettings().resolve("judge") is None


def test_resolve_prefers_the_configured_role_over_worker() -> None:
    models = ModelsSettings(
        worker=ModelRole(provider="azure", model="gpt-5.4"),
        summary=ModelRole(provider="azure", model="gpt-5.4-mini"),
    )
    resolved = models.resolve("summary")
    assert resolved is not None
    assert resolved.model == "gpt-5.4-mini"


def test_meta_role_resolves_when_configured_and_never_falls_back_to_worker() -> None:
    """The proposer role is opt-in: unset meta means the caller's default, not the worker.

    The worker is picked for scenario-generation quality, not the proposer's
    long-context/long-output needs, so it must not leak into delta proposal silently.
    """
    worker = ModelRole(provider="azure", model="gpt-5.4")
    meta = ModelRole(provider="azure", model="gpt-5.5", deployment="gpt-5-5")
    assert ModelsSettings(worker=worker).resolve("meta") is None
    assert ModelsSettings(worker=worker, meta=meta).resolve("meta") is meta


def test_meta_role_round_trips_through_toml(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    settings = ProjectSettings(
        models=ModelsSettings(
            meta=ModelRole(
                provider="azure",
                model="gpt-5.5",
                endpoint="https://x.example",
                deployment="gpt-5-5",
            )
        )
    )
    save_settings(settings, root)
    loaded = load_settings(root)
    assert loaded.models.meta is not None
    assert loaded.models.meta.model == "gpt-5.5"
    assert loaded.models.meta.deployment == "gpt-5-5"
    assert "[models.meta]" in settings_path(root).read_text(encoding="utf-8")


def test_resolve_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown model role"):
        ModelsSettings().resolve("teacher")
