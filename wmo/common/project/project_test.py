"""Tests for project TOML loading and immutable initialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.project import (
    AgentConfiguration,
    ProjectConfig,
    ProjectConfigError,
    ProjectProviderFreeStage,
    ProjectTracePreparationSettings,
    load_project_config,
    write_project_config,
)

_DIGEST = "a" * 64


def test_trace_first_config_omits_late_setup_without_changing_existing_defaults(
    tmp_path: Path,
) -> None:
    """Trace-first Projects omit model-era setup while ordinary local configs keep defaults."""
    existing_default = ProjectConfig(project_id="existing-project")
    assert existing_default.retrieval is not None
    assert existing_default.budgets is not None
    path = tmp_path / "project.toml"
    trace_first = ProjectConfig(
        project_id="trace-first-project",
        trace_preparation=ProjectTracePreparationSettings(source_kind="otlp"),
        retrieval=None,
        budgets=None,
    )

    write_project_config(path, trace_first)

    assert load_project_config(path) == trace_first
    payload = path.read_text(encoding="utf-8")
    assert "retrieval" not in payload
    assert "budgets" not in payload


def test_provider_free_contract_keeps_settings_and_stage_ownership_minimal() -> None:
    """Project owns preparation settings while the selected stage stores only exact pointers."""
    trace = ArtifactInput(artifact_id="trace-dataset", sha256=_DIGEST)
    task = ArtifactInput(artifact_id="task-set", sha256=_DIGEST)
    stage = ProjectProviderFreeStage(trace_dataset=trace, task_set=task)

    assert set(ProjectTracePreparationSettings.model_fields) == {
        "source_kind",
        "fit_task_budget",
        "held_out_task_budget",
        "descriptor_dimensions",
    }
    assert stage.model_dump(mode="json") == {
        "schema_version": 1,
        "trace_dataset": trace.model_dump(mode="json"),
        "task_set": task.model_dump(mode="json"),
    }
    with pytest.raises(ValueError, match="trace preparation settings"):
        ProjectConfig(project_id="missing-settings", provider_free_stage=stage)


def test_project_config_round_trip_preserves_safe_local_metadata(tmp_path: Path) -> None:
    """Project TOML contains customer wiring metadata but no provider credential references."""
    path = tmp_path / "project.toml"
    config = ProjectConfig(
        project_id="support-project",
        agent=AgentConfiguration(factory="acme_support.wmo:create_agent_runtime"),
        redacted_field_names=("email", "phone"),
    )

    write_project_config(path, config)

    assert load_project_config(path) == config
    assert "code_revision" not in path.read_text(encoding="utf-8")


def test_custom_agent_revision_round_trip_is_explicit(tmp_path: Path) -> None:
    """Persist an explicit custom-agent revision without changing revisionless serialization.

    Args:
        tmp_path: Isolated project configuration directory.
    """
    path = tmp_path / "project.toml"
    config = ProjectConfig(
        project_id="support-project",
        agent=AgentConfiguration(
            factory="acme_support.wmo:create_agent_runtime",
            code_revision="agent-release-42",
        ),
    )

    write_project_config(path, config)

    assert load_project_config(path) == config
    assert 'code_revision = "agent-release-42"' in path.read_text(encoding="utf-8")


def test_project_config_rejects_secret_reference(tmp_path: Path) -> None:
    """Project TOML cannot become an alternate home for model credential configuration."""
    path = tmp_path / "project.toml"
    path.write_text(
        """
project_id = "support-project"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ProjectConfigError, match="api_key_env"):
        load_project_config(path)
