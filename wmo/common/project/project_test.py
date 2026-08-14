"""Tests for project TOML loading and immutable initialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.project import (
    AgentConfiguration,
    ProjectConfig,
    ProjectConfigError,
    load_project_config,
    write_project_config,
)


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
