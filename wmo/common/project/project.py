"""Project configuration contract and TOML loading for the new local layout."""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w
from pydantic import Field

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    SecretBoundaryError,
    assert_secret_free,
)
from wmo.common.core.files import write_text_atomic


class ProjectConfigError(ValueError):
    """A project configuration file was absent, malformed, or violated its local contract."""


class AgentConfiguration(ContractModel):
    """Optional explicit factory for a customer-provided injectable agent runtime."""

    factory: str = Field(min_length=1, max_length=512)


class ProjectConfig(ContractModel):
    """Project-local configuration that names no provider credentials or secret references."""

    schema_version: int = Field(default=1, ge=1)
    project_id: ArtifactId
    agent: AgentConfiguration | None = None
    model_optimization_config: ArtifactInput | None = None
    redacted_field_names: tuple[str, ...] = ()


def load_project_config(path: Path) -> ProjectConfig:
    """Load and validate one project TOML file.

    Args:
        path: Path to ``project.toml``.

    Returns:
        The immutable typed configuration.

    Raises:
        ProjectConfigError: The file is missing, malformed, or does not satisfy the contract.
    """
    try:
        with path.open("rb") as handle:
            raw_config = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ProjectConfigError(f"project configuration does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProjectConfigError(f"project configuration is invalid TOML: {path}") from exc
    try:
        config = ProjectConfig.model_validate(raw_config)
        assert_secret_free(config)
    except (SecretBoundaryError, ValueError) as exc:
        raise ProjectConfigError(f"project configuration is invalid: {exc}") from exc
    return config


def write_project_config(path: Path, config: ProjectConfig) -> None:
    """Atomically materialize a typed project configuration.

    Args:
        path: Destination ``project.toml`` path.
        config: Validated project configuration.

    Raises:
        ProjectConfigError: The configuration violates the no-secret project boundary.
    """
    try:
        assert_secret_free(config)
    except SecretBoundaryError as exc:
        raise ProjectConfigError(f"project configuration is invalid: {exc}") from exc
    payload = tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    write_text_atomic(path, payload)
