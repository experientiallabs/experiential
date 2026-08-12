"""Tests for local model-catalog TOML loading and credential boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.models import (
    ConnectionConfig,
    ModelCatalog,
    ModelCatalogError,
    ModelRecord,
    ModelRoles,
    load_model_catalog,
    write_model_catalog,
)


def _catalog() -> ModelCatalog:
    return ModelCatalog(
        connections={
            "openrouter": ConnectionConfig(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
            )
        },
        models={
            "candidate-economy": ModelRecord(
                connection="openrouter",
                model="deepseek/deepseek-v4-flash",
            )
        },
        roles=ModelRoles(candidates=("candidate-economy",), incumbent="candidate-economy"),
    )


def test_model_catalog_round_trip_preserves_aliases_and_environment_name(tmp_path: Path) -> None:
    """Models TOML records aliases and an environment variable name, never its value."""
    path = tmp_path / "models.toml"
    catalog = _catalog()

    write_model_catalog(path, catalog)

    assert load_model_catalog(path) == catalog
    assert "OPENROUTER_API_KEY" in path.read_text(encoding="utf-8")
    assert "api_key =" not in path.read_text(encoding="utf-8")


def test_model_catalog_rejects_credential_values_and_embedded_url_credentials(
    tmp_path: Path,
) -> None:
    """The catalog permits a credential environment name but no secret value or URL credentials."""
    raw_key_path = tmp_path / "raw-key.toml"
    raw_key_path.write_text(
        """
[connections.openrouter]
provider = "openrouter"
api_key = "sk-abcdefghijklmnopqrstuvwxyz123456"

[models.candidate-economy]
connection = "openrouter"
model = "deepseek/deepseek-v4-flash"
""".strip(),
        encoding="utf-8",
    )
    embedded_credential_path = tmp_path / "embedded.toml"
    embedded_credential_path.write_text(
        """
[connections.private-world-model]
provider = "openai-compatible"
base_url = "https://user:password@models.example.com/v1"

[models.world-model]
connection = "private-world-model"
model = "private-world-model"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ModelCatalogError, match="api_key"):
        load_model_catalog(raw_key_path)
    with pytest.raises(ModelCatalogError, match="embed credentials"):
        load_model_catalog(embedded_credential_path)
