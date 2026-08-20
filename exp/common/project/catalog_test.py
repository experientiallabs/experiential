"""Tests for the credential-free Project model-catalog artifact seam."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.models import BillingSource, ModelCapabilities, ModelSnapshot
from exp.common.project import ProjectStore
from exp.common.project.catalog import (
    ProjectCatalogModel,
    ProjectModelCatalog,
    load_project_model_catalog,
    persist_project_model_catalog,
)

_CREATED_AT = datetime(2026, 8, 17, tzinfo=UTC)


def _model(alias: str) -> ProjectCatalogModel:
    """Build one internally consistent secret-free catalog entry."""
    capabilities = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=16_384,
        maximum_output_tokens=2_048,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
    )
    return ProjectCatalogModel(
        alias=alias,
        model=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="openai",
            model_id=f"model-{alias}",
            capabilities_sha256=capabilities.identity_sha256(),
            connection_sha256="b" * 64,
        ),
        capabilities=capabilities,
    )


def test_project_catalog_persists_and_replays_as_one_exact_artifact(tmp_path: Path) -> None:
    """The project-scoped snapshot has a stable ID and never reads ambient models.toml."""
    store = ProjectStore(tmp_path / ".exp", "project-1")
    catalog = ProjectModelCatalog(
        project_id="project-1",
        models=(_model("baseline"), _model("candidate")),
    )
    ambient = store.model_catalog_path
    ambient.parent.mkdir(parents=True)
    ambient.write_text("api_key_env = 'DO_NOT_READ'\n", encoding="utf-8")

    first = persist_project_model_catalog(
        store.artifacts,
        catalog,
        created_at=_CREATED_AT,
        code_revision="producer-revision",
    )
    ambient.write_text("changed = true\n", encoding="utf-8")
    replay = persist_project_model_catalog(
        store.artifacts,
        catalog,
        created_at=_CREATED_AT,
        code_revision="producer-revision",
    )

    assert replay == first
    assert load_project_model_catalog(store.artifacts, first) == catalog
    assert store.artifacts.list_ids() == (first.artifact_id,)


def test_project_catalog_requires_sorted_unique_models_and_matching_capability_digest() -> None:
    """Alias order and immutable capability identities fail closed before persistence."""
    baseline = _model("baseline")
    candidate = _model("candidate")
    with pytest.raises(ValidationError, match="sorted by alias"):
        ProjectModelCatalog(
            project_id="project-1",
            models=(candidate, baseline),
        )
    with pytest.raises(ValidationError, match="must not repeat"):
        ProjectModelCatalog(
            project_id="project-1",
            models=(baseline, baseline),
        )
    with pytest.raises(ValidationError, match="capability digest"):
        ProjectCatalogModel(
            alias="baseline",
            model=baseline.model.model_copy(update={"capabilities_sha256": "c" * 64}),
            capabilities=baseline.capabilities,
        )


def test_project_catalog_has_no_credential_or_connection_reference_fields() -> None:
    """The portable catalog carries model identity, capabilities, and no secret handle seam."""
    assert set(ProjectCatalogModel.model_fields) == {"alias", "model", "capabilities"}
    assert set(ProjectModelCatalog.model_fields) == {
        "schema_version",
        "project_id",
        "models",
    }
