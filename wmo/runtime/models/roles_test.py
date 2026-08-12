"""Tests for model-role preflight behavior consumed by the future build command."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from wmo.common.models import ConnectionConfig, ModelCatalog, ModelRecord, ModelRoles
from wmo.runtime.models.preflight import CapabilityRequirement
from wmo.runtime.models.registry import RuntimeModelCatalog
from wmo.runtime.models.roles import (
    DEFAULT_BUILD_REQUIRED_ROLES,
    MissingModelRolesError,
    ModelRole,
    preflight_model_roles,
)


def _catalog(roles: ModelRoles) -> ModelCatalog:
    """Build one all-purpose compatible catalog with caller-selected roles."""
    return ModelCatalog(
        connections={
            "openai": ConnectionConfig(
                provider="openai",
                api_key_env="FIXTURE_API_KEY",
            )
        },
        models={
            "candidate": ModelRecord(connection="openai", model="candidate-model"),
            "world": ModelRecord(connection="openai", model="world-model"),
            "judge": ModelRecord(connection="openai", model="judge-model"),
            "embedder": ModelRecord(connection="openai", model="embedding-model"),
            "teacher": ModelRecord(connection="openai", model="teacher-model"),
        },
        roles=roles,
    )


@dataclass
class _Configurator:
    """Returns prepared interactive selections and records the exact missing roles."""

    configured: ModelCatalog
    seen: tuple[ModelRole, ...] = ()

    def configure(
        self,
        catalog: ModelCatalog,
        missing_roles: tuple[ModelRole, ...],
    ) -> ModelCatalog:
        """Return the caller's prepared role selections."""
        del catalog
        self.seen = missing_roles
        return self.configured


def test_noninteractive_preflight_lists_every_missing_role_in_catalog_order() -> None:
    """Build callers get one exact actionable missing-role error without guessing."""
    catalog = _catalog(ModelRoles())
    resolver = RuntimeModelCatalog(catalog, environment={})

    with pytest.raises(MissingModelRolesError) as raised:
        preflight_model_roles(catalog, resolver)

    assert raised.value.missing_roles == DEFAULT_BUILD_REQUIRED_ROLES
    assert str(raised.value) == (
        "missing model roles: candidates, world_model, judge, rubric_proposer, embedder, teacher. "
        "Configure them in .wmo/models.toml or run wmo build interactively."
    )


def test_interactive_preflight_uses_updated_catalog_and_role_requirements() -> None:
    """Interactive selection rebuilds the resolver and checks every selected alias locally."""
    initial = _catalog(ModelRoles())
    configured = _catalog(
        ModelRoles(
            candidates=("candidate",),
            world_model="world",
            judge="judge",
            rubric_proposer="judge",
            embedder="embedder",
            teacher="teacher",
        )
    )
    configurator = _Configurator(configured)
    resolver = RuntimeModelCatalog(initial, environment={"FIXTURE_API_KEY": "fixture-key"})

    result = preflight_model_roles(
        initial,
        resolver,
        non_interactive=False,
        configurator=configurator,
        requirements={
            ModelRole.WORLD_MODEL: CapabilityRequirement(requires_tools=True),
        },
    )

    assert configurator.seen == DEFAULT_BUILD_REQUIRED_ROLES
    assert result.catalog == configured
    assert tuple(result.models) == DEFAULT_BUILD_REQUIRED_ROLES
    assert result.models[ModelRole.CANDIDATES][0].snapshot.model_id == "candidate-model"
    assert result.models[ModelRole.EMBEDDER][0].embedding_client is not None
