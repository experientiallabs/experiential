"""Tests for model-role preflight behavior consumed by the future build command."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from wmo.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
)
from wmo.runtime.models.preflight import CapabilityRequirement
from wmo.runtime.models.registry import RuntimeModelCatalog
from wmo.runtime.models.roles import (
    DEFAULT_BUILD_REQUIRED_ROLES,
    MissingModelRolesError,
    ModelRole,
    ModelRoleWorkflow,
    preflight_model_roles,
    required_model_roles,
)

_CAPABILITIES = ModelCapabilities(
    supports_tools=True,
    supports_embeddings=True,
    context_window_tokens=128_000,
    maximum_output_tokens=16_000,
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
            "candidate": ModelRecord(
                connection="openai", model="candidate-model", capabilities=_CAPABILITIES
            ),
            "world": ModelRecord(
                connection="openai", model="world-model", capabilities=_CAPABILITIES
            ),
            "judge": ModelRecord(
                connection="openai", model="judge-model", capabilities=_CAPABILITIES
            ),
            "embedder": ModelRecord(
                connection="openai", model="embedding-model", capabilities=_CAPABILITIES
            ),
            "teacher": ModelRecord(
                connection="openai", model="teacher-model", capabilities=_CAPABILITIES
            ),
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
        "missing model roles: candidates, world_model, judge, embedder. "
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
            embedder="embedder",
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

    assert configurator.seen == (
        ModelRole.CANDIDATES,
        ModelRole.WORLD_MODEL,
        ModelRole.JUDGE,
        ModelRole.EMBEDDER,
    )
    assert result.catalog == configured
    assert tuple(result.models) == DEFAULT_BUILD_REQUIRED_ROLES
    assert result.models[ModelRole.CANDIDATES][0].snapshot.model_id == "candidate-model"
    assert result.models[ModelRole.EMBEDDER][0].embedding_client is not None


def test_router_build_does_not_require_optional_rubric_or_teacher_roles() -> None:
    """The router build has every role it uses even when optional future workflows are unset."""
    catalog = _catalog(
        ModelRoles(
            candidates=("candidate",),
            world_model="world",
            judge="judge",
            embedder="embedder",
        )
    )
    resolver = RuntimeModelCatalog(catalog, environment={"FIXTURE_API_KEY": "fixture-key"})

    result = preflight_model_roles(catalog, resolver)

    assert tuple(result.models) == DEFAULT_BUILD_REQUIRED_ROLES
    assert ModelRole.RUBRIC_PROPOSER not in result.models
    assert ModelRole.TEACHER not in result.models


def test_sft_and_judging_preflight_request_only_their_own_roles() -> None:
    """SFT and judging retain exact independent role requirements after router setup narrows."""
    sft_catalog = _catalog(ModelRoles(teacher="teacher"))
    judging_catalog = _catalog(ModelRoles(judge="judge"))
    sft_resolver = RuntimeModelCatalog(sft_catalog, environment={"FIXTURE_API_KEY": "fixture-key"})
    judging_resolver = RuntimeModelCatalog(
        judging_catalog,
        environment={"FIXTURE_API_KEY": "fixture-key"},
    )

    sft = preflight_model_roles(
        sft_catalog,
        sft_resolver,
        workflow=ModelRoleWorkflow.SFT,
    )
    judging = preflight_model_roles(
        judging_catalog,
        judging_resolver,
        workflow=ModelRoleWorkflow.JUDGING,
    )

    assert tuple(sft.models) == (ModelRole.TEACHER,)
    assert tuple(judging.models) == (ModelRole.JUDGE,)
    assert required_model_roles(ModelRoleWorkflow.RUBRIC_PROPOSAL) == (ModelRole.RUBRIC_PROPOSER,)
    with pytest.raises(MissingModelRolesError, match="teacher"):
        preflight_model_roles(
            judging_catalog,
            judging_resolver,
            workflow=ModelRoleWorkflow.SFT,
        )
