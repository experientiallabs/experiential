"""Model-role validation and interactive-or-noninteractive build preflight service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from wmo.common.models import ModelCatalog
from wmo.runtime.models.preflight import CapabilityRequirement
from wmo.runtime.models.registry import ResolvedModel, RuntimeModelCatalog


class ModelRole(StrEnum):
    """Named model purposes in the local catalog."""

    CANDIDATES = "candidates"
    WORLD_MODEL = "world_model"
    JUDGE = "judge"
    RUBRIC_PROPOSER = "rubric_proposer"
    EMBEDDER = "embedder"
    TEACHER = "teacher"


class ModelRoleWorkflow(StrEnum):
    """Named workflows whose model roles must be configured before execution."""

    ROUTER_BUILD = "router_build"
    JUDGING = "judging"
    RUBRIC_PROPOSAL = "rubric_proposal"
    SFT_PRODUCTION_TRACES = "sft_production_traces"
    SFT_TEACHER_ROLLOUTS = "sft_teacher_rollouts"


_WORKFLOW_REQUIRED_ROLES: Mapping[ModelRoleWorkflow, tuple[ModelRole, ...]] = {
    ModelRoleWorkflow.ROUTER_BUILD: (
        ModelRole.CANDIDATES,
        ModelRole.WORLD_MODEL,
        ModelRole.JUDGE,
        ModelRole.EMBEDDER,
    ),
    ModelRoleWorkflow.JUDGING: (ModelRole.JUDGE,),
    ModelRoleWorkflow.RUBRIC_PROPOSAL: (ModelRole.RUBRIC_PROPOSER,),
    ModelRoleWorkflow.SFT_PRODUCTION_TRACES: (),
    ModelRoleWorkflow.SFT_TEACHER_ROLLOUTS: (ModelRole.TEACHER,),
}

DEFAULT_BUILD_WORKFLOW = ModelRoleWorkflow.ROUTER_BUILD
DEFAULT_BUILD_REQUIRED_ROLES = _WORKFLOW_REQUIRED_ROLES[DEFAULT_BUILD_WORKFLOW]


class MissingModelRolesError(ValueError):
    """The catalog is missing one or more roles required by the requested workflow."""

    def __init__(self, missing_roles: Sequence[ModelRole]) -> None:
        self.missing_roles = tuple(missing_roles)
        rendered = ", ".join(role.value for role in self.missing_roles)
        super().__init__(
            f"missing model roles: {rendered}. Configure them in .wmo/models.toml or run "
            "wmo build interactively."
        )


class ModelRoleConfigurator(Protocol):
    """Prompts a local user for missing role assignments and returns an updated catalog."""

    def configure(
        self,
        catalog: ModelCatalog,
        missing_roles: tuple[ModelRole, ...],
    ) -> ModelCatalog:
        """Return catalog metadata after collecting exactly the named missing roles.

        Args:
            catalog: Current validated catalog before interactive updates.
            missing_roles: Exact workflow-scoped assignments to collect from the local user.

        Returns:
            A validated catalog containing the requested role assignments.
        """


@dataclass(frozen=True)
class ModelRolePreflightResult:
    """Resolved runtime clients grouped by the catalog role that selected them."""

    catalog: ModelCatalog
    models: Mapping[ModelRole, tuple[ResolvedModel, ...]]


def missing_model_roles(
    catalog: ModelCatalog,
    *,
    workflow: ModelRoleWorkflow = DEFAULT_BUILD_WORKFLOW,
) -> tuple[ModelRole, ...]:
    """Return the requested workflow's unassigned roles in deterministic order.

    Args:
        catalog: Catalog whose assignments are being checked.
        workflow: Product workflow about to use model roles.

    Returns:
        Required roles that do not yet name a model alias.
    """
    missing: list[ModelRole] = []
    for role in required_model_roles(workflow):
        if role is ModelRole.CANDIDATES:
            if not catalog.roles.candidates:
                missing.append(role)
        elif getattr(catalog.roles, role.value) is None:
            missing.append(role)
    return tuple(missing)


def preflight_model_roles(
    catalog: ModelCatalog,
    resolver: RuntimeModelCatalog,
    *,
    workflow: ModelRoleWorkflow = DEFAULT_BUILD_WORKFLOW,
    requirements: Mapping[ModelRole, CapabilityRequirement] | None = None,
    non_interactive: bool = True,
    configurator: ModelRoleConfigurator | None = None,
) -> ModelRolePreflightResult:
    """Validate roles, optionally prompt through a caller-owned UI, and resolve local clients.

    Args:
        catalog: Initial local model catalog metadata.
        resolver: Explicit runtime catalog constructor for the same catalog state.
        workflow: Product workflow whose roles must be ready.
        requirements: Optional capability constraints by role.
        non_interactive: When true, report every missing role without guessing.
        configurator: Caller-owned interactive prompt implementation when interaction is allowed.

    Returns:
        The effective catalog and resolved models grouped by role.

    Raises:
        MissingModelRolesError: Required roles are absent after optional interactive collection.
        ValueError: Interactive mode has no caller-provided prompt implementation.
    """
    effective_catalog = catalog
    required_roles = required_model_roles(workflow)
    missing = missing_model_roles(effective_catalog, workflow=workflow)
    if missing and not non_interactive:
        if configurator is None:
            raise ValueError("interactive model-role preflight needs a ModelRoleConfigurator")
        effective_catalog = configurator.configure(effective_catalog, missing)
        missing = missing_model_roles(effective_catalog, workflow=workflow)
    if missing:
        raise MissingModelRolesError(missing)
    if effective_catalog != catalog:
        resolver = resolver.with_catalog(effective_catalog)
    constraints = requirements or {}
    resolved: dict[ModelRole, tuple[ResolvedModel, ...]] = {}
    for role in required_roles:
        aliases = _role_aliases(effective_catalog, role)
        requirement = constraints.get(
            role,
            CapabilityRequirement(requires_embeddings=role is ModelRole.EMBEDDER),
        )
        resolved[role] = tuple(resolver.preflight(alias, requirement) for alias in aliases)
    return ModelRolePreflightResult(catalog=effective_catalog, models=resolved)


def required_model_roles(workflow: ModelRoleWorkflow) -> tuple[ModelRole, ...]:
    """Return the exact catalog roles that one product workflow consumes.

    Args:
        workflow: Product workflow about to use model roles.

    Returns:
        Ordered role assignments required before that workflow can run.
    """
    return _WORKFLOW_REQUIRED_ROLES[workflow]


def _role_aliases(catalog: ModelCatalog, role: ModelRole) -> tuple[str, ...]:
    """Return known role assignments after missing-role validation has completed."""
    if role is ModelRole.CANDIDATES:
        return catalog.roles.candidates
    alias = getattr(catalog.roles, role.value)
    if not isinstance(alias, str):
        raise MissingModelRolesError((role,))
    return (alias,)
