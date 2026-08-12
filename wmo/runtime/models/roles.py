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


DEFAULT_BUILD_REQUIRED_ROLES: tuple[ModelRole, ...] = (
    ModelRole.CANDIDATES,
    ModelRole.WORLD_MODEL,
    ModelRole.JUDGE,
    ModelRole.RUBRIC_PROPOSER,
    ModelRole.EMBEDDER,
    ModelRole.TEACHER,
)


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
        """Return catalog metadata after collecting exactly the named missing roles."""


@dataclass(frozen=True)
class ModelRolePreflightResult:
    """Resolved runtime clients grouped by the catalog role that selected them."""

    catalog: ModelCatalog
    models: Mapping[ModelRole, tuple[ResolvedModel, ...]]


def missing_model_roles(
    catalog: ModelCatalog,
    required_roles: Sequence[ModelRole] = DEFAULT_BUILD_REQUIRED_ROLES,
) -> tuple[ModelRole, ...]:
    """Return required roles without an assigned alias in deterministic caller order."""
    missing: list[ModelRole] = []
    for role in required_roles:
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
    required_roles: Sequence[ModelRole] = DEFAULT_BUILD_REQUIRED_ROLES,
    requirements: Mapping[ModelRole, CapabilityRequirement] | None = None,
    non_interactive: bool = True,
    configurator: ModelRoleConfigurator | None = None,
) -> ModelRolePreflightResult:
    """Validate roles, optionally prompt through a caller-owned UI, and resolve local clients.

    Args:
        catalog: Initial local model catalog metadata.
        resolver: Explicit runtime catalog constructor for the same catalog state.
        required_roles: Roles required by this caller's workflow.
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
    missing = missing_model_roles(effective_catalog, required_roles)
    if missing and not non_interactive:
        if configurator is None:
            raise ValueError("interactive model-role preflight needs a ModelRoleConfigurator")
        effective_catalog = configurator.configure(effective_catalog, missing)
        missing = missing_model_roles(effective_catalog, required_roles)
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


def _role_aliases(catalog: ModelCatalog, role: ModelRole) -> tuple[str, ...]:
    """Return known role assignments after missing-role validation has completed."""
    if role is ModelRole.CANDIDATES:
        return catalog.roles.candidates
    alias = getattr(catalog.roles, role.value)
    if not isinstance(alias, str):
        raise MissingModelRolesError((role,))
    return (alias,)
