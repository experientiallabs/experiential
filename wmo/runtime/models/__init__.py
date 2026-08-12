"""Explicit local model resolution and focused provider adapters."""

from wmo.runtime.models.preflight import CapabilityRequirement, ModelCapabilityError
from wmo.runtime.models.registry import (
    ModelConnectionError,
    ResolvedModel,
    RuntimeModelCatalog,
)
from wmo.runtime.models.roles import (
    DEFAULT_BUILD_WORKFLOW,
    MissingModelRolesError,
    ModelRole,
    ModelRolePreflightResult,
    ModelRoleWorkflow,
    preflight_model_roles,
    required_model_roles,
)

__all__ = [
    "CapabilityRequirement",
    "DEFAULT_BUILD_WORKFLOW",
    "MissingModelRolesError",
    "ModelCapabilityError",
    "ModelConnectionError",
    "ModelRole",
    "ModelRolePreflightResult",
    "ModelRoleWorkflow",
    "ResolvedModel",
    "RuntimeModelCatalog",
    "preflight_model_roles",
    "required_model_roles",
]
