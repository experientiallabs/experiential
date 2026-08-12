"""Explicit local model resolution and focused provider adapters."""

from wmo.runtime.models.preflight import CapabilityRequirement, ModelCapabilityError
from wmo.runtime.models.registry import (
    ModelConnectionError,
    ResolvedModel,
    RuntimeModelCatalog,
)
from wmo.runtime.models.roles import (
    MissingModelRolesError,
    ModelRole,
    ModelRolePreflightResult,
    preflight_model_roles,
)

__all__ = [
    "CapabilityRequirement",
    "MissingModelRolesError",
    "ModelCapabilityError",
    "ModelConnectionError",
    "ModelRole",
    "ModelRolePreflightResult",
    "ResolvedModel",
    "RuntimeModelCatalog",
    "preflight_model_roles",
]
