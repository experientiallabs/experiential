"""Explicit local model resolution and focused provider adapters."""

from wmo.runtime.models.preflight import CapabilityRequirement, ModelCapabilityError
from wmo.runtime.models.registry import (
    CatalogRoleName,
    ModelConnectionError,
    ResolvedModel,
    RuntimeModelCatalog,
)

__all__ = [
    "CapabilityRequirement",
    "CatalogRoleName",
    "ModelCapabilityError",
    "ModelConnectionError",
    "ResolvedModel",
    "RuntimeModelCatalog",
]
