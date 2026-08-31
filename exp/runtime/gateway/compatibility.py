"""Versioned public-field compatibility manifests shared by the API surfaces.

Split from :mod:`exp.runtime.gateway.contracts` for the module line
budget: each public surface declares one closed manifest of explicit
field decisions here, and its decoder enforces it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.runtime.gateway.contracts import GatewayApiSurface


class CompatibilityDisposition(StrEnum):
    """How one installed public request field is handled by the gateway.

    ``IGNORED`` keeps a compatibility field valid at the public boundary while
    deliberately omitting it from provider dispatch when the normalized gateway
    response cannot preserve its result.
    """

    SUPPORTED = "supported"
    CONDITIONALLY_SUPPORTED = "conditionally_supported"
    METADATA_ONLY = "metadata_only"
    IGNORED = "ignored"
    UNSUPPORTED = "unsupported"


class CompatibilityField(ContractModel):
    """One explicit public-field decision in a versioned compatibility manifest."""

    field_path: str = Field(min_length=1, max_length=512)
    disposition: CompatibilityDisposition
    capability: str | None = Field(default=None, min_length=1, max_length=256)


class CompatibilityManifest(ContractModel):
    """Closed field-classification contract for one public API surface."""

    schema_version: int = Field(ge=1)
    surface: GatewayApiSurface
    fields: tuple[CompatibilityField, ...]

    @model_validator(mode="after")
    def _require_unique_field_paths(self) -> CompatibilityManifest:
        """Reject duplicate field decisions that could make parsing ambiguous.

        Returns:
            The validated manifest.

        Raises:
            ValueError: A field path appears more than once.
        """
        paths = tuple(field.field_path for field in self.fields)
        if len(set(paths)) != len(paths):
            raise ValueError("compatibility manifest field paths must be unique")
        return self
