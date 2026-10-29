"""Required local alias fields and the shared lifecycle validation error."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exp.runtime.gateway.management import GatewayAliasView


class GatewayLifecycleError(ValueError):
    """Local gateway configuration cannot form one ready execution snapshot."""


def required_revision(alias: GatewayAliasView) -> tuple[str, str]:
    """Return required alias revision and catalog digest values."""
    return (
        required(alias.revision_id, "revision ID", alias),
        required(alias.catalog_sha256, "catalog digest", alias),
    )


def required(value: str | None, name: str, alias: GatewayAliasView) -> str:
    """Return one required active-alias field or fail with safe context."""
    if value is None:
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} is missing {name}")
    return value
