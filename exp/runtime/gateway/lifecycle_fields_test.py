"""Lifecycle field extraction preserves error identity and alias validation."""

import pytest

from exp.runtime.gateway import lifecycle, lifecycle_fields
from exp.runtime.gateway.management import GatewayAliasView


def test_lifecycle_field_exports_retain_the_same_classes_and_callables() -> None:
    """Existing fallback imports share the exact lifecycle validation authority."""
    assert lifecycle.GatewayLifecycleError is lifecycle_fields.GatewayLifecycleError
    assert lifecycle._required is lifecycle_fields.required
    assert lifecycle._required_revision is lifecycle_fields.required_revision
    alias = GatewayAliasView(alias_id="alias", alias_name="alias", active=True)
    with pytest.raises(lifecycle.GatewayLifecycleError, match="revision ID"):
        lifecycle_fields.required_revision(alias)
