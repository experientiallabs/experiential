"""Tests for the public surface enum and its re-export."""

from __future__ import annotations

from exp.runtime.gateway import contracts
from exp.runtime.gateway.surfaces import GatewayApiSurface


def test_contracts_re_exports_the_same_enum() -> None:
    """Every existing ``contracts.GatewayApiSurface`` import names this enum."""
    assert contracts.GatewayApiSurface is GatewayApiSurface


def test_the_video_surface_is_named_video() -> None:
    """The ledger stores the surface value; video is ``video``."""
    assert GatewayApiSurface.VIDEO.value == "video"
