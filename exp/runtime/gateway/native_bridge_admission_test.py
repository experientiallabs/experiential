# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for native control-plane admission composition."""

from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_bridge_admission import NativeAdmissionMixin


def test_native_control_plane_admission_has_one_implementation() -> None:
    assert NativeControlPlane.admit is NativeAdmissionMixin.admit
