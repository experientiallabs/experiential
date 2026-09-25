"""Capability errors preserve caller-facing fields across native surfaces."""

from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.native_bridge_errors import capability_param


def test_streaming_tool_arguments_names_the_tools_field() -> None:
    """Partial tool argument support is disclosed on tools rather than stream."""
    for surface in GatewayApiSurface:
        assert capability_param("streaming_tool_arguments", surface) == "tools"
