"""Shared generation-parameter validation helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.reasoning_compat import supported_reasoning_efforts


def effective_profile_reasoning_effort(
    profile: GatewayWireProfile,
    requested_effort: str | None,
) -> str | None:
    """Return an explicit caller effort or one wire's required default."""
    if requested_effort is not None:
        return requested_effort
    return profile.reasoning_effort if profile.reasoning_effort_required else None


def profile_reasoning_efforts(profile: GatewayWireProfile) -> tuple[str, ...]:
    """Return exact accepted efforts for one deployment wire profile."""
    if not profile.supports_reasoning or profile.reasoning_wire_format == "none":
        return ()
    return supported_reasoning_efforts(
        profile.model_id,
        profile.reasoning_wire_format,
        configured_effort=profile.reasoning_effort,
        explicit_efforts=profile.supported_reasoning_efforts or None,
    )


def require_route_numeric_parameter(
    profiles: Sequence[GatewayWireProfile],
    *,
    param: str,
    value: float | int,
    supported: Callable[[GatewayWireProfile], bool],
    minimum: Callable[[GatewayWireProfile], float | int],
    maximum: Callable[[GatewayWireProfile], float | int | None],
) -> None:
    """Require every waterfall rung to accept one exact numeric control."""
    if not all(supported(profile) for profile in profiles):
        raise ProviderParameterError(
            message=(
                f"The parameter {param!r} is not supported by this model route. "
                "Remove the field or choose a different model."
            ),
            param=param,
            code="unsupported_parameter",
        )
    route_minimum = max(minimum(profile) for profile in profiles)
    maxima = tuple(bound for profile in profiles if (bound := maximum(profile)) is not None)
    route_maximum = min(maxima) if maxima else None
    if value >= route_minimum and (route_maximum is None or value <= route_maximum):
        return
    range_text = (
        f"{route_minimum} or greater"
        if route_maximum is None
        else f"between {route_minimum} and {route_maximum}"
    )
    raise ProviderParameterError(
        message=(
            f"The value {value!r} for {param!r} is not supported by this model route. "
            f"Supported values are {range_text}."
        ),
        param=param,
        code="invalid_parameter",
    )
