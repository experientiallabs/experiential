"""Virtual-key authentication callback for the native data plane."""

from __future__ import annotations

import json
from typing import Protocol

from exp.runtime.gateway.native_accounting import authority_error
from exp.runtime.gateway.native_components import NativeGatewayComponents


class _HasComponents(Protocol):
    _components: NativeGatewayComponents


def authenticate_raw_key(components: NativeGatewayComponents, raw_key: str) -> None:
    """Authenticate one virtual key and map storage failures to public errors."""
    try:
        components.store.authenticate_key(raw_key=raw_key)
    except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
        raise authority_error(exc) from exc


class NativeAuthenticationMixin:
    """Authenticate virtual keys before request-body decoding."""

    def authenticate(self: _HasComponents, argument: str) -> str:
        """Authenticate one virtual key before the data plane decodes the body.

        Args:
            argument: JSON object with ``raw_key``.

        Returns:
            An empty JSON object on success.

        Raises:
            NativeBridgeError: The key is invalid, expired, or revoked.
        """
        data = json.loads(argument)
        authenticate_raw_key(self._components, data["raw_key"])
        return "{}"
