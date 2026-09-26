"""Virtual-key authentication callback for the native data plane."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from typing import Protocol

from exp.runtime.gateway.native_accounting import authority_error
from exp.runtime.gateway.native_components import NativeGatewayComponents


class _HasComponents(Protocol):
    _components: NativeGatewayComponents


_CHAT_PREFLIGHT = threading.local()


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
        _clear_chat_preflight()
        data = json.loads(argument)
        try:
            store = self._components.store
            preflight_authenticate = getattr(store, "authenticate_key_for_preflight", None)
            if callable(preflight_authenticate):
                preflight_authenticate(raw_key=data["raw_key"])
            else:
                store.authenticate_key(raw_key=data["raw_key"])
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise authority_error(exc) from exc
        return "{}"

    def authenticate_for_chat_admission(self: _HasComponents, argument: str) -> str:
        """Authenticate a Chat body gate and retain its exact key identity for admission.

        Rust calls this method and ``admit`` sequentially on the same bounded
        bridge worker. Admission revalidates the current key row and key
        fingerprint by its exact identity, so it avoids only the prefix scan.
        """
        _clear_chat_preflight()
        data = json.loads(argument)
        raw_key = str(data["raw_key"])
        try:
            store = self._components.store
            preflight_authenticate = getattr(store, "authenticate_key_for_preflight", None)
            if callable(preflight_authenticate):
                authenticated = preflight_authenticate(raw_key=raw_key)
                if (
                    isinstance(authenticated, tuple)
                    and len(authenticated) == 3
                    and all(isinstance(value, str) for value in authenticated)
                ):
                    _CHAT_PREFLIGHT.value = (
                        id(self),
                        hashlib.sha256(raw_key.encode("utf-8")).digest(),
                        authenticated,
                    )
            else:
                store.authenticate_key(raw_key=raw_key)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise authority_error(exc) from exc
        return "{}"

    def _take_chat_admission_preflight(
        self: _HasComponents,
        raw_key: str,
    ) -> tuple[str, str, str] | None:
        """Consume the key proof minted by this worker for the same raw key."""
        value = getattr(_CHAT_PREFLIGHT, "value", None)
        _clear_chat_preflight()
        if not isinstance(value, tuple) or len(value) != 3 or value[0] != id(self):
            return None
        digest = hashlib.sha256(raw_key.encode("utf-8")).digest()
        if not hmac.compare_digest(digest, value[1]):
            return None
        identity = value[2]
        if not isinstance(identity, tuple) or len(identity) != 3:
            return None
        if not all(isinstance(item, str) for item in identity):
            return None
        return identity


def _clear_chat_preflight() -> None:
    """Drop any request-local auth hint before starting a different bridge call."""
    if hasattr(_CHAT_PREFLIGHT, "value"):
        del _CHAT_PREFLIGHT.value
