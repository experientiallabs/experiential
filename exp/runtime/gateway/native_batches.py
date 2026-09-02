"""Bridge-callable relays for the optional /v1/batches lane.

The native routes call these methods by name; a composition without a batch
plane answers every one with a uniform not-enabled envelope. The relay only
forwards strings: all batch semantics live in the batch package's
``BatchControlPlane``.
"""

from __future__ import annotations

import json
from typing import Protocol

from exp.runtime.gateway.native_accounting import NativeBridgeError
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


class _BatchPlane(Protocol):
    """The duck-typed surface the relay forwards to."""

    def batch_create(self, argument: str) -> str: ...

    def batch_retrieve(self, argument: str) -> str: ...

    def batch_list(self, argument: str) -> str: ...

    def batch_cancel(self, argument: str) -> str: ...

    def file_create(self, argument: str) -> str: ...

    def file_retrieve(self, argument: str) -> str: ...

    def file_content(self, argument: str) -> str: ...

    def is_batch_model(self, *, alias: str) -> bool: ...


class NativeBatchRelayMixin:
    """Relay the batch route methods onto an optionally injected plane."""

    _batches: _BatchPlane | None

    def _batch_pointer_error(self, *, alias: str, mapped: Exception) -> Exception | None:
        """Return the did-you-mean pointer for an authorized batch-only miss.

        Only an authenticated identity learns that a name is batch-only: an
        authentication failure (the mapped error carries status 401) keeps
        its generic envelope so an invalid key cannot enumerate the batch
        catalog. Returns None when the pointer does not apply.
        """
        if self._batches is None or not isinstance(mapped, NativeBridgeError):
            return None
        if json.loads(mapped.public_error_json)["status_code"] == 401:
            return None
        if not self._batches.is_batch_model(alias=alias):
            return None
        return NativeBridgeError(
            OpenAIProtocolError(
                status_code=404,
                code="model_requires_batch",
                message=(
                    f"The model {alias!r} is only available through the "
                    "Batch API. Submit it explicitly via /v1/batches."
                ),
                error_type="invalid_request_error",
            )
        )

    def _batches_disabled(self) -> str:
        """Render the uniform envelope for gateways without the batch lane."""
        return json.dumps(
            {
                "status": 404,
                "body": {
                    "error": {
                        "message": "batches are not enabled on this gateway",
                        "type": "invalid_request_error",
                        "code": "not_found",
                    }
                },
            }
        )

    def batch_create(self, argument: str) -> str:
        """Relay one batch submission to the optional batch plane."""
        if self._batches is None:
            return self._batches_disabled()
        return self._batches.batch_create(argument)

    def batch_retrieve(self, argument: str) -> str:
        """Relay one batch read to the optional batch plane."""
        if self._batches is None:
            return self._batches_disabled()
        return self._batches.batch_retrieve(argument)

    def batch_list(self, argument: str) -> str:
        """Relay one batch listing to the optional batch plane."""
        if self._batches is None:
            return self._batches_disabled()
        return self._batches.batch_list(argument)

    def batch_cancel(self, argument: str) -> str:
        """Relay one batch cancellation to the optional batch plane."""
        if self._batches is None:
            return self._batches_disabled()
        return self._batches.batch_cancel(argument)

    def file_create(self, argument: str) -> str:
        """Relay one batch file upload to the optional batch plane."""
        if self._batches is None:
            return self._batches_disabled()
        return self._batches.file_create(argument)

    def file_retrieve(self, argument: str) -> str:
        """Relay one batch file read to the optional batch plane."""
        if self._batches is None:
            return self._batches_disabled()
        return self._batches.file_retrieve(argument)

    def file_content(self, argument: str) -> str:
        """Relay one batch file download to the optional batch plane."""
        if self._batches is None:
            return self._batches_disabled()
        return self._batches.file_content(argument)
