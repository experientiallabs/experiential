"""Bridge-callable relays for the optional /v1/batches lane.

The native routes call these methods by name; a composition without a batch
plane answers every one with a uniform not-enabled envelope. The relay only
forwards strings: all batch semantics live in the batch package's
``BatchControlPlane``.
"""

from __future__ import annotations

import json
from typing import Protocol


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
