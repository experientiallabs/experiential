"""Neutral completion-service contract between HTTP adapters and durable runtimes."""

from __future__ import annotations

from typing import Protocol

from wmo.common.core.artifacts import StructuredFailure
from wmo.common.models import ModelRequest
from wmo.runtime.router.runtime import RoutedModelResponse


class RouterCompletionConflictError(ValueError):
    """A caller key conflicts with a previously accepted logical interaction."""


class RouterCompletionInProgressError(RuntimeError):
    """A durable interaction is already being completed and may be retried."""


class RouterCompletionFailedError(RuntimeError):
    """A durable interaction has a stable terminal provider failure."""

    def __init__(self, failure: StructuredFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class RouterCompletionService(Protocol):
    """Durable idempotent completion boundary used by public router adapters."""

    def complete(
        self,
        request: ModelRequest,
        *,
        idempotency_key: str,
        conversation_id: str | None = None,
    ) -> RoutedModelResponse:
        """Complete or replay one durable caller-keyed interaction."""
