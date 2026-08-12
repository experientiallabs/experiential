"""Sampling-only adapter for an already-completed Tinker trained-model handle."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from wmo.common.core.artifacts import ContractModel
from wmo.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    Usage,
)


class TinkerSample(ContractModel):
    """One completed sampled Tinker turn, independent of training lifecycle state."""

    output: AssistantAction
    usage: Usage | None = None
    served_model_id: str | None = None


@runtime_checkable
class TinkerSampler(Protocol):
    """The narrow completed-handle sampling operation WMO needs at runtime."""

    def sample(self, request: ModelRequest) -> TinkerSample:
        """Sample one complete assistant action from the completed trained handle.

        Args:
            request: Typed WMO request to render for the trained model.

        Returns:
            Parsed output and any observed token accounting.
        """


class TinkerSamplingClient:
    """Adapts a completed Tinker sampler to the common non-streaming model protocol."""

    def __init__(self, *, model: ModelSnapshot, sampler: TinkerSampler) -> None:
        """Bind one completed trained-model identity to its sampling handle.

        Args:
            model: Catalog identity, typically with a ``tinker://`` model handle.
            sampler: Already-created sampling handle. This adapter never creates, trains, saves,
                promotes, or deploys a Tinker model.
        """
        self._model = model
        self._sampler = sampler

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Sample one action and attach local observed latency to the shared response."""
        started_at = time.monotonic()
        sample = self._sampler.sample(request)
        model = (
            self._model.model_copy(update={"model_id": sample.served_model_id})
            if sample.served_model_id
            else self._model
        )
        return ModelResponse(
            output=sample.output,
            model=model,
            economics=OperationEconomics(
                usage=sample.usage,
                latency_seconds=NumericMeasurement(
                    value=time.monotonic() - started_at,
                    provenance="observed",
                ),
            ),
        )
