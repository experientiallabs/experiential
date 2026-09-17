"""Resident run ownership shared by local and remotely hosted learning services."""

from __future__ import annotations

from typing import Literal, Protocol

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.optimize.claas.training_contracts import (
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingCheckpoint,
    TrainingResult,
    TrainingSession,
)

RunMode = Literal["burst", "run"]


class LearnerRuntime(TrainingSession, Protocol):
    """One run's resident optimizer and optional student generation engine."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Sample original student tokens; burst-only runtimes reject generation."""
        ...

    async def train(self, batch: TrainingBatch) -> TrainingResult:
        """Persist an update before acknowledging it; retrying its ID is idempotent."""
        ...


class LearnerRuntimeFactory(Protocol):
    """Allocate once per run and keep engine lifecycle inside the selected runtime."""

    async def open(
        self,
        spec: ClaasTrainingSpec,
        resume: TrainingCheckpoint | None = None,
        *,
        mode: RunMode,
    ) -> LearnerRuntime:
        """Restore committed state and initialize the engines needed by this run."""
        ...
