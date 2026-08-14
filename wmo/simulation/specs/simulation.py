"""Immutable sparse specifications shared by concrete simulation engines.

The specification deliberately contains one common envelope and one optional settings object per
mode.  A concrete simulator validates the settings it consumes before it starts any provider work.
"""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    Sha256,
    sha256_json,
)
from wmo.common.models import EmbeddingCostReservation
from wmo.common.rollouts import SimulationMode


class WorldModelSettings(ContractModel):
    """Versioned controls for one text-only world-model simulation run.

    The settings retain the normal artifact envelope so a selected prompt and provider alias can
    be compared across immutable runs without introducing a second configuration file format.
    """

    world_model_alias: ArtifactId
    prompt_version: str = Field(min_length=1, max_length=256)
    query_embedding: EmbeddingCostReservation | None = None
    maximum_output_tokens: int = Field(default=16_000, ge=8_000)
    allow_tools: Literal[False] = False


class SandboxSettings(ContractModel):
    """Versioned executable-environment identity and per-episode wall-clock limit."""

    environment_id: ArtifactId
    environment_sha256: Sha256
    maximum_time_seconds: float = Field(default=300.0, gt=0)

    @field_validator("maximum_time_seconds")
    @classmethod
    def _require_finite_time_limit(cls, value: float) -> float:
        """Reject a wall-clock limit that cannot stop an executable episode."""
        if not math.isfinite(value):
            raise ValueError("sandbox maximum_time_seconds must be finite")
        return value


class MixedRealitySettings(ContractModel):
    """Reserved future settings that no v1 simulator is allowed to execute."""

    policy_id: ArtifactId


class SimulationSpec(ArtifactEnvelope):
    """One immutable sparse selection of evaluation cells and exactly one simulator mode.

    Args:
        simulation_id: Stable identity for the persisted simulation specification.
        evaluation_plan_id: Immutable plan whose cells this run explicitly selects.
        cell_ids: Sorted exact simulated plan cell IDs.  The simulator never expands this set.
        agent_id: Customer agent implementation selected for this run.
        mode: Concrete simulator mode chosen for every selected cell.
        world_model: Text world-model settings when ``mode`` is ``world_model``.
        sandbox: Executable environment settings when ``mode`` is ``sandbox``.
        mixed_reality: Reserved settings for an intentionally unimplemented future mode.
        seed: Pinned random seed preserved in each rollout artifact.
        maximum_steps: Strict upper bound on candidate model turns per episode.
        maximum_concurrency: Maximum number of episode workers allowed at once.
        maximum_cost_usd: Optional run-wide provider spend ceiling in US dollars.
    """

    simulation_id: ArtifactId
    evaluation_plan_id: ArtifactId
    cell_ids: tuple[ArtifactId, ...]
    agent_id: str = Field(min_length=1, max_length=256)
    mode: SimulationMode
    world_model: WorldModelSettings | None = None
    sandbox: SandboxSettings | None = None
    mixed_reality: MixedRealitySettings | None = None
    seed: int
    maximum_steps: int = Field(ge=1)
    maximum_concurrency: int = Field(default=1, ge=1)
    maximum_cost_usd: float | None = Field(default=None, gt=0)

    @field_validator("maximum_cost_usd")
    @classmethod
    def _require_finite_cost_limit(cls, value: float | None) -> float | None:
        """Reject a spend ceiling that cannot provide a finite admission boundary."""
        if value is not None and not math.isfinite(value):
            raise ValueError("simulation maximum_cost_usd must be finite")
        return value

    @field_validator("cell_ids")
    @classmethod
    def _require_sorted_unique_cells(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if not value:
            raise ValueError("a simulation specification needs at least one explicit cell ID")
        if len(set(value)) != len(value):
            raise ValueError("simulation specification cell IDs must not repeat")
        if value != tuple(sorted(value)):
            raise ValueError("simulation specification cell IDs must be sorted")
        return value

    @model_validator(mode="after")
    def _require_selected_mode_settings(self) -> Self:
        selected = {
            SimulationMode.WORLD_MODEL: self.world_model,
            SimulationMode.SANDBOX: self.sandbox,
            SimulationMode.MIXED_REALITY: self.mixed_reality,
        }
        if selected[self.mode] is None:
            raise ValueError(f"missing settings for {self.mode.value} simulation mode")
        inactive = [
            mode.value
            for mode, settings in selected.items()
            if mode is not self.mode and settings is not None
        ]
        if inactive:
            rendered = ", ".join(inactive)
            raise ValueError(f"settings for inactive simulation modes must be unset: {rendered}")
        return self


def simulation_spec_digest(spec: SimulationSpec) -> Sha256:
    """Return the stable content digest used to bind simulated rollout provenance.

    Args:
        spec: Fully validated immutable simulation recipe.

    Returns:
        SHA-256 digest of the canonical specification serialization.
    """
    return sha256_json(spec)
