"""Typed evaluation inputs shared with router composition, without policy-fitting fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactId, ArtifactInput, ContractModel
from exp.common.evaluations import EvaluationProtocol, ObservedProductionCell
from exp.common.judging import Judge
from exp.common.models import ModelSnapshot, RoutedCandidateSnapshot
from exp.optimize.evaluation.simulation import SimulatorFactory
from exp.simulation.specs import WorldModelSettings


class EvaluationSetup(ContractModel):
    """Frozen worker, environment, judge and execution inputs independent of router fitting."""

    candidates: tuple[RoutedCandidateSnapshot, ...] = Field(min_length=1)
    observed_cells: tuple[ObservedProductionCell, ...] = ()
    production_protocol: EvaluationProtocol
    simulation_protocol: EvaluationProtocol
    fit_rag_input: ArtifactInput
    pricing_snapshot_id: ArtifactId
    judgment_status: Literal["provisional", "human_calibrated"]
    world_model_settings: WorldModelSettings
    simulation_completion_input: ArtifactInput | None = None
    agent_id: str = Field(min_length=1, max_length=256)
    seed: int
    maximum_steps: int = Field(gt=0)
    maximum_concurrency: int = Field(gt=0)


class EvaluationBudget(ContractModel):
    """Finite ceilings for simulation plus judging; execution never opts out of enforcement."""

    maximum_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    maximum_judgments: int = Field(gt=0)


class EvaluationExecutionContract(ArtifactEnvelope):
    """Hash-bound execution settings and authorization included in the evaluation identity."""

    contract_id: ArtifactId
    setup: EvaluationSetup
    budget: EvaluationBudget


class JudgmentReferences(Protocol):
    """Artifact identities consumed by the shared durable judgment executor."""

    @property
    def rubric_id(self) -> str:
        """Return the frozen rubric artifact identity."""

    @property
    def calibration_id(self) -> str:
        """Return the frozen calibration artifact identity."""


@dataclass(frozen=True)
class EvaluationJudge:
    """Verified persisted judge references, never caller-authored calibration evidence."""

    rubric_id: str
    calibration_id: str


@dataclass(frozen=True)
class EvaluationServices:
    """Injected model-backed services; evaluation orchestration remains Experiential-owned."""

    simulator_factory: SimulatorFactory
    judge: EvaluationRuntimeJudge
    plan_inputs: tuple[ArtifactInput, ...] = ()


class EvaluationRuntimeJudge(Judge, Protocol):
    """Judge whose configured provider identity can be checked before any paid work."""

    @property
    def model(self) -> ModelSnapshot:
        """Return the exact model bound by the runtime's reservation-enforcing client."""
