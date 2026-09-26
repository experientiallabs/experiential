"""Typed evaluation inputs shared with router composition, without policy-fitting fields."""

from __future__ import annotations

from collections.abc import Callable
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
    """Frozen worker, environment, judge and execution inputs independent of router fitting.

    Attributes:
        candidates: Nonempty tuple of pinned worker identities.
        observed_cells: Optional production evidence, empty by default.
        production_protocol: Frozen interpretation of production evidence.
        simulation_protocol: Frozen simulated-evaluation protocol.
        fit_rag_input: Exact fit-only grounding artifact.
        pricing_snapshot_id: Frozen catalog prices used for comparison.
        judgment_status: Provisional or human-calibrated status of the selected judge.
        world_model_settings: Environment model and retrieval settings.
        simulation_completion_input: Optional immutable provider reservation contract.
        agent_id: Nonempty identity of the rollout agent.
        seed: Scenario randomization seed.
        run_id: Optional execution namespace; distinct values collect independent evidence.
        maximum_steps: Positive candidate-turn ceiling.
        continuation_of: Optional parent evaluation retained during budget continuation.
        maximum_rollout_output_tokens: Positive cumulative generation ceiling, default one million.
        maximum_concurrency: Positive maximum number of simultaneous rollouts.
        repeats: Independent runs per scenario/model pair, default one and separate from retries.
    """

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
    run_id: str | None = Field(default=None, min_length=1, max_length=256)
    maximum_steps: int = Field(gt=0)
    continuation_of: ArtifactInput | None = None
    maximum_rollout_output_tokens: int = Field(default=1_000_000, gt=0)
    maximum_concurrency: int = Field(gt=0)
    repeats: int = Field(default=1, ge=1)


class EvaluationBudget(ContractModel):
    """Finite ceilings for simulation plus judging; execution never opts out of enforcement.

    Attributes:
        maximum_cost_usd: Positive finite provider-spend ceiling.
        maximum_judgments: Positive ceiling on durable cell judgments.
    """

    maximum_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    maximum_judgments: int = Field(gt=0)


class EvaluationExecutionContract(ArtifactEnvelope):
    """Hash-bound execution ceilings included in the evaluation identity.

    Attributes:
        contract_id: Content-derived execution identity.
        setup: Frozen model, environment and judge inputs.
        budget: Frozen finite execution ceilings. Catalog-backed execution also enforces
            the separately approved request allowance in its durable spending ledger.
    """

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
    """Verified persisted judge references, never caller-authored calibration evidence.

    Attributes:
        rubric_id: Verified immutable rubric identity.
        calibration_id: Verified immutable calibration identity.
    """

    rubric_id: str
    calibration_id: str


@dataclass(frozen=True)
class EvaluationServices:
    """Injected model-backed services; evaluation orchestration remains Experiential-owned.

    Attributes:
        simulator_factory: Builds the selected simulation engine for one frozen plan.
        judge: Provider-bound, reservation-enforcing judge.
        plan_inputs: Additional immutable execution inputs, empty by default.
        judging_protocol: Optional explicit fresh judging pass over saved rollouts.
        judging_input: Immutable reviewed judging revision, independent of simulation identity.
        spending_limit_usd: Optional request-ledger allowance, independent of plan identity.
        judge_spend: Optional authoritative request-ledger reconciliation for saved rollouts.
    """

    simulator_factory: SimulatorFactory
    judge: EvaluationRuntimeJudge
    plan_inputs: tuple[ArtifactInput, ...] = ()
    judging_protocol: EvaluationProtocol | None = None
    judging_input: ArtifactInput | None = None
    spending_limit_usd: float | None = None
    judge_spend: Callable[[tuple[str, ...]], float] | None = None


class EvaluationRuntimeJudge(Judge, Protocol):
    """Judge whose configured provider identity can be checked before any paid work."""

    @property
    def model(self) -> ModelSnapshot:
        """Return the exact model bound by the runtime's reservation-enforcing client."""
