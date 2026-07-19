"""Finite Monte Carlo gate for a preregistered paired-confirmation simulation."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from collections.abc import Iterable
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path
from statistics import fmean
from typing import Literal, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from scipy.stats import beta

import wmh.evals.paired as paired_analysis
from wmh.evals.paired import (
    PAIRED_ANALYSIS_VERSION,
    BoundedMeanBet,
    PairedEvaluationDesign,
    PairedPanelPlan,
    paired_primary_decision_passed,
)

PAIRED_POWER_GATE_VERSION: Literal["2"] = "2"
PAIRED_POWER_SIMULATION_VERSION: Literal["1"] = "1"
PAIRED_POWER_DGP_VERSION: Literal["1"] = "1"
PAIRED_POWER_TASK_PROFILE_VERSION: Literal["1"] = "1"
PAIRED_POWER_ARTIFACT_VERSION: Literal["1"] = "1"
PAIRED_POWER_CHUNK_VERSION: Literal["1"] = "1"
PAIRED_POWER_BITSET_ENCODING: Literal["base64-lsb0-bitset-v1"] = "base64-lsb0-bitset-v1"
PAIRED_POWER_RNG: Literal["numpy-pcg64dxsm-v1"] = "numpy-pcg64dxsm-v1"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_ARTIFACT_MODE = 0o600
_CP_MIN_DECIMAL_PRECISION = 120
_CP_MAX_OUTWARD_STEPS = 65_536


class PairedPowerLaneBaseline(BaseModel):
    """One fixed task success probability for one frozen lane."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    panel_member: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("panel_member")
    @classmethod
    def _require_canonical_member(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("power profile lane identity cannot have surrounding whitespace")
        return value

    @field_validator("probability", mode="before")
    @classmethod
    def _reject_boolean_probability(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("power profile probability cannot be boolean")
        return value


class PairedPowerTaskProfileEntry(BaseModel):
    """Private simulation metadata and fixed baseline rates for one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    stratum: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    lane_baselines: tuple[PairedPowerLaneBaseline, ...]

    @field_validator("task_id", "stratum", "group_id")
    @classmethod
    def _require_canonical_identity(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("power task profile identities cannot have surrounding whitespace")
        return value

    @model_validator(mode="after")
    def _require_canonical_lanes(self) -> Self:
        if not self.lane_baselines:
            raise ValueError("power task profile needs at least one lane baseline")
        canonical = tuple(sorted(self.lane_baselines, key=lambda item: item.panel_member))
        if self.lane_baselines != canonical:
            raise ValueError("power task lane baselines must be unique and in canonical order")
        names = [item.panel_member for item in self.lane_baselines]
        if len(names) != len(set(names)):
            raise ValueError("power task profile contains duplicate lane baselines")
        return self


class PairedPowerTaskProfileManifest(BaseModel):
    """Private immutable task strata, semantic groups, and nuisance rates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_version: Literal["1"] = PAIRED_POWER_TASK_PROFILE_VERSION
    tasks: tuple[PairedPowerTaskProfileEntry, ...]

    @model_validator(mode="after")
    def _require_canonical_tasks(self) -> Self:
        if not self.tasks:
            raise ValueError("power task profile cannot be empty")
        canonical = tuple(sorted(self.tasks, key=lambda item: item.task_id))
        if self.tasks != canonical:
            raise ValueError("power task profile must be unique and in canonical task order")
        names = [item.task_id for item in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("power task profile contains duplicate task identities")
        lane_sets = {
            tuple(item.panel_member for item in task.lane_baselines) for task in self.tasks
        }
        if len(lane_sets) != 1:
            raise ValueError("every power task profile entry must contain the same lane set")
        return self

    @property
    def lane_set(self) -> tuple[str, ...]:
        """Return the canonical frozen lane identities."""
        return tuple(item.panel_member for item in self.tasks[0].lane_baselines)

    @property
    def digest(self) -> str:
        """Return the private task-profile identity."""
        return _canonical_digest(self.model_dump(mode="json"))

    @property
    def metadata_digest(self) -> str:
        """Bind task, stratum, and group metadata without publishing identities."""
        return _canonical_digest(
            [
                {"task_id": task.task_id, "stratum": task.stratum, "group_id": task.group_id}
                for task in self.tasks
            ]
        )


class PairedPowerEffectAtom(BaseModel):
    """One frozen semantic-group effect multiplier and its probability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    multiplier: float = Field(ge=-10.0, le=10.0, allow_inf_nan=False)
    probability: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("multiplier", "probability", mode="before")
    @classmethod
    def _reject_boolean_values(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("power effect values cannot be boolean")
        return value


class PairedPowerEffectShapeManifest(BaseModel):
    """Categorical effect heterogeneity shared by each semantic group."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atoms: tuple[PairedPowerEffectAtom, ...]

    @model_validator(mode="after")
    def _require_canonical_mean_one_shape(self) -> Self:
        if not self.atoms:
            raise ValueError("power effect shape cannot be empty")
        canonical = tuple(sorted(self.atoms, key=lambda atom: atom.multiplier))
        if self.atoms != canonical:
            raise ValueError("power effect atoms must be unique and in canonical order")
        multipliers = [atom.multiplier for atom in self.atoms]
        if len(multipliers) != len(set(multipliers)):
            raise ValueError("power effect shape contains duplicate multipliers")
        probability = math.fsum(atom.probability for atom in self.atoms)
        if not math.isclose(probability, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("power effect probabilities must sum to one")
        mean = math.fsum(atom.probability * atom.multiplier for atom in self.atoms)
        if not math.isclose(mean, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("power effect multipliers must have probability-weighted mean one")
        return self


class PairedPowerDependenceManifest(BaseModel):
    """Frozen dependence assumptions for complete simulated task vectors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    residual_attempt_intraclass_correlation: float = Field(
        ge=0.0,
        lt=1.0,
        allow_inf_nan=False,
    )
    attempt_model: Literal["beta-binomial-exchangeable"] = "beta-binomial-exchangeable"
    paired_arm_model: Literal["conditionally-independent"] = "conditionally-independent"
    effect_sharing: Literal["semantic-group-across-lanes"] = "semantic-group-across-lanes"
    task_vector_assumption: Literal[
        "independent-conditional-on-frozen-rates-and-realized-effects"
    ] = "independent-conditional-on-frozen-rates-and-realized-effects"

    @field_validator("residual_attempt_intraclass_correlation", mode="before")
    @classmethod
    def _reject_boolean_icc(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("residual attempt dependence cannot be boolean")
        return value


class PairedPowerDgpManifest(BaseModel):
    """Complete immutable data-generating assumptions for paired task means."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dgp_version: Literal["1"] = PAIRED_POWER_DGP_VERSION
    baseline_model: Literal["fixed-private-task-probabilities"] = "fixed-private-task-probabilities"
    candidate_model: Literal["clipped-additive-calibrated-equal-task-effect"] = (
        "clipped-additive-calibrated-equal-task-effect"
    )
    effect_shape: PairedPowerEffectShapeManifest
    dependence: PairedPowerDependenceManifest


class PairedPowerSeedManifest(BaseModel):
    """Root seed and counter-domain identity for deterministic chunks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_seed: str = Field(min_length=16)
    rng: Literal["numpy-pcg64dxsm-v1"] = PAIRED_POWER_RNG

    @field_validator("root_seed")
    @classmethod
    def _require_canonical_seed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("power root seed cannot have surrounding whitespace")
        return value


class PairedPowerReplicationManifest(BaseModel):
    """Fixed horizon and deterministic chunk layout for both scenarios."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    replications_per_scenario: StrictInt = Field(ge=1)
    chunk_size: StrictInt = Field(ge=1)

    @field_validator("replications_per_scenario", "chunk_size", mode="before")
    @classmethod
    def _reject_boolean_counts(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("power replication counts cannot be boolean")
        return value

    @model_validator(mode="after")
    def _bound_chunk_size(self) -> Self:
        if self.chunk_size > self.replications_per_scenario:
            raise ValueError("power chunk size cannot exceed the scenario horizon")
        return self

    @property
    def chunk_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return inclusive one-based ranges in frozen canonical order."""
        return tuple(
            (first, min(first + self.chunk_size - 1, self.replications_per_scenario))
            for first in range(1, self.replications_per_scenario + 1, self.chunk_size)
        )


class PairedPowerScenarioManifest(BaseModel):
    """One named boundary-null or target-effect simulation scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario: Literal["weak-null", "target-alternative"]
    equal_task_effect: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("equal_task_effect", mode="before")
    @classmethod
    def _reject_boolean_effect(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("power scenario effect cannot be boolean")
        return value

    @model_validator(mode="after")
    def _require_scenario_boundary(self) -> Self:
        if self.scenario == "weak-null" and self.equal_task_effect != 0.0:
            raise ValueError("weak-null scenario must sit at the zero-effect boundary")
        if self.scenario == "target-alternative" and self.equal_task_effect <= 0.0:
            raise ValueError("target-alternative scenario needs a positive effect")
        return self


class PairedPowerSimulationManifest(BaseModel):
    """Digest-bound simulator, private profile, matrix, DGP, and horizon contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_version: Literal["1"] = PAIRED_POWER_SIMULATION_VERSION
    simulator_source_digest: str = Field(pattern=_DIGEST_PATTERN)
    simulator_schema_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_analysis_source_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_analysis_version: Literal["5"]
    paired_evaluation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    task_profile_digest: str = Field(pattern=_DIGEST_PATTERN)
    task_strata_group_metadata_digest: str = Field(pattern=_DIGEST_PATTERN)
    task_count: StrictInt = Field(ge=1)
    lane_set: tuple[str, ...]
    attempts_by_lane: tuple[PairedPanelPlan, ...]
    primary_e_value_bets: tuple[BoundedMeanBet, ...]
    alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    minimum_equal_task_member_delta: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    dgp: PairedPowerDgpManifest
    seeds: PairedPowerSeedManifest
    replications: PairedPowerReplicationManifest
    scenarios: tuple[PairedPowerScenarioManifest, ...]

    @property
    def digest(self) -> str:
        """Return the identity consumed by every chunk and power gate."""
        return _canonical_digest(self.model_dump(mode="json"))

    @classmethod
    def create(
        cls,
        *,
        evaluation_design: PairedEvaluationDesign,
        task_profile: PairedPowerTaskProfileManifest,
        dgp: PairedPowerDgpManifest,
        seeds: PairedPowerSeedManifest,
        replications: PairedPowerReplicationManifest,
        scenarios: tuple[PairedPowerScenarioManifest, ...],
    ) -> PairedPowerSimulationManifest:
        """Freeze a simulator manifest from the exact private analysis inputs."""
        manifest = cls(
            simulator_source_digest=_module_source_digest(Path(__file__)),
            simulator_schema_digest=_simulation_schema_digest(),
            paired_analysis_source_digest=_module_source_digest(Path(paired_analysis.__file__)),
            paired_analysis_version=PAIRED_ANALYSIS_VERSION,
            paired_evaluation_design_digest=evaluation_design.digest,
            task_profile_digest=task_profile.digest,
            task_strata_group_metadata_digest=task_profile.metadata_digest,
            task_count=len(evaluation_design.tasks),
            lane_set=evaluation_design.panel_members,
            attempts_by_lane=evaluation_design.panel,
            primary_e_value_bets=evaluation_design.primary_e_value_bets,
            alpha=evaluation_design.alpha,
            minimum_equal_task_member_delta=(evaluation_design.minimum_equal_task_member_delta),
            dgp=dgp,
            seeds=seeds,
            replications=replications,
            scenarios=scenarios,
        )
        manifest.validate_frozen_inputs(evaluation_design, task_profile)
        return manifest

    @model_validator(mode="after")
    def _require_canonical_matrix_and_scenarios(self) -> Self:
        if self.lane_set != tuple(sorted(set(self.lane_set))):
            raise ValueError("power simulation lanes must be unique and in canonical order")
        if tuple(item.panel_member for item in self.attempts_by_lane) != self.lane_set:
            raise ValueError("power simulation attempts must exactly match the lane set")
        if tuple(item.scenario for item in self.scenarios) != (
            "weak-null",
            "target-alternative",
        ):
            raise ValueError("power simulation must freeze weak-null then target-alternative")
        return self

    def validate_frozen_inputs(
        self,
        evaluation_design: PairedEvaluationDesign,
        task_profile: PairedPowerTaskProfileManifest,
    ) -> None:
        """Reject source, schema, design, or private-profile drift before simulation."""
        expected = {
            "simulator source digest": (
                self.simulator_source_digest,
                _module_source_digest(Path(__file__)),
            ),
            "simulator schema digest": (
                self.simulator_schema_digest,
                _simulation_schema_digest(),
            ),
            "paired analysis source digest": (
                self.paired_analysis_source_digest,
                _module_source_digest(Path(paired_analysis.__file__)),
            ),
            "paired evaluation design digest": (
                self.paired_evaluation_design_digest,
                evaluation_design.digest,
            ),
            "task profile digest": (self.task_profile_digest, task_profile.digest),
            "task metadata digest": (
                self.task_strata_group_metadata_digest,
                task_profile.metadata_digest,
            ),
        }
        for label, (frozen, observed) in expected.items():
            if frozen != observed:
                raise ValueError(f"power simulation {label} differs from the frozen manifest")
        design_group_ids = {task.task_id: task.group_id for task in evaluation_design.tasks}
        profile_group_ids = {task.task_id: task.group_id for task in task_profile.tasks}
        if design_group_ids != profile_group_ids:
            raise ValueError("power task profile differs from the evaluation task/group roster")
        if task_profile.lane_set != evaluation_design.panel_members:
            raise ValueError("power task profile lanes differ from the evaluation panel")
        if self.task_count != len(evaluation_design.tasks):
            raise ValueError("power task count differs from the evaluation roster")
        if self.attempts_by_lane != evaluation_design.panel:
            raise ValueError("power attempt profile differs from the evaluation design")
        if self.primary_e_value_bets != evaluation_design.primary_e_value_bets:
            raise ValueError("power bet mixture differs from the evaluation design")
        if self.alpha != evaluation_design.alpha:
            raise ValueError("power alpha differs from the evaluation design")
        if (
            self.minimum_equal_task_member_delta
            != evaluation_design.minimum_equal_task_member_delta
        ):
            raise ValueError("power effect floor differs from the evaluation design")


class PairedPowerGateDesign(BaseModel):
    """Frozen operating-characteristic thresholds for one locked simulator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate_version: Literal["2"] = PAIRED_POWER_GATE_VERSION
    simulation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_evaluation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    target_effect: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    maximum_type_i_error: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    minimum_power: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    monte_carlo_alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    replications_per_scenario: StrictInt = Field(ge=1)

    @field_validator(
        "target_effect",
        "maximum_type_i_error",
        "minimum_power",
        "monte_carlo_alpha",
        mode="before",
    )
    @classmethod
    def _reject_boolean_thresholds(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("paired power thresholds cannot be boolean")
        return value

    @field_validator("replications_per_scenario", mode="before")
    @classmethod
    def _reject_boolean_replications(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("paired power replications cannot be boolean")
        return value

    @property
    def digest(self) -> str:
        """Return the canonical identity of this complete gate."""
        return _canonical_digest(self.model_dump(mode="json"))


class PairedPowerTrial(BaseModel):
    """One simulator replicate projected to the frozen primary decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_evaluation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    scenario: Literal["weak-null", "target-alternative"]
    replicate: StrictInt = Field(ge=1)
    primary_passed: StrictBool

    @field_validator("replicate", mode="before")
    @classmethod
    def _reject_boolean_replicates(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("paired power replicate identities cannot be boolean")
        return value


class PairedPowerDecisionBits(BaseModel):
    """Canonical compact primary-decision bitmap for one ordered range."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    encoding: Literal["base64-lsb0-bitset-v1"] = PAIRED_POWER_BITSET_ENCODING
    count: StrictInt = Field(ge=1)
    payload: str = Field(min_length=4)
    payload_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("count", mode="before")
    @classmethod
    def _reject_boolean_count(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("power bitset count cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_canonical_payload(self) -> Self:
        decoded = _decode_bit_payload(self.payload)
        expected_bytes = (self.count + 7) // 8
        if len(decoded) != expected_bytes:
            raise ValueError("power bitset byte length differs from its declared count")
        unused = expected_bytes * 8 - self.count
        if unused and decoded[-1] >> (8 - unused):
            raise ValueError("power bitset contains nonzero padding bits")
        if self.payload != base64.b64encode(decoded).decode("ascii"):
            raise ValueError("power bitset payload is not canonical base64")
        if self.payload_digest != _bytes_digest(decoded):
            raise ValueError("power bitset payload digest does not match its bytes")
        return self

    @classmethod
    def from_decisions(cls, decisions: tuple[bool, ...]) -> PairedPowerDecisionBits:
        """Pack ordered decisions into the canonical least-significant-bit-first encoding."""
        if not decisions:
            raise ValueError("power decision bitset cannot be empty")
        payload = bytearray((len(decisions) + 7) // 8)
        for index, decision in enumerate(decisions):
            if decision:
                payload[index // 8] |= 1 << (index % 8)
        frozen = bytes(payload)
        return cls(
            count=len(decisions),
            payload=base64.b64encode(frozen).decode("ascii"),
            payload_digest=_bytes_digest(frozen),
        )

    def decisions(self) -> tuple[bool, ...]:
        """Decode every ordered decision after model-level integrity validation."""
        payload = _decode_bit_payload(self.payload)
        return tuple(bool(payload[index // 8] & (1 << (index % 8))) for index in range(self.count))

    @property
    def rejection_count(self) -> int:
        """Return the number of true primary decisions without materializing trials."""
        return sum(byte.bit_count() for byte in _decode_bit_payload(self.payload))


class PairedPowerTrialChunk(BaseModel):
    """One deterministic, independently resumable scenario chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_version: Literal["1"] = PAIRED_POWER_CHUNK_VERSION
    simulation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_evaluation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    scenario: Literal["weak-null", "target-alternative"]
    first_replicate: StrictInt = Field(ge=1)
    last_replicate: StrictInt = Field(ge=1)
    decisions: PairedPowerDecisionBits

    @field_validator("first_replicate", "last_replicate", mode="before")
    @classmethod
    def _reject_boolean_replicates(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("power chunk replicate bounds cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.last_replicate < self.first_replicate:
            raise ValueError("power chunk replicate range is reversed")
        if self.decisions.count != self.last_replicate - self.first_replicate + 1:
            raise ValueError("power chunk bitset does not fill its replicate range")
        return self

    @property
    def key(self) -> tuple[str, int, int]:
        """Return the canonical merge identity."""
        return self.scenario, self.first_replicate, self.last_replicate

    @property
    def digest(self) -> str:
        """Return the complete chunk identity."""
        return _canonical_digest(self.model_dump(mode="json"))


class PairedPowerScenarioArtifact(BaseModel):
    """One complete compact scenario bitmap in canonical replicate order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario: Literal["weak-null", "target-alternative"]
    decisions: PairedPowerDecisionBits


class PairedPowerTrialArtifact(BaseModel):
    """Compact complete digest-bound trials consumable by the exact power gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_version: Literal["1"] = PAIRED_POWER_ARTIFACT_VERSION
    simulator_schema_digest: str = Field(pattern=_DIGEST_PATTERN)
    simulation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_evaluation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    target_effect: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    replications_per_scenario: StrictInt = Field(ge=1)
    scenarios: tuple[PairedPowerScenarioArtifact, ...]
    chunk_digests: tuple[str, ...]

    @field_validator("replications_per_scenario", mode="before")
    @classmethod
    def _reject_boolean_replications(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("power artifact replication count cannot be boolean")
        return value

    @field_validator("target_effect", mode="before")
    @classmethod
    def _reject_boolean_target_effect(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("power artifact target effect cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_complete_scenarios(self) -> Self:
        if tuple(item.scenario for item in self.scenarios) != (
            "weak-null",
            "target-alternative",
        ):
            raise ValueError("power artifact must contain both scenarios in canonical order")
        if any(item.decisions.count != self.replications_per_scenario for item in self.scenarios):
            raise ValueError("power artifact scenarios must fill the frozen horizon")
        if not self.chunk_digests or len(self.chunk_digests) != len(set(self.chunk_digests)):
            raise ValueError("power artifact chunk digests must be nonempty and unique")
        if any(not _is_digest(value) for value in self.chunk_digests):
            raise ValueError("power artifact contains an invalid chunk digest")
        return self

    @property
    def digest(self) -> str:
        """Return the canonical complete trial-artifact identity."""
        return _canonical_digest(self.model_dump(mode="json"))

    @property
    def trial_evidence_digest(self) -> str:
        """Return the v2 gate identity of the represented expanded trial records."""
        return _artifact_trial_evidence_digest(self)

    def rejection_count(self, scenario: Literal["weak-null", "target-alternative"]) -> int:
        """Return primary passes for one exact frozen scenario."""
        for item in self.scenarios:
            if item.scenario == scenario:
                return item.decisions.rejection_count
        raise ValueError(f"power artifact omits scenario {scenario}")


class _PairedPowerChunkEnvelope(BaseModel):
    """On-disk chunk plus a digest over its complete canonical payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact: PairedPowerTrialChunk
    artifact_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_digest(self) -> Self:
        if self.artifact_digest != self.artifact.digest:
            raise ValueError("paired power chunk artifact digest does not match its payload")
        return self


class _PairedPowerTrialEnvelope(BaseModel):
    """On-disk complete trial artifact plus its canonical payload digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact: PairedPowerTrialArtifact
    artifact_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_digest(self) -> Self:
        if self.artifact_digest != self.artifact.digest:
            raise ValueError("paired power trial artifact digest does not match its payload")
        return self


class PairedPowerGateReport(BaseModel):
    """Reload-safe Monte Carlo evidence, bounds, and decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate_version: Literal["2"]
    design: PairedPowerGateDesign
    trial_evidence_digest: str = Field(pattern=_DIGEST_PATTERN)
    null_rejections: StrictInt = Field(ge=0)
    target_rejections: StrictInt = Field(ge=0)
    empirical_type_i_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    type_i_error_upper_bound: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    empirical_power: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    power_lower_bound: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    type_i_error_passed: bool
    power_passed: bool
    report_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_derived_evidence(self) -> Self:
        count = self.design.replications_per_scenario
        if self.null_rejections > count or self.target_rejections > count:
            raise ValueError("paired power rejection counts cannot exceed frozen replications")
        expected_type_i_error = self.null_rejections / count
        expected_power = self.target_rejections / count
        expected_type_i_upper = _binomial_upper_bound(
            self.null_rejections,
            count,
            alpha=self.design.monte_carlo_alpha,
        )
        expected_power_lower = _binomial_lower_bound(
            self.target_rejections,
            count,
            alpha=self.design.monte_carlo_alpha,
        )
        derived = {
            "empirical_type_i_error": expected_type_i_error,
            "type_i_error_upper_bound": expected_type_i_upper,
            "empirical_power": expected_power,
            "power_lower_bound": expected_power_lower,
            "type_i_error_passed": (expected_type_i_upper <= self.design.maximum_type_i_error),
            "power_passed": expected_power_lower >= self.design.minimum_power,
        }
        for field, expected in derived.items():
            if getattr(self, field) != expected:
                raise ValueError(f"paired power report {field} differs from its frozen evidence")
        expected_digest = _canonical_digest(self.model_dump(mode="json", exclude={"report_digest"}))
        if self.report_digest != expected_digest:
            raise ValueError("paired power report digest differs from its frozen evidence")
        return self

    @property
    def passed(self) -> bool:
        """Return whether both preregistered operating-characteristic gates pass."""
        return self.type_i_error_passed and self.power_passed

    @property
    def digest(self) -> str:
        """Return the canonical identity binding the design, evidence, and decisions."""
        return self.report_digest


def evaluate_paired_power_gate(
    design: PairedPowerGateDesign,
    trials: tuple[PairedPowerTrial, ...] | PairedPowerTrialArtifact,
) -> PairedPowerGateReport:
    """Evaluate complete locked null and target simulations without optional stopping.

    The simulator, its data-generating assumptions, seeds, exact paired analysis,
    and mapping from a replicate to ``primary_passed`` live behind the frozen
    ``simulation_design_digest``. The separate paired-evaluation digest binds the
    exact roster, lane attempts, e-value bets, and observed floor exercised by the
    simulator. This gate rejects digest drift, duplicate or missing replicate
    identities, and extra replicates. It uses one-sided exact Clopper-Pearson bounds
    at the preregistered Monte Carlo alpha. Passing supports only the design's frozen
    target effect and assumptions; it does not establish power for an untested
    effect size.
    """
    if isinstance(trials, PairedPowerTrialArtifact):
        null_rejections, target_rejections = _artifact_rejection_counts(design, trials)
        trial_evidence_digest = trials.trial_evidence_digest
    else:
        null_rejections, target_rejections = _expanded_trial_rejection_counts(design, trials)
        trial_evidence_digest = _expanded_trial_evidence_digest(trials)
    count = design.replications_per_scenario
    empirical_type_i_error = null_rejections / count
    empirical_power = target_rejections / count
    type_i_error_upper_bound = _binomial_upper_bound(
        null_rejections,
        count,
        alpha=design.monte_carlo_alpha,
    )
    power_lower_bound = _binomial_lower_bound(
        target_rejections,
        count,
        alpha=design.monte_carlo_alpha,
    )
    report_payload: dict[str, JsonValue] = {
        "gate_version": PAIRED_POWER_GATE_VERSION,
        "design": design.model_dump(mode="json"),
        "trial_evidence_digest": trial_evidence_digest,
        "null_rejections": null_rejections,
        "target_rejections": target_rejections,
        "empirical_type_i_error": empirical_type_i_error,
        "type_i_error_upper_bound": type_i_error_upper_bound,
        "empirical_power": empirical_power,
        "power_lower_bound": power_lower_bound,
        "type_i_error_passed": type_i_error_upper_bound <= design.maximum_type_i_error,
        "power_passed": power_lower_bound >= design.minimum_power,
    }
    return PairedPowerGateReport.model_validate(
        {**report_payload, "report_digest": _canonical_digest(report_payload)}
    )


def run_paired_power_chunk(
    manifest: PairedPowerSimulationManifest,
    evaluation_design: PairedEvaluationDesign,
    task_profile: PairedPowerTaskProfileManifest,
    *,
    scenario: Literal["weak-null", "target-alternative"],
    first_replicate: int,
    last_replicate: int,
) -> PairedPowerTrialChunk:
    """Run one exact frozen chunk through the production v5 primary decision."""
    manifest.validate_frozen_inputs(evaluation_design, task_profile)
    if isinstance(first_replicate, bool) or isinstance(last_replicate, bool):
        raise ValueError("power chunk replicate bounds cannot be boolean")
    if (first_replicate, last_replicate) not in manifest.replications.chunk_ranges:
        raise ValueError("power chunk range is not one of the frozen deterministic ranges")
    scenario_manifest = _scenario_manifest(manifest, scenario)
    decisions = _simulate_primary_decisions(
        manifest,
        evaluation_design,
        task_profile,
        scenario_manifest,
        first_replicate=first_replicate,
        last_replicate=last_replicate,
    )
    return PairedPowerTrialChunk(
        simulation_design_digest=manifest.digest,
        paired_evaluation_design_digest=evaluation_design.digest,
        scenario=scenario,
        first_replicate=first_replicate,
        last_replicate=last_replicate,
        decisions=PairedPowerDecisionBits.from_decisions(decisions),
    )


def merge_paired_power_chunks(
    manifest: PairedPowerSimulationManifest,
    chunks: tuple[PairedPowerTrialChunk, ...],
) -> PairedPowerTrialArtifact:
    """Merge only the exact complete frozen chunk matrix into a compact artifact."""
    if any(chunk.simulation_design_digest != manifest.digest for chunk in chunks):
        raise ValueError("power chunk simulation design digest differs from the manifest")
    if any(
        chunk.paired_evaluation_design_digest != manifest.paired_evaluation_design_digest
        for chunk in chunks
    ):
        raise ValueError("power chunk evaluation design digest differs from the manifest")
    keys = [chunk.key for chunk in chunks]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"power chunks contain duplicate deterministic ranges: {duplicates}")
    expected = tuple(
        (scenario, first, last)
        for scenario in ("weak-null", "target-alternative")
        for first, last in manifest.replications.chunk_ranges
    )
    if set(keys) != set(expected) or len(keys) != len(expected):
        raise ValueError("power chunks must exactly fill both frozen scenario horizons")
    by_key = {chunk.key: chunk for chunk in chunks}
    scenario_artifacts: list[PairedPowerScenarioArtifact] = []
    ordered_chunks: list[PairedPowerTrialChunk] = []
    for scenario in ("weak-null", "target-alternative"):
        decisions: list[bool] = []
        for first, last in manifest.replications.chunk_ranges:
            chunk = by_key[(scenario, first, last)]
            ordered_chunks.append(chunk)
            decisions.extend(chunk.decisions.decisions())
        scenario_artifacts.append(
            PairedPowerScenarioArtifact(
                scenario=scenario,
                decisions=PairedPowerDecisionBits.from_decisions(tuple(decisions)),
            )
        )
    return PairedPowerTrialArtifact(
        simulator_schema_digest=manifest.simulator_schema_digest,
        simulation_design_digest=manifest.digest,
        paired_evaluation_design_digest=manifest.paired_evaluation_design_digest,
        target_effect=_scenario_manifest(manifest, "target-alternative").equal_task_effect,
        replications_per_scenario=manifest.replications.replications_per_scenario,
        scenarios=tuple(scenario_artifacts),
        chunk_digests=tuple(chunk.digest for chunk in ordered_chunks),
    )


def resume_paired_power_simulation(
    manifest: PairedPowerSimulationManifest,
    evaluation_design: PairedEvaluationDesign,
    task_profile: PairedPowerTaskProfileManifest,
    chunk_directory: str | Path,
) -> PairedPowerTrialArtifact:
    """Resume deterministic chunks, refusing corrupt or drifted existing work."""
    manifest.validate_frozen_inputs(evaluation_design, task_profile)
    directory = Path(chunk_directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected_names = {
        _chunk_filename(scenario, first, last)
        for scenario in ("weak-null", "target-alternative")
        for first, last in manifest.replications.chunk_ranges
    }
    observed_names = {
        path.name for path in directory.iterdir() if path.is_file() and path.suffix == ".json"
    }
    unexpected_names = sorted(observed_names - expected_names)
    if unexpected_names:
        raise ValueError(
            f"power chunk directory contains unexpected JSON files: {unexpected_names}"
        )
    existing_chunks: dict[tuple[str, int, int], PairedPowerTrialChunk] = {}
    for scenario in ("weak-null", "target-alternative"):
        for first, last in manifest.replications.chunk_ranges:
            path = directory / _chunk_filename(scenario, first, last)
            if not path.exists():
                continue
            chunk = load_paired_power_chunk(path)
            if chunk.key != (scenario, first, last):
                raise ValueError("existing power chunk identity differs from its frozen path")
            if chunk.simulation_design_digest != manifest.digest:
                raise ValueError("existing power chunk simulation digest differs from the manifest")
            if chunk.paired_evaluation_design_digest != manifest.paired_evaluation_design_digest:
                raise ValueError("existing power chunk evaluation digest differs from the manifest")
            existing_chunks[chunk.key] = chunk
    chunks: list[PairedPowerTrialChunk] = []
    for scenario in ("weak-null", "target-alternative"):
        for first, last in manifest.replications.chunk_ranges:
            path = directory / _chunk_filename(scenario, first, last)
            key = (scenario, first, last)
            chunk = existing_chunks.get(key)
            if chunk is None:
                chunk = run_paired_power_chunk(
                    manifest,
                    evaluation_design,
                    task_profile,
                    scenario=scenario,
                    first_replicate=first,
                    last_replicate=last,
                )
                write_paired_power_chunk(path, chunk)
            chunks.append(chunk)
    return merge_paired_power_chunks(manifest, tuple(chunks))


def write_paired_power_chunk(path: str | Path, chunk: PairedPowerTrialChunk) -> None:
    """Atomically persist one complete chunk with a canonical payload digest."""
    envelope = _PairedPowerChunkEnvelope(artifact=chunk, artifact_digest=chunk.digest)
    _atomic_write_json(Path(path), envelope.model_dump(mode="json"))


def load_paired_power_chunk(path: str | Path) -> PairedPowerTrialChunk:
    """Load one bounded chunk and verify both bitset and envelope integrity."""
    payload = _read_bounded_file(Path(path))
    return _PairedPowerChunkEnvelope.model_validate_json(payload).artifact


def write_paired_power_trial_artifact(
    path: str | Path,
    artifact: PairedPowerTrialArtifact,
) -> None:
    """Atomically persist a compact complete trial artifact."""
    envelope = _PairedPowerTrialEnvelope(artifact=artifact, artifact_digest=artifact.digest)
    _atomic_write_json(Path(path), envelope.model_dump(mode="json"))


def load_paired_power_trial_artifact(path: str | Path) -> PairedPowerTrialArtifact:
    """Load and fully verify a compact complete trial artifact."""
    payload = _read_bounded_file(Path(path))
    return _PairedPowerTrialEnvelope.model_validate_json(payload).artifact


def _simulate_primary_decisions(
    manifest: PairedPowerSimulationManifest,
    evaluation_design: PairedEvaluationDesign,
    task_profile: PairedPowerTaskProfileManifest,
    scenario: PairedPowerScenarioManifest,
    *,
    first_replicate: int,
    last_replicate: int,
) -> tuple[bool, ...]:
    replicate_count = last_replicate - first_replicate + 1
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            _chunk_seed(manifest, scenario.scenario, first_replicate, last_replicate)
        )
    )
    group_ids = tuple(sorted({task.group_id for task in task_profile.tasks}))
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    task_group_indices = np.array(
        [group_index[task.group_id] for task in task_profile.tasks],
        dtype=np.int64,
    )
    atoms = manifest.dgp.effect_shape.atoms
    atom_values = np.array([atom.multiplier for atom in atoms], dtype=np.float64)
    atom_probabilities = np.array([atom.probability for atom in atoms], dtype=np.float64)
    group_atom_indices = rng.choice(
        len(atoms),
        size=(replicate_count, len(group_ids)),
        p=atom_probabilities,
    )
    task_multipliers = atom_values[group_atom_indices[:, task_group_indices]]
    task_deltas_by_member: list[np.ndarray] = []
    for lane_index, panel_member in enumerate(evaluation_design.panel_members):
        attempts = evaluation_design.panel[lane_index].attempts
        baseline = np.array(
            [_baseline_probability(task, panel_member) for task in task_profile.tasks],
            dtype=np.float64,
        )
        scale = _calibrated_effect_scale(
            baseline,
            atoms,
            target_effect=scenario.equal_task_effect,
        )
        candidate_probability = np.clip(
            baseline[None, :] + scale * task_multipliers,
            0.0,
            1.0,
        )
        baseline_probability = np.broadcast_to(
            baseline[None, :],
            candidate_probability.shape,
        )
        baseline_count = _sample_attempt_counts(
            rng,
            baseline_probability,
            attempts=attempts,
            intraclass_correlation=(
                manifest.dgp.dependence.residual_attempt_intraclass_correlation
            ),
        )
        candidate_count = _sample_attempt_counts(
            rng,
            candidate_probability,
            attempts=attempts,
            intraclass_correlation=(
                manifest.dgp.dependence.residual_attempt_intraclass_correlation
            ),
        )
        task_deltas_by_member.append((candidate_count - baseline_count) / attempts)
    return tuple(
        paired_primary_decision_passed(
            evaluation_design,
            tuple(
                tuple(float(delta) for delta in member_deltas[replicate])
                for member_deltas in task_deltas_by_member
            ),
        )
        for replicate in range(replicate_count)
    )


def _sample_attempt_counts(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    *,
    attempts: int,
    intraclass_correlation: float,
) -> np.ndarray:
    if intraclass_correlation == 0.0:
        return rng.binomial(attempts, probabilities)
    concentration = 1.0 / intraclass_correlation - 1.0
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    realized = rng.beta(clipped * concentration, (1.0 - clipped) * concentration)
    counts = rng.binomial(attempts, realized)
    counts = np.where(probabilities <= 0.0, 0, counts)
    return np.where(probabilities >= 1.0, attempts, counts)


def _calibrated_effect_scale(
    baseline: np.ndarray,
    atoms: tuple[PairedPowerEffectAtom, ...],
    *,
    target_effect: float,
) -> float:
    if target_effect == 0.0:
        return 0.0

    def expected_effect(scale: float) -> float:
        return fmean(
            math.fsum(
                atom.probability
                * (min(1.0, max(0.0, probability + scale * atom.multiplier)) - probability)
                for atom in atoms
            )
            for probability in baseline
        )

    lower = 0.0
    upper = target_effect
    while expected_effect(upper) < target_effect and upper < 16.0:
        upper *= 2.0
    if expected_effect(upper) < target_effect:
        raise ValueError("target effect is infeasible under the frozen baseline and effect shape")
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if expected_effect(midpoint) < target_effect:
            lower = midpoint
        else:
            upper = midpoint
    calibrated = (lower + upper) / 2.0
    if not math.isclose(expected_effect(calibrated), target_effect, abs_tol=1e-12):
        raise RuntimeError("power effect calibration did not reach the frozen target")
    return calibrated


def _baseline_probability(task: PairedPowerTaskProfileEntry, panel_member: str) -> float:
    for baseline in task.lane_baselines:
        if baseline.panel_member == panel_member:
            return baseline.probability
    raise ValueError(f"power task profile omits lane {panel_member}")


def _scenario_manifest(
    manifest: PairedPowerSimulationManifest,
    scenario: Literal["weak-null", "target-alternative"],
) -> PairedPowerScenarioManifest:
    for item in manifest.scenarios:
        if item.scenario == scenario:
            return item
    raise ValueError(f"power simulation manifest omits scenario {scenario}")


def _chunk_seed(
    manifest: PairedPowerSimulationManifest,
    scenario: str,
    first_replicate: int,
    last_replicate: int,
) -> int:
    payload = json.dumps(
        {
            "root_seed": manifest.seeds.root_seed,
            "simulation_design_digest": manifest.digest,
            "scenario": scenario,
            "first_replicate": first_replicate,
            "last_replicate": last_replicate,
            "rng": manifest.seeds.rng,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _chunk_filename(scenario: str, first_replicate: int, last_replicate: int) -> str:
    return f"{scenario}-{first_replicate:09d}-{last_replicate:09d}.json"


def _expanded_trial_rejection_counts(
    design: PairedPowerGateDesign,
    trials: tuple[PairedPowerTrial, ...],
) -> tuple[int, int]:
    expected_replicates = tuple(range(1, design.replications_per_scenario + 1))
    expected_scenarios = ("weak-null", "target-alternative")
    if any(trial.simulation_design_digest != design.simulation_design_digest for trial in trials):
        raise ValueError("paired power trial simulation design digest differs from the gate")
    if any(
        trial.paired_evaluation_design_digest != design.paired_evaluation_design_digest
        for trial in trials
    ):
        raise ValueError("paired power trial evaluation design digest differs from the gate")
    keys = [(trial.scenario, trial.replicate) for trial in trials]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"paired power trials contain duplicate replicate identities: {duplicates}"
        )
    observed = {
        scenario: tuple(sorted(trial.replicate for trial in trials if trial.scenario == scenario))
        for scenario in expected_scenarios
    }
    expected = {scenario: expected_replicates for scenario in expected_scenarios}
    if observed != expected:
        raise ValueError("paired power trials must exactly fill both frozen scenarios")
    return (
        sum(trial.primary_passed for trial in trials if trial.scenario == "weak-null"),
        sum(trial.primary_passed for trial in trials if trial.scenario == "target-alternative"),
    )


def _artifact_rejection_counts(
    design: PairedPowerGateDesign,
    artifact: PairedPowerTrialArtifact,
) -> tuple[int, int]:
    if artifact.simulation_design_digest != design.simulation_design_digest:
        raise ValueError("paired power artifact simulation design digest differs from the gate")
    if artifact.paired_evaluation_design_digest != design.paired_evaluation_design_digest:
        raise ValueError("paired power artifact evaluation design digest differs from the gate")
    if artifact.replications_per_scenario != design.replications_per_scenario:
        raise ValueError("paired power artifact does not fill the gate replication horizon")
    if artifact.target_effect != design.target_effect:
        raise ValueError("paired power artifact target effect differs from the gate")
    return (
        artifact.rejection_count("weak-null"),
        artifact.rejection_count("target-alternative"),
    )


def _expanded_trial_evidence_digest(trials: tuple[PairedPowerTrial, ...]) -> str:
    canonical_trials = sorted(trials, key=lambda trial: (trial.scenario, trial.replicate))
    return _canonical_sequence_digest(
        _trial_evidence_record(
            simulation_design_digest=trial.simulation_design_digest,
            paired_evaluation_design_digest=trial.paired_evaluation_design_digest,
            scenario=trial.scenario,
            replicate=trial.replicate,
            primary_passed=trial.primary_passed,
        )
        for trial in canonical_trials
    )


def _artifact_trial_evidence_digest(artifact: PairedPowerTrialArtifact) -> str:
    canonical_scenarios = sorted(artifact.scenarios, key=lambda item: item.scenario)
    return _canonical_sequence_digest(
        _trial_evidence_record(
            simulation_design_digest=artifact.simulation_design_digest,
            paired_evaluation_design_digest=artifact.paired_evaluation_design_digest,
            scenario=scenario.scenario,
            replicate=replicate,
            primary_passed=primary_passed,
        )
        for scenario in canonical_scenarios
        for replicate, primary_passed in enumerate(scenario.decisions.decisions(), start=1)
    )


def _trial_evidence_record(
    *,
    simulation_design_digest: str,
    paired_evaluation_design_digest: str,
    scenario: str,
    replicate: int,
    primary_passed: bool,
) -> dict[str, JsonValue]:
    return {
        "simulation_design_digest": simulation_design_digest,
        "paired_evaluation_design_digest": paired_evaluation_design_digest,
        "scenario": scenario,
        "replicate": replicate,
        "primary_passed": primary_passed,
    }


def _canonical_sequence_digest(values: Iterable[JsonValue]) -> str:
    """Hash a canonical JSON array incrementally without materializing all records."""
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, value in enumerate(values):
        if index:
            digest.update(b",")
        digest.update(_canonical_json_bytes(value))
    digest.update(b"]")
    return "sha256:" + digest.hexdigest()


def _binomial_upper_bound(successes: int, trials: int, *, alpha: float) -> float:
    if successes == trials:
        return 1.0
    candidate = float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))
    threshold = Decimal.from_float(alpha)
    for _ in range(_CP_MAX_OUTWARD_STEPS):
        if _binomial_lower_tail_upper(successes, trials, probability=candidate) <= threshold:
            return candidate
        next_candidate = math.nextafter(candidate, math.inf)
        if next_candidate == candidate or next_candidate > 1.0:
            break
        candidate = next_candidate
    raise RuntimeError("could not certify an outward Clopper-Pearson upper bound")


def _binomial_lower_bound(successes: int, trials: int, *, alpha: float) -> float:
    if successes == 0:
        return 0.0
    candidate = float(beta.ppf(alpha, successes, trials - successes + 1))
    threshold = Decimal.from_float(alpha)
    for _ in range(_CP_MAX_OUTWARD_STEPS):
        if _binomial_upper_tail_upper(successes, trials, probability=candidate) <= threshold:
            return candidate
        next_candidate = math.nextafter(candidate, -math.inf)
        if next_candidate == candidate or next_candidate < 0.0:
            break
        candidate = next_candidate
    raise RuntimeError("could not certify an outward Clopper-Pearson lower bound")


def _binomial_lower_tail_upper(
    successes: int,
    trials: int,
    *,
    probability: float,
) -> Decimal:
    """Return a directed-rounding upper bound on ``P[X <= successes]``.

    The recurrence starts from ``q**n`` and advances through positive terms.
    Every division, multiplication, and addition rounds toward positive infinity,
    so the result is an upper bound. The local precision is large enough to make
    ``q = 1-p`` exact for the binary-float candidate.
    """
    if probability <= 0.0:
        return Decimal(1)
    if probability >= 1.0:
        return Decimal(0) if successes < trials else Decimal(1)
    probability_decimal = Decimal.from_float(probability)
    with localcontext() as context:
        context.prec = _cp_decimal_precision(probability_decimal)
        context.rounding = ROUND_CEILING
        one = Decimal(1)
        complement = context.subtract(one, probability_decimal)
        if context.add(probability_decimal, complement) != one:
            raise RuntimeError("Clopper-Pearson decimal precision did not preserve q = 1-p")
        term = context.power(complement, trials)
        total = term
        odds_upper = context.divide(probability_decimal, complement)
        for observed in range(successes):
            count_ratio_upper = context.divide(
                Decimal(trials - observed),
                Decimal(observed + 1),
            )
            term = context.multiply(
                context.multiply(term, count_ratio_upper),
                odds_upper,
            )
            total = context.add(total, term)
        return +total


def _binomial_upper_tail_upper(
    successes: int,
    trials: int,
    *,
    probability: float,
) -> Decimal:
    """Return a directed-rounding upper bound on ``P[X >= successes]``.

    The recurrence starts from ``p**n`` and moves backward through positive
    terms, with every operation rounded toward positive infinity.
    """
    if probability <= 0.0:
        return Decimal(0) if successes > 0 else Decimal(1)
    if probability >= 1.0:
        return Decimal(1)
    probability_decimal = Decimal.from_float(probability)
    with localcontext() as context:
        context.prec = _cp_decimal_precision(probability_decimal)
        context.rounding = ROUND_CEILING
        one = Decimal(1)
        complement = context.subtract(one, probability_decimal)
        if context.add(probability_decimal, complement) != one:
            raise RuntimeError("Clopper-Pearson decimal precision did not preserve q = 1-p")
        term = context.power(probability_decimal, trials)
        total = term
        reverse_odds_upper = context.divide(complement, probability_decimal)
        for observed in range(trials, successes, -1):
            count_ratio_upper = context.divide(
                Decimal(observed),
                Decimal(trials - observed + 1),
            )
            term = context.multiply(
                context.multiply(term, count_ratio_upper),
                reverse_odds_upper,
            )
            total = context.add(total, term)
        return +total


def _cp_decimal_precision(value: Decimal) -> int:
    """Return enough precision to form the exact unit complement of ``value``."""
    value_tuple = value.as_tuple()
    if not isinstance(value_tuple.exponent, int):
        raise RuntimeError("Clopper-Pearson certification requires a finite probability")
    exact_complement_digits = max(len(value_tuple.digits), 1 - value_tuple.exponent)
    return max(_CP_MIN_DECIMAL_PRECISION, exact_complement_digits + 8)


def _simulation_schema_digest() -> str:
    return _canonical_digest(
        {
            "simulation_manifest": PairedPowerSimulationManifest.model_json_schema(),
            "task_profile": PairedPowerTaskProfileManifest.model_json_schema(),
            "chunk": PairedPowerTrialChunk.model_json_schema(),
            "trial_artifact": PairedPowerTrialArtifact.model_json_schema(),
        }
    )


def _module_source_digest(path: Path) -> str:
    return _bytes_digest(path.read_bytes())


def _decode_bit_payload(payload: str) -> bytes:
    try:
        return base64.b64decode(payload.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("power bitset payload is not valid base64") from exc


def _bytes_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _is_digest(value: str) -> bool:
    return (
        len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _atomic_write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise ValueError("paired power artifact exceeds the bounded file size")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, _ARTIFACT_MODE)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_bounded_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_size > _MAX_ARTIFACT_BYTES:
            raise ValueError("paired power artifact exceeds the bounded file size")
        payload = bytearray()
        while len(payload) <= _MAX_ARTIFACT_BYTES:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_ARTIFACT_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > _MAX_ARTIFACT_BYTES:
            raise ValueError("paired power artifact exceeds the bounded file size")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _canonical_digest(value: JsonValue) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
