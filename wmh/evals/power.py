"""Finite Monte Carlo gate for a preregistered paired-confirmation simulation."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import stat
import sys
import uuid
from collections import Counter
from collections.abc import Iterable
from decimal import ROUND_CEILING, Decimal, localcontext
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import numpy as np
import pydantic
import scipy
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
    paired_member_primary_decision_passed,
    paired_primary_decision_passed,
)

PAIRED_POWER_GATE_VERSION: Literal["3"] = "3"
PAIRED_POWER_SIMULATION_VERSION: Literal["2"] = "2"
PAIRED_POWER_DGP_VERSION: Literal["1"] = "1"
PAIRED_POWER_TASK_PROFILE_VERSION: Literal["1"] = "1"
PAIRED_POWER_ARTIFACT_VERSION: Literal["2"] = "2"
PAIRED_POWER_CHUNK_VERSION: Literal["2"] = "2"
PAIRED_POWER_BITSET_ENCODING: Literal["base64-lsb0-bitset-v1"] = "base64-lsb0-bitset-v1"
PAIRED_POWER_RNG: Literal["numpy-pcg64dxsm-v1"] = "numpy-pcg64dxsm-v1"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_ARTIFACT_MODE = 0o600
_ARTIFACT_DIRECTORY_MODE = 0o700
_CP_MIN_DECIMAL_PRECISION = 120
_CP_MAX_OUTWARD_STEPS = 65_536


class PairedPowerRuntimeManifest(BaseModel):
    """Exact interpreter and numerical-library identity for reproducible simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_implementation: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    python_cache_tag: str = Field(min_length=1)
    python_executable_digest: str = Field(pattern=_DIGEST_PATTERN)
    platform_system: str = Field(min_length=1)
    platform_machine: str = Field(min_length=1)
    platform_release: str = Field(min_length=1)
    platform_version: str = Field(min_length=1)
    numpy_version: str = Field(min_length=1)
    numpy_distribution_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    scipy_version: str = Field(min_length=1)
    scipy_distribution_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    pydantic_version: str = Field(min_length=1)
    pydantic_distribution_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    pydantic_core_version: str = Field(min_length=1)
    pydantic_core_distribution_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    rng_implementation: Literal["numpy.random.PCG64DXSM-v1"] = "numpy.random.PCG64DXSM-v1"
    effect_solver_implementation: Literal["clipped-piecewise-linear-v1"] = (
        "clipped-piecewise-linear-v1"
    )
    cp_certifier_implementation: Literal["decimal-directed-tail-sum-v1"] = (
        "decimal-directed-tail-sum-v1"
    )

    @classmethod
    @cache
    def current(cls) -> PairedPowerRuntimeManifest:
        """Capture the exact runtime used by chunks and certified gate bounds."""
        cache_tag = sys.implementation.cache_tag
        if cache_tag is None:
            raise RuntimeError("paired power simulation requires a Python cache tag")
        executable = Path(sys.executable)
        if not executable.is_file():
            raise RuntimeError("paired power simulation requires a readable Python executable")
        return cls(
            python_implementation=platform.python_implementation(),
            python_version=sys.version,
            python_cache_tag=cache_tag,
            python_executable_digest=_bytes_digest(executable.read_bytes()),
            platform_system=platform.system(),
            platform_machine=platform.machine(),
            platform_release=platform.release(),
            platform_version=platform.version(),
            numpy_version=np.__version__,
            numpy_distribution_record_digest=_distribution_record_digest("numpy"),
            scipy_version=scipy.__version__,
            scipy_distribution_record_digest=_distribution_record_digest("scipy"),
            pydantic_version=pydantic.__version__,
            pydantic_distribution_record_digest=_distribution_record_digest("pydantic"),
            pydantic_core_version=importlib.metadata.version("pydantic-core"),
            pydantic_core_distribution_record_digest=_distribution_record_digest("pydantic-core"),
        )

    def validate_current(self) -> None:
        """Reject numerical execution under anything but the frozen runtime."""
        if self != self.current():
            raise ValueError("paired power runtime differs from the frozen numerical runtime")


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
    effect_assignment: Literal["deterministic-stratified-semantic-group-v1"] = (
        "deterministic-stratified-semantic-group-v1"
    )
    lane_sampling_model: Literal["independent-conditional-on-fixed-profile"] = (
        "independent-conditional-on-fixed-profile"
    )
    iut_null_evaluation: Literal["memberwise-marginal-conservative-upper-bound-v1"] = (
        "memberwise-marginal-conservative-upper-bound-v1"
    )
    other_lane_nuisance_bound: Literal["all-other-member-decisions-pass"] = (
        "all-other-member-decisions-pass"
    )
    task_vector_assumption: Literal["independent-conditional-on-frozen-rates-and-fixed-effects"] = (
        "independent-conditional-on-frozen-rates-and-fixed-effects"
    )

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
    """Fixed horizon and deterministic chunk layout for every simulation configuration."""

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

    simulation_version: Literal["2"] = PAIRED_POWER_SIMULATION_VERSION
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
    runtime: PairedPowerRuntimeManifest
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
            runtime=PairedPowerRuntimeManifest.current(),
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
        self.validate_execution_environment()
        expected = {
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

    def validate_execution_environment(self) -> None:
        """Reject runtime or executable source/schema drift without opening private inputs."""
        self.runtime.validate_current()
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
        }
        for label, (frozen, observed) in expected.items():
            if frozen != observed:
                raise ValueError(f"power simulation {label} differs from the frozen manifest")


class PairedPowerGateDesign(BaseModel):
    """Frozen operating-characteristic thresholds for one locked simulator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate_version: Literal["3"] = PAIRED_POWER_GATE_VERSION
    simulation_manifest: PairedPowerSimulationManifest
    simulation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_evaluation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    runtime: PairedPowerRuntimeManifest
    null_configurations: tuple[str, ...]
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

    @model_validator(mode="after")
    def _require_canonical_null_configurations(self) -> Self:
        if not self.null_configurations:
            raise ValueError("paired power gate needs at least one IUT null configuration")
        if self.null_configurations != tuple(sorted(set(self.null_configurations))):
            raise ValueError("paired power IUT null configurations must be unique and canonical")
        self.validate_frozen_manifest()
        return self

    def validate_frozen_manifest(self) -> None:
        """Bind every caller-visible gate parameter to the executable simulation manifest."""
        manifest = self.simulation_manifest
        manifest.validate_execution_environment()
        target = _scenario_manifest(manifest, "target-alternative").equal_task_effect
        expected: tuple[tuple[str, object, object], ...] = (
            ("simulation digest", self.simulation_design_digest, manifest.digest),
            (
                "paired evaluation design digest",
                self.paired_evaluation_design_digest,
                manifest.paired_evaluation_design_digest,
            ),
            ("runtime", self.runtime, manifest.runtime),
            ("null configurations", self.null_configurations, manifest.lane_set),
            ("target effect", self.target_effect, target),
            (
                "replication horizon",
                self.replications_per_scenario,
                manifest.replications.replications_per_scenario,
            ),
        )
        for label, observed, frozen in expected:
            if observed != frozen:
                verb = "differ" if label.endswith("s") else "differs"
                raise ValueError(f"paired power gate {label} {verb} from the manifest")

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
    null_configuration: str | None
    replicate: StrictInt = Field(ge=1)
    primary_passed: StrictBool

    @field_validator("replicate", mode="before")
    @classmethod
    def _reject_boolean_replicates(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("paired power replicate identities cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_scenario_configuration(self) -> Self:
        if self.scenario == "weak-null" and not self.null_configuration:
            raise ValueError("weak-null power trials require an IUT null configuration")
        if self.scenario == "target-alternative" and self.null_configuration is not None:
            raise ValueError("target power trials cannot carry an IUT null configuration")
        return self


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

    chunk_version: Literal["2"] = PAIRED_POWER_CHUNK_VERSION
    simulation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_evaluation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    scenario: Literal["weak-null", "target-alternative"]
    null_configuration: str | None
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
        if self.scenario == "weak-null" and not self.null_configuration:
            raise ValueError("weak-null power chunks require an IUT null configuration")
        if self.scenario == "target-alternative" and self.null_configuration is not None:
            raise ValueError("target power chunks cannot carry an IUT null configuration")
        return self

    @property
    def key(self) -> tuple[str, str | None, int, int]:
        """Return the canonical merge identity."""
        return self.scenario, self.null_configuration, self.first_replicate, self.last_replicate

    @property
    def digest(self) -> str:
        """Return the complete chunk identity."""
        return _canonical_digest(self.model_dump(mode="json"))


class PairedPowerScenarioArtifact(BaseModel):
    """One complete compact scenario bitmap in canonical replicate order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario: Literal["weak-null", "target-alternative"]
    null_configuration: str | None
    decisions: PairedPowerDecisionBits

    @model_validator(mode="after")
    def _validate_scenario_configuration(self) -> Self:
        if self.scenario == "weak-null" and not self.null_configuration:
            raise ValueError("weak-null artifacts require an IUT null configuration")
        if self.scenario == "target-alternative" and self.null_configuration is not None:
            raise ValueError("target artifacts cannot carry an IUT null configuration")
        return self


class PairedPowerTrialArtifact(BaseModel):
    """Compact complete digest-bound trials consumable by the exact power gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_version: Literal["2"] = PAIRED_POWER_ARTIFACT_VERSION
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
        weak_nulls = tuple(item for item in self.scenarios if item.scenario == "weak-null")
        targets = tuple(item for item in self.scenarios if item.scenario == "target-alternative")
        null_configurations = tuple(item.null_configuration for item in weak_nulls)
        if not weak_nulls or len(targets) != 1:
            raise ValueError("power artifact must contain memberwise nulls and one target scenario")
        if null_configurations != tuple(sorted(set(null_configurations))):
            raise ValueError("power artifact IUT null configurations must be unique and canonical")
        if self.scenarios != (*weak_nulls, targets[0]):
            raise ValueError("power artifact scenarios must use canonical null-then-target order")
        if any(item.decisions.count != self.replications_per_scenario for item in self.scenarios):
            raise ValueError("power artifact scenarios must fill the frozen horizon")
        if not self.chunk_digests or len(self.chunk_digests) != len(set(self.chunk_digests)):
            raise ValueError("power artifact chunk digests must be nonempty and unique")
        if any(not _is_digest(value) for value in self.chunk_digests):
            raise ValueError("power artifact contains an invalid chunk digest")
        return self

    @property
    def null_configurations(self) -> tuple[str, ...]:
        """Return the frozen memberwise IUT null identities."""
        return tuple(
            item.null_configuration
            for item in self.scenarios
            if item.scenario == "weak-null" and item.null_configuration is not None
        )

    @property
    def digest(self) -> str:
        """Return the canonical complete trial-artifact identity."""
        return _canonical_digest(self.model_dump(mode="json"))

    @property
    def trial_evidence_digest(self) -> str:
        """Return the v3 gate identity of the represented expanded trial records."""
        return _artifact_trial_evidence_digest(self)

    def rejection_count(
        self,
        scenario: Literal["weak-null", "target-alternative"],
        *,
        null_configuration: str | None = None,
    ) -> int:
        """Return primary passes for one exact frozen scenario."""
        for item in self.scenarios:
            if item.scenario == scenario and item.null_configuration == null_configuration:
                return item.decisions.rejection_count
        raise ValueError(f"power artifact omits scenario {scenario}/{null_configuration}")


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


class PairedPowerNullConfigurationReport(BaseModel):
    """One memberwise IUT boundary-null Monte Carlo result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    null_configuration: str = Field(min_length=1)
    rejections: StrictInt = Field(ge=0)
    empirical_type_i_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    type_i_error_upper_bound: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class PairedPowerGateReport(BaseModel):
    """Environment-bound Monte Carlo evidence, bounds, and decisions.

    Reload validation requires the exact frozen runtime, executable, source, and
    schema identities embedded by the design. Transfer to a different environment
    fails closed even when the serialized evidence itself is unchanged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate_version: Literal["3"]
    design: PairedPowerGateDesign
    trial_evidence_digest: str = Field(pattern=_DIGEST_PATTERN)
    null_configuration_monte_carlo_alpha: float = Field(
        gt=0.0,
        lt=1.0,
        allow_inf_nan=False,
    )
    null_configurations: tuple[PairedPowerNullConfigurationReport, ...]
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
        self.design.validate_frozen_manifest()
        count = self.design.replications_per_scenario
        if self.null_rejections > count or self.target_rejections > count:
            raise ValueError("paired power rejection counts cannot exceed frozen replications")
        observed_configurations = tuple(
            item.null_configuration for item in self.null_configurations
        )
        if observed_configurations != self.design.null_configurations:
            raise ValueError("paired power report IUT null configurations differ from the design")
        null_alpha = _divide_float_downward(
            self.design.monte_carlo_alpha,
            len(self.design.null_configurations),
        )
        if self.null_configuration_monte_carlo_alpha != null_alpha:
            raise ValueError(
                "paired power null Monte Carlo alpha differs from its frozen allocation"
            )
        for item in self.null_configurations:
            if item.rejections > count:
                raise ValueError("paired power null rejections cannot exceed frozen replications")
            expected_rate = item.rejections / count
            expected_upper = _binomial_upper_bound(item.rejections, count, alpha=null_alpha)
            if item.empirical_type_i_error != expected_rate:
                raise ValueError("paired power null empirical rate differs from its evidence")
            if item.type_i_error_upper_bound != expected_upper:
                raise ValueError("paired power null upper bound differs from its evidence")
        expected_null_rejections = max(item.rejections for item in self.null_configurations)
        expected_type_i_error = max(
            item.empirical_type_i_error for item in self.null_configurations
        )
        expected_type_i_upper = max(
            item.type_i_error_upper_bound for item in self.null_configurations
        )
        if self.null_rejections != expected_null_rejections:
            raise ValueError("paired power null rejection maximum differs from its evidence")
        expected_power = self.target_rejections / count
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
    identities, and extra replicates. It uses one-sided exact Clopper-Pearson bounds,
    with the preregistered Monte Carlo alpha divided downward across the complete
    memberwise null roster. Passing supports only the design's frozen target effect
    and assumptions; it does not establish power for an untested effect size.
    """
    design.validate_frozen_manifest()
    if isinstance(trials, PairedPowerTrialArtifact):
        null_counts, target_rejections = _artifact_rejection_counts(design, trials)
        trial_evidence_digest = trials.trial_evidence_digest
    else:
        null_counts, target_rejections = _expanded_trial_rejection_counts(design, trials)
        trial_evidence_digest = _expanded_trial_evidence_digest(trials)
    count = design.replications_per_scenario
    null_alpha = _divide_float_downward(
        design.monte_carlo_alpha,
        len(design.null_configurations),
    )
    null_configuration_reports = tuple(
        PairedPowerNullConfigurationReport(
            null_configuration=null_configuration,
            rejections=null_counts[null_configuration],
            empirical_type_i_error=null_counts[null_configuration] / count,
            type_i_error_upper_bound=_binomial_upper_bound(
                null_counts[null_configuration],
                count,
                alpha=null_alpha,
            ),
        )
        for null_configuration in design.null_configurations
    )
    null_rejections = max(null_counts.values())
    empirical_type_i_error = max(item.empirical_type_i_error for item in null_configuration_reports)
    empirical_power = target_rejections / count
    type_i_error_upper_bound = max(
        item.type_i_error_upper_bound for item in null_configuration_reports
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
        "null_configuration_monte_carlo_alpha": null_alpha,
        "null_configurations": [
            item.model_dump(mode="json") for item in null_configuration_reports
        ],
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
    null_configuration: str | None = None,
    first_replicate: int,
    last_replicate: int,
) -> PairedPowerTrialChunk:
    """Run one exact frozen chunk through the production v5 primary decision."""
    manifest.validate_frozen_inputs(evaluation_design, task_profile)
    if isinstance(first_replicate, bool) or isinstance(last_replicate, bool):
        raise ValueError("power chunk replicate bounds cannot be boolean")
    if (first_replicate, last_replicate) not in manifest.replications.chunk_ranges:
        raise ValueError("power chunk range is not one of the frozen deterministic ranges")
    if (scenario, null_configuration) not in _simulation_configurations(manifest):
        raise ValueError("power chunk scenario/configuration is not frozen in the manifest")
    scenario_manifest = _scenario_manifest(manifest, scenario)
    decisions = _simulate_primary_decisions(
        manifest,
        evaluation_design,
        task_profile,
        scenario_manifest,
        null_configuration=null_configuration,
        first_replicate=first_replicate,
        last_replicate=last_replicate,
    )
    return PairedPowerTrialChunk(
        simulation_design_digest=manifest.digest,
        paired_evaluation_design_digest=evaluation_design.digest,
        scenario=scenario,
        null_configuration=null_configuration,
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
        (scenario, null_configuration, first, last)
        for scenario, null_configuration in _simulation_configurations(manifest)
        for first, last in manifest.replications.chunk_ranges
    )
    if set(keys) != set(expected) or len(keys) != len(expected):
        raise ValueError("power chunks must exactly fill all frozen configuration horizons")
    by_key = {chunk.key: chunk for chunk in chunks}
    scenario_artifacts: list[PairedPowerScenarioArtifact] = []
    ordered_chunks: list[PairedPowerTrialChunk] = []
    for scenario, null_configuration in _simulation_configurations(manifest):
        decisions: list[bool] = []
        for first, last in manifest.replications.chunk_ranges:
            chunk = by_key[(scenario, null_configuration, first, last)]
            ordered_chunks.append(chunk)
            decisions.extend(chunk.decisions.decisions())
        scenario_artifacts.append(
            PairedPowerScenarioArtifact(
                scenario=scenario,
                null_configuration=null_configuration,
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
    directory_descriptor, directory_identity = _open_secure_directory(directory, create=True)
    primary_error: BaseException | None = None
    try:
        expected_names = {
            _chunk_filename(manifest, scenario, null_configuration, first, last)
            for scenario, null_configuration in _simulation_configurations(manifest)
            for first, last in manifest.replications.chunk_ranges
        }
        observed_names = set(os.listdir(directory_descriptor))
        unexpected_names = sorted(
            name for name in observed_names - expected_names if name.endswith(".json")
        )
        if unexpected_names:
            raise ValueError(
                f"power chunk directory contains unexpected JSON files: {unexpected_names}"
            )
        existing_chunks: dict[tuple[str, str | None, int, int], PairedPowerTrialChunk] = {}
        for scenario, null_configuration in _simulation_configurations(manifest):
            for first, last in manifest.replications.chunk_ranges:
                name = _chunk_filename(
                    manifest,
                    scenario,
                    null_configuration,
                    first,
                    last,
                )
                if name not in observed_names:
                    continue
                chunk = _load_paired_power_chunk_at(directory_descriptor, name)
                if chunk.key != (scenario, null_configuration, first, last):
                    raise ValueError("existing power chunk identity differs from its frozen path")
                if chunk.simulation_design_digest != manifest.digest:
                    raise ValueError(
                        "existing power chunk simulation digest differs from the manifest"
                    )
                if (
                    chunk.paired_evaluation_design_digest
                    != manifest.paired_evaluation_design_digest
                ):
                    raise ValueError(
                        "existing power chunk evaluation digest differs from the manifest"
                    )
                existing_chunks[chunk.key] = chunk
        generated: dict[tuple[str, str | None, int, int], PairedPowerTrialChunk] = {}
        for scenario, null_configuration in _simulation_configurations(manifest):
            for first, last in manifest.replications.chunk_ranges:
                key = (scenario, null_configuration, first, last)
                if key in existing_chunks:
                    continue
                chunk = run_paired_power_chunk(
                    manifest,
                    evaluation_design,
                    task_profile,
                    scenario=scenario,
                    null_configuration=null_configuration,
                    first_replicate=first,
                    last_replicate=last,
                )
                name = _chunk_filename(
                    manifest,
                    scenario,
                    null_configuration,
                    first,
                    last,
                )
                _write_paired_power_chunk_at(directory_descriptor, name, chunk)
                generated[key] = chunk
        chunks: list[PairedPowerTrialChunk] = []
        for scenario, null_configuration in _simulation_configurations(manifest):
            for first, last in manifest.replications.chunk_ranges:
                key = (scenario, null_configuration, first, last)
                name = _chunk_filename(
                    manifest,
                    scenario,
                    null_configuration,
                    first,
                    last,
                )
                reloaded = _load_paired_power_chunk_at(directory_descriptor, name)
                if reloaded != (existing_chunks | generated)[key]:
                    raise ValueError("power chunk changed during deterministic resume")
                chunks.append(reloaded)
        return merge_paired_power_chunks(manifest, tuple(chunks))
    except BaseException:
        primary_error = sys.exception()
        raise
    finally:
        _validate_and_close_directory(
            directory_descriptor,
            directory,
            directory_identity,
            primary_error=primary_error,
        )


def write_paired_power_chunk(path: str | Path, chunk: PairedPowerTrialChunk) -> None:
    """Atomically persist one complete chunk with a canonical payload digest."""
    envelope = _PairedPowerChunkEnvelope(artifact=chunk, artifact_digest=chunk.digest)
    _atomic_write_json(Path(path), envelope.model_dump(mode="json"))


def _write_paired_power_chunk_at(
    directory_descriptor: int,
    name: str,
    chunk: PairedPowerTrialChunk,
) -> None:
    envelope = _PairedPowerChunkEnvelope(artifact=chunk, artifact_digest=chunk.digest)
    _atomic_write_json_at(directory_descriptor, name, envelope.model_dump(mode="json"))


def load_paired_power_chunk(path: str | Path) -> PairedPowerTrialChunk:
    """Load one bounded chunk and verify both bitset and envelope integrity."""
    payload = _read_bounded_file(Path(path))
    return _PairedPowerChunkEnvelope.model_validate_json(payload).artifact


def _load_paired_power_chunk_at(
    directory_descriptor: int,
    name: str,
) -> PairedPowerTrialChunk:
    payload = _read_bounded_file_at(directory_descriptor, name)
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
    null_configuration: str | None,
    first_replicate: int,
    last_replicate: int,
) -> tuple[bool, ...]:
    """Simulate exact target decisions or a conservative memberwise IUT null bound.

    For null member ``j``, the all-member rejection event is a subset of member
    ``j``'s rejection event. The frozen DGP makes member ``j``'s marginal invariant
    to every other lane's nuisance effect. We therefore simulate only ``j`` at its
    boundary and project its exact decision, which is the Boolean configuration
    where every other member decision passes. No finite other-lane effect is needed.
    """
    replicate_count = last_replicate - first_replicate + 1
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            _chunk_seed(
                manifest,
                scenario.scenario,
                null_configuration,
                first_replicate,
                last_replicate,
            )
        )
    )
    task_multipliers = _fixed_task_effect_multipliers(manifest, task_profile)
    member_index = (
        evaluation_design.panel_members.index(null_configuration)
        if null_configuration is not None
        else None
    )
    lane_indices = (
        (member_index,) if member_index is not None else tuple(range(len(manifest.lane_set)))
    )
    task_deltas_by_member: dict[int, np.ndarray] = {}
    for lane_index in lane_indices:
        panel_member = evaluation_design.panel_members[lane_index]
        attempts = evaluation_design.panel[lane_index].attempts
        baseline = np.array(
            [_baseline_probability(task, panel_member) for task in task_profile.tasks],
            dtype=np.float64,
        )
        scale = _calibrated_fixed_effect_scale(
            baseline,
            task_multipliers,
            target_effect=scenario.equal_task_effect,
        )
        fixed_candidate_probability = np.clip(
            baseline + scale * task_multipliers,
            0.0,
            1.0,
        )
        candidate_probability = np.broadcast_to(
            fixed_candidate_probability[None, :],
            (replicate_count, len(task_profile.tasks)),
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
        task_deltas_by_member[lane_index] = (candidate_count - baseline_count) / attempts
    decisions: list[bool] = []
    for replicate in range(replicate_count):
        if member_index is None:
            task_deltas = tuple(
                tuple(float(delta) for delta in task_deltas_by_member[index][replicate])
                for index in range(len(manifest.lane_set))
            )
            decisions.append(paired_primary_decision_passed(evaluation_design, task_deltas))
        else:
            decisions.append(
                paired_member_primary_decision_passed(
                    evaluation_design,
                    evaluation_design.panel_members[member_index],
                    tuple(float(delta) for delta in task_deltas_by_member[member_index][replicate]),
                )
            )
    return tuple(decisions)


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
    pair_baselines = np.repeat(baseline, len(atoms))
    multipliers = np.tile(
        np.array([atom.multiplier for atom in atoms], dtype=np.float64),
        len(baseline),
    )
    weights = np.tile(
        np.array([atom.probability for atom in atoms], dtype=np.float64),
        len(baseline),
    ) / len(baseline)
    return _solve_clipped_additive_scale(
        pair_baselines,
        multipliers,
        weights,
        target_effect=target_effect,
    )


def _calibrated_fixed_effect_scale(
    baseline: np.ndarray,
    task_multipliers: np.ndarray,
    *,
    target_effect: float,
) -> float:
    return _solve_clipped_additive_scale(
        baseline,
        task_multipliers,
        np.full(len(baseline), 1.0 / len(baseline), dtype=np.float64),
        target_effect=target_effect,
    )


def _solve_clipped_additive_scale(
    baselines: np.ndarray,
    multipliers: np.ndarray,
    weights: np.ndarray,
    *,
    target_effect: float,
) -> float:
    """Find the first feasible scale over every piecewise-linear clipping segment."""
    if target_effect == 0.0:
        return 0.0
    if not (len(baselines) == len(multipliers) == len(weights)) or not len(baselines):
        raise ValueError("power effect calibration vectors must be nonempty and aligned")

    def expected_effect(scale: float) -> float:
        deltas = np.clip(baselines + scale * multipliers, 0.0, 1.0) - baselines
        return math.fsum(
            float(weight * delta) for weight, delta in zip(weights, deltas, strict=True)
        )

    breakpoints = {0.0}
    for baseline, multiplier in zip(baselines, multipliers, strict=True):
        if multiplier > 0.0:
            breakpoint = (1.0 - float(baseline)) / float(multiplier)
        elif multiplier < 0.0:
            breakpoint = float(baseline) / -float(multiplier)
        else:
            continue
        if math.isfinite(breakpoint) and breakpoint >= 0.0:
            breakpoints.add(breakpoint)
    ordered = sorted(breakpoints)
    values = [expected_effect(point) for point in ordered]
    for point, value in zip(ordered, values, strict=True):
        if value == target_effect:
            return point
    for left, right, left_value, right_value in zip(
        ordered,
        ordered[1:],
        values,
        values[1:],
        strict=True,
    ):
        if target_effect < min(left_value, right_value):
            continue
        if target_effect > max(left_value, right_value):
            continue
        slope = (right_value - left_value) / (right - left)
        if slope == 0.0:
            continue
        candidate = left + (target_effect - left_value) / slope
        candidate = min(right, max(left, candidate))
        achieved = expected_effect(candidate)
        if target_effect > 0.0 and achieved <= 0.0:
            continue
        clipped = np.clip(baselines + candidate * multipliers, 0.0, 1.0)
        rounding_bound = math.fsum(
            abs(float(weight))
            * (
                math.ulp(float(baseline))
                + math.ulp(float(candidate_probability))
                + abs(float(multiplier)) * math.ulp(candidate)
            )
            for baseline, candidate_probability, multiplier, weight in zip(
                baselines,
                clipped,
                multipliers,
                weights,
                strict=True,
            )
        ) + math.ulp(target_effect)
        if abs(achieved - target_effect) <= rounding_bound:
            return candidate
    maximum = max(values)
    raise ValueError(
        "target effect is infeasible under the frozen baseline and effect shape "
        f"(maximum={maximum:.17g})"
    )


def _fixed_task_effect_multipliers(
    manifest: PairedPowerSimulationManifest,
    task_profile: PairedPowerTaskProfileManifest,
) -> np.ndarray:
    """Assign one deterministic stratified atom to each semantic group."""
    group_ids = tuple(sorted({task.group_id for task in task_profile.tasks}))
    domain = {
        "domain": "paired-power-fixed-semantic-effects-v1",
        "root_seed": manifest.seeds.root_seed,
        "task_profile_digest": manifest.task_profile_digest,
        "effect_shape": manifest.dgp.effect_shape.model_dump(mode="json"),
    }

    def group_order_key(group_id: str) -> bytes:
        return hashlib.sha256(_canonical_json_bytes({**domain, "group_id": group_id})).digest()

    ordered_groups = sorted(group_ids, key=group_order_key)
    atoms = manifest.dgp.effect_shape.atoms
    assignment: dict[str, float] = {}
    for rank, group_id in enumerate(ordered_groups):
        quantile = (rank + 0.5) / len(ordered_groups)
        cumulative = 0.0
        selected = atoms[-1]
        for atom in atoms:
            cumulative += atom.probability
            if quantile <= cumulative:
                selected = atom
                break
        assignment[group_id] = selected.multiplier
    return np.array([assignment[task.group_id] for task in task_profile.tasks], dtype=np.float64)


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
    null_configuration: str | None,
    first_replicate: int,
    last_replicate: int,
) -> int:
    payload = json.dumps(
        {
            "root_seed": manifest.seeds.root_seed,
            "simulation_design_digest": manifest.digest,
            "scenario": scenario,
            "null_configuration": null_configuration,
            "first_replicate": first_replicate,
            "last_replicate": last_replicate,
            "rng": manifest.seeds.rng,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _simulation_configurations(
    manifest: PairedPowerSimulationManifest,
) -> tuple[tuple[Literal["weak-null", "target-alternative"], str | None], ...]:
    return tuple(("weak-null", panel_member) for panel_member in manifest.lane_set) + (
        ("target-alternative", None),
    )


def _chunk_filename(
    manifest: PairedPowerSimulationManifest,
    scenario: str,
    null_configuration: str | None,
    first_replicate: int,
    last_replicate: int,
) -> str:
    if null_configuration is None:
        stem = scenario
    else:
        stem = f"{scenario}-member-{manifest.lane_set.index(null_configuration) + 1:03d}"
    return f"{stem}-{first_replicate:09d}-{last_replicate:09d}.json"


def _expanded_trial_rejection_counts(
    design: PairedPowerGateDesign,
    trials: tuple[PairedPowerTrial, ...],
) -> tuple[dict[str, int], int]:
    expected_replicates = tuple(range(1, design.replications_per_scenario + 1))
    if any(trial.simulation_design_digest != design.simulation_design_digest for trial in trials):
        raise ValueError("paired power trial simulation design digest differs from the gate")
    if any(
        trial.paired_evaluation_design_digest != design.paired_evaluation_design_digest
        for trial in trials
    ):
        raise ValueError("paired power trial evaluation design digest differs from the gate")
    keys = [(trial.scenario, trial.null_configuration, trial.replicate) for trial in trials]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"paired power trials contain duplicate replicate identities: {duplicates}"
        )
    expected_scenarios = tuple(
        ("weak-null", null_configuration) for null_configuration in design.null_configurations
    ) + (("target-alternative", None),)
    observed = {
        scenario_key: tuple(
            sorted(
                trial.replicate
                for trial in trials
                if (trial.scenario, trial.null_configuration) == scenario_key
            )
        )
        for scenario_key in expected_scenarios
    }
    observed_keys = {(trial.scenario, trial.null_configuration) for trial in trials}
    expected = {scenario_key: expected_replicates for scenario_key in expected_scenarios}
    if observed != expected or observed_keys != set(expected_scenarios):
        raise ValueError("paired power trials must exactly fill all frozen configurations")
    return (
        {
            null_configuration: sum(
                trial.primary_passed
                for trial in trials
                if trial.scenario == "weak-null" and trial.null_configuration == null_configuration
            )
            for null_configuration in design.null_configurations
        },
        sum(trial.primary_passed for trial in trials if trial.scenario == "target-alternative"),
    )


def _artifact_rejection_counts(
    design: PairedPowerGateDesign,
    artifact: PairedPowerTrialArtifact,
) -> tuple[dict[str, int], int]:
    if artifact.simulator_schema_digest != design.simulation_manifest.simulator_schema_digest:
        raise ValueError("paired power artifact simulator schema digest differs from the gate")
    if artifact.simulation_design_digest != design.simulation_design_digest:
        raise ValueError("paired power artifact simulation design digest differs from the gate")
    if artifact.paired_evaluation_design_digest != design.paired_evaluation_design_digest:
        raise ValueError("paired power artifact evaluation design digest differs from the gate")
    if artifact.replications_per_scenario != design.replications_per_scenario:
        raise ValueError("paired power artifact does not fill the gate replication horizon")
    if artifact.target_effect != design.target_effect:
        raise ValueError("paired power artifact target effect differs from the gate")
    if artifact.null_configurations != design.null_configurations:
        raise ValueError("paired power artifact IUT null configurations differ from the gate")
    return (
        {
            null_configuration: artifact.rejection_count(
                "weak-null",
                null_configuration=null_configuration,
            )
            for null_configuration in design.null_configurations
        },
        artifact.rejection_count("target-alternative", null_configuration=None),
    )


def _expanded_trial_evidence_digest(trials: tuple[PairedPowerTrial, ...]) -> str:
    canonical_trials = sorted(
        trials,
        key=lambda trial: (
            trial.scenario,
            trial.null_configuration or "",
            trial.replicate,
        ),
    )
    return _canonical_sequence_digest(
        _trial_evidence_record(
            simulation_design_digest=trial.simulation_design_digest,
            paired_evaluation_design_digest=trial.paired_evaluation_design_digest,
            scenario=trial.scenario,
            null_configuration=trial.null_configuration,
            replicate=trial.replicate,
            primary_passed=trial.primary_passed,
        )
        for trial in canonical_trials
    )


def _artifact_trial_evidence_digest(artifact: PairedPowerTrialArtifact) -> str:
    canonical_scenarios = sorted(
        artifact.scenarios,
        key=lambda item: (item.scenario, item.null_configuration or ""),
    )
    return _canonical_sequence_digest(
        _trial_evidence_record(
            simulation_design_digest=artifact.simulation_design_digest,
            paired_evaluation_design_digest=artifact.paired_evaluation_design_digest,
            scenario=scenario.scenario,
            null_configuration=scenario.null_configuration,
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
    null_configuration: str | None,
    replicate: int,
    primary_passed: bool,
) -> dict[str, JsonValue]:
    return {
        "simulation_design_digest": simulation_design_digest,
        "paired_evaluation_design_digest": paired_evaluation_design_digest,
        "scenario": scenario,
        "null_configuration": null_configuration,
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


def _divide_float_downward(numerator: float, denominator: int) -> float:
    """Divide a positive binary float without exceeding the exact real quotient."""
    if numerator <= 0.0 or denominator <= 0:
        raise ValueError("downward float division requires positive operands")
    candidate = numerator / denominator
    threshold = Decimal.from_float(numerator)
    divisor = Decimal(denominator)
    while Decimal.from_float(candidate) * divisor > threshold:
        next_candidate = math.nextafter(candidate, -math.inf)
        if next_candidate <= 0.0 or next_candidate == candidate:
            raise RuntimeError("could not round the Monte Carlo alpha allocation downward")
        candidate = next_candidate
    return candidate


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


def _distribution_record_digest(distribution_name: str) -> str:
    """Bind relocation-invariant installed contents for one numerical dependency."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"paired power runtime is missing distribution {distribution_name}"
        ) from exc
    records = tuple(entry for entry in distribution.files or () if entry.name == "RECORD")
    if len(records) != 1:
        raise RuntimeError(
            f"paired power runtime cannot identify {distribution_name} distribution RECORD"
        )
    record_path = Path(str(distribution.locate_file(records[0])))
    if not record_path.is_file():
        raise RuntimeError(
            f"paired power runtime cannot read {distribution_name} distribution RECORD"
        )
    return _canonical_distribution_record_digest(record_path)


def _canonical_distribution_record_digest(record_path: Path) -> str:
    """Hash actual in-distribution files selected by a wheel RECORD.

    Installers rewrite external console scripts with environment-specific shebangs
    and record them through paths such as ``../../../bin/f2py``. Those paths are
    outside the distribution root and cannot identify the wheel or its numerical
    implementation, so they are excluded. Safe in-root dot components are normalized
    before hashing. Every internal file is read, checked against its recorded hash and
    size, and bound by canonical relative path and actual SHA-256 digest. The
    self-referential RECORD row is excluded.
    """
    if record_path.is_symlink() or not record_path.is_file():
        raise RuntimeError("paired power distribution RECORD must be a regular file")
    distribution_root = record_path.parent.parent
    try:
        record_text = record_path.read_text(encoding="utf-8")
        record_relative_path = PurePosixPath(record_path.relative_to(distribution_root).as_posix())
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("paired power distribution RECORD cannot be read canonically") from exc
    inventory: list[dict[str, JsonValue]] = []
    observed_paths: set[str] = set()
    for row in csv.reader(io.StringIO(record_text, newline="")):
        if len(row) != 3:
            raise RuntimeError("paired power distribution RECORD row is malformed")
        relative_text, recorded_hash, recorded_size = row
        relative_path = PurePosixPath(relative_text)
        if not relative_text or relative_path.is_absolute() or ".." in relative_path.parts:
            continue
        canonical_relative = relative_path.as_posix()
        if relative_path == record_relative_path:
            continue
        if canonical_relative in observed_paths:
            raise RuntimeError("paired power distribution RECORD contains duplicate paths")
        observed_paths.add(canonical_relative)
        installed_path = distribution_root.joinpath(*relative_path.parts)
        if installed_path.is_symlink() or not installed_path.is_file():
            raise RuntimeError("paired power distribution RECORD internal file is unavailable")
        try:
            payload = installed_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                "paired power distribution RECORD internal file cannot be read"
            ) from exc
        actual_hash = hashlib.sha256(payload).digest()
        if recorded_hash:
            algorithm, separator, encoded_hash = recorded_hash.partition("=")
            if separator != "=" or algorithm != "sha256":
                raise RuntimeError("paired power distribution RECORD hash is not SHA-256")
            expected_hash = base64.urlsafe_b64encode(actual_hash).rstrip(b"=").decode()
            if encoded_hash != expected_hash:
                raise RuntimeError("paired power distribution content does not match RECORD hash")
        if recorded_size:
            try:
                expected_size = int(recorded_size)
            except ValueError as exc:
                raise RuntimeError("paired power distribution RECORD size is malformed") from exc
            if expected_size != len(payload):
                raise RuntimeError("paired power distribution content does not match RECORD size")
        inventory.append(
            {
                "path": canonical_relative,
                "sha256": "sha256:" + actual_hash.hex(),
                "size": len(payload),
            }
        )
    if not inventory:
        raise RuntimeError("paired power distribution RECORD has no internal files")
    return _canonical_digest(sorted(inventory, key=lambda item: str(item["path"])))


def _bytes_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _is_digest(value: str) -> bool:
    return (
        len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _atomic_write_json(path: Path, value: JsonValue) -> None:
    directory_descriptor, directory_identity = _open_secure_directory(path.parent, create=True)
    primary_error: BaseException | None = None
    try:
        _atomic_write_json_at(directory_descriptor, path.name, value)
    except BaseException:
        primary_error = sys.exception()
        raise
    finally:
        _validate_and_close_directory(
            directory_descriptor,
            path.parent,
            directory_identity,
            primary_error=primary_error,
        )


def _read_bounded_file(path: Path) -> bytes:
    directory_descriptor, directory_identity = _open_secure_directory(path.parent, create=False)
    primary_error: BaseException | None = None
    try:
        return _read_bounded_file_at(directory_descriptor, path.name)
    except BaseException:
        primary_error = sys.exception()
        raise
    finally:
        _validate_and_close_directory(
            directory_descriptor,
            path.parent,
            directory_identity,
            primary_error=primary_error,
        )


def _open_secure_directory(path: Path, *, create: bool) -> tuple[int, tuple[int, int]]:
    _require_descriptor_relative_filesystem()
    absolute_path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(absolute_path.anchor, flags)
    components = absolute_path.parts[1:]
    try:
        for index, component in enumerate(components):
            final_component = index == len(components) - 1
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create:
                    raise ValueError("paired power artifact directory is unavailable") from None
                try:
                    os.mkdir(component, _ARTIFACT_DIRECTORY_MODE, dir_fd=descriptor)
                except FileExistsError:
                    pass
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    "paired power artifact directory path contains a symlink or non-directory"
                )
            try:
                opened_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ValueError(
                    "paired power artifact directory component cannot be opened securely"
                ) from exc
            try:
                opened = os.fstat(opened_descriptor)
                if not stat.S_ISDIR(opened.st_mode) or _inode_identity(before) != _inode_identity(
                    opened
                ):
                    raise ValueError("paired power artifact directory changed while opening")
            except BaseException:
                os.close(opened_descriptor)
                raise
            os.close(descriptor)
            descriptor = opened_descriptor
            if final_component:
                _validate_directory_metadata(opened)
        if not components:
            _validate_directory_metadata(os.fstat(descriptor))
        opened = os.fstat(descriptor)
        return descriptor, _inode_identity(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _validate_directory_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("paired power artifact directory is not a directory")
    if metadata.st_uid != os.getuid():
        raise ValueError("paired power artifact directory owner is not the current user")
    if stat.S_IMODE(metadata.st_mode) != _ARTIFACT_DIRECTORY_MODE:
        raise ValueError("paired power artifact directory mode must be 0700")


def _validate_directory_identity(
    descriptor: int,
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    opened = os.fstat(descriptor)
    _validate_directory_metadata(opened)
    if _inode_identity(opened) != expected_identity:
        raise ValueError("paired power artifact directory descriptor changed identity")
    try:
        current_descriptor, current_identity = _open_secure_directory(path, create=False)
    except (OSError, ValueError) as exc:
        raise ValueError("paired power artifact directory path was replaced") from exc
    try:
        if current_identity != expected_identity:
            raise ValueError("paired power artifact directory path was replaced")
    finally:
        os.close(current_descriptor)


def _validate_and_close_directory(
    descriptor: int,
    path: Path,
    expected_identity: tuple[int, int],
    *,
    primary_error: BaseException | None,
) -> None:
    """Validate and close without replacing an active primary exception."""
    cleanup_errors: list[tuple[str, Exception]] = []
    try:
        _validate_directory_identity(descriptor, path, expected_identity)
    except Exception as exc:  # noqa: BLE001 - cleanup cannot replace an active error
        cleanup_errors.append(("directory identity validation", exc))
    try:
        os.close(descriptor)
    except OSError as exc:
        cleanup_errors.append(("directory descriptor close", exc))
    if not cleanup_errors:
        return
    if primary_error is not None:
        for operation, error in cleanup_errors:
            primary_error.add_note(f"secondary {operation} failed: {type(error).__name__}: {error}")
        return
    primary_cleanup_error = cleanup_errors[0][1]
    for operation, error in cleanup_errors[1:]:
        primary_cleanup_error.add_note(
            f"secondary {operation} failed: {type(error).__name__}: {error}"
        )
    raise primary_cleanup_error


def _atomic_write_json_at(directory_descriptor: int, name: str, value: JsonValue) -> None:
    _validate_leaf_name(name)
    payload = _canonical_json_bytes(value)
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise ValueError("paired power artifact exceeds the bounded file size")
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(name)
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(
        temporary,
        flags,
        _ARTIFACT_MODE,
        dir_fd=directory_descriptor,
    )
    try:
        os.fchmod(descriptor, _ARTIFACT_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("paired power artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        written_metadata = os.fstat(descriptor)
        _validate_artifact_metadata(written_metadata)
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
        published = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        _validate_artifact_metadata(published)
        if _inode_identity(published) != _inode_identity(written_metadata):
            raise ValueError("paired power artifact changed during publication")
    except Exception:
        try:
            os.unlink(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        raise


def _read_bounded_file_at(directory_descriptor: int, name: str) -> bytes:
    _validate_leaf_name(name)
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("paired power artifact file is unavailable") from exc
    _validate_artifact_metadata(before)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise ValueError("paired power artifact file cannot be opened securely") from exc
    try:
        opened = os.fstat(descriptor)
        _validate_artifact_metadata(opened)
        expected_identity = _stable_file_identity(before)
        if _stable_file_identity(opened) != expected_identity:
            raise ValueError("paired power artifact changed while opening")
        payload = bytearray()
        while len(payload) <= _MAX_ARTIFACT_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_ARTIFACT_BYTES + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > _MAX_ARTIFACT_BYTES:
            raise ValueError("paired power artifact exceeds the bounded file size")
        after = os.fstat(descriptor)
        if _stable_file_identity(after) != expected_identity:
            raise ValueError("paired power artifact changed while reading")
    finally:
        os.close(descriptor)
    try:
        current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("paired power artifact was replaced while reading") from exc
    if _stable_file_identity(current) != expected_identity:
        raise ValueError("paired power artifact was replaced while reading")
    return bytes(payload)


def _validate_artifact_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("paired power artifact must be a regular file")
    if metadata.st_uid != os.getuid():
        raise ValueError("paired power artifact owner is not the current user")
    if stat.S_IMODE(metadata.st_mode) != _ARTIFACT_MODE:
        raise ValueError("paired power artifact mode must be 0600")
    if metadata.st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("paired power artifact exceeds the bounded file size")


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _validate_leaf_name(name: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("paired power artifact name must be one safe path component")


def _require_descriptor_relative_filesystem() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise RuntimeError("paired power artifacts require POSIX no-follow filesystem flags")
    required_dir_fd = (os.open, os.stat, os.mkdir, os.link, os.unlink)
    if any(function not in os.supports_dir_fd for function in required_dir_fd):
        raise RuntimeError("paired power artifacts require descriptor-relative filesystem calls")


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
