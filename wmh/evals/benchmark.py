"""Benchmark-neutral contracts for ground-truth task evaluation.

These models keep benchmark evidence independent from the orchestrator that produced it. A scored
zero is data, while a missing score, cancellation, and infrastructure failure remain distinct. The
contract deliberately preserves the verifier's complete reward mapping instead of choosing a
benchmark-specific headline key.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import StrEnum
from statistics import fmean
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RewardValue = float | int
Rewards = dict[str, RewardValue]
_SHA256_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class BenchmarkTaskEnvironment(StrEnum):
    """Where Harbor runs the ground-truth task environment."""

    DOCKER = "docker"
    E2B = "e2b"


class BenchmarkTrialStatus(StrEnum):
    """Whether one planned benchmark cell is usable, pending, or operationally failed."""

    SCORED = "scored"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    TASK_TIMEOUT = "task_timeout"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    UNCLASSIFIED_ERROR = "unclassified_error"


class BenchmarkFailureKind(StrEnum):
    """Portable failure classes that must not be collapsed into reward zero."""

    CANCELLED = "cancelled"
    TASK_TIMEOUT = "task_timeout"
    ENVIRONMENT = "environment"
    PROVIDER = "provider"
    VERIFIER = "verifier"
    MALFORMED_RESULT = "malformed_result"
    UNCLASSIFIED = "unclassified"


class BenchmarkCandidateStatus(StrEnum):
    """Whether trusted adapter evidence observed candidate completion or failure."""

    UNKNOWN = "unknown"
    COMPLETED = "completed"
    FAILED = "failed"


class BenchmarkCandidateStage(StrEnum):
    """Backend-neutral phase in which a candidate-controlled failure occurred."""

    SETUP = "setup"
    EXECUTION = "execution"


class BenchmarkCandidateTerminalReason(StrEnum):
    """Bounded reason a candidate execution reached its ordinary terminal boundary."""

    COMPLETED = "completed"
    LIMIT_REACHED = "limit_reached"
    ABORTED = "aborted"


class BenchmarkCandidateOutcome(BaseModel):
    """Typed candidate outcome retained without copying arbitrary backend metadata."""

    status: BenchmarkCandidateStatus = BenchmarkCandidateStatus.UNKNOWN
    stage: BenchmarkCandidateStage | None = None
    terminal_reason: BenchmarkCandidateTerminalReason | None = None

    @model_validator(mode="after")
    def _validate_details(self) -> Self:
        if self.status is BenchmarkCandidateStatus.UNKNOWN:
            if self.stage is not None or self.terminal_reason is not None:
                raise ValueError("unknown candidate outcome cannot carry details")
        elif self.status is BenchmarkCandidateStatus.COMPLETED:
            if self.stage is not None:
                raise ValueError("completed candidate outcome cannot carry a failure stage")
        elif self.terminal_reason is not None:
            raise ValueError("failed candidate outcome cannot carry a terminal reason")
        return self


class BenchmarkCell(BaseModel):
    """Stable identity of one task attempt, independent of a backend's directory names."""

    model_config = ConfigDict(frozen=True)

    task_key: str = Field(
        min_length=1,
        description="Opaque, dataset-qualified task identity used to prevent name collisions",
    )
    task_name: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    config_digest: str = Field(
        pattern=_SHA256_DIGEST_PATTERN,
        description="Opaque digest of the resolved execution config for this cell",
    )


class BenchmarkError(BaseModel):
    """Portable exception evidence retained from an evaluator backend."""

    kind: BenchmarkFailureKind
    type: str = Field(min_length=1)
    message: str = ""
    traceback: str | None = None


class BenchmarkUsage(BaseModel):
    """Token and cost totals, with missing metering kept distinct from a measured zero."""

    input_tokens: int | None = Field(default=None, ge=0)
    cache_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("input_tokens", "cache_tokens", "output_tokens", "cost_usd", mode="before")
    @classmethod
    def _reject_boolean_measurements(cls, value: int | float | None) -> int | float | None:
        if isinstance(value, bool):
            raise ValueError("usage measurements cannot be boolean")
        return value


def aggregate_benchmark_usage(usages: Iterable[BenchmarkUsage]) -> BenchmarkUsage:
    """Aggregate exact trial usage while preserving unknown metering as missing."""
    collected = list(usages)
    return BenchmarkUsage(
        input_tokens=_sum_optional_int(usage.input_tokens for usage in collected),
        cache_tokens=_sum_optional_int(usage.cache_tokens for usage in collected),
        output_tokens=_sum_optional_int(usage.output_tokens for usage in collected),
        cost_usd=_sum_optional_float(usage.cost_usd for usage in collected),
    )


class BenchmarkRunIdentity(BaseModel):
    """Execution identity that every observed trial must match."""

    candidate_hash: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    task_environment: BenchmarkTaskEnvironment
    runner_image: str = Field(min_length=1)
    run_config_digest: str = Field(
        pattern=_SHA256_DIGEST_PATTERN,
        description="Opaque digest of replay-critical configuration shared by the run",
    )


class BenchmarkTrialResult(BaseModel):
    """Backend-neutral evidence for one task attempt."""

    cell: BenchmarkCell
    task_identity: str = Field(min_length=1)
    task_checksum: str = Field(min_length=1)
    source: str | None = None
    status: BenchmarkTrialStatus
    rewards: Rewards | None = None
    error: BenchmarkError | None = None
    usage: BenchmarkUsage = Field(default_factory=BenchmarkUsage)
    candidate_outcome: BenchmarkCandidateOutcome = Field(default_factory=BenchmarkCandidateOutcome)

    @field_validator("rewards", mode="before")
    @classmethod
    def _require_finite_numeric_rewards(cls, value: Rewards | None) -> Rewards | None:
        if value is None:
            return None
        for key, reward in value.items():
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                raise ValueError(f"reward {key!r} must be an integer or float, not boolean")
            if not math.isfinite(reward):
                raise ValueError(f"reward {key!r} must be finite")
        return value

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        if self.status is BenchmarkTrialStatus.SCORED:
            if self.rewards is None:
                raise ValueError(
                    "scored trial must carry rewards, including an empty reward mapping"
                )
            if self.error is not None and self.error.kind is not BenchmarkFailureKind.TASK_TIMEOUT:
                raise ValueError("scored trial only permits task-timeout error evidence")
            return self
        if self.rewards is not None:
            raise ValueError("non-scored trial cannot carry verifier rewards")
        if self.status is BenchmarkTrialStatus.INCOMPLETE:
            if self.error is not None:
                raise ValueError("incomplete trial cannot carry error evidence")
            return self
        if self.error is None:
            raise ValueError(f"{self.status.value} trial must carry an error")
        expected_kinds = {
            BenchmarkTrialStatus.TASK_TIMEOUT: {BenchmarkFailureKind.TASK_TIMEOUT},
            BenchmarkTrialStatus.CANCELLED: {BenchmarkFailureKind.CANCELLED},
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR: {
                BenchmarkFailureKind.ENVIRONMENT,
                BenchmarkFailureKind.PROVIDER,
                BenchmarkFailureKind.VERIFIER,
                BenchmarkFailureKind.MALFORMED_RESULT,
            },
            BenchmarkTrialStatus.UNCLASSIFIED_ERROR: {BenchmarkFailureKind.UNCLASSIFIED},
        }
        if self.error.kind not in expected_kinds[self.status]:
            raise ValueError(
                f"{self.status.value} trial cannot carry {self.error.kind.value} error evidence"
            )
        return self


class BenchmarkRunResult(BaseModel):
    """Portable result for a fixed benchmark job and all observed task attempts."""

    job_name: str = Field(min_length=1)
    identity: BenchmarkRunIdentity
    expected_cells: list[BenchmarkCell] = Field(default_factory=list)
    trials: list[BenchmarkTrialResult] = Field(default_factory=list)
    usage: BenchmarkUsage = Field(default_factory=BenchmarkUsage)

    @field_validator("expected_cells")
    @classmethod
    def _order_expected_cells(cls, value: list[BenchmarkCell]) -> list[BenchmarkCell]:
        return sorted(value, key=lambda cell: (cell.task_key, cell.attempt, cell.task_name))

    @field_validator("trials")
    @classmethod
    def _order_trials(cls, value: list[BenchmarkTrialResult]) -> list[BenchmarkTrialResult]:
        return sorted(
            value,
            key=lambda trial: (trial.cell.task_key, trial.cell.attempt, trial.cell.task_name),
        )

    @model_validator(mode="after")
    def _validate_trials(self) -> Self:
        expected_keys = [(cell.task_key, cell.attempt) for cell in self.expected_cells]
        duplicate_expected = sorted({key for key in expected_keys if expected_keys.count(key) > 1})
        if duplicate_expected:
            raise ValueError(f"duplicate expected benchmark cell(s): {duplicate_expected}")
        observed_keys = [(trial.cell.task_key, trial.cell.attempt) for trial in self.trials]
        duplicate_observed = sorted({key for key in observed_keys if observed_keys.count(key) > 1})
        if duplicate_observed:
            raise ValueError(f"duplicate observed benchmark cell(s): {duplicate_observed}")
        if set(observed_keys) != set(expected_keys):
            missing = sorted(set(expected_keys) - set(observed_keys))
            extra = sorted(set(observed_keys) - set(expected_keys))
            raise ValueError(
                f"benchmark cells do not match manifest: missing={missing}, extra={extra}"
            )
        expected_by_key = {(cell.task_key, cell.attempt): cell for cell in self.expected_cells}
        observed_by_key = {
            (trial.cell.task_key, trial.cell.attempt): trial.cell for trial in self.trials
        }
        mismatched = sorted(
            key for key, cell in observed_by_key.items() if cell != expected_by_key[key]
        )
        if mismatched:
            raise ValueError(
                f"observed benchmark cell identity differs from manifest: {mismatched}"
            )
        aggregate_usage = aggregate_benchmark_usage(trial.usage for trial in self.trials)
        if "usage" not in self.model_fields_set:
            self.usage = aggregate_usage
        elif self.usage != aggregate_usage:
            raise ValueError("benchmark run usage must equal the aggregate of trial usage")
        return self

    @property
    def expected_trials(self) -> int:
        """Return the exact number of cells in the trusted manifest."""
        return len(self.expected_cells)

    @property
    def n_scored(self) -> int:
        """Return the number of trials with verifier rewards, including zero rewards."""
        return sum(trial.status is BenchmarkTrialStatus.SCORED for trial in self.trials)

    @property
    def n_infrastructure_errors(self) -> int:
        """Return the number of terminal evaluator or environment failures."""
        return sum(
            trial.status is BenchmarkTrialStatus.INFRASTRUCTURE_ERROR for trial in self.trials
        )

    @property
    def n_incomplete(self) -> int:
        """Return planned cells that have not reached a terminal outcome."""
        return sum(trial.status is BenchmarkTrialStatus.INCOMPLETE for trial in self.trials)

    @property
    def n_cancelled(self) -> int:
        """Return terminal cancelled attempts that require a newly named job to rerun."""
        return sum(trial.status is BenchmarkTrialStatus.CANCELLED for trial in self.trials)

    @property
    def n_task_timeouts(self) -> int:
        """Return task attempts that exhausted their agent execution budget."""
        return sum(
            trial.error is not None and trial.error.kind is BenchmarkFailureKind.TASK_TIMEOUT
            for trial in self.trials
        )

    @property
    def n_unclassified_errors(self) -> int:
        """Return errors that were retained without being mislabeled as infrastructure."""
        return sum(trial.status is BenchmarkTrialStatus.UNCLASSIFIED_ERROR for trial in self.trials)

    @property
    def is_complete(self) -> bool:
        """Return whether every planned cell has reached any terminal outcome."""
        return self.n_incomplete == 0

    def mean_reward(self, key: str) -> float:
        """Return one verifier-reward mean only when the entire planned matrix is scoreable."""
        if not key:
            raise ValueError("reward key must be non-empty")
        if not self.trials:
            raise ValueError("cannot aggregate an empty benchmark matrix")
        invalid = [
            trial.cell.task_key
            for trial in self.trials
            if trial.status is not BenchmarkTrialStatus.SCORED
        ]
        if invalid:
            raise ValueError(
                f"cannot aggregate reward {key!r}; {len(invalid)} planned cells are not scored"
            )
        missing = [
            trial.cell.task_key
            for trial in self.trials
            if trial.rewards is None or key not in trial.rewards
        ]
        if missing:
            raise ValueError(
                f"cannot aggregate reward {key!r}; {len(missing)} scored cells omit that key"
            )
        values = [
            float(trial.rewards[key])
            for trial in self.trials
            if trial.rewards is not None and key in trial.rewards
        ]
        return fmean(values)


def _sum_optional_int(values: Iterable[int | None]) -> int | None:
    collected = list(values)
    if not collected or any(value is None for value in collected):
        return None
    return sum(value for value in collected if value is not None)


def _sum_optional_float(values: Iterable[float | None]) -> float | None:
    collected = list(values)
    if not collected or any(value is None for value in collected):
        return None
    return sum(value for value in collected if value is not None)
