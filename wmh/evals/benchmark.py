"""Benchmark-neutral contracts for ground-truth task evaluation.

These models keep benchmark evidence independent from the orchestrator that produced it. A scored
zero is data, while a missing score, cancellation, and infrastructure failure remain distinct. The
contract deliberately preserves the verifier's complete reward mapping instead of choosing a
benchmark-specific headline key.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from statistics import fmean
from typing import Self, TypeGuard

from llm_waterfall import ReasoningEffort, ResponseTranslationFailure
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wmh.core.types import JsonObject
from wmh.providers.failure_attribution import ProviderFailureReason, ProviderFailureStage

RewardValue = float | int
Rewards = dict[str, RewardValue]
_SHA256_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
MAX_BENCHMARK_TASK_INSTRUCTION_CHARS = 8_000


def is_sha256_digest(value: object) -> TypeGuard[str]:
    """Return whether ``value`` is one canonical lowercase sha256 digest."""
    return isinstance(value, str) and re.fullmatch(_SHA256_DIGEST_PATTERN, value) is not None


class BenchmarkTaskEnvironment(StrEnum):
    """Where Harbor runs the ground-truth task environment."""

    DOCKER = "docker"
    E2B = "e2b"


class BenchmarkTrialStatus(StrEnum):
    """Whether one planned benchmark cell is usable, pending, or operationally failed."""

    SCORED = "scored"
    CANDIDATE_FAILURE = "candidate_failure"
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
    ENVIRONMENT_CONFIRMATION_REQUIRED = "environment_confirmation_required"
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


class BenchmarkCandidateFailureReason(StrEnum):
    """Portable reason a candidate-controlled execution failed."""

    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    RUNTIME_ERROR = "runtime_error"
    INVALID_REQUEST = "invalid_request"


class BenchmarkRunHealth(StrEnum):
    """Whether one trial is valid evidence or requires operational handling."""

    VALID = "valid"
    CANDIDATE_DAMAGED = "candidate_damaged"
    RETRY_REQUIRED = "retry_required"
    UNKNOWN = "unknown"


class BenchmarkCandidateTerminalReason(StrEnum):
    """Bounded reason a candidate execution reached its ordinary terminal boundary."""

    COMPLETED = "completed"
    LIMIT_REACHED = "limit_reached"
    ABORTED = "aborted"


class BenchmarkUsageStatus(StrEnum):
    """Whether one usage value is exact, a known lower bound, or unavailable."""

    EXACT = "exact"
    LOWER_BOUND = "lower_bound"
    UNAVAILABLE = "unavailable"


class BenchmarkCandidateOutcome(BaseModel):
    """Typed candidate outcome retained without copying arbitrary backend metadata."""

    status: BenchmarkCandidateStatus = BenchmarkCandidateStatus.UNKNOWN
    stage: BenchmarkCandidateStage | None = None
    failure_reason: BenchmarkCandidateFailureReason | None = None
    terminal_reason: BenchmarkCandidateTerminalReason | None = None

    @model_validator(mode="after")
    def _validate_details(self) -> Self:
        if self.status is BenchmarkCandidateStatus.UNKNOWN:
            if (
                self.stage is not None
                or self.failure_reason is not None
                or self.terminal_reason is not None
            ):
                raise ValueError("unknown candidate outcome cannot carry details")
        elif self.status is BenchmarkCandidateStatus.COMPLETED:
            if self.stage is not None or self.failure_reason is not None:
                raise ValueError("completed candidate outcome cannot carry failure details")
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
    """Token and cost evidence without conflating exact, partial, and missing metering."""

    calls: int | None = Field(default=None, ge=0)
    calls_status: BenchmarkUsageStatus = BenchmarkUsageStatus.UNAVAILABLE
    input_tokens: int | None = Field(default=None, ge=0)
    input_tokens_status: BenchmarkUsageStatus = BenchmarkUsageStatus.UNAVAILABLE
    cache_tokens: int | None = Field(default=None, ge=0)
    cache_tokens_status: BenchmarkUsageStatus = BenchmarkUsageStatus.UNAVAILABLE
    output_tokens: int | None = Field(default=None, ge=0)
    output_tokens_status: BenchmarkUsageStatus = BenchmarkUsageStatus.UNAVAILABLE
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cost_usd_status: BenchmarkUsageStatus = BenchmarkUsageStatus.UNAVAILABLE

    @field_validator(
        "calls",
        "input_tokens",
        "cache_tokens",
        "output_tokens",
        "cost_usd",
        mode="before",
    )
    @classmethod
    def _reject_boolean_measurements(cls, value: int | float | None) -> int | float | None:
        if isinstance(value, bool):
            raise ValueError("usage measurements cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_measurement_statuses(self) -> Self:
        self.calls_status = _resolve_usage_status(
            "calls",
            self.calls,
            self.calls_status,
            explicit="calls_status" in self.model_fields_set,
        )
        self.input_tokens_status = _resolve_usage_status(
            "input_tokens",
            self.input_tokens,
            self.input_tokens_status,
            explicit="input_tokens_status" in self.model_fields_set,
        )
        self.cache_tokens_status = _resolve_usage_status(
            "cache_tokens",
            self.cache_tokens,
            self.cache_tokens_status,
            explicit="cache_tokens_status" in self.model_fields_set,
        )
        self.output_tokens_status = _resolve_usage_status(
            "output_tokens",
            self.output_tokens,
            self.output_tokens_status,
            explicit="output_tokens_status" in self.model_fields_set,
        )
        self.cost_usd_status = _resolve_usage_status(
            "cost_usd",
            self.cost_usd,
            self.cost_usd_status,
            explicit="cost_usd_status" in self.model_fields_set,
        )
        return self


def _resolve_usage_status(
    field: str,
    value: int | float | None,
    status: BenchmarkUsageStatus,
    *,
    explicit: bool,
) -> BenchmarkUsageStatus:
    if not explicit:
        return BenchmarkUsageStatus.EXACT if value is not None else BenchmarkUsageStatus.UNAVAILABLE
    if value is None and status is not BenchmarkUsageStatus.UNAVAILABLE:
        raise ValueError(f"{field} without a value must be unavailable")
    if value is not None and status is BenchmarkUsageStatus.UNAVAILABLE:
        raise ValueError(f"{field} with a value cannot be unavailable")
    return status


def aggregate_benchmark_usage(usages: Iterable[BenchmarkUsage]) -> BenchmarkUsage:
    """Aggregate usage while retaining observed lower bounds from incomplete metering."""
    collected = list(usages)
    calls, calls_status = _aggregate_int_measurements(
        (usage.calls, usage.calls_status) for usage in collected
    )
    input_tokens, input_status = _aggregate_int_measurements(
        (usage.input_tokens, usage.input_tokens_status) for usage in collected
    )
    cache_tokens, cache_status = _aggregate_int_measurements(
        (usage.cache_tokens, usage.cache_tokens_status) for usage in collected
    )
    output_tokens, output_status = _aggregate_int_measurements(
        (usage.output_tokens, usage.output_tokens_status) for usage in collected
    )
    cost_usd, cost_status = _aggregate_float_measurements(
        (usage.cost_usd, usage.cost_usd_status) for usage in collected
    )
    return BenchmarkUsage(
        calls=calls,
        calls_status=calls_status,
        input_tokens=input_tokens,
        input_tokens_status=input_status,
        cache_tokens=cache_tokens,
        cache_tokens_status=cache_status,
        output_tokens=output_tokens,
        output_tokens_status=output_status,
        cost_usd=cost_usd,
        cost_usd_status=cost_status,
    )


class BenchmarkRunIdentity(BaseModel):
    """Execution identity that every observed trial must match."""

    candidate_hash: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    reasoning_effort: ReasoningEffort | None = None
    task_environment: BenchmarkTaskEnvironment
    runner_config_digest: str = Field(
        pattern=_SHA256_DIGEST_PATTERN,
        description="Canonical configuration identity for the isolated agent runner",
    )
    runner_environment_digest: str = Field(
        pattern=_SHA256_DIGEST_PATTERN,
        description="Stable attestation expected from every successfully opened runner",
    )
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
    task_instruction: str = Field(default="", max_length=MAX_BENCHMARK_TASK_INSTRUCTION_CHARS)
    task_environment_digest: str | None = Field(
        default=None,
        pattern=_SHA256_DIGEST_PATTERN,
        description=(
            "Opaque digest of the immutable executed task-environment definition and bytes"
        ),
    )
    task_environment_attestation: JsonObject | None = Field(
        default=None,
        description="Bounded stable evidence for the task environment that actually ran",
    )
    task_environment_lease_receipt: JsonObject | None = Field(
        default=None,
        description="Terminal task-environment resource lifecycle receipt retained for audit",
    )
    runner_environment_digest: str | None = Field(
        default=None,
        pattern=_SHA256_DIGEST_PATTERN,
        description="Stable attestation digest for the isolated agent runner actually opened",
    )
    runner_environment_attestation: JsonObject | None = Field(
        default=None,
        description="Bounded stable evidence for the exact runner environment",
    )
    runner_lease_receipts: list[JsonObject] = Field(
        default_factory=list,
        description="Terminal per-resource lifecycle receipts retained for audit",
    )
    status: BenchmarkTrialStatus
    rewards: Rewards | None = None
    error: BenchmarkError | None = None
    usage: BenchmarkUsage = Field(default_factory=BenchmarkUsage)
    candidate_outcome: BenchmarkCandidateOutcome = Field(default_factory=BenchmarkCandidateOutcome)
    run_health: BenchmarkRunHealth = BenchmarkRunHealth.UNKNOWN
    provider_failure_stage: ProviderFailureStage | None = None
    provider_failure_reason: ProviderFailureReason | None = None
    provider_response_translation_failure: ResponseTranslationFailure | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

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
        if (self.provider_failure_stage is None) != (self.provider_failure_reason is None):
            raise ValueError("provider failure stage and reason must be supplied together")
        if (
            self.provider_failure_stage is not ProviderFailureStage.RESPONSE_TRANSLATION
            and self.provider_response_translation_failure is not None
        ):
            raise ValueError("response translation failure requires the response_translation stage")
        if self.provider_failure_stage is not None and (
            self.error is None or self.error.kind is not BenchmarkFailureKind.PROVIDER
        ):
            raise ValueError("provider failure attribution requires a provider error")
        if self.run_health is BenchmarkRunHealth.CANDIDATE_DAMAGED and (
            self.candidate_outcome.status is not BenchmarkCandidateStatus.FAILED
        ):
            raise ValueError("candidate-damaged run health requires a failed candidate outcome")
        if self.status is BenchmarkTrialStatus.SCORED:
            if self.rewards is None:
                raise ValueError(
                    "scored trial must carry rewards, including an empty reward mapping"
                )
            allowed_candidate_damage = (
                self.run_health is BenchmarkRunHealth.CANDIDATE_DAMAGED
                and self.candidate_outcome.status is BenchmarkCandidateStatus.FAILED
                and self.error is not None
                and self.error.kind
                in {
                    BenchmarkFailureKind.ENVIRONMENT,
                    BenchmarkFailureKind.VERIFIER,
                    BenchmarkFailureKind.UNCLASSIFIED,
                }
            )
            if (
                self.error is not None
                and self.error.kind is not BenchmarkFailureKind.TASK_TIMEOUT
                and not allowed_candidate_damage
            ):
                raise ValueError(
                    "scored trial only permits task-timeout or typed candidate-damage "
                    "error evidence"
                )
            return self
        if self.status is BenchmarkTrialStatus.CANDIDATE_FAILURE:
            if self.rewards is not None:
                raise ValueError("candidate_failure trial cannot carry verifier rewards")
            if (
                self.candidate_outcome.status is not BenchmarkCandidateStatus.FAILED
                or self.run_health is not BenchmarkRunHealth.CANDIDATE_DAMAGED
            ):
                raise ValueError(
                    "candidate_failure trial requires a failed candidate outcome and "
                    "candidate-damaged run health"
                )
            if self.error is not None and self.error.kind not in {
                BenchmarkFailureKind.ENVIRONMENT,
                BenchmarkFailureKind.VERIFIER,
                BenchmarkFailureKind.UNCLASSIFIED,
            }:
                raise ValueError(
                    "candidate_failure trial only permits environment, verifier, or "
                    "unclassified terminal error evidence"
                )
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
                BenchmarkFailureKind.ENVIRONMENT_CONFIRMATION_REQUIRED,
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
        duplicate_expected = sorted(
            key for key, count in Counter(expected_keys).items() if count > 1
        )
        if duplicate_expected:
            raise ValueError(f"duplicate expected benchmark cell(s): {duplicate_expected}")
        observed_keys = [(trial.cell.task_key, trial.cell.attempt) for trial in self.trials]
        duplicate_observed = sorted(
            key for key, count in Counter(observed_keys).items() if count > 1
        )
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
    def n_scoreable(self) -> int:
        """Return trials usable in reward means, including typed candidate-failure zeroes."""
        return sum(
            trial.status in {BenchmarkTrialStatus.SCORED, BenchmarkTrialStatus.CANDIDATE_FAILURE}
            for trial in self.trials
        )

    @property
    def n_candidate_failure_zeroes(self) -> int:
        """Return candidate-damaged trials that are valid zeroes without verifier rewards."""
        return sum(trial.status is BenchmarkTrialStatus.CANDIDATE_FAILURE for trial in self.trials)

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
    def n_candidate_failures(self) -> int:
        """Return candidate executions that failed while leaving gradeable task state."""
        return sum(
            trial.candidate_outcome.status is BenchmarkCandidateStatus.FAILED
            for trial in self.trials
        )

    @property
    def n_candidate_timeouts(self) -> int:
        """Return candidate executions stopped by the evaluator-owned runtime deadline."""
        return sum(
            trial.candidate_outcome.failure_reason is BenchmarkCandidateFailureReason.TIMEOUT
            for trial in self.trials
        )

    @property
    def n_candidate_resource_limits(self) -> int:
        """Return candidate executions stopped by a bounded evaluator resource limit."""
        return sum(
            trial.candidate_outcome.failure_reason is BenchmarkCandidateFailureReason.RESOURCE_LIMIT
            for trial in self.trials
        )

    @property
    def n_candidate_runtime_errors(self) -> int:
        """Return candidate executions stopped by a candidate-controlled runtime error."""
        return sum(
            trial.candidate_outcome.failure_reason is BenchmarkCandidateFailureReason.RUNTIME_ERROR
            for trial in self.trials
        )

    @property
    def n_candidate_unclassified_failures(self) -> int:
        """Return failed candidate executions without a portable reason."""
        return sum(
            trial.candidate_outcome.status is BenchmarkCandidateStatus.FAILED
            and trial.candidate_outcome.failure_reason is None
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
        """Return the primary reward mean, assigning failed candidates and timeouts zero."""
        if not key:
            raise ValueError("reward key must be non-empty")
        if not self.trials:
            raise ValueError("cannot aggregate an empty benchmark matrix")
        unhealthy = [
            trial.cell.task_key
            for trial in self.trials
            if trial.run_health
            not in {BenchmarkRunHealth.VALID, BenchmarkRunHealth.CANDIDATE_DAMAGED}
        ]
        if unhealthy:
            raise ValueError(
                f"cannot aggregate reward {key!r}; {len(unhealthy)} planned cells' "
                "run health is not valid"
            )
        invalid = [
            trial.cell.task_key
            for trial in self.trials
            if trial.status
            not in {BenchmarkTrialStatus.SCORED, BenchmarkTrialStatus.CANDIDATE_FAILURE}
        ]
        if invalid:
            raise ValueError(
                f"cannot aggregate reward {key!r}; {len(invalid)} planned cells are not scored"
            )
        missing = [
            trial.cell.task_key
            for trial in self.trials
            if trial.status is BenchmarkTrialStatus.SCORED
            and (trial.rewards is None or key not in trial.rewards)
        ]
        if missing:
            raise ValueError(
                f"cannot aggregate reward {key!r}; {len(missing)} scored cells omit that key"
            )
        values: list[float] = []
        for trial in self.trials:
            if trial.candidate_outcome.status is BenchmarkCandidateStatus.FAILED or (
                trial.error is not None and trial.error.kind is BenchmarkFailureKind.TASK_TIMEOUT
            ):
                values.append(0.0)
                continue
            assert trial.rewards is not None
            values.append(float(trial.rewards[key]))
        return fmean(values)


def _aggregate_int_measurements(
    measurements: Iterable[tuple[int | None, BenchmarkUsageStatus]],
) -> tuple[int | None, BenchmarkUsageStatus]:
    collected = list(measurements)
    observed = [value for value, _status in collected if value is not None]
    if not observed:
        return None, BenchmarkUsageStatus.UNAVAILABLE
    if all(
        value is not None and status is BenchmarkUsageStatus.EXACT for value, status in collected
    ):
        return sum(observed), BenchmarkUsageStatus.EXACT
    return sum(observed), BenchmarkUsageStatus.LOWER_BOUND


def _aggregate_float_measurements(
    measurements: Iterable[tuple[float | None, BenchmarkUsageStatus]],
) -> tuple[float | None, BenchmarkUsageStatus]:
    collected = list(measurements)
    observed = [value for value, _status in collected if value is not None]
    if not observed:
        return None, BenchmarkUsageStatus.UNAVAILABLE
    if all(
        value is not None and status is BenchmarkUsageStatus.EXACT for value, status in collected
    ):
        return sum(observed), BenchmarkUsageStatus.EXACT
    return sum(observed), BenchmarkUsageStatus.LOWER_BOUND
