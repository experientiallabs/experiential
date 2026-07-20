"""Tests for benchmark-neutral result contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmh.evals.benchmark import (
    BenchmarkCandidateFailureReason,
    BenchmarkCandidateOutcome,
    BenchmarkCandidateStage,
    BenchmarkCandidateStatus,
    BenchmarkCandidateTerminalReason,
    BenchmarkCell,
    BenchmarkError,
    BenchmarkFailureKind,
    BenchmarkRunHealth,
    BenchmarkRunIdentity,
    BenchmarkRunResult,
    BenchmarkTaskEnvironment,
    BenchmarkTrialResult,
    BenchmarkTrialStatus,
    BenchmarkUsage,
    BenchmarkUsageStatus,
    is_sha256_digest,
)

_RUN_CONFIG_DIGEST = "sha256:" + "a" * 64
_CELL_CONFIG_DIGEST = "sha256:" + "b" * 64
_RUNNER_CONFIG_DIGEST = "sha256:" + "c" * 64
_RUNNER_ENVIRONMENT_DIGEST = "sha256:" + "d" * 64
_IDENTITY = BenchmarkRunIdentity(
    candidate_hash="candidate-hash",
    agent_name="wmh-pi",
    agent_version="0.1.0",
    provider="bedrock",
    model_name="model",
    task_environment=BenchmarkTaskEnvironment.DOCKER,
    runner_config_digest=_RUNNER_CONFIG_DIGEST,
    runner_environment_digest=_RUNNER_ENVIRONMENT_DIGEST,
    run_config_digest=_RUN_CONFIG_DIGEST,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("sha256:" + "a" * 64, True),
        ("sha256:" + "A" * 64, False),
        ("sha256:" + "a" * 63, False),
        ("md5:" + "a" * 64, False),
        (None, False),
    ],
)
def test_sha256_digest_validation(value: object, expected: bool) -> None:
    assert is_sha256_digest(value) is expected


def _cell(task_name: str, attempt: int = 1, *, task_key: str | None = None) -> BenchmarkCell:
    return BenchmarkCell(
        task_key=task_key or f"dataset/{task_name}@checksum",
        task_name=task_name,
        attempt=attempt,
        config_digest=_CELL_CONFIG_DIGEST,
    )


def _trial(
    cell: BenchmarkCell,
    status: BenchmarkTrialStatus,
    *,
    rewards: dict[str, float | int] | None = None,
    error: BenchmarkError | None = None,
    candidate_outcome: BenchmarkCandidateOutcome | None = None,
    usage: BenchmarkUsage | None = None,
    run_health: BenchmarkRunHealth = BenchmarkRunHealth.VALID,
) -> BenchmarkTrialResult:
    return BenchmarkTrialResult(
        cell=cell,
        task_identity=cell.task_name,
        task_checksum="task-checksum",
        status=status,
        rewards=rewards,
        error=error,
        candidate_outcome=candidate_outcome or BenchmarkCandidateOutcome(),
        usage=usage or BenchmarkUsage(),
        run_health=run_health,
    )


def test_zero_and_arbitrary_reward_keys_are_scored_data() -> None:
    trial = _trial(
        _cell("task-a"),
        BenchmarkTrialStatus.SCORED,
        rewards={"reward": 0, "partial_credit": 0.25, "tests_passed": 3},
    )

    assert trial.rewards == {"reward": 0, "partial_credit": 0.25, "tests_passed": 3}


def test_success_json_omits_absent_response_translation_discriminator() -> None:
    trial = _trial(
        _cell("task-a"),
        BenchmarkTrialStatus.SCORED,
        rewards={"reward": 1},
    )

    assert "provider_response_translation_failure" not in trial.model_dump(mode="json")
    assert '"provider_response_translation_failure"' not in trial.model_dump_json()


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf"), True, False])
def test_rewards_reject_non_finite_and_boolean_values(reward: float | bool) -> None:
    with pytest.raises(ValidationError, match="reward .* (finite|boolean)"):
        _trial(
            _cell("task-a"),
            BenchmarkTrialStatus.SCORED,
            rewards={"reward": reward},
        )


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), float("-inf"), True])
def test_usage_rejects_non_finite_and_boolean_costs(cost: float | bool) -> None:
    with pytest.raises(ValidationError):
        BenchmarkUsage(cost_usd=cost)


def test_usage_requires_explicit_lower_bounds_instead_of_exact_unknown_totals() -> None:
    exact = BenchmarkUsage(input_tokens=0)
    lower_bound = BenchmarkUsage(
        input_tokens=7,
        input_tokens_status=BenchmarkUsageStatus.LOWER_BOUND,
    )

    assert exact.input_tokens_status is BenchmarkUsageStatus.EXACT
    assert lower_bound.input_tokens_status is BenchmarkUsageStatus.LOWER_BOUND
    with pytest.raises(ValidationError, match="without a value must be unavailable"):
        BenchmarkUsage(input_tokens_status=BenchmarkUsageStatus.LOWER_BOUND)
    with pytest.raises(ValidationError, match="with a value cannot be unavailable"):
        BenchmarkUsage(
            input_tokens=0,
            input_tokens_status=BenchmarkUsageStatus.UNAVAILABLE,
        )


def test_usage_tracks_successful_call_count_with_the_same_status_contract() -> None:
    exact = BenchmarkUsage(calls=0)
    lower_bound = BenchmarkUsage(calls=2, calls_status=BenchmarkUsageStatus.LOWER_BOUND)

    assert exact.calls_status is BenchmarkUsageStatus.EXACT
    assert lower_bound.calls_status is BenchmarkUsageStatus.LOWER_BOUND
    with pytest.raises(ValidationError):
        BenchmarkUsage(calls=True)
    with pytest.raises(ValidationError, match="without a value must be unavailable"):
        BenchmarkUsage(calls_status=BenchmarkUsageStatus.EXACT)


def test_run_counts_incomplete_and_infrastructure_cells_separately() -> None:
    cells = [
        _cell("task-a"),
        _cell("task-b"),
        _cell("task-c"),
    ]
    result = BenchmarkRunResult(
        job_name="ground-truth-evaluation",
        identity=_IDENTITY,
        expected_cells=cells,
        trials=[
            _trial(
                cells[0],
                BenchmarkTrialStatus.SCORED,
                rewards={"reward": 0},
            ),
            _trial(
                cells[1],
                BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
                error=BenchmarkError(
                    kind=BenchmarkFailureKind.ENVIRONMENT,
                    type="EnvironmentStartTimeoutError",
                    message="timed out",
                ),
            ),
            _trial(cells[2], BenchmarkTrialStatus.INCOMPLETE),
        ],
    )

    assert result.n_scored == 1
    assert result.n_infrastructure_errors == 1
    assert result.n_incomplete == 1
    assert not result.is_complete


def test_run_usage_is_derived_from_trials_and_rejects_a_conflicting_total() -> None:
    cells = [_cell("task-a"), _cell("task-b")]
    trials = [
        _trial(
            cells[0],
            BenchmarkTrialStatus.SCORED,
            rewards={"reward": 1},
            usage=BenchmarkUsage(
                calls=2,
                input_tokens=10,
                cache_tokens=3,
                output_tokens=4,
                cost_usd=0.25,
            ),
        ),
        _trial(
            cells[1],
            BenchmarkTrialStatus.SCORED,
            rewards={"reward": 0},
            usage=BenchmarkUsage(
                calls=3,
                input_tokens=20,
                cache_tokens=5,
                output_tokens=6,
                cost_usd=0.5,
            ),
        ),
    ]

    result = BenchmarkRunResult(
        job_name="metered",
        identity=_IDENTITY,
        expected_cells=cells,
        trials=trials,
    )

    assert result.usage == BenchmarkUsage(
        calls=5,
        input_tokens=30,
        cache_tokens=8,
        output_tokens=10,
        cost_usd=0.75,
    )
    with pytest.raises(ValidationError, match="aggregate of trial usage"):
        BenchmarkRunResult(
            job_name="metered",
            identity=_IDENTITY,
            expected_cells=cells,
            trials=trials,
            usage=BenchmarkUsage(input_tokens=999),
        )


def test_run_usage_retains_known_lower_bound_when_one_trial_is_incompletely_metered() -> None:
    cells = [_cell("task-a"), _cell("task-b")]
    result = BenchmarkRunResult(
        job_name="partially-metered",
        identity=_IDENTITY,
        expected_cells=cells,
        trials=[
            _trial(
                cells[0],
                BenchmarkTrialStatus.SCORED,
                rewards={"reward": 1},
                usage=BenchmarkUsage(input_tokens=10, output_tokens=4),
            ),
            _trial(
                cells[1],
                BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
                error=BenchmarkError(
                    kind=BenchmarkFailureKind.PROVIDER,
                    type="WmhPiProviderError",
                ),
                usage=BenchmarkUsage(
                    input_tokens=7,
                    input_tokens_status=BenchmarkUsageStatus.LOWER_BOUND,
                    output_tokens=2,
                    output_tokens_status=BenchmarkUsageStatus.LOWER_BOUND,
                ),
            ),
        ],
    )

    assert result.usage.input_tokens == 17
    assert result.usage.input_tokens_status is BenchmarkUsageStatus.LOWER_BOUND
    assert result.usage.output_tokens == 6
    assert result.usage.output_tokens_status is BenchmarkUsageStatus.LOWER_BOUND
    assert result.usage.cost_usd is None
    assert result.usage.cost_usd_status is BenchmarkUsageStatus.UNAVAILABLE


def test_scored_trial_requires_a_reward_mapping_even_when_empty() -> None:
    with pytest.raises(ValidationError, match="scored trial must carry rewards"):
        _trial(
            _cell("task-a"),
            BenchmarkTrialStatus.SCORED,
        )


def test_infrastructure_trial_requires_error_evidence() -> None:
    with pytest.raises(ValidationError, match="infrastructure_error trial must carry an error"):
        _trial(
            _cell("task-a"),
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
        )


@pytest.mark.parametrize(
    ("status", "kind", "rewards", "message"),
    [
        (
            BenchmarkTrialStatus.SCORED,
            BenchmarkFailureKind.PROVIDER,
            {"reward": 1},
            "scored trial only permits task-timeout",
        ),
        (
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.PROVIDER,
            {"reward": 0},
            "non-scored trial cannot carry verifier rewards",
        ),
        (
            BenchmarkTrialStatus.INCOMPLETE,
            BenchmarkFailureKind.PROVIDER,
            None,
            "incomplete trial cannot carry error evidence",
        ),
        (
            BenchmarkTrialStatus.TASK_TIMEOUT,
            BenchmarkFailureKind.PROVIDER,
            None,
            "task_timeout trial cannot carry provider",
        ),
        (
            BenchmarkTrialStatus.UNCLASSIFIED_ERROR,
            BenchmarkFailureKind.ENVIRONMENT,
            None,
            "unclassified_error trial cannot carry environment",
        ),
    ],
)
def test_contradictory_trial_evidence_is_rejected(
    status: BenchmarkTrialStatus,
    kind: BenchmarkFailureKind,
    rewards: dict[str, float | int] | None,
    message: str,
) -> None:
    error = BenchmarkError(kind=kind, type="ExampleError")

    with pytest.raises(ValidationError, match=message):
        _trial(_cell("task-a"), status, rewards=rewards, error=error)


def test_scored_timeout_is_counted_as_both_scored_and_timed_out() -> None:
    cell = _cell("task-a")
    trial = _trial(
        cell,
        BenchmarkTrialStatus.SCORED,
        rewards={"reward": 0},
        error=BenchmarkError(
            kind=BenchmarkFailureKind.TASK_TIMEOUT,
            type="AgentTimeoutError",
        ),
    )
    result = BenchmarkRunResult(
        job_name="timeout-with-score",
        identity=_IDENTITY,
        expected_cells=[cell],
        trials=[trial],
    )

    assert result.n_scored == 1
    assert result.n_task_timeouts == 1


def test_mean_reward_requires_every_planned_cell_and_requested_key() -> None:
    cells = [_cell("task-a"), _cell("task-b")]
    complete = BenchmarkRunResult(
        job_name="complete",
        identity=_IDENTITY,
        expected_cells=cells,
        trials=[
            _trial(cells[0], BenchmarkTrialStatus.SCORED, rewards={"score": 0}),
            _trial(cells[1], BenchmarkTrialStatus.SCORED, rewards={"score": 0.5}),
        ],
    )

    assert complete.mean_reward("score") == 0.25
    with pytest.raises(ValueError, match="omit that key"):
        complete.mean_reward("other")

    invalid = complete.model_copy(deep=True)
    invalid.trials[1] = _trial(
        cells[1],
        BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
        error=BenchmarkError(
            kind=BenchmarkFailureKind.PROVIDER,
            type="WmhPiProviderError",
        ),
    )
    with pytest.raises(ValueError, match="planned cells are not scored"):
        invalid.mean_reward("score")


def test_candidate_damaged_environment_is_an_explicit_valid_zero() -> None:
    cell = _cell("candidate-killed-container")
    trial = _trial(
        cell,
        BenchmarkTrialStatus.CANDIDATE_FAILURE,
        error=BenchmarkError(
            kind=BenchmarkFailureKind.ENVIRONMENT,
            type="WmhPiEnvironmentError",
            message="candidate task environment was destroyed",
        ),
        candidate_outcome=BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.FAILED,
            stage=BenchmarkCandidateStage.EXECUTION,
            failure_reason=BenchmarkCandidateFailureReason.RESOURCE_LIMIT,
        ),
        run_health=BenchmarkRunHealth.CANDIDATE_DAMAGED,
    )
    result = BenchmarkRunResult(
        job_name="candidate-zero",
        identity=_IDENTITY,
        expected_cells=[cell],
        trials=[trial],
    )

    assert result.n_scored == 0
    assert result.n_scoreable == 1
    assert result.n_candidate_failure_zeroes == 1
    assert result.n_infrastructure_errors == 0
    assert result.mean_reward("reward") == 0.0


def test_candidate_damage_preserves_a_later_verifier_reward_as_diagnostic_data() -> None:
    trial = _trial(
        _cell("candidate-damaged-but-verified"),
        BenchmarkTrialStatus.SCORED,
        rewards={"reward": 1},
        error=BenchmarkError(
            kind=BenchmarkFailureKind.ENVIRONMENT,
            type="WmhPiEnvironmentError",
        ),
        candidate_outcome=BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.FAILED,
            stage=BenchmarkCandidateStage.EXECUTION,
            failure_reason=BenchmarkCandidateFailureReason.RESOURCE_LIMIT,
        ),
        run_health=BenchmarkRunHealth.CANDIDATE_DAMAGED,
    )

    assert trial.rewards == {"reward": 1}
    assert trial.error is not None
    assert trial.error.kind is BenchmarkFailureKind.ENVIRONMENT
    result = BenchmarkRunResult(
        job_name="candidate-diagnostic",
        identity=_IDENTITY,
        expected_cells=[trial.cell],
        trials=[trial],
    )
    assert result.mean_reward("reward") == 0.0


def test_candidate_failure_zero_requires_failed_candidate_and_damaged_health() -> None:
    cell = _cell("candidate-killed-container")

    with pytest.raises(ValidationError, match="candidate_failure trial requires"):
        _trial(
            cell,
            BenchmarkTrialStatus.CANDIDATE_FAILURE,
            candidate_outcome=BenchmarkCandidateOutcome(
                status=BenchmarkCandidateStatus.FAILED,
                stage=BenchmarkCandidateStage.EXECUTION,
                failure_reason=BenchmarkCandidateFailureReason.RESOURCE_LIMIT,
            ),
            run_health=BenchmarkRunHealth.VALID,
        )


@pytest.mark.parametrize(
    "run_health",
    [BenchmarkRunHealth.RETRY_REQUIRED, BenchmarkRunHealth.UNKNOWN],
)
def test_unhealthy_or_unknown_evidence_cannot_enter_reward_aggregation(
    run_health: BenchmarkRunHealth,
) -> None:
    cell = _cell("ambiguous")
    result = BenchmarkRunResult(
        job_name="ambiguous",
        identity=_IDENTITY,
        expected_cells=[cell],
        trials=[
            _trial(
                cell,
                BenchmarkTrialStatus.SCORED,
                rewards={"reward": 1},
                run_health=run_health,
            )
        ],
    )

    with pytest.raises(ValueError, match="run health is not valid"):
        result.mean_reward("reward")


def test_duplicate_cells_are_rejected() -> None:
    cell = _cell("task-a")
    trial = _trial(cell, BenchmarkTrialStatus.INCOMPLETE)

    with pytest.raises(ValidationError, match="duplicate expected benchmark cell"):
        BenchmarkRunResult(
            job_name="ground-truth-evaluation",
            identity=_IDENTITY,
            expected_cells=[cell, cell],
            trials=[trial],
        )


def test_cancelled_is_terminal_and_distinct_from_missing_evidence() -> None:
    cancelled = _trial(
        _cell("cancelled"),
        BenchmarkTrialStatus.CANCELLED,
        error=BenchmarkError(
            kind=BenchmarkFailureKind.CANCELLED,
            type="CancelledError",
        ),
    )
    missing = _trial(_cell("missing"), BenchmarkTrialStatus.INCOMPLETE)
    result = BenchmarkRunResult(
        job_name="ground-truth-evaluation",
        identity=_IDENTITY,
        expected_cells=[cancelled.cell, missing.cell],
        trials=[cancelled, missing],
    )

    assert result.n_cancelled == 1
    assert result.n_incomplete == 1
    assert result.is_complete is False


def test_candidate_outcome_is_typed_independently_from_verifier_score() -> None:
    failed_but_scored = _trial(
        _cell("task-a"),
        BenchmarkTrialStatus.SCORED,
        rewards={"reward": 0.5},
        candidate_outcome=BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.FAILED,
            stage=BenchmarkCandidateStage.EXECUTION,
            failure_reason=BenchmarkCandidateFailureReason.TIMEOUT,
        ),
    )
    completed = _trial(
        _cell("task-b"),
        BenchmarkTrialStatus.SCORED,
        rewards={"reward": 1},
        candidate_outcome=BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.COMPLETED,
            terminal_reason=BenchmarkCandidateTerminalReason.COMPLETED,
        ),
    )

    assert failed_but_scored.candidate_outcome.status is BenchmarkCandidateStatus.FAILED
    assert failed_but_scored.candidate_outcome.stage is BenchmarkCandidateStage.EXECUTION
    assert (
        failed_but_scored.candidate_outcome.failure_reason
        is BenchmarkCandidateFailureReason.TIMEOUT
    )
    assert completed.candidate_outcome.terminal_reason is (
        BenchmarkCandidateTerminalReason.COMPLETED
    )


@pytest.mark.parametrize(
    "outcome",
    [
        BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.UNKNOWN,
        ).model_copy(update={"stage": BenchmarkCandidateStage.SETUP}),
        BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.COMPLETED,
        ).model_copy(update={"stage": BenchmarkCandidateStage.EXECUTION}),
        BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.FAILED,
        ).model_copy(update={"terminal_reason": BenchmarkCandidateTerminalReason.ABORTED}),
        BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.COMPLETED,
        ).model_copy(update={"failure_reason": BenchmarkCandidateFailureReason.TIMEOUT}),
    ],
)
def test_candidate_outcome_rejects_contradictory_details(
    outcome: BenchmarkCandidateOutcome,
) -> None:
    with pytest.raises(ValidationError, match="candidate outcome"):
        BenchmarkCandidateOutcome.model_validate(outcome.model_dump())


def test_candidate_timeout_count_is_distinct_from_harbor_task_timeout() -> None:
    candidate_timeout = _trial(
        _cell("candidate-timeout"),
        BenchmarkTrialStatus.SCORED,
        rewards={"reward": 0},
        candidate_outcome=BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.FAILED,
            stage=BenchmarkCandidateStage.EXECUTION,
            failure_reason=BenchmarkCandidateFailureReason.TIMEOUT,
        ),
    )
    task_timeout = _trial(
        _cell("task-timeout"),
        BenchmarkTrialStatus.TASK_TIMEOUT,
        error=BenchmarkError(
            kind=BenchmarkFailureKind.TASK_TIMEOUT,
            type="AgentTimeoutError",
        ),
    )
    result = BenchmarkRunResult(
        job_name="timeout-accounting",
        identity=_IDENTITY,
        expected_cells=[candidate_timeout.cell, task_timeout.cell],
        trials=[candidate_timeout, task_timeout],
    )

    assert result.n_candidate_timeouts == 1
    assert result.n_task_timeouts == 1


def test_candidate_failure_counts_cover_each_reason_and_unclassified_evidence() -> None:
    reasons = [
        BenchmarkCandidateFailureReason.TIMEOUT,
        BenchmarkCandidateFailureReason.RESOURCE_LIMIT,
        BenchmarkCandidateFailureReason.RUNTIME_ERROR,
        None,
    ]
    failures = [
        _trial(
            _cell(f"candidate-failure-{index}"),
            BenchmarkTrialStatus.SCORED,
            rewards={"reward": 0},
            candidate_outcome=BenchmarkCandidateOutcome(
                status=BenchmarkCandidateStatus.FAILED,
                stage=BenchmarkCandidateStage.EXECUTION,
                failure_reason=reason,
            ),
        )
        for index, reason in enumerate(reasons)
    ]
    completed = _trial(
        _cell("candidate-completed"),
        BenchmarkTrialStatus.SCORED,
        rewards={"reward": 1},
        candidate_outcome=BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.COMPLETED,
            terminal_reason=BenchmarkCandidateTerminalReason.COMPLETED,
        ),
    )
    trials = [*failures, completed]
    result = BenchmarkRunResult(
        job_name="candidate-failure-accounting",
        identity=_IDENTITY,
        expected_cells=[trial.cell for trial in trials],
        trials=trials,
    )

    assert result.n_candidate_failures == 4
    assert result.n_candidate_timeouts == 1
    assert result.n_candidate_resource_limits == 1
    assert result.n_candidate_runtime_errors == 1
    assert result.n_candidate_unclassified_failures == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_name", "tampered-name"),
        ("config_digest", "sha256:" + "c" * 64),
    ],
)
def test_observed_cell_must_match_full_expected_identity(field: str, value: str) -> None:
    expected = _cell("task-a")
    observed = expected.model_copy(update={field: value})

    with pytest.raises(ValidationError, match="cell identity differs from manifest"):
        BenchmarkRunResult(
            job_name="tampered-cell",
            identity=_IDENTITY,
            expected_cells=[expected],
            trials=[_trial(observed, BenchmarkTrialStatus.INCOMPLETE)],
        )


def test_same_display_name_from_different_datasets_is_not_a_cell_collision() -> None:
    cells = [
        _cell("shared-name", task_key="dataset-a/shared-name@checksum-a"),
        _cell("shared-name", task_key="dataset-b/shared-name@checksum-b"),
    ]

    result = BenchmarkRunResult(
        job_name="multi-dataset",
        identity=_IDENTITY,
        expected_cells=cells,
        trials=[
            _trial(cells[0], BenchmarkTrialStatus.INCOMPLETE),
            _trial(cells[1], BenchmarkTrialStatus.INCOMPLETE),
        ],
    )

    assert len(result.trials) == 2
