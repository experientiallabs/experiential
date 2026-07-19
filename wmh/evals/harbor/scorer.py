"""Project strict Harbor task matrices into benchmark-neutral harness scores."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Self

from llm_waterfall import ChatProviderReceipt
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from wmh.core.text import normalize_durable_text, validate_durable_text
from wmh.core.types import JsonObject
from wmh.evals.benchmark import (
    BenchmarkCandidateFailureReason,
    BenchmarkCandidateOutcome,
    BenchmarkCandidateStatus,
    BenchmarkCell,
    BenchmarkFailureKind,
    BenchmarkRunHealth,
    BenchmarkRunResult,
    BenchmarkTaskEnvironment,
    BenchmarkTrialResult,
    BenchmarkTrialStatus,
    BenchmarkUsageStatus,
    Rewards,
    is_sha256_digest,
)
from wmh.evals.harbor.agent import WMH_PI_AGENT_VERSION
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.evaluator import HarborEvaluator, harbor_run_expectation
from wmh.evals.harbor.receipt_trace import validate_provider_receipt_trace
from wmh.evals.harbor.results import HarborTrialLocator, LoadedHarborJobResult
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import EventKind
from wmh.harness.pi_local import PI_CONTAINER_IMAGE, validate_pi_container_image
from wmh.harness.scoring import (
    MAX_TASK_EVIDENCE_CHARS,
    HarnessScoreReport,
    ScoreCapabilities,
    ScoreRequest,
    ScoreRunHealth,
    TaskScore,
)
from wmh.harness.tools import READ_SKILL
from wmh.providers.base import ProviderConfig

MAX_HARBOR_TASK_EVIDENCE_CHARS = MAX_TASK_EVIDENCE_CHARS
_MAX_TRACE_FILE_BYTES = 16 * 1024 * 1024
_TRACE_FILENAME = "wmh-events.jsonl"
_AsyncRunner = asyncio.Runner


class HarborAgentComputeEnvelope(BaseModel):
    """Candidate-controlled compute settings held fixed across every scored harness."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    runtime_kind: str
    max_turns: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=2.0)
    effective_tools: tuple[str, ...]
    turn_timeout_s: float = Field(gt=0.0)


class _TraceRecord(BaseModel):
    """One trusted WMH agent trace event, with no backend-defined extra fields."""

    model_config = ConfigDict(extra="forbid")

    kind: EventKind
    payload: JsonObject


class _EvaluationCellIdentity(BaseModel):
    """Canonical score evidence that one evaluation ID commits to."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    cell: BenchmarkCell
    task_identity: str
    task_checksum: str
    source: str | None
    task_instruction_digest: str
    task_environment_digest: str
    status: BenchmarkTrialStatus
    rewards_digest: str
    verifier_reward: float | None = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    candidate_outcome: BenchmarkCandidateOutcome
    run_health: BenchmarkRunHealth
    error_kind: BenchmarkFailureKind | None
    trace_digest: str


@dataclass(frozen=True)
class _TraceEvidence:
    text: str
    digest: str
    provider_receipts: tuple[ChatProviderReceipt, ...]
    provider_call_indexes: tuple[int, ...]


@dataclass(frozen=True)
class AdmittedHarborTrial:
    """One exact Harbor trial admitted as binary analysis evidence."""

    trial: BenchmarkTrialResult
    verifier_reward: float | None
    score: float
    trace_text: str
    trace_digest: str
    provider_receipts: tuple[ChatProviderReceipt, ...]
    provider_receipt_call_indexes: tuple[int, ...]


class HarborHarnessScorer:
    """Score harnesses through one frozen, exact Harbor task matrix.

    The scorer deliberately advertises no subset or attempt override support. Search callers must
    disable screening and confirmation requests, so every candidate runs the same task cells.
    """

    capabilities = ScoreCapabilities(task_subsets=False, attempt_overrides=False)

    def __init__(
        self,
        *,
        job_spec: HarborJobSpec,
        provider_config: ProviderConfig,
        reference_harness: HarnessDoc,
        task_ids: tuple[str, ...],
        task_keys: tuple[str, ...],
        task_environment_digests: tuple[str, ...],
        reward_key: str,
        runner_image: str = PI_CONTAINER_IMAGE,
        turn_timeout_s: float = 300.0,
    ) -> None:
        """Freeze task selection, provider route, backend, image, and agent compute."""
        validate_pi_container_image(runner_image)
        if not math.isfinite(turn_timeout_s) or turn_timeout_s <= 0:
            raise ValueError("turn_timeout_s must be finite and positive")
        if not reward_key.strip():
            raise ValueError("reward_key must be non-empty")
        validate_durable_text(reward_key, field="Harbor reward key")
        frozen_task_ids = tuple(task_ids)
        if not frozen_task_ids:
            raise ValueError("task_ids must be non-empty")
        if len(set(frozen_task_ids)) != len(frozen_task_ids):
            raise ValueError("task_ids must be unique")
        for task_id in frozen_task_ids:
            if not task_id or len(task_id) > 512:
                raise ValueError("each task_id must contain between 1 and 512 characters")
            validate_durable_text(task_id, field="Harbor task id")
        frozen_task_keys = tuple(task_keys)
        if len(frozen_task_keys) != len(frozen_task_ids):
            raise ValueError("task_keys must contain exactly one key per task_id")
        if len(set(frozen_task_keys)) != len(frozen_task_keys):
            raise ValueError("task_keys must be unique")
        for task_key in frozen_task_keys:
            if not is_sha256_digest(task_key):
                raise ValueError(
                    "each task_key must be a sha256 digest from a Harbor qualification manifest"
                )
        frozen_environment_digests = tuple(task_environment_digests)
        if len(frozen_environment_digests) != len(frozen_task_ids):
            raise ValueError("task_environment_digests must contain exactly one digest per task_id")
        for environment_digest in frozen_environment_digests:
            if not is_sha256_digest(environment_digest):
                raise ValueError(
                    "each task_environment_digest must be a sha256 digest from a Harbor "
                    "qualification run"
                )

        spec = HarborJobSpec.model_validate(job_spec.model_dump())
        spec = spec.model_copy(
            update={"jobs_dir": spec.jobs_dir.expanduser().resolve()},
            deep=True,
        )
        _validate_job_prefix(spec.job_name)
        validate_exact_harbor_dataset_selection(spec, frozen_task_ids)
        if spec.environment_backend is HarborEnvironmentBackend.E2B:
            raise ValueError(
                "HarborHarnessScorer cannot validate E2B runs until Harbor exposes the immutable "
                "build ID used to create each sandbox; use local Docker for scored search"
            )
        provider = ProviderConfig.model_validate(provider_config.model_dump())
        envelope = harbor_agent_compute_envelope(
            reference_harness,
            turn_timeout_s=turn_timeout_s,
        )
        if envelope.runtime_kind != "pi-node":
            raise ValueError(
                "HarborHarnessScorer requires a pi-node reference harness, got "
                f"{envelope.runtime_kind!r}"
            )

        self._job_spec = spec
        self._provider_config = provider
        self._task_ids = frozen_task_ids
        self._task_keys = frozen_task_keys
        self._task_environment_digests = frozen_environment_digests
        self._reward_key = reward_key
        self._runner_image = runner_image
        self._compute_envelope = envelope
        self._runner: asyncio.Runner | None = None
        self._closed = False

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return the exact ordered task identities frozen by this scorer."""
        return self._task_ids

    @property
    def default_attempts(self) -> int:
        """Return the immutable number of attempts in every scored task cell."""
        return self._job_spec.n_attempts

    @property
    def compute_envelope(self) -> HarborAgentComputeEnvelope:
        """Return the immutable candidate-controlled compute envelope."""
        return self._compute_envelope

    @property
    def task_keys(self) -> tuple[str, ...]:
        """Return the dataset-qualified content identities paired with ``task_ids``."""
        return self._task_keys

    @property
    def task_environment_digests(self) -> tuple[str, ...]:
        """Return the executed task-environment identities frozen by qualification."""
        return self._task_environment_digests

    @property
    def environment_backend(self) -> HarborEnvironmentBackend:
        """Return the frozen Harbor task-environment backend."""
        return self._job_spec.environment_backend

    def validate_candidate(self, candidate: HarnessDoc) -> str | None:
        """Reject candidates that change the frozen runtime or agent compute envelope."""
        try:
            actual = harbor_agent_compute_envelope(
                candidate,
                turn_timeout_s=self._compute_envelope.turn_timeout_s,
            )
        except (TypeError, ValueError) as exc:
            return f"candidate compute envelope is invalid: {exc}"
        if actual != self._compute_envelope:
            return (
                "candidate changes the frozen Harbor agent compute envelope; only harness "
                "source may vary between candidates"
            )
        return None

    def before_proposal_batch(self) -> None:
        """Release the idle event loop before a potentially long proposal call."""
        self._require_open()
        self._release_runner()

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
        """Evaluate one candidate and fail closed on any non-scoreable Harbor cell."""
        self._require_open()
        if request.task_ids is not None:
            raise ValueError("HarborHarnessScorer does not support task subset requests")
        if request.attempts is not None:
            raise ValueError("HarborHarnessScorer does not support attempt override requests")
        rejection = self.validate_candidate(candidate)
        if rejection is not None:
            raise ValueError(rejection)

        job_name = self._job_name(candidate)
        spec = self._job_spec.model_copy(update={"job_name": job_name}, deep=True)
        evaluator = HarborEvaluator(
            spec,
            self._provider_config.model_copy(deep=True),
            runner_image=self._runner_image,
            turn_timeout_s=self._compute_envelope.turn_timeout_s,
        )
        try:
            loaded = self._run_evaluation(evaluator, candidate)
            return self._project(candidate, spec, loaded, request=request)
        except BaseException:
            self._release_runner()
            raise

    def close(self) -> None:
        """Close the owned event loop; repeated calls are safe."""
        if self._closed:
            return
        self._closed = True
        self._release_runner()

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _run_evaluation(
        self,
        evaluator: HarborEvaluator,
        candidate: HarnessDoc,
    ) -> LoadedHarborJobResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "HarborHarnessScorer.score cannot run inside a running event loop; "
                "call the synchronous search API from a worker thread"
            )
        if self._runner is None:
            self._runner = _AsyncRunner()
        try:
            return self._runner.run(evaluator.evaluate(candidate))
        except BaseException:
            self._release_runner()
            raise

    def _project(
        self,
        candidate: HarnessDoc,
        spec: HarborJobSpec,
        loaded: LoadedHarborJobResult,
        *,
        request: ScoreRequest,
    ) -> HarnessScoreReport:
        result = loaded.result
        validate_harbor_run_identity(
            result,
            candidate=candidate,
            spec=spec,
            provider_config=self._provider_config,
            runner_image=self._runner_image,
            turn_timeout_s=self._compute_envelope.turn_timeout_s,
        )
        ordered = admit_harbor_matrix(
            loaded,
            task_ids=self._task_ids,
            task_keys=self._task_keys,
            task_environment_digests=self._task_environment_digests,
            attempts=self.default_attempts,
            reward_key=self._reward_key,
            provider_config=self._provider_config,
            compute_envelope=self._compute_envelope,
        )
        per_task: dict[str, TaskScore] = {}
        identity_cells: list[_EvaluationCellIdentity] = []
        for task_id in self._task_ids:
            trials = ordered[task_id]
            scores = [item.score for item in trials]
            task_score = fmean(scores)
            description = trials[0].trial.task_instruction
            mechanisms = _failure_mechanisms(trials)
            evidence = _render_task_evidence(trials)
            per_task[task_id] = TaskScore(
                task_id=task_id,
                score=task_score,
                secondary_score=task_score,
                passed=all(score == 1.0 for score in scores),
                description=description,
                mechanisms=mechanisms,
                evidence=evidence,
            )
            for item in trials:
                trial = item.trial
                identity_cells.append(
                    _EvaluationCellIdentity(
                        cell=trial.cell,
                        task_identity=trial.task_identity,
                        task_checksum=trial.task_checksum,
                        source=trial.source,
                        task_instruction_digest=_sha256_text(trial.task_instruction),
                        task_environment_digest=_required_task_environment_digest(trial),
                        status=trial.status,
                        rewards_digest=_rewards_digest(trial.rewards),
                        verifier_reward=item.verifier_reward,
                        score=item.score,
                        candidate_outcome=trial.candidate_outcome,
                        run_health=trial.run_health,
                        error_kind=trial.error.kind if trial.error is not None else None,
                        trace_digest=item.trace_digest,
                    )
                )

        aggregate = fmean(task.score for task in per_task.values())
        evaluation_id = _evaluation_id(
            result=result,
            request=request,
            reward_key=self._reward_key,
            task_ids=self._task_ids,
            task_keys=self._task_keys,
            task_environment_digests=self._task_environment_digests,
            attempts=self.default_attempts,
            compute_envelope=self._compute_envelope,
            cells=identity_cells,
        )
        return HarnessScoreReport(
            evaluation_id=evaluation_id,
            label=result.job_name,
            score=aggregate,
            secondary_score=aggregate,
            attempts=self.default_attempts,
            run_health=ScoreRunHealth.VALID,
            per_task=per_task,
        )

    def _release_runner(self) -> None:
        runner = self._runner
        self._runner = None
        if runner is not None:
            runner.close()

    def _job_name(self, candidate: HarnessDoc) -> str:
        payload = {
            "schema_version": 1,
            "job_spec": {
                **self._job_spec.model_dump(
                    mode="json",
                    exclude={"jobs_dir", "datasets"},
                ),
                "datasets": [
                    {"task_names": list(dataset.task_names or [])}
                    for dataset in self._job_spec.datasets
                ],
            },
            "provider_config": self._provider_config.model_dump(mode="json"),
            "candidate": candidate.model_dump(mode="json"),
            "task_ids": list(self._task_ids),
            "task_keys": list(self._task_keys),
            "task_environment_digests": list(self._task_environment_digests),
            "reward_key": self._reward_key,
            "runner_image": self._runner_image,
            "compute_envelope": self._compute_envelope.model_dump(mode="json"),
            "agent_version": WMH_PI_AGENT_VERSION,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return "wmh-score-" + hashlib.sha256(canonical.encode()).hexdigest()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("HarborHarnessScorer is closed")


def harbor_agent_compute_envelope(
    harness: HarnessDoc,
    *,
    turn_timeout_s: float,
) -> HarborAgentComputeEnvelope:
    """Project the candidate-controlled settings that must stay equal across arms."""
    tools = harness.tools()
    if harness.skills() and READ_SKILL.name not in tools:
        tools.append(READ_SKILL.name)
    return HarborAgentComputeEnvelope(
        runtime_kind=harness.runtime_kind(),
        max_turns=harness.max_turns(),
        max_output_tokens=harness.max_output_tokens(),
        temperature=harness.temperature(),
        effective_tools=tuple(tools),
        turn_timeout_s=turn_timeout_s,
    )


def _validate_job_prefix(job_name: str) -> None:
    if (
        job_name in {".", ".."}
        or "\0" in job_name
        or Path(job_name).is_absolute()
        or "/" in job_name
        or "\\" in job_name
    ):
        raise ValueError("Harbor scorer job_name must be a single safe path component")
    if len(job_name.encode()) > 200:
        raise ValueError("Harbor scorer job_name must be at most 200 UTF-8 bytes")


def validate_exact_harbor_dataset_selection(
    spec: HarborJobSpec,
    task_ids: tuple[str, ...],
) -> None:
    """Require explicit, ordered, preflightable local task selection."""
    selected: list[str] = []
    for dataset in spec.datasets:
        if not dataset.is_local():
            raise ValueError(
                "HarborHarnessScorer requires preflightable local dataset paths; remote, "
                "registry, and package acquisition are not safe scoring inputs"
            )
        if dataset.task_names is None:
            raise ValueError("HarborHarnessScorer requires explicit task_names on every dataset")
        if dataset.exclude_task_names is not None or dataset.n_tasks is not None:
            raise ValueError(
                "HarborHarnessScorer forbids exclude_task_names and n_tasks; use one exact "
                "task_names list"
            )
        for task_name in dataset.task_names:
            if any(marker in task_name for marker in ("*", "?", "[", "]")):
                raise ValueError("HarborHarnessScorer task_names cannot contain glob patterns")
            selected.append(task_name)
    if tuple(selected) != task_ids:
        raise ValueError("Harbor dataset task_names must exactly match task_ids in order")


def validate_harbor_run_identity(
    result: BenchmarkRunResult,
    *,
    candidate: HarnessDoc,
    spec: HarborJobSpec,
    provider_config: ProviderConfig,
    runner_image: str,
    turn_timeout_s: float,
    require_exact_run_config: bool = False,
) -> None:
    """Require loaded evidence to match its frozen harness and execution route."""
    if result.job_name != spec.job_name:
        raise ValueError(
            f"Harbor result job name {result.job_name!r} does not match {spec.job_name!r}"
        )
    expected_environment = (
        BenchmarkTaskEnvironment.E2B
        if spec.environment_backend is HarborEnvironmentBackend.E2B
        else BenchmarkTaskEnvironment.DOCKER
    )
    identity = result.identity
    mismatches: list[str] = []
    if identity.candidate_hash != candidate.execution_hash:
        mismatches.append("candidate hash")
    if identity.agent_name != "wmh-pi" or identity.agent_version != WMH_PI_AGENT_VERSION:
        mismatches.append("agent")
    if (
        identity.provider != provider_config.kind.value
        or identity.model_name != provider_config.model
    ):
        mismatches.append("provider route")
    if identity.task_environment is not expected_environment:
        mismatches.append("task backend")
    if identity.runner_image != runner_image:
        mismatches.append("runner image")
    if require_exact_run_config:
        expectation = harbor_run_expectation(
            candidate=candidate,
            spec=spec,
            provider_config=provider_config,
            runner_image=runner_image,
            turn_timeout_s=turn_timeout_s,
        )
        if identity != expectation.identity:
            mismatches.append("exact run config")
    if mismatches:
        raise ValueError(f"Harbor result identity mismatches frozen scorer: {sorted(mismatches)}")


def admit_harbor_matrix(
    loaded: LoadedHarborJobResult,
    *,
    task_ids: tuple[str, ...],
    task_keys: tuple[str, ...],
    task_environment_digests: tuple[str, ...],
    attempts: int,
    reward_key: str,
    provider_config: ProviderConfig,
    compute_envelope: HarborAgentComputeEnvelope,
) -> dict[str, list[AdmittedHarborTrial]]:
    """Admit an exact Harbor matrix and map candidate-owned failures to analysis zero."""
    result = loaded.result
    expected_keys = [(cell.task_key, cell.attempt) for cell in result.expected_cells]
    observed_keys = [(trial.cell.task_key, trial.cell.attempt) for trial in result.trials]
    duplicate_expected = sorted(key for key, count in Counter(expected_keys).items() if count > 1)
    if duplicate_expected:
        raise ValueError(f"duplicate expected Harbor cell(s): {duplicate_expected}")
    duplicate_observed = sorted(key for key, count in Counter(observed_keys).items() if count > 1)
    if duplicate_observed:
        raise ValueError(f"duplicate observed Harbor cell(s): {duplicate_observed}")
    if set(expected_keys) != set(observed_keys):
        missing = sorted(set(expected_keys) - set(observed_keys))
        extra = sorted(set(observed_keys) - set(expected_keys))
        raise ValueError(
            f"Harbor result cells differ from its plan: missing={missing}, extra={extra}"
        )
    expected_by_key = {(cell.task_key, cell.attempt): cell for cell in result.expected_cells}
    for trial in result.trials:
        key = (trial.cell.task_key, trial.cell.attempt)
        if trial.cell != expected_by_key[key]:
            raise ValueError(f"observed Harbor cell identity differs from plan: {key}")

    locator_keys = [(locator.cell.task_key, locator.cell.attempt) for locator in loaded.locators]
    duplicate_locators = sorted(key for key, count in Counter(locator_keys).items() if count > 1)
    if duplicate_locators:
        raise ValueError(f"duplicate Harbor result locator(s): {duplicate_locators}")
    if set(locator_keys) != set(observed_keys):
        missing = sorted(set(observed_keys) - set(locator_keys))
        extra = sorted(set(locator_keys) - set(observed_keys))
        raise ValueError(
            f"Harbor result locators differ from cells: missing={missing}, extra={extra}"
        )
    locators = {
        (locator.cell.task_key, locator.cell.attempt): locator for locator in loaded.locators
    }
    for trial in result.trials:
        key = (trial.cell.task_key, trial.cell.attempt)
        if locators[key].cell != trial.cell:
            raise ValueError(f"Harbor locator cell identity differs from trial: {key}")

    by_task: defaultdict[str, list[BenchmarkTrialResult]] = defaultdict(list)
    for trial in result.trials:
        by_task[trial.task_identity].append(trial)
    expected_identities = set(task_ids)
    found_identities = set(by_task)
    if found_identities != expected_identities:
        missing = sorted(expected_identities - found_identities)
        extra = sorted(found_identities - expected_identities)
        raise ValueError(
            f"Harbor task identities differ from frozen scorer: missing={missing}, extra={extra}"
        )

    ordered: dict[str, list[AdmittedHarborTrial]] = {}
    expected_attempts = list(range(1, attempts + 1))
    expected_keys_by_task = dict(zip(task_ids, task_keys, strict=True))
    expected_environment_by_task = dict(zip(task_ids, task_environment_digests, strict=True))
    provider_request_ids: set[str] = set()
    for task_id in task_ids:
        trials = sorted(by_task[task_id], key=lambda trial: trial.cell.attempt)
        found_attempts = [trial.cell.attempt for trial in trials]
        if found_attempts != expected_attempts:
            raise ValueError(
                f"Harbor task {task_id!r} attempts differ from frozen scorer: "
                f"expected={expected_attempts}, found={found_attempts}"
            )
        _validate_task_attempt_identity(
            task_id,
            trials,
            expected_task_key=expected_keys_by_task[task_id],
            expected_environment_digest=expected_environment_by_task[task_id],
        )
        scored: list[AdmittedHarborTrial] = []
        for trial in trials:
            if trial.run_health not in {
                BenchmarkRunHealth.VALID,
                BenchmarkRunHealth.CANDIDATE_DAMAGED,
            }:
                raise ValueError(
                    f"Harbor task {task_id!r} attempt {trial.cell.attempt} has "
                    f"run health {trial.run_health.value!r}; retry or invalidate the cell"
                )
            if trial.status not in {
                BenchmarkTrialStatus.SCORED,
                BenchmarkTrialStatus.CANDIDATE_FAILURE,
            }:
                detail = trial.error.kind.value if trial.error is not None else trial.status.value
                raise ValueError(
                    f"Harbor task {task_id!r} attempt {trial.cell.attempt} is not scored: {detail}"
                )
            if trial.run_health is BenchmarkRunHealth.CANDIDATE_DAMAGED and (
                trial.candidate_outcome.status is not BenchmarkCandidateStatus.FAILED
            ):
                raise ValueError(
                    f"Harbor task {task_id!r} attempt {trial.cell.attempt} has "
                    "candidate-damaged health without a failed candidate outcome"
                )
            if (
                trial.status is BenchmarkTrialStatus.SCORED
                and trial.error is not None
                and trial.error.kind is not BenchmarkFailureKind.TASK_TIMEOUT
                and not (
                    trial.run_health is BenchmarkRunHealth.CANDIDATE_DAMAGED
                    and trial.candidate_outcome.status is BenchmarkCandidateStatus.FAILED
                    and trial.error.kind
                    in {
                        BenchmarkFailureKind.ENVIRONMENT,
                        BenchmarkFailureKind.VERIFIER,
                        BenchmarkFailureKind.UNCLASSIFIED,
                    }
                )
            ):
                raise ValueError(
                    f"Harbor task {task_id!r} attempt {trial.cell.attempt} carries "
                    f"infrastructure error {trial.error.kind.value!r} despite scored status"
                )
            if trial.status is BenchmarkTrialStatus.CANDIDATE_FAILURE and (
                trial.run_health is not BenchmarkRunHealth.CANDIDATE_DAMAGED
                or trial.candidate_outcome.status is not BenchmarkCandidateStatus.FAILED
                or trial.rewards is not None
                or (
                    trial.error is not None
                    and trial.error.kind
                    not in {
                        BenchmarkFailureKind.ENVIRONMENT,
                        BenchmarkFailureKind.VERIFIER,
                        BenchmarkFailureKind.UNCLASSIFIED,
                    }
                )
            ):
                raise ValueError(
                    f"Harbor task {task_id!r} attempt {trial.cell.attempt} has invalid "
                    "candidate-failure evidence"
                )
            if trial.candidate_outcome.status is BenchmarkCandidateStatus.UNKNOWN and not (
                trial.error is not None and trial.error.kind is BenchmarkFailureKind.TASK_TIMEOUT
            ):
                raise ValueError(
                    f"Harbor task {task_id!r} attempt {trial.cell.attempt} lacks trusted "
                    "candidate outcome metadata"
                )
            verifier_reward, score = harbor_trial_analysis_values(
                trial,
                reward_key=reward_key,
            )
            if (
                trial.usage.calls is None
                or trial.usage.calls_status is not BenchmarkUsageStatus.EXACT
            ):
                raise ValueError(
                    f"Harbor task {task_id!r} attempt {trial.cell.attempt} lacks an exact "
                    "successful provider call count"
                )
            locator = locators[(trial.cell.task_key, trial.cell.attempt)]
            trace = _read_trace(
                loaded,
                locator,
                allow_missing=(
                    trial.error is not None
                    and trial.error.kind is BenchmarkFailureKind.TASK_TIMEOUT
                ),
                expected_calls=trial.usage.calls,
                provider_config=provider_config,
                compute_envelope=compute_envelope,
            )
            request_ids = {receipt.provider_request_id for receipt in trace.provider_receipts}
            if provider_request_ids.intersection(request_ids):
                raise ValueError("Harbor provider request identity was reused across trials")
            provider_request_ids.update(request_ids)
            scored.append(
                AdmittedHarborTrial(
                    trial=trial,
                    verifier_reward=verifier_reward,
                    score=score,
                    trace_text=trace.text,
                    trace_digest=trace.digest,
                    provider_receipts=trace.provider_receipts,
                    provider_receipt_call_indexes=trace.provider_call_indexes,
                )
            )
        ordered[task_id] = scored
    return ordered


def _validate_task_attempt_identity(
    task_id: str,
    trials: list[BenchmarkTrialResult],
    *,
    expected_task_key: str,
    expected_environment_digest: str,
) -> None:
    for trial in trials:
        if trial.task_instruction != normalize_durable_text(trial.task_instruction):
            raise ValueError(f"Harbor task {task_id!r} instruction is not durable text")
    identities = {
        (
            trial.cell.task_key,
            trial.cell.task_name,
            trial.task_identity,
            trial.task_checksum,
            trial.source,
            trial.task_instruction,
            trial.task_environment_digest,
            trial.cell.config_digest,
        )
        for trial in trials
    }
    if len(identities) != 1:
        raise ValueError(f"Harbor task {task_id!r} attempts have inconsistent identity")
    if trials[0].cell.task_key != expected_task_key:
        raise ValueError(
            f"Harbor task {task_id!r} key differs from the frozen qualification manifest"
        )
    if _required_task_environment_digest(trials[0]) != expected_environment_digest:
        raise ValueError(
            f"Harbor task {task_id!r} environment differs from the frozen qualification run"
        )


def _read_trace(
    loaded: LoadedHarborJobResult,
    locator: HarborTrialLocator,
    *,
    allow_missing: bool,
    expected_calls: int,
    provider_config: ProviderConfig,
    compute_envelope: HarborAgentComputeEnvelope,
) -> _TraceEvidence:
    relative_path = locator.trial_dir / _TRACE_FILENAME
    trace_path = loaded.resolve_path(relative_path)
    if not trace_path.exists():
        if not allow_missing or expected_calls != 0:
            raise ValueError(f"completed Harbor trial is missing its WMH trace: {relative_path}")
        return _TraceEvidence(
            text="(trace unavailable)",
            digest="missing",
            provider_receipts=(),
            provider_call_indexes=(),
        )
    if not trace_path.is_file():
        raise ValueError(f"Harbor trace must be a regular file: {relative_path}")
    with trace_path.open("rb") as handle:
        payload = handle.read(_MAX_TRACE_FILE_BYTES + 1)
    if len(payload) > _MAX_TRACE_FILE_BYTES:
        raise ValueError(
            f"Harbor trace exceeds the {_MAX_TRACE_FILE_BYTES}-byte evidence limit: {relative_path}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Harbor trace is not valid UTF-8: {relative_path}") from exc
    canonical_lines: list[str] = []
    receipt_payloads: list[JsonObject] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(
                f"Harbor trace contains an empty JSONL record at line {line_number}: "
                f"{relative_path}"
            )
        try:
            record = _TraceRecord.model_validate_json(line)
        except PydanticValidationError as exc:
            raise ValueError(
                f"Harbor trace contains an invalid event at line {line_number}: {relative_path}"
            ) from exc
        canonical_lines.append(
            json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        if record.kind == "provider_receipt":
            receipt_payloads.append(record.payload)
    try:
        receipt_trace = validate_provider_receipt_trace(
            receipt_payloads,
            expected_calls=expected_calls,
            provider_config=provider_config,
            requested_temperature=compute_envelope.temperature,
            max_tokens=compute_envelope.max_output_tokens,
        )
    except (TypeError, ValueError, PydanticValidationError) as exc:
        raise ValueError(
            f"Harbor trace contains invalid provider-call evidence: {relative_path}"
        ) from exc
    canonical = "\n".join(canonical_lines)
    return _TraceEvidence(
        text=normalize_durable_text(canonical) if canonical else "(trace contained no events)",
        digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        provider_receipts=receipt_trace.receipts,
        provider_call_indexes=receipt_trace.call_indexes,
    )


def _failure_mechanisms(trials: list[AdmittedHarborTrial]) -> tuple[str, ...]:
    mechanisms: set[str] = set()
    for item in trials:
        if item.score == 1.0:
            continue
        outcome = item.trial.candidate_outcome
        if outcome.status is BenchmarkCandidateStatus.FAILED:
            reason = outcome.failure_reason
            labels = {
                BenchmarkCandidateFailureReason.TIMEOUT: "candidate timeout",
                BenchmarkCandidateFailureReason.RESOURCE_LIMIT: "candidate resource limit",
                BenchmarkCandidateFailureReason.RUNTIME_ERROR: "candidate runtime error",
                BenchmarkCandidateFailureReason.INVALID_REQUEST: "candidate invalid request",
                None: "candidate failure",
            }
            mechanisms.add(labels[reason])
        elif (
            item.trial.error is not None
            and item.trial.error.kind is BenchmarkFailureKind.TASK_TIMEOUT
        ):
            mechanisms.add("agent task timeout")
        else:
            mechanisms.add("ground-truth reward below one")
    return tuple(sorted(mechanisms))


def _render_task_evidence(trials: list[AdmittedHarborTrial]) -> str:
    separator_chars = 2 * (len(trials) - 1)
    per_attempt_limit = max(
        0,
        (MAX_HARBOR_TASK_EVIDENCE_CHARS - separator_chars) // len(trials),
    )
    sections: list[str] = []
    for item in trials:
        trial = item.trial
        outcome = trial.candidate_outcome
        details = [
            f"## Attempt {trial.cell.attempt}",
            f"score={item.score:.1f}",
            (
                "verifier_reward=unavailable"
                if item.verifier_reward is None
                else f"verifier_reward={item.verifier_reward:.1f}"
            ),
            f"candidate_status={outcome.status.value}",
        ]
        if outcome.failure_reason is not None:
            details.append(f"candidate_failure_reason={outcome.failure_reason.value}")
        if outcome.terminal_reason is not None:
            details.append(f"candidate_terminal_reason={outcome.terminal_reason.value}")
        if trial.error is not None:
            details.append(f"trial_error={trial.error.kind.value}")
        details.extend([f"trace_digest={item.trace_digest}", item.trace_text])
        sections.append(
            _bound_text(
                "\n".join(details),
                limit=per_attempt_limit,
                label=f"attempt {trial.cell.attempt} evidence",
            )
        )
    return _bound_text(
        "\n\n".join(sections),
        limit=MAX_HARBOR_TASK_EVIDENCE_CHARS,
        label="task evidence",
    )


def _bound_text(value: str, *, limit: int, label: str) -> str:
    if len(value) <= limit:
        return value
    marker = f"\n...[{label} truncated; original_chars={len(value)}]...\n"
    retained = limit - len(marker)
    if retained <= 0:
        return marker[:limit]
    head = retained // 2
    tail = retained - head
    return value[:head] + marker + value[-tail:]


def _analysis_score(trial: BenchmarkTrialResult, *, verifier_reward: float) -> float:
    """Apply the frozen primary rule while retaining Harbor's reward as diagnostic evidence."""
    if trial.candidate_outcome.status is BenchmarkCandidateStatus.FAILED:
        return 0.0
    if trial.error is not None and trial.error.kind is BenchmarkFailureKind.TASK_TIMEOUT:
        return 0.0
    return verifier_reward


def harbor_trial_analysis_values(
    trial: BenchmarkTrialResult,
    *,
    reward_key: str,
) -> tuple[float | None, float]:
    """Recompute the canonical binary reward and attributed analysis score.

    Reports can call this on JSON reload so a stored score cannot silently disagree with the full
    typed trial status and candidate-failure attribution that justified it.
    """
    if trial.status is BenchmarkTrialStatus.CANDIDATE_FAILURE:
        return None, 0.0
    if trial.status is not BenchmarkTrialStatus.SCORED:
        raise ValueError("only admitted scored or candidate-failure trials have analysis values")
    if trial.rewards is None or reward_key not in trial.rewards:
        raise ValueError(
            f"Harbor task {trial.task_identity!r} attempt {trial.cell.attempt} omits binary reward "
            f"{reward_key!r}"
        )
    raw_reward = trial.rewards[reward_key]
    if isinstance(raw_reward, bool) or raw_reward not in (0, 0.0, 1, 1.0):
        raise ValueError(
            f"Harbor task {trial.task_identity!r} attempt {trial.cell.attempt} reward "
            f"{reward_key!r} must be binary, got {raw_reward!r}"
        )
    verifier_reward = float(raw_reward)
    return verifier_reward, _analysis_score(trial, verifier_reward=verifier_reward)


def _evaluation_id(
    *,
    result: BenchmarkRunResult,
    request: ScoreRequest,
    reward_key: str,
    task_ids: tuple[str, ...],
    task_keys: tuple[str, ...],
    task_environment_digests: tuple[str, ...],
    attempts: int,
    compute_envelope: HarborAgentComputeEnvelope,
    cells: list[_EvaluationCellIdentity],
) -> str:
    payload = {
        "schema_version": 1,
        "job_name": result.job_name,
        "run_identity": result.identity.model_dump(mode="json"),
        "score_request": request.model_dump(mode="json"),
        "reward_key": reward_key,
        "task_ids": list(task_ids),
        "task_keys": list(task_keys),
        "task_environment_digests": list(task_environment_digests),
        "attempts": attempts,
        "compute_envelope": compute_envelope.model_dump(mode="json"),
        "cells": [cell.model_dump(mode="json") for cell in cells],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return "harbor-sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _required_task_environment_digest(trial: BenchmarkTrialResult) -> str:
    digest = trial.task_environment_digest
    if digest is None:
        raise ValueError(
            f"Harbor task {trial.task_identity!r} attempt {trial.cell.attempt} omits trusted "
            "task environment identity"
        )
    return digest


def _rewards_digest(rewards: Rewards | None) -> str:
    if rewards is not None:
        for key, value in rewards.items():
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    "Harbor reward mappings must contain string keys and numeric values"
                )
            if not math.isfinite(value):
                raise ValueError("Harbor reward mappings must contain only finite values")
    canonical = json.dumps(
        rewards,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
