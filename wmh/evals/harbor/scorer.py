"""Project strict Harbor task matrices into benchmark-neutral harness scores."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
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
)
from wmh.evals.harbor.agent import WMH_PI_AGENT_VERSION
from wmh.evals.harbor.config import (
    SUPPORTED_HARBOR_VERSION,
    HarborEnvironmentBackend,
    HarborJobSpec,
)
from wmh.evals.harbor.e2b_environment import (
    ExactE2BBuildRecord,
    require_exact_e2b_build_record,
)
from wmh.evals.harbor.evaluator import (
    HARBOR_EVALUATOR_VERSION,
    HarborEvaluator,
    harbor_run_expectation,
)
from wmh.evals.harbor.qualification_types import (
    QualifiedE2BBuildIdentity,
    QualifiedHarborTask,
)
from wmh.evals.harbor.receipt_trace import (
    ProviderReceiptTrace,
    validate_provider_receipt_trace,
)
from wmh.evals.harbor.results import HarborTrialLocator, LoadedHarborJobResult
from wmh.harness.cost import (
    SearchComponentCostBinding,
    SearchComponentCostRuntime,
    SearchComponentRole,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import EventKind
from wmh.harness.pi_runner_backend import (
    LocalPiRunnerSpec,
    PiRunnerBackendSpec,
    e2b_runner_resource_class,
    validate_pi_runner_turn_timeout,
)
from wmh.harness.scoring import (
    MAX_TASK_EVIDENCE_CHARS,
    HarnessScoreReport,
    ScoreCapabilities,
    ScoreRequest,
    ScoreRunHealth,
    TaskScore,
)
from wmh.harness.tools import READ_SKILL
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.receipt import ProviderResponseIdentity
from wmh.tracking.budget import (
    BudgetAccount,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceRole,
    validate_timed_resource_class,
)
from wmh.tracking.rate_limit import (
    ExternalDispatchRateAuthority,
    ExternalDispatchRateBinding,
    ExternalDispatchRatePolicy,
    bind_external_dispatch_rate_authority,
    validate_e2b_sandbox_create_rate_policy,
)

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


class HarborHarnessScorePlan(BaseModel):
    """Path-free scorer semantics that survive host and worker relocation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["wmh.harbor-harness-score-plan.v4"] = "wmh.harbor-harness-score-plan.v4"
    implementation: str = Field(min_length=1)
    job_name: str = Field(min_length=1)
    n_attempts: int = Field(ge=1)
    n_concurrent_trials: int = Field(ge=1)
    agent_n_concurrent: int | None = Field(default=None, ge=1)
    environment_backend: HarborEnvironmentBackend
    create_rate_policy: ExternalDispatchRatePolicy | None = None
    create_rate_binding: ExternalDispatchRateBinding | None = None
    allow_preexisting_e2b_builds: bool
    max_retries: int = Field(ge=0)
    retry_exceptions: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    provider_config: ProviderConfig
    response_identity: ProviderResponseIdentity
    qualified_tasks: tuple[QualifiedHarborTask, ...]
    reward_key: str
    runner_spec: PiRunnerBackendSpec
    compute_envelope: HarborAgentComputeEnvelope
    agent_version: str
    evaluator_version: str
    harbor_version: str

    @property
    def configuration_id(self) -> str:
        """Return the canonical identity used by checkpoints and cost bindings."""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return qualified task identities in frozen scoring order."""
        return tuple(task.task_id for task in self.qualified_tasks)

    @property
    def task_keys(self) -> tuple[str, ...]:
        """Return qualified content keys in frozen scoring order."""
        return tuple(task.task_key for task in self.qualified_tasks)

    @property
    def task_environment_digests(self) -> tuple[str, ...]:
        """Return qualified environment identities in frozen scoring order."""
        return tuple(task.task_environment_digest for task in self.qualified_tasks)

    @property
    def task_resource_classes(self) -> tuple[TimedResourceClass | None, ...]:
        """Return qualified external task classes in frozen scoring order."""
        return tuple(task.task_resource_class for task in self.qualified_tasks)


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
    runner_environment_digest: str
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
    provider_receipt_trace: ProviderReceiptTrace


@dataclass(frozen=True)
class AdmittedHarborTrial:
    """One exact Harbor trial admitted as binary analysis evidence."""

    trial: BenchmarkTrialResult
    verifier_reward: float | None
    score: float
    trace_text: str
    trace_digest: str
    provider_receipt_trace: ProviderReceiptTrace


@dataclass(frozen=True)
class _FrozenScorerInputs:
    spec: HarborJobSpec
    provider: ProviderConfig
    response_identity: ProviderResponseIdentity
    qualified_tasks: tuple[QualifiedHarborTask, ...]
    reward_key: str
    runner_spec: PiRunnerBackendSpec
    compute_envelope: HarborAgentComputeEnvelope
    create_rate_binding: ExternalDispatchRateBinding | None

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.qualified_tasks)

    @property
    def task_keys(self) -> tuple[str, ...]:
        return tuple(task.task_key for task in self.qualified_tasks)

    @property
    def task_environment_digests(self) -> tuple[str, ...]:
        return tuple(task.task_environment_digest for task in self.qualified_tasks)

    @property
    def task_resource_classes(self) -> tuple[TimedResourceClass | None, ...]:
        return tuple(task.task_resource_class for task in self.qualified_tasks)


def harbor_harness_score_plan(
    *,
    job_spec: HarborJobSpec,
    provider_config: ProviderConfig,
    response_identity: ProviderResponseIdentity | None = None,
    reference_harness: HarnessDoc,
    qualified_tasks: tuple[QualifiedHarborTask, ...],
    reward_key: str,
    runner_spec: PiRunnerBackendSpec | JsonObject | None = None,
    turn_timeout_s: float = 300.0,
    create_rate_binding: ExternalDispatchRateBinding | None = None,
) -> HarborHarnessScorePlan:
    """Freeze path-free scorer semantics before cost accounts or workers are created."""
    frozen = _freeze_scorer_inputs(
        job_spec=job_spec,
        provider_config=provider_config,
        response_identity=response_identity,
        reference_harness=reference_harness,
        qualified_tasks=qualified_tasks,
        reward_key=reward_key,
        runner_spec=runner_spec,
        turn_timeout_s=turn_timeout_s,
        create_rate_binding=create_rate_binding,
    )
    return _score_plan(frozen, scorer_type=HarborHarnessScorer)


class HarborHarnessScorer:
    """Score harnesses through one frozen, exact Harbor task matrix.

    The scorer deliberately advertises no subset or attempt override support. Search callers must
    disable screening and confirmation requests, so every candidate runs the same task cells.
    """

    capabilities = ScoreCapabilities(task_subsets=False, attempt_overrides=False)
    requires_search_cost_binding = True

    def __init__(
        self,
        *,
        job_spec: HarborJobSpec,
        provider_config: ProviderConfig,
        response_identity: ProviderResponseIdentity | None = None,
        reference_harness: HarnessDoc,
        qualified_tasks: tuple[QualifiedHarborTask, ...],
        reward_key: str,
        runner_spec: PiRunnerBackendSpec | JsonObject | None = None,
        turn_timeout_s: float = 300.0,
        cost_runtime: SearchComponentCostRuntime | None = None,
        create_rate_authority: ExternalDispatchRateAuthority | None = None,
    ) -> None:
        """Freeze task selection, provider route, backend, runner, and agent compute."""
        create_rate_binding = _validated_create_rate_binding(
            job_spec,
            runner_spec=runner_spec,
            turn_timeout_s=turn_timeout_s,
            authority=create_rate_authority,
        )
        frozen = _freeze_scorer_inputs(
            job_spec=job_spec,
            provider_config=provider_config,
            response_identity=response_identity,
            reference_harness=reference_harness,
            qualified_tasks=qualified_tasks,
            reward_key=reward_key,
            runner_spec=runner_spec,
            turn_timeout_s=turn_timeout_s,
            create_rate_binding=create_rate_binding,
        )
        self._job_spec = frozen.spec
        self._provider_config = frozen.provider
        self._response_identity = frozen.response_identity
        self._qualified_tasks = frozen.qualified_tasks
        self._task_ids = frozen.task_ids
        self._task_keys = frozen.task_keys
        self._task_environment_digests = frozen.task_environment_digests
        self._task_resource_classes = frozen.task_resource_classes
        self._reward_key = frozen.reward_key
        self._runner_spec = frozen.runner_spec
        self._compute_envelope = frozen.compute_envelope
        self._plan = _score_plan(frozen, scorer_type=type(self))
        self._cost_runtime = cost_runtime
        self._create_rate_authority = create_rate_authority
        if cost_runtime is not None:
            self.search_cost_binding = self._validate_cost_runtime(cost_runtime)
            self._resolved_cost_accounts()
        elif self.requires_search_cost_binding:
            raise ValueError("HarborHarnessScorer requires a complete search cost runtime")
        elif _requires_external_resources(self._job_spec, self._runner_spec):
            raise ValueError("E2B Harbor scoring requires a complete search cost runtime")
        self._runner: asyncio.Runner | None = None
        self._closed = False

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return the exact ordered task identities frozen by this scorer."""
        return self._task_ids

    @property
    def configuration_id(self) -> str:
        """Return an opaque digest of the exact scorer and task-matrix configuration."""
        return self._plan.configuration_id

    @property
    def plan(self) -> HarborHarnessScorePlan:
        """Return a detached path-free scorer plan suitable for durable commitments."""
        return HarborHarnessScorePlan.model_validate(self._plan.model_dump())

    @property
    def create_rate_binding(self) -> ExternalDispatchRateBinding | None:
        """Return the shared external dispatch authority used by E2B task and runner creates."""
        binding = self._plan.create_rate_binding
        if binding is None:
            return None
        return ExternalDispatchRateBinding.model_validate(binding.model_dump())

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
        budget_account, task_accounts, runner_account = self._resolved_cost_accounts()
        create_rate_authority = self._validated_create_rate_authority()
        rejection = self.validate_candidate(candidate)
        if rejection is not None:
            raise ValueError(rejection)

        job_name = self._job_name(candidate)
        spec = self._job_spec.model_copy(update={"job_name": job_name}, deep=True)
        provider_config = self._provider_config.model_copy(deep=True)
        if budget_account is None:
            evaluator = HarborEvaluator(
                spec,
                provider_config,
                runner_spec=self._runner_spec,
                turn_timeout_s=self._compute_envelope.turn_timeout_s,
                response_identity=self._response_identity,
                qualified_tasks=self._qualified_tasks,
            )
        else:
            evaluator = HarborEvaluator(
                spec,
                provider_config,
                runner_spec=self._runner_spec,
                turn_timeout_s=self._compute_envelope.turn_timeout_s,
                response_identity=self._response_identity,
                budget_account=budget_account,
                task_resource_budget_accounts=task_accounts,
                runner_resource_budget_account=runner_account,
                create_rate_authority=create_rate_authority,
                qualified_tasks=self._qualified_tasks,
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
        budget_account, _task_accounts, _runner_account = self._resolved_cost_accounts()
        result = loaded.result
        validate_harbor_run_identity(
            result,
            candidate=candidate,
            spec=spec,
            provider_config=self._provider_config,
            runner_spec=self._runner_spec,
            turn_timeout_s=self._compute_envelope.turn_timeout_s,
            budget_policy_digest=(
                budget_account.policy.policy_digest if budget_account is not None else None
            ),
            response_identity=self._response_identity,
            require_exact_run_config=True,
        )
        ordered = admit_harbor_matrix(
            loaded,
            task_ids=self._task_ids,
            task_keys=self._task_keys,
            task_environment_digests=self._task_environment_digests,
            attempts=self.default_attempts,
            reward_key=self._reward_key,
            provider_config=self._provider_config,
            response_identity=self._response_identity,
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
                        runner_environment_digest=_required_runner_environment_digest(trial),
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

    def _validate_cost_runtime(
        self,
        runtime: SearchComponentCostRuntime,
    ) -> SearchComponentCostBinding:
        """Validate exact scorer cost provenance without creating a Harbor evaluator."""
        if not isinstance(runtime, SearchComponentCostRuntime):
            raise TypeError("HarborHarnessScorer cost_runtime must be a SearchComponentCostRuntime")
        binding = SearchComponentCostBinding.model_validate(runtime.binding.model_dump())
        if binding.role not in {
            SearchComponentRole.SCORER,
            SearchComponentRole.HOLDOUT_SCORER,
        }:
            raise ValueError("HarborHarnessScorer cost runtime must use a scorer role")
        if binding.configuration_id != self.configuration_id:
            raise ValueError("HarborHarnessScorer configuration_id differs from its cost runtime")
        if len(binding.providers) != 1:
            raise ValueError(
                "HarborHarnessScorer cost runtime must bind exactly one provider account"
            )
        if binding.providers[0].provider_config != self._provider_config:
            raise ValueError("Harbor scorer provider config differs from its cost binding")
        if binding.providers[0].response_identity != self._response_identity:
            raise ValueError("Harbor scorer response identity differs from its cost binding")
        return binding

    def _resolved_cost_accounts(
        self,
    ) -> tuple[
        BudgetAccount | None,
        tuple[TimedResourceBudgetAccount, ...],
        TimedResourceBudgetAccount | None,
    ]:
        """Reaudit and resolve exact provider and resource accounts before dispatch."""
        runtime = self._cost_runtime
        if runtime is None:
            if self.requires_search_cost_binding:
                raise ValueError("HarborHarnessScorer requires a complete search cost runtime")
            if _requires_external_resources(self._job_spec, self._runner_spec):
                raise ValueError("E2B Harbor scoring requires a complete search cost runtime")
            return None, (), None
        binding = self._validate_cost_runtime(runtime)
        provider_account = runtime.provider_account(binding.providers[0])
        expected_task_classes = {
            resource_class.digest: resource_class
            for resource_class in self._task_resource_classes
            if resource_class is not None
        }
        task_accounts_by_class: dict[str, TimedResourceBudgetAccount] = {}
        runner_account: TimedResourceBudgetAccount | None = None
        for resource in binding.timed_resources:
            account = runtime.timed_resource_account(resource)
            if resource.resource_type == TimedResourceRole.TASK_ENVIRONMENT.value:
                if resource.resource_class_digest in task_accounts_by_class:
                    raise ValueError(
                        "Harbor scorer cost runtime has duplicate task resource class accounts"
                    )
                task_accounts_by_class[resource.resource_class_digest] = account
            elif resource.resource_type == TimedResourceRole.AGENT_RUNNER.value:
                if runner_account is not None:
                    raise ValueError("Harbor scorer cost runtime has duplicate runner accounts")
                runner_account = account
            else:
                raise ValueError(
                    "Harbor scorer cost runtime contains a non-scoring resource account"
                )
        if set(task_accounts_by_class) != set(expected_task_classes):
            raise ValueError(
                "Harbor scorer task resource accounts must exactly cover qualified classes"
            )
        task_accounts: list[TimedResourceBudgetAccount] = []
        for resource_digest in sorted(expected_task_classes):
            account = task_accounts_by_class[resource_digest]
            validate_timed_resource_class(account, expected_task_classes[resource_digest])
            task_accounts.append(account)
        self._require_qualified_task_builds(task_accounts_by_class)
        if isinstance(self._runner_spec, LocalPiRunnerSpec):
            if runner_account is not None:
                raise ValueError("local Pi scorer cannot consume a runner resource account")
        elif runner_account is None:
            raise ValueError("E2B Pi scorer requires one exact runner resource account")
        else:
            validate_timed_resource_class(
                runner_account,
                e2b_runner_resource_class(self._runner_spec),
            )
        return provider_account, tuple(task_accounts), runner_account

    def _require_qualified_task_builds(
        self,
        task_accounts_by_class: dict[str, TimedResourceBudgetAccount],
    ) -> None:
        """Reject E2B build-record drift before constructing a paid evaluator."""
        build_records: dict[str, ExactE2BBuildRecord] = {}
        for qualification in self._qualified_tasks:
            if qualification.environment_backend is HarborEnvironmentBackend.LOCAL:
                continue
            identity = qualification.e2b_build_identity
            resource_class = qualification.task_resource_class
            if identity is None or resource_class is None:
                raise ValueError("E2B scorer task lacks exact qualification evidence")
            record = build_records.get(identity.build_config_digest)
            if record is None:
                account = task_accounts_by_class[resource_class.digest]
                record = require_exact_e2b_build_record(
                    jobs_dir=self._job_spec.jobs_dir,
                    environment_id=identity.environment_id,
                    build_context_digest=identity.build_context_digest,
                    docker_image=identity.docker_image,
                    cpu_count=identity.cpu_count,
                    memory_mb=identity.memory_mb,
                    expected_budget_authority=account,
                    allow_preexisting_outside_study=False,
                )
                build_records[identity.build_config_digest] = record
            _validate_qualified_build_record(record, qualification)

    def _validated_create_rate_authority(self) -> ExternalDispatchRateAuthority | None:
        """Revalidate the shared E2B create authority before evaluator construction."""
        authority = self._create_rate_authority
        binding = _validated_create_rate_binding(
            self._job_spec,
            runner_spec=self._runner_spec,
            turn_timeout_s=self._compute_envelope.turn_timeout_s,
            authority=authority,
        )
        if binding != self._plan.create_rate_binding:
            raise ValueError("Harbor scorer create-rate authority changed after construction")
        return authority

    def _release_runner(self) -> None:
        runner = self._runner
        self._runner = None
        if runner is not None:
            runner.close()

    def _job_name(self, candidate: HarnessDoc) -> str:
        cost_binding_digest = (
            self._cost_runtime.search_binding.digest if self._cost_runtime is not None else None
        )
        payload = {
            "schema_version": 2,
            "score_plan": self._plan.model_dump(mode="json"),
            "candidate": candidate.model_dump(mode="json"),
            "search_cost_binding_digest": cost_binding_digest,
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


def _freeze_scorer_inputs(
    *,
    job_spec: HarborJobSpec,
    provider_config: ProviderConfig,
    response_identity: ProviderResponseIdentity | None,
    reference_harness: HarnessDoc,
    qualified_tasks: tuple[QualifiedHarborTask, ...],
    reward_key: str,
    runner_spec: PiRunnerBackendSpec | JsonObject | None,
    turn_timeout_s: float,
    create_rate_binding: ExternalDispatchRateBinding | None,
) -> _FrozenScorerInputs:
    """Validate host runtime inputs and detach their path-free scoring semantics."""
    validated_runner = TypeAdapter(PiRunnerBackendSpec).validate_python(
        runner_spec if runner_spec is not None else LocalPiRunnerSpec().model_dump(mode="json")
    )
    validate_pi_runner_turn_timeout(validated_runner, turn_timeout_s=turn_timeout_s)
    if not reward_key.strip():
        raise ValueError("reward_key must be non-empty")
    validate_durable_text(reward_key, field="Harbor reward key")
    frozen_qualified_tasks = tuple(
        QualifiedHarborTask.model_validate(task.model_dump()) for task in qualified_tasks
    )
    frozen_task_ids = tuple(task.task_id for task in frozen_qualified_tasks)
    if not frozen_task_ids:
        raise ValueError("qualified_tasks must be non-empty")
    if len(set(frozen_task_ids)) != len(frozen_task_ids):
        raise ValueError("qualified task IDs must be unique")
    frozen_task_keys = tuple(task.task_key for task in frozen_qualified_tasks)
    if len(set(frozen_task_keys)) != len(frozen_task_keys):
        raise ValueError("qualified task keys must be unique")
    spec = HarborJobSpec.model_validate(job_spec.model_dump())
    spec = spec.model_copy(
        update={"jobs_dir": spec.jobs_dir.expanduser().resolve()},
        deep=True,
    )
    _validate_job_prefix(spec.job_name)
    validate_exact_harbor_dataset_selection(spec, frozen_task_ids)
    if any(
        task.environment_backend is not spec.environment_backend for task in frozen_qualified_tasks
    ):
        raise ValueError("qualified Harbor task backends differ from the scorer backend")
    build_identities: dict[str, QualifiedE2BBuildIdentity] = {}
    for task in frozen_qualified_tasks:
        identity = task.e2b_build_identity
        if identity is None:
            continue
        prior = build_identities.setdefault(identity.build_config_digest, identity)
        if prior != identity:
            raise ValueError(
                "one qualified E2B build config cannot name conflicting immutable builds"
            )
    for artifact_path in spec.artifact_paths:
        candidate = PurePosixPath(artifact_path)
        if (
            not artifact_path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or "\\" in artifact_path
            or candidate.as_posix() != artifact_path
        ):
            raise ValueError("Harbor artifact_paths must be project-relative canonical paths")
    requires_rate = _requires_external_resources(spec, validated_runner)
    if requires_rate:
        if spec.create_rate_policy is None or create_rate_binding is None:
            raise ValueError("E2B Harbor scoring requires a frozen create-rate binding")
        validate_e2b_sandbox_create_rate_policy(spec.create_rate_policy)
        if create_rate_binding.policy_digest != spec.create_rate_policy.digest:
            raise ValueError("Harbor scorer create-rate binding differs from its policy")
    elif spec.create_rate_policy is not None or create_rate_binding is not None:
        raise ValueError("local Harbor scoring cannot carry an E2B create-rate binding")
    provider = ProviderConfig.model_validate(provider_config.model_dump())
    frozen_response_identity = _resolve_provider_response_identity(
        provider,
        response_identity,
    )
    envelope = harbor_agent_compute_envelope(
        reference_harness,
        turn_timeout_s=turn_timeout_s,
    )
    if envelope.runtime_kind != "pi-node":
        raise ValueError(
            "HarborHarnessScorer requires a pi-node reference harness, got "
            f"{envelope.runtime_kind!r}"
        )
    return _FrozenScorerInputs(
        spec=spec,
        provider=provider,
        response_identity=frozen_response_identity,
        qualified_tasks=frozen_qualified_tasks,
        reward_key=reward_key,
        runner_spec=validated_runner,
        compute_envelope=envelope,
        create_rate_binding=(
            None
            if create_rate_binding is None
            else ExternalDispatchRateBinding.model_validate(create_rate_binding.model_dump())
        ),
    )


def _score_plan(
    frozen: _FrozenScorerInputs,
    *,
    scorer_type: type[object],
) -> HarborHarnessScorePlan:
    """Project validated runtime inputs into a path-free durable plan."""
    spec = frozen.spec
    return HarborHarnessScorePlan(
        implementation=f"{scorer_type.__module__}.{scorer_type.__qualname__}",
        job_name=spec.job_name,
        n_attempts=spec.n_attempts,
        n_concurrent_trials=spec.n_concurrent_trials,
        agent_n_concurrent=spec.agent_n_concurrent,
        environment_backend=spec.environment_backend,
        create_rate_policy=spec.create_rate_policy,
        create_rate_binding=frozen.create_rate_binding,
        allow_preexisting_e2b_builds=spec.allow_preexisting_e2b_builds,
        max_retries=spec.max_retries,
        retry_exceptions=tuple(sorted(spec.retry_exceptions)),
        artifact_paths=tuple(spec.artifact_paths),
        provider_config=frozen.provider,
        response_identity=frozen.response_identity,
        qualified_tasks=frozen.qualified_tasks,
        reward_key=frozen.reward_key,
        runner_spec=frozen.runner_spec,
        compute_envelope=frozen.compute_envelope,
        agent_version=WMH_PI_AGENT_VERSION,
        evaluator_version=HARBOR_EVALUATOR_VERSION,
        harbor_version=SUPPORTED_HARBOR_VERSION,
    )


def _requires_external_resources(
    spec: HarborJobSpec,
    runner_spec: PiRunnerBackendSpec,
) -> bool:
    return spec.environment_backend is HarborEnvironmentBackend.E2B or not isinstance(
        runner_spec, LocalPiRunnerSpec
    )


def _resolve_provider_response_identity(
    provider_config: ProviderConfig,
    response_identity: ProviderResponseIdentity | None,
) -> ProviderResponseIdentity:
    """Freeze exact served-model evidence, deriving only Bedrock's explicit null shape."""
    if response_identity is None:
        if provider_config.kind is not ProviderKind.BEDROCK:
            raise ValueError(
                "OpenAI-shaped Harbor scoring requires an exact provider response identity"
            )
        return ProviderResponseIdentity(provider=provider_config.kind)
    frozen = ProviderResponseIdentity.model_validate(response_identity.model_dump())
    if frozen.provider is not provider_config.kind:
        raise ValueError("provider response identity differs from the Harbor provider route")
    return frozen


def _validate_qualified_build_record(
    record: ExactE2BBuildRecord,
    qualification: QualifiedHarborTask,
) -> None:
    """Require the current registry record to equal pre-run qualification evidence."""
    identity = qualification.e2b_build_identity
    if identity is None:
        raise ValueError("E2B scorer task lacks an exact qualified build identity")
    if (
        record.build_config_digest != identity.build_config_digest
        or record.digest != identity.build_record_digest
        or record.template_id != identity.template_id
        or record.build_id != identity.build_id
    ):
        raise ValueError("scored E2B task build differs from its qualification evidence")


def _validated_create_rate_binding(
    job_spec: HarborJobSpec,
    *,
    runner_spec: PiRunnerBackendSpec | JsonObject | None,
    turn_timeout_s: float,
    authority: ExternalDispatchRateAuthority | None,
) -> ExternalDispatchRateBinding | None:
    """Validate and register one shared E2B create authority without dispatching."""
    spec = HarborJobSpec.model_validate(job_spec.model_dump())
    validated_runner = TypeAdapter(PiRunnerBackendSpec).validate_python(
        runner_spec if runner_spec is not None else LocalPiRunnerSpec().model_dump(mode="json")
    )
    validate_pi_runner_turn_timeout(validated_runner, turn_timeout_s=turn_timeout_s)
    requires_rate = _requires_external_resources(spec, validated_runner)
    if not requires_rate:
        if authority is not None:
            raise ValueError("local Harbor scoring cannot carry an E2B create-rate authority")
        return None
    if authority is None or spec.create_rate_policy is None:
        raise ValueError("E2B Harbor scoring requires a frozen create-rate authority")
    frozen_policy = validate_e2b_sandbox_create_rate_policy(authority.policy)
    if frozen_policy != spec.create_rate_policy:
        raise ValueError("Harbor scorer create-rate authority differs from its policy")
    return bind_external_dispatch_rate_authority(authority)


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
    runner_spec: PiRunnerBackendSpec | JsonObject | None = None,
    runner_image: str | None = None,
    turn_timeout_s: float,
    require_exact_run_config: bool = False,
    budget_policy_digest: str | None = None,
    response_identity: ProviderResponseIdentity | None = None,
) -> None:
    """Require loaded evidence to match its frozen harness and execution route."""
    if runner_spec is not None and runner_image is not None:
        raise ValueError("runner_spec and runner_image are mutually exclusive")
    validated_runner = TypeAdapter(PiRunnerBackendSpec).validate_python(
        runner_spec
        if runner_spec is not None
        else (
            LocalPiRunnerSpec(image=runner_image)
            if runner_image is not None
            else LocalPiRunnerSpec()
        )
    )
    validate_pi_runner_turn_timeout(validated_runner, turn_timeout_s=turn_timeout_s)
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
    if identity.runner_config_digest != validated_runner.config_digest:
        mismatches.append("runner configuration")
    if identity.runner_environment_digest != validated_runner.attestation.digest:
        mismatches.append("runner environment")
    if require_exact_run_config:
        expectation = harbor_run_expectation(
            candidate=candidate,
            spec=spec,
            provider_config=provider_config,
            runner_spec=validated_runner,
            turn_timeout_s=turn_timeout_s,
            response_identity=response_identity,
            budget_policy_digest=budget_policy_digest,
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
    response_identity: ProviderResponseIdentity | None = None,
    compute_envelope: HarborAgentComputeEnvelope,
) -> dict[str, list[AdmittedHarborTrial]]:
    """Admit an exact Harbor matrix and map candidate-owned failures to analysis zero."""
    frozen_response_identity = _resolve_provider_response_identity(
        provider_config,
        response_identity,
    )
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
        if _required_runner_environment_digest(trial) != (
            result.identity.runner_environment_digest
        ):
            raise ValueError(f"observed Harbor runner identity differs from plan: {key}")

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
                response_identity=frozen_response_identity,
                compute_envelope=compute_envelope,
            )
            request_ids = {
                receipt.provider_request_id for receipt in trace.provider_receipt_trace.receipts
            }
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
                    provider_receipt_trace=trace.provider_receipt_trace,
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
    response_identity: ProviderResponseIdentity,
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
            provider_receipt_trace=ProviderReceiptTrace(receipts=(), call_indexes=()),
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
            response_identity=response_identity,
        )
    except (TypeError, ValueError, PydanticValidationError) as exc:
        raise ValueError(
            f"Harbor trace contains invalid provider-call evidence: {relative_path}"
        ) from exc
    canonical = "\n".join(canonical_lines)
    return _TraceEvidence(
        text=normalize_durable_text(canonical) if canonical else "(trace contained no events)",
        digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        provider_receipt_trace=receipt_trace,
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


def _required_runner_environment_digest(trial: BenchmarkTrialResult) -> str:
    digest = trial.runner_environment_digest
    if digest is None:
        raise ValueError(
            f"Harbor task {trial.task_identity!r} attempt {trial.cell.attempt} omits trusted "
            "runner environment identity"
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
