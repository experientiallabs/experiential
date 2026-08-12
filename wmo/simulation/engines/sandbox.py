"""Atomic, resumable executable simulation over explicit evaluation-plan cells."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.evaluations import EvaluationCell, EvaluationPlan
from wmo.common.models import AssistantAction, NumericMeasurement, OperationEconomics
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ArtifactStore,
    artifact_input,
)
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SandboxSimulationCellBinding,
    SandboxSimulatorSnapshot,
    SimulationArtifactSet,
    SimulationMode,
    StopReason,
)
from wmo.common.tasks import LoadedTaskSet, load_task_set
from wmo.runtime.agents import AgentEpisode, AgentRuntime
from wmo.runtime.environments import EnvironmentRuntime
from wmo.simulation.engines.sandbox_bindings import (
    SANDBOX_SIMULATOR_ID,
    CandidateBinding,
    EnvironmentCostBinding,
    SandboxSimulationResolution,
    make_sandbox_cell_binding,
    make_sandbox_resolution,
    sandbox_binding_digest,
    sandbox_lease_id,
    sandbox_rollout_id,
)
from wmo.simulation.engines.sandbox_recording import (
    SandboxExecutionEvidence,
    SandboxTimeLimitError,
    execute_bounded_sandbox_episode,
    merge_sandbox_spans,
    require_hard_wall_timeout_support,
)
from wmo.simulation.engines.text.leases import (
    TextCellLeaseError,
    TextCellLeaseState,
    TextCellLeaseStore,
)
from wmo.simulation.orchestration import require_implemented_mode
from wmo.simulation.specs import SimulationSpec

_SPEC_FILE = "simulation-spec.json"
_RESOLUTION_FILE = "sandbox-simulation-resolution.json"
_ROLLOUT_FILE = "rollout.json"
_ARTIFACT_SET_FILE = "artifact-set.json"
_ARTIFACT_IDS_FILE = "artifact-ids.jsonl"


class SandboxSimulationError(ValueError):
    """A sandbox recipe or exact immutable input binding is inconsistent."""


class SandboxResumeError(RuntimeError):
    """Existing immutable sandbox evidence cannot safely be resumed or reused."""


class SandboxContentionError(SandboxResumeError):
    """Another live process owns an executable cell that may be retried later."""


class SandboxSimulator:
    """Run exact sandbox cells and persist each rollout before admitting another episode.

    Args:
        store: Immutable project-local artifact store.
        evaluation_plan: Frozen sparse plan whose simulated cells may execute.
        evaluation_plan_input: Exact persisted plan manifest identity.
        task_set_input: Exact persisted task-set manifest identity.
        candidates: Resolved candidate clients keyed by plan alias.
        agent_factory: Creates a fresh customer agent runtime for every episode.
        environment_runtime: Simulator-owned executable environment factory.
        environment_cost: Proven environment cost ceiling and observability capability.
        environment_id: Stable environment identity expected in sandbox settings.
        environment_sha256: Exact environment plan or image digest expected in settings.
        source_run_id: Durable run identity retained by every rollout.
        clock: Aware artifact and event time source.
        monotonic: Monotonic latency and deadline time source.
    """

    def __init__(
        self,
        *,
        store: ArtifactStore,
        evaluation_plan: EvaluationPlan,
        evaluation_plan_input: ArtifactInput,
        task_set_input: ArtifactInput,
        candidates: Mapping[str, CandidateBinding],
        agent_factory: Callable[[], AgentRuntime],
        environment_runtime: EnvironmentRuntime,
        environment_cost: EnvironmentCostBinding | None = None,
        environment_id: str,
        environment_sha256: str,
        source_run_id: str,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind immutable inputs and verify stored task and evaluation evidence immediately."""
        if evaluation_plan_input.artifact_id != evaluation_plan.plan_id:
            raise SandboxSimulationError(
                "evaluation_plan_input must name the supplied sandbox evaluation plan"
            )
        if not environment_id or not source_run_id:
            raise ValueError("sandbox environment_id and source_run_id must be nonempty")
        if len(environment_sha256) != 64:
            raise ValueError("sandbox environment_sha256 must be a SHA-256 digest")
        self._store = store
        self._plan = evaluation_plan
        self._plan_input = evaluation_plan_input
        self._task_set_input = task_set_input
        loaded = self._load_and_verify_task_set(task_set_input)
        self._task_set = loaded.task_set
        self._tasks = {task.task_id: task for task in loaded.tasks}
        self._candidates = dict(candidates)
        self._agent_factory = agent_factory
        self._environment_runtime = environment_runtime
        self._environment_cost = environment_cost or EnvironmentCostBinding()
        self._environment_id = environment_id
        self._environment_sha256 = environment_sha256
        self._source_run_id = source_run_id
        self._clock = clock or _utc_now
        self._monotonic = monotonic
        self._leases = TextCellLeaseStore(store.project_directory, clock=self._clock)
        self._verify_persisted_evaluation_plan()

    def run(self, spec: SimulationSpec) -> SimulationArtifactSet:
        """Run or resume every selected cell with durable per-cell exact-once evidence.

        Args:
            spec: Validated sandbox recipe over sorted explicit plan cell IDs.

        Returns:
            Immutable final index of one independently stored rollout artifact per cell.

        Raises:
            SandboxSimulationError: Inputs disagree or finite cost cannot be enforced safely.
            SandboxResumeError: Existing immutable evidence has conflicting provenance.
            SandboxContentionError: Another live process owns the same in-flight cell.
        """
        require_implemented_mode(spec, SimulationMode.SANDBOX)
        cells = self._validate_spec_and_bindings(spec)
        try:
            require_hard_wall_timeout_support()
        except SandboxTimeLimitError as exc:
            raise SandboxSimulationError(
                "sandbox simulation cannot start without enforceable hard wall-time limits"
            ) from exc
        spec_input = self._persist_specification(spec)
        resolution, resolution_input, bindings = self._persist_resolution(
            spec,
            spec_input,
            cells,
        )
        completed = self._load_completed_rollouts(cells, bindings, resolution_input)
        for cell in cells:
            if cell.cell_id in completed:
                continue
            completed[cell.cell_id] = self._execute_and_persist_cell(
                spec,
                cell,
                resolution,
                resolution_input,
                bindings[cell.cell_id],
            )
        ordered = tuple(completed[cell.cell_id] for cell in cells)
        return self._persist_artifact_set(spec, spec_input, resolution_input, ordered)

    def _load_and_verify_task_set(self, task_set_input: ArtifactInput) -> LoadedTaskSet:
        """Load the exact task set and reject an ID paired with another manifest digest."""
        try:
            stored = self._store.read(task_set_input.artifact_id)
        except ArtifactCorruptionError as exc:
            raise SandboxSimulationError(
                f"sandbox task set {task_set_input.artifact_id!r} cannot be read safely"
            ) from exc
        if (
            stored.manifest.artifact_type != "task-set"
            or artifact_input(stored.manifest) != task_set_input
        ):
            raise SandboxSimulationError(
                "task_set_input must name the exact persisted task-set manifest"
            )
        try:
            return load_task_set(self._store, task_set_input.artifact_id)
        except ArtifactCorruptionError as exc:
            raise SandboxSimulationError("sandbox task-set content is invalid") from exc

    def _verify_persisted_evaluation_plan(self) -> None:
        """Reject an in-memory plan that differs from its persisted immutable bytes."""
        try:
            stored = self._store.read(self._plan_input.artifact_id)
            persisted = EvaluationPlan.model_validate_json(
                self._store.read_bytes(self._plan_input.artifact_id, "evaluation-plan.json")
            )
        except (ArtifactCorruptionError, ValueError) as exc:
            raise SandboxSimulationError("sandbox evaluation plan cannot be read safely") from exc
        if (
            stored.manifest.artifact_type != "evaluation-plan"
            or artifact_input(stored.manifest) != self._plan_input
            or persisted != self._plan
        ):
            raise SandboxSimulationError(
                "supplied evaluation plan differs from its immutable persisted manifest"
            )
        if self._plan.task_set_id != self._task_set.task_set_id:
            raise SandboxSimulationError(
                "sandbox evaluation plan task_set_id does not match the immutable task set"
            )

    def _validate_spec_and_bindings(
        self,
        spec: SimulationSpec,
    ) -> tuple[EvaluationCell, ...]:
        """Validate every local binding and finite-cost capability before opening an environment."""
        if spec.evaluation_plan_id != self._plan.plan_id:
            raise SandboxSimulationError(
                "sandbox spec evaluation_plan_id does not match the supplied plan"
            )
        if self._plan_input not in spec.inputs or self._task_set_input not in spec.inputs:
            raise SandboxSimulationError(
                "sandbox spec inputs must include exact evaluation-plan and task-set manifests"
            )
        settings = spec.sandbox
        if settings is None:  # pragma: no cover - selected settings are validated by the spec
            raise SandboxSimulationError("sandbox settings are missing")
        if (
            settings.environment_id != self._environment_id
            or settings.environment_sha256 != self._environment_sha256
        ):
            raise SandboxSimulationError(
                "sandbox environment identity does not match the configured runtime"
            )
        plan_cells = {cell.cell_id: cell for cell in self._plan.cells}
        expected_candidates = {
            snapshot.alias: snapshot.model for snapshot in self._plan.candidate_snapshots
        }
        cells: list[EvaluationCell] = []
        for cell_id in spec.cell_ids:
            cell = plan_cells.get(cell_id)
            if cell is None:
                raise SandboxSimulationError(
                    f"sandbox spec selected cell {cell_id!r} outside its evaluation plan"
                )
            if cell.execution != "simulate":
                raise SandboxSimulationError(
                    f"sandbox spec selected observed cell {cell.cell_id!r}; only simulate cells run"
                )
            if cell.task_id not in self._tasks:
                raise SandboxSimulationError(
                    f"sandbox task {cell.task_id!r} is unavailable from the pinned task set"
                )
            candidate = self._candidates.get(cell.candidate_alias)
            if candidate is None or candidate.alias != cell.candidate_alias:
                raise SandboxSimulationError(
                    f"sandbox has no exact candidate binding for {cell.candidate_alias!r}"
                )
            if candidate.snapshot != expected_candidates[cell.candidate_alias]:
                raise SandboxSimulationError(
                    f"sandbox candidate {cell.candidate_alias!r} differs from the plan snapshot"
                )
            if spec.maximum_cost_usd is not None and (
                not candidate.cost_is_observable or candidate.maximum_call_cost_usd is None
            ):
                raise SandboxSimulationError(
                    "finite sandbox budgets require observable candidate cost and a "
                    "maximum call cost"
                )
            if spec.maximum_cost_usd is not None and (
                not self._environment_cost.cost_is_observable
                or self._environment_cost.maximum_episode_cost_usd is None
            ):
                raise SandboxSimulationError(
                    "finite sandbox budgets require observable environment cost and a "
                    "maximum episode cost"
                )
            cells.append(cell)
        return tuple(cells)

    def _persist_specification(self, spec: SimulationSpec) -> ArtifactInput:
        """Write or verify the immutable specification before any executable work."""
        try:
            manifest = self._store.write_json(
                artifact_id=spec.simulation_id,
                artifact_type="simulation-spec",
                envelope=spec,
                files={_SPEC_FILE: spec},
            )
            return artifact_input(manifest)
        except ArtifactAlreadyExistsError as exc:
            stored = self._store.read(spec.simulation_id)
            try:
                persisted = SimulationSpec.model_validate_json(
                    self._store.read_bytes(spec.simulation_id, _SPEC_FILE)
                )
            except (ArtifactCorruptionError, ValueError) as error:
                raise SandboxResumeError("sandbox simulation specification is invalid") from error
            if stored.manifest.artifact_type != "simulation-spec" or persisted != spec:
                raise SandboxResumeError(
                    f"simulation ID {spec.simulation_id!r} already names another specification"
                ) from exc
            return artifact_input(stored.manifest)

    def _persist_resolution(
        self,
        spec: SimulationSpec,
        spec_input: ArtifactInput,
        cells: Sequence[EvaluationCell],
    ) -> tuple[
        SandboxSimulationResolution,
        ArtifactInput,
        dict[ArtifactId, SandboxSimulationCellBinding],
    ]:
        """Persist exact task, candidate, environment, and manifest bindings before execution."""
        bindings = {
            cell.cell_id: make_sandbox_cell_binding(
                spec=spec,
                simulation_spec_input=spec_input,
                evaluation_plan_input=self._plan_input,
                task_set_input=self._task_set_input,
                task_set=self._task_set,
                cell=cell,
                task=self._tasks[cell.task_id],
                candidate=self._candidates[cell.candidate_alias],
                environment_cost=self._environment_cost,
            )
            for cell in cells
        }
        resolution = make_sandbox_resolution(
            spec=spec,
            simulation_spec_input=spec_input,
            evaluation_plan_input=self._plan_input,
            task_set_input=self._task_set_input,
            bindings=tuple(bindings[cell.cell_id] for cell in cells),
            created_at=_timestamp(self._clock),
        )
        try:
            manifest = self._store.write_json(
                artifact_id=resolution.resolution_id,
                artifact_type="sandbox-simulation-resolution",
                envelope=resolution,
                files={_RESOLUTION_FILE: resolution},
            )
            return resolution, artifact_input(manifest), bindings
        except ArtifactAlreadyExistsError as exc:
            stored = self._store.read(resolution.resolution_id)
            try:
                existing = SandboxSimulationResolution.model_validate_json(
                    self._store.read_bytes(resolution.resolution_id, _RESOLUTION_FILE)
                )
            except (ArtifactCorruptionError, ValueError) as error:
                raise SandboxResumeError("sandbox simulation resolution is invalid") from error
            if (
                stored.manifest.artifact_type != "sandbox-simulation-resolution"
                or existing.model_copy(update={"created_at": resolution.created_at}) != resolution
            ):
                raise SandboxResumeError(
                    "sandbox aliases, task, environment, or inputs changed after resolution"
                ) from exc
            return existing, artifact_input(stored.manifest), bindings

    def _load_completed_rollouts(
        self,
        cells: Sequence[EvaluationCell],
        bindings: Mapping[ArtifactId, SandboxSimulationCellBinding],
        resolution_input: ArtifactInput,
    ) -> dict[ArtifactId, RolloutArtifact]:
        """Load independently completed cells whose exact immutable binding still matches."""
        completed: dict[ArtifactId, RolloutArtifact] = {}
        for cell in cells:
            binding = bindings[cell.cell_id]
            rollout = self._load_optional_rollout(sandbox_rollout_id(binding))
            if rollout is not None:
                self._validate_resume_rollout(rollout, cell, binding, resolution_input)
                completed[cell.cell_id] = rollout
        return completed

    def _execute_and_persist_cell(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        resolution: SandboxSimulationResolution,
        resolution_input: ArtifactInput,
        binding: SandboxSimulationCellBinding,
    ) -> RolloutArtifact:
        """Claim, execute, and atomically persist one rollout before releasing its lease."""
        rollout_id = sandbox_rollout_id(binding)
        existing = self._load_optional_rollout(rollout_id)
        if existing is not None:
            self._validate_resume_rollout(existing, cell, binding, resolution_input)
            return existing
        try:
            claim = self._leases.acquire(
                lease_id=sandbox_lease_id(resolution, binding),
                resolution_id=resolution.resolution_id,
                simulation_id=spec.simulation_id,
                rollout_id=rollout_id,
                binding_sha256=sandbox_binding_digest(binding),
                maximum_cost_usd=spec.maximum_cost_usd,
                rollout_completed=lambda item: self._load_optional_rollout(item) is not None,
                observed_spend_usd=lambda: self._known_resolution_spend(
                    resolution.cell_bindings,
                    resolution_input,
                ),
            )
        except TextCellLeaseError as exc:
            raise SandboxResumeError(f"sandbox cell {cell.cell_id!r} cannot be admitted") from exc
        if claim.retryable:
            raise SandboxContentionError("sandbox cell is contended; retry this run")
        if claim.state == TextCellLeaseState.COMPLETED:
            completed = self._load_optional_rollout(rollout_id)
            if completed is None:
                raise SandboxResumeError("completed sandbox claim has no readable rollout")
            self._validate_resume_rollout(completed, cell, binding, resolution_input)
            return completed
        if claim.state == TextCellLeaseState.STALE:
            rollout = self._admission_failure_rollout(
                spec,
                cell,
                binding,
                resolution_input,
                phase="paid_cell_stale_lease",
                message="a prior sandbox cell claim expired; WMO will not replay it",
                spent=None,
            )
            return self._persist_rollout(rollout, cell, binding, resolution_input)
        if claim.state == TextCellLeaseState.BUDGET_BLOCKED:
            rollout = self._admission_failure_rollout(
                spec,
                cell,
                binding,
                resolution_input,
                phase="paid_cell_admission",
                message="sandbox spend is unknown or exhausted the finite run ceiling",
                spent=claim.observed_spend_usd,
            )
            return self._persist_rollout(rollout, cell, binding, resolution_input)
        if claim.lease is None:
            raise SandboxResumeError("owned sandbox admission omitted its durable lease")
        lease = claim.lease
        remaining = (
            None
            if spec.maximum_cost_usd is None
            else max(0.0, spec.maximum_cost_usd - (claim.observed_spend_usd or 0.0))
        )
        environment_reservation = binding.environment_maximum_episode_cost_usd
        dispatch_may_have_started = False
        try:
            if (
                remaining is not None
                and environment_reservation is not None
                and environment_reservation > remaining
            ):
                rollout = self._admission_failure_rollout(
                    spec,
                    cell,
                    binding,
                    resolution_input,
                    phase="environment_cost_admission",
                    message="sandbox environment reservation exceeds the remaining run ceiling",
                    spent=claim.observed_spend_usd,
                )
            else:
                dispatch_may_have_started = True
                rollout = self._execute_cell(spec, cell, binding, resolution_input, remaining)
            persisted = self._persist_rollout(rollout, cell, binding, resolution_input)
        except BaseException:
            if dispatch_may_have_started:
                self._leases.retain_non_replay_tombstone(lease)
            else:
                self._leases.release(lease)
            raise
        else:
            self._leases.release(lease)
            return persisted

    def _known_resolution_spend(
        self,
        bindings: Sequence[SandboxSimulationCellBinding],
        resolution_input: ArtifactInput,
    ) -> float | None:
        """Return candidate spend, or unknown when any dispatched model call lacks cost."""
        total = 0.0
        cells = {cell.cell_id: cell for cell in self._plan.cells}
        for binding in bindings:
            rollout = self._load_optional_rollout(sandbox_rollout_id(binding))
            if rollout is None:
                continue
            cell = cells[binding.cell_id]
            self._validate_resume_rollout(rollout, cell, binding, resolution_input)
            spend = _rollout_spend(rollout)
            if spend is None:
                return None
            total += spend
        return total

    def _execute_cell(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        binding: SandboxSimulationCellBinding,
        resolution_input: ArtifactInput,
        remaining_cost_usd: float | None,
    ) -> RolloutArtifact:
        """Execute one bounded customer episode and convert every normal failure into evidence."""
        started_at = _timestamp(self._clock)
        started = self._monotonic()
        candidate = self._candidates[cell.candidate_alias]
        settings = spec.sandbox
        if settings is None:  # pragma: no cover - run validation requires sandbox settings
            raise SandboxSimulationError("sandbox settings are missing")
        try:
            episode, evidence = execute_bounded_sandbox_episode(
                agent_factory=self._agent_factory,
                task=self._tasks[cell.task_id],
                candidate=candidate.client,
                candidate_snapshot=candidate.snapshot,
                environment_runtime=self._environment_runtime,
                environment_maximum_episode_cost_usd=(binding.environment_maximum_episode_cost_usd),
                environment_cost_is_observable=binding.environment_cost_is_observable,
                maximum_steps=spec.maximum_steps,
                maximum_time_seconds=settings.maximum_time_seconds,
                remaining_cost_usd=remaining_cost_usd,
                maximum_call_cost_usd=candidate.maximum_call_cost_usd,
                cost_is_observable=candidate.cost_is_observable,
                clock=self._clock,
                monotonic=self._monotonic,
            )
        except SandboxTimeLimitError as exc:
            failure = StructuredFailure(
                code=FailureCode.TIMEOUT,
                message="sandbox episode could not enforce or exceeded maximum_time_seconds",
                exception_type=type(exc).__name__,
                attribution=FailureAttribution.AGENT,
                details={"phase": "maximum_time"},
            )
            episode = AgentEpisode(stop_reason=StopReason.FAILURE, failure=failure)
            evidence = SandboxExecutionEvidence(
                candidate_spans=(),
                environment_spans=(),
                candidate_economics=OperationEconomics(),
                sandbox_economics=OperationEconomics(),
                limit_stop_reason=StopReason.MAXIMUM_TIME,
                limit_failure=failure,
            )
        except Exception as exc:  # noqa: BLE001 - construction failures are cell evidence
            failure = StructuredFailure(
                code=FailureCode.INTERNAL,
                message=f"sandbox agent construction failed with {type(exc).__name__}",
                exception_type=type(exc).__name__,
                attribution=FailureAttribution.AGENT,
                details={"phase": "agent_construction"},
            )
            episode = AgentEpisode(stop_reason=StopReason.FAILURE, failure=failure)
            evidence = SandboxExecutionEvidence(
                candidate_spans=(),
                environment_spans=(),
                candidate_economics=OperationEconomics(),
                sandbox_economics=OperationEconomics(),
                limit_stop_reason=None,
                limit_failure=None,
            )
        stop_reason, failure = _terminal_state(episode, evidence)
        ended_at = _timestamp(self._clock, not_before=started_at)
        spans = merge_sandbox_spans(episode, evidence)
        if not spans:
            spans = (_terminal_span(started_at, ended_at, failure),)
        return self._make_rollout(
            spec,
            cell,
            binding,
            resolution_input,
            stop_reason=stop_reason,
            failure=failure,
            final_output=episode.final_action,
            spans=spans,
            candidate_economics=evidence.candidate_economics,
            sandbox_economics=evidence.sandbox_economics,
            orchestration_economics=_latency_economics(max(0.0, self._monotonic() - started)),
        )

    def _admission_failure_rollout(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        binding: SandboxSimulationCellBinding,
        resolution_input: ArtifactInput,
        *,
        phase: str,
        message: str,
        spent: float | None,
    ) -> RolloutArtifact:
        """Represent a cell rejected before environment or candidate dispatch."""
        failure = StructuredFailure(
            code=FailureCode.BUDGET,
            message=message,
            attribution=FailureAttribution.MODEL,
            details={"phase": phase, "observed_spend_usd": spent},
        )
        now = _timestamp(self._clock)
        return self._make_rollout(
            spec,
            cell,
            binding,
            resolution_input,
            stop_reason=StopReason.MAXIMUM_COST,
            failure=failure,
            final_output=None,
            spans=(_terminal_span(now, now, failure),),
            candidate_economics=OperationEconomics(),
            sandbox_economics=OperationEconomics(),
            orchestration_economics=_latency_economics(0.0),
        )

    def _make_rollout(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        binding: SandboxSimulationCellBinding,
        resolution_input: ArtifactInput,
        *,
        stop_reason: StopReason,
        failure: StructuredFailure | None,
        final_output: AssistantAction | None,
        spans: tuple[RolloutSpan, ...],
        candidate_economics: OperationEconomics,
        sandbox_economics: OperationEconomics,
        orchestration_economics: OperationEconomics,
    ) -> RolloutArtifact:
        """Compose one canonical sandbox rollout from exact binding and runtime evidence."""
        rollout_id = sandbox_rollout_id(binding)
        return RolloutArtifact(
            schema_version=1,
            created_at=_timestamp(self._clock),
            inputs=_sorted_inputs(
                self._plan_input,
                self._task_set_input,
                binding.simulation_spec_input,
                resolution_input,
            ),
            code_revision=spec.code_revision,
            artifact_id=rollout_id,
            simulation_id=spec.simulation_id,
            cell_id=cell.cell_id,
            mode=SimulationMode.SANDBOX,
            rollout_id=rollout_id,
            trace_id=hashlib.sha256(sandbox_binding_digest(binding).encode()).hexdigest()[:32],
            evidence_source="sandbox",
            source_run_id=self._source_run_id,
            task_id=cell.task_id,
            candidate=binding.candidate,
            agent_id=spec.agent_id,
            simulator=SandboxSimulatorSnapshot(
                simulator_id=SANDBOX_SIMULATOR_ID,
                environment_id=binding.environment_id,
                environment_sha256=binding.environment_sha256,
            ),
            seed=spec.seed,
            repeat=cell.repeat,
            spans=spans,
            final_output=final_output,
            stop_reason=stop_reason,
            failure=failure,
            candidate_economics=candidate_economics,
            sandbox_economics=sandbox_economics,
            orchestration_economics=orchestration_economics,
            simulation_spec_sha256=binding.simulation_spec_sha256,
            sandbox_binding=binding,
        )

    def _persist_rollout(
        self,
        rollout: RolloutArtifact,
        cell: EvaluationCell,
        binding: SandboxSimulationCellBinding,
        resolution_input: ArtifactInput,
    ) -> RolloutArtifact:
        """Persist one cell atomically and accept only an exactly bound completed counterpart."""
        try:
            self._store.write_json(
                artifact_id=rollout.artifact_id,
                artifact_type="rollout",
                envelope=rollout,
                files={_ROLLOUT_FILE: rollout},
            )
            return rollout
        except ArtifactAlreadyExistsError:
            existing = self._load_rollout(rollout.artifact_id)
            self._validate_resume_rollout(existing, cell, binding, resolution_input)
            return existing

    def _load_rollout(self, rollout_id: ArtifactId) -> RolloutArtifact:
        """Load one verified canonical rollout artifact."""
        stored = self._store.read(rollout_id)
        if stored.manifest.artifact_type != "rollout":
            raise ArtifactCorruptionError(f"artifact {rollout_id!r} is not a rollout")
        try:
            return RolloutArtifact.model_validate_json(
                self._store.read_bytes(rollout_id, _ROLLOUT_FILE)
            )
        except (ArtifactCorruptionError, ValueError) as exc:
            raise ArtifactCorruptionError(f"sandbox rollout {rollout_id!r} is invalid") from exc

    def _load_optional_rollout(self, rollout_id: ArtifactId) -> RolloutArtifact | None:
        """Load a completed rollout while distinguishing absence from immutable corruption."""
        if rollout_id not in self._store.list_ids():
            return None
        return self._load_rollout(rollout_id)

    def _validate_resume_rollout(
        self,
        rollout: RolloutArtifact,
        cell: EvaluationCell,
        binding: SandboxSimulationCellBinding,
        resolution_input: ArtifactInput,
    ) -> None:
        """Require every resumed cell to match its exact task, model, environment, and inputs."""
        expected_inputs = _sorted_inputs(
            self._plan_input,
            self._task_set_input,
            binding.simulation_spec_input,
            resolution_input,
        )
        if (
            binding.cell_id != cell.cell_id
            or binding.task_id != cell.task_id
            or binding.purpose != cell.purpose
            or rollout.cell_id != cell.cell_id
            or rollout.task_id != cell.task_id
            or rollout.mode != SimulationMode.SANDBOX
            or rollout.artifact_id != sandbox_rollout_id(binding)
            or rollout.rollout_id != sandbox_rollout_id(binding)
            or rollout.sandbox_binding != binding
            or rollout.inputs != expected_inputs
        ):
            raise SandboxResumeError(
                f"stored rollout {rollout.artifact_id!r} does not match its sandbox cell"
            )

    def _persist_artifact_set(
        self,
        spec: SimulationSpec,
        spec_input: ArtifactInput,
        resolution_input: ArtifactInput,
        rollouts: Sequence[RolloutArtifact],
    ) -> SimulationArtifactSet:
        """Write or verify the terminal index after every per-cell artifact is durable."""
        artifact_ids = tuple(rollout.artifact_id for rollout in rollouts)
        payload = _jsonl_bytes(tuple({"artifact_id": item} for item in artifact_ids))
        artifact_set_id = stable_id(
            "simulation-artifact-set",
            {"simulation_id": spec.simulation_id, "artifact_ids": artifact_ids},
        )
        artifact_set = SimulationArtifactSet(
            schema_version=1,
            created_at=_timestamp(self._clock),
            inputs=_sorted_inputs(
                self._plan_input,
                self._task_set_input,
                spec_input,
                resolution_input,
            ),
            code_revision=spec.code_revision,
            artifact_set_id=artifact_set_id,
            simulation_id=spec.simulation_id,
            artifact_ids=artifact_ids,
            artifacts_path=_ARTIFACT_IDS_FILE,
            artifacts_sha256=hashlib.sha256(payload).hexdigest(),
        )
        try:
            self._store.write(
                artifact_id=artifact_set_id,
                artifact_type="simulation-artifact-set",
                envelope=artifact_set,
                files={
                    _ARTIFACT_SET_FILE: canonical_json_bytes(artifact_set),
                    _ARTIFACT_IDS_FILE: payload,
                },
            )
            return artifact_set
        except ArtifactAlreadyExistsError as exc:
            stored = self._store.read(artifact_set_id)
            try:
                existing = SimulationArtifactSet.model_validate_json(
                    self._store.read_bytes(artifact_set_id, _ARTIFACT_SET_FILE)
                )
            except (ArtifactCorruptionError, ValueError) as error:
                raise SandboxResumeError("sandbox artifact set is invalid") from error
            same_content = (
                existing.model_copy(update={"created_at": artifact_set.created_at}) == artifact_set
            )
            if stored.manifest.artifact_type != "simulation-artifact-set" or not same_content:
                raise SandboxResumeError(
                    f"artifact set ID {artifact_set_id!r} names different rollout evidence"
                ) from exc
            return existing


def _terminal_state(
    episode: AgentEpisode,
    evidence: SandboxExecutionEvidence,
) -> tuple[StopReason, StructuredFailure | None]:
    """Prefer an enforced limit while retaining ordinary agent and cleanup terminal evidence."""
    if episode.failure is not None and episode.failure.attribution == FailureAttribution.CLEANUP:
        return StopReason.FAILURE, episode.failure
    if evidence.limit_stop_reason is not None:
        return evidence.limit_stop_reason, evidence.limit_failure or episode.failure
    failure = episode.failure
    if failure is not None:
        names = {
            "SandboxStepLimitError": StopReason.MAXIMUM_STEPS,
            "SandboxTimeLimitError": StopReason.MAXIMUM_TIME,
            "SandboxCostLimitError": StopReason.MAXIMUM_COST,
            "SandboxUnknownCostError": StopReason.MAXIMUM_COST,
        }
        mapped = names.get(failure.exception_type or "")
        if mapped is not None:
            return mapped, failure
    return episode.stop_reason, failure


def _rollout_spend(rollout: RolloutArtifact) -> float | None:
    """Return candidate plus sandbox spend without treating unknown cost as zero."""
    if rollout.failure is not None and (
        rollout.failure.details.get("provider_dispatch_unknown_spend") is True
        or rollout.failure.details.get("environment_dispatch_unknown_spend") is True
        or rollout.failure.details.get("phase") == "paid_cell_stale_lease"
    ):
        return None
    total = 0.0
    made_candidate_call = any(
        span.kind == RolloutEventKind.AGENT_MODEL_CALL for span in rollout.spans
    )
    if made_candidate_call:
        candidate_cost = rollout.candidate_economics.cost_usd
        if candidate_cost is None:
            return None
        total += candidate_cost.value
    binding = rollout.sandbox_binding
    if binding is None:
        return None
    if binding.environment_maximum_episode_cost_usd == 0:
        return total
    sandbox_cost = rollout.sandbox_economics.cost_usd if rollout.sandbox_economics else None
    if sandbox_cost is None:
        return None
    return total + sandbox_cost.value


def _terminal_span(
    started_at: datetime,
    ended_at: datetime,
    failure: StructuredFailure | None,
) -> RolloutSpan:
    """Create minimum lifecycle evidence for an episode with no operation-level events."""
    return RolloutSpan(
        span_id="sandbox-terminal",
        kind=RolloutEventKind.LIFECYCLE,
        started_at=started_at,
        ended_at=ended_at,
        payload={"phase": "terminal"},
        failure=failure,
    )


def _latency_economics(duration_seconds: float) -> OperationEconomics:
    """Record directly observed orchestration time without manufacturing cost."""
    return OperationEconomics(
        latency_seconds=NumericMeasurement(value=duration_seconds, provenance="observed")
    )


def _sorted_inputs(*inputs: ArtifactInput) -> tuple[ArtifactInput, ...]:
    """Return one exact immutable input per artifact ID in stable order."""
    by_id = {item.artifact_id: item for item in inputs}
    if len(by_id) != len(inputs):
        raise SandboxSimulationError("sandbox artifact inputs must have distinct IDs")
    return tuple(by_id[item] for item in sorted(by_id))


def _jsonl_bytes(records: Sequence[Mapping[str, str]]) -> bytes:
    """Render deterministic small JSONL artifact-index records."""
    payload = b"\n".join(canonical_json_bytes(dict(record)) for record in records)
    return payload + (b"\n" if payload else b"")


def _timestamp(
    clock: Callable[[], datetime],
    *,
    not_before: datetime | None = None,
) -> datetime:
    """Return an aware timestamp that cannot precede an earlier event."""
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("sandbox simulation clock must return timezone-aware datetimes")
    return not_before if not_before is not None and value < not_before else value


def _utc_now() -> datetime:
    """Return the current aware UTC timestamp."""
    return datetime.now(UTC)


__all__ = [
    "CandidateBinding",
    "EnvironmentCostBinding",
    "SandboxContentionError",
    "SandboxResumeError",
    "SandboxSimulationError",
    "SandboxSimulator",
]
