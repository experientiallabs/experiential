"""Atomic, resumable text world-model simulation over explicit evaluation-plan cells."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import cast

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
    canonical_json_bytes,
    sha256_json,
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
    SimulationArtifactSet,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from wmo.common.tasks import TaskCase
from wmo.runtime.agents import AgentEpisode, AgentRuntime, execute_agent_episode
from wmo.runtime.models import ResolvedModel
from wmo.simulation.engines.text.environment import (
    TextOnlyEnvironmentRuntime,
    TextOnlyToolUseError,
)
from wmo.simulation.engines.text.prompt import (
    WORLD_MODEL_TEXT_PROMPT_ID,
    WORLD_MODEL_TEXT_PROMPT_VERSION,
)
from wmo.simulation.engines.text.recording import (
    RecordingCandidateClient,
    TokenCounter,
    Utf8UpperBoundTokenCounter,
)
from wmo.simulation.engines.text.redaction import (
    redact_action,
    redact_failure,
    redact_span,
    redacted_field_set,
)
from wmo.simulation.orchestration import require_implemented_mode
from wmo.simulation.specs import SimulationSpec, simulation_spec_digest

_SIMULATOR_ID = "text-world-model-v1"
_SPEC_FILE = "simulation-spec.json"
_ROLLOUT_FILE = "rollout.json"
_ARTIFACT_SET_FILE = "artifact-set.json"
_ARTIFACT_IDS_FILE = "artifact-ids.jsonl"


class SimulationConfigurationError(ValueError):
    """A sparse simulation recipe cannot be executed against supplied local bindings."""


class SimulationResumeError(RuntimeError):
    """An immutable simulation artifact cannot safely be resumed or reused."""


class WorldModelSimulator:
    """Execute a text-only customer agent against a remote world-model provider.

    The simulator deliberately owns only one concrete mode. It invokes an independently resolved
    candidate client and world-model client, gives the agent an execute-only no-tools environment,
    and persists one immutable rollout per selected evaluation cell. It never exposes a mutable
    world-model session, sends no tools to the world model, and records candidate economics apart
    from simulator operating cost.

    Args:
        store: Immutable local artifact store receiving specifications and rollout artifacts.
        evaluation_plan: Frozen plan whose explicit simulated cells may be selected.
        evaluation_plan_input: Verified persisted-plan manifest reference.
        tasks: Canonical tasks keyed by task ID.
        candidate_models: Independently resolved candidate models keyed by plan alias.
        world_models: Independently resolved world-model providers keyed by local alias.
        agent_factory: Creates an isolated customer-agent runtime for each episode worker.
        redacted_field_names: Project-configured labels removed before evidence persists.
        clock: Time source for artifact and span timestamps.
        monotonic: Monotonic time source for orchestration latency measurements.
        token_counter: Optional full-request preflight counter. The default never truncates input.
    """

    def __init__(
        self,
        *,
        store: ArtifactStore,
        evaluation_plan: EvaluationPlan,
        evaluation_plan_input: ArtifactInput,
        tasks: Mapping[str, TaskCase],
        candidate_models: Mapping[str, ResolvedModel],
        world_models: Mapping[str, ResolvedModel],
        agent_factory: Callable[[], AgentRuntime],
        redacted_field_names: tuple[str, ...] = (),
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Bind one immutable plan and all explicit runtime dependencies.

        Args:
            store: Immutable store scoped to one `.wmo` project.
            evaluation_plan: Exact plan selected by later simulation specifications.
            evaluation_plan_input: Manifest identity for ``evaluation_plan``.
            tasks: Canonical task records available to selected plan cells.
            candidate_models: Candidate aliases resolved through the canonical runtime package.
            world_models: Remote world-model aliases resolved through the canonical runtime package.
            agent_factory: Per-episode customer-agent construction seam.
            redacted_field_names: Local field labels removed before artifact persistence.
            clock: Time source, injectable for deterministic tests.
            monotonic: Duration source, injectable for deterministic tests.
            token_counter: Full-request counter used before each provider call.

        Raises:
            SimulationConfigurationError: The supplied plan input does not name this plan.
        """
        if evaluation_plan_input.artifact_id != evaluation_plan.plan_id:
            raise SimulationConfigurationError(
                "evaluation_plan_input must name the evaluation plan supplied to the simulator"
            )
        self._store = store
        self._plan = evaluation_plan
        self._plan_input = evaluation_plan_input
        self._tasks = dict(tasks)
        self._candidate_models = dict(candidate_models)
        self._world_models = dict(world_models)
        self._agent_factory = agent_factory
        self._redacted_field_names = redacted_field_set(redacted_field_names)
        self._clock = clock or _utc_now
        self._monotonic = monotonic
        self._token_counter = token_counter or Utf8UpperBoundTokenCounter()

    def run(self, spec: SimulationSpec) -> SimulationArtifactSet:
        """Run or resume exactly the sparse simulated cells selected by ``spec``.

        A finite spend ceiling serializes new episode admission so later cells can be marked as
        structured budget failures after observed provider spend reaches the ceiling. Provider
        pricing is observed after a call in the v1 model contract, so the first episode that
        crosses a ceiling is retained honestly rather than fabricated as a zero-cost estimate.

        Args:
            spec: Frozen world-model recipe selecting exact evaluation-plan cells.

        Returns:
            Immutable index of one rollout artifact for every selected cell.

        Raises:
            SimulationConfigurationError: The spec, plan, models, or tasks disagree before calls.
            SimulationResumeError: Existing immutable artifacts do not match this simulation.
        """
        require_implemented_mode(spec, SimulationMode.WORLD_MODEL)
        cells, world_model = self._validate_spec_and_bindings(spec)
        spec_input = self._persist_specification(spec)
        spec_digest = simulation_spec_digest(spec)
        completed = self._load_completed_rollouts(spec, cells, spec_digest)
        pending = tuple(cell for cell in cells if cell.cell_id not in completed)

        if pending:
            if spec.maximum_cost_usd is None:
                completed.update(
                    self._run_without_spend_limit(
                        spec,
                        pending,
                        world_model,
                        spec_input,
                        spec_digest,
                    )
                )
            else:
                completed.update(
                    self._run_with_spend_limit(
                        spec,
                        pending,
                        world_model,
                        spec_input,
                        spec_digest,
                        tuple(completed.values()),
                    )
                )
        ordered_rollouts = tuple(completed[cell.cell_id] for cell in cells)
        return self._persist_artifact_set(spec, spec_input, ordered_rollouts)

    def _validate_spec_and_bindings(
        self,
        spec: SimulationSpec,
    ) -> tuple[tuple[EvaluationCell, ...], ResolvedModel]:
        """Validate all local inputs before any artifact write or provider call."""
        if spec.evaluation_plan_id != self._plan.plan_id:
            raise SimulationConfigurationError(
                "simulation spec evaluation_plan_id does not match the supplied evaluation plan"
            )
        if self._plan_input not in spec.inputs:
            raise SimulationConfigurationError(
                "simulation spec inputs must include the exact evaluation-plan manifest reference"
            )
        settings = spec.world_model
        if settings is None:  # pragma: no cover - selected settings are validated by SimulationSpec
            raise SimulationConfigurationError("world-model simulation settings are missing")
        if settings.prompt_version != WORLD_MODEL_TEXT_PROMPT_VERSION:
            raise SimulationConfigurationError(
                "text-world-model simulator supports only "
                f"prompt version {WORLD_MODEL_TEXT_PROMPT_VERSION!r}"
            )
        world_model = self._world_models.get(settings.world_model_alias)
        if world_model is None:
            raise SimulationConfigurationError(
                f"no resolved world model is configured for alias {settings.world_model_alias!r}"
            )
        plan_cells = {cell.cell_id: cell for cell in self._plan.cells}
        cells = []
        expected_candidates = {
            snapshot.alias: snapshot.model for snapshot in self._plan.candidate_snapshots
        }
        for cell_id in spec.cell_ids:
            cell = plan_cells.get(cell_id)
            if cell is None:
                raise SimulationConfigurationError(
                    f"simulation spec selected cell {cell_id!r} outside the evaluation plan"
                )
            if cell.execution != "simulate":
                raise SimulationConfigurationError(
                    f"simulation spec selected observed cell {cell.cell_id!r}; "
                    "only simulate cells run"
                )
            task = self._tasks.get(cell.task_id)
            if task is None or task.task_id != cell.task_id:
                raise SimulationConfigurationError(
                    f"simulation task {cell.task_id!r} is unavailable or has a mismatched identity"
                )
            candidate = self._candidate_models.get(cell.candidate_alias)
            if candidate is None:
                raise SimulationConfigurationError(
                    f"no resolved candidate is configured for alias {cell.candidate_alias!r}"
                )
            expected = expected_candidates[cell.candidate_alias]
            if candidate.snapshot != expected:
                raise SimulationConfigurationError(
                    f"candidate {cell.candidate_alias!r} does not match the plan's pinned snapshot"
                )
            if candidate.client is world_model.client:
                raise SimulationConfigurationError(
                    "candidate and world model must be resolved to independent model-client objects"
                )
            cells.append(cell)
        return tuple(cells), world_model

    def _persist_specification(self, spec: SimulationSpec) -> ArtifactInput:
        """Atomically write or verify the immutable specification before rollout execution."""
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
            if stored.manifest.artifact_type != "simulation-spec":
                raise SimulationResumeError(
                    f"artifact {spec.simulation_id!r} exists but is not a simulation specification"
                ) from exc
            try:
                persisted = SimulationSpec.model_validate_json(
                    self._store.read_bytes(spec.simulation_id, _SPEC_FILE)
                )
            except (ArtifactCorruptionError, ValueError) as exc:
                raise SimulationResumeError(
                    f"simulation specification {spec.simulation_id!r} cannot be read safely"
                ) from exc
            if persisted != spec:
                raise SimulationResumeError(
                    f"simulation ID {spec.simulation_id!r} already names a different immutable spec"
                ) from exc
            return artifact_input(stored.manifest)

    def _load_completed_rollouts(
        self,
        spec: SimulationSpec,
        cells: Sequence[EvaluationCell],
        spec_digest: str,
    ) -> dict[ArtifactId, RolloutArtifact]:
        """Load valid completed episodes, leaving missing cells available for atomic resume."""
        completed: dict[ArtifactId, RolloutArtifact] = {}
        for cell in cells:
            rollout_id = _rollout_id(spec_digest, cell.cell_id)
            rollout = self._load_optional_rollout(rollout_id)
            if rollout is None:
                continue
            self._validate_resume_rollout(rollout, spec, cell, spec_digest)
            completed[cell.cell_id] = rollout
        return completed

    def _run_without_spend_limit(
        self,
        spec: SimulationSpec,
        cells: Sequence[EvaluationCell],
        world_model: ResolvedModel,
        spec_input: ArtifactInput,
        spec_digest: str,
    ) -> dict[ArtifactId, RolloutArtifact]:
        """Run pending cells with the configured bounded worker concurrency."""
        if spec.maximum_concurrency == 1:
            return {
                cell.cell_id: self._execute_and_persist_cell(
                    spec,
                    cell,
                    world_model,
                    spec_input,
                    spec_digest,
                )
                for cell in cells
            }
        with ThreadPoolExecutor(
            max_workers=spec.maximum_concurrency,
            thread_name_prefix="wmo-text-simulation",
        ) as executor:
            futures = {
                cell.cell_id: executor.submit(
                    self._execute_and_persist_cell,
                    spec,
                    cell,
                    world_model,
                    spec_input,
                    spec_digest,
                )
                for cell in cells
            }
            return {cell.cell_id: futures[cell.cell_id].result() for cell in cells}

    def _run_with_spend_limit(
        self,
        spec: SimulationSpec,
        cells: Sequence[EvaluationCell],
        world_model: ResolvedModel,
        spec_input: ArtifactInput,
        spec_digest: str,
        completed: Sequence[RolloutArtifact],
    ) -> dict[ArtifactId, RolloutArtifact]:
        """Admit sequential episodes until observed total provider spend reaches the ceiling."""
        ceiling = spec.maximum_cost_usd
        if ceiling is None:  # pragma: no cover - caller branches on this exact value
            raise ValueError("a spend-limited simulation needs a finite maximum_cost_usd")
        spent = _known_total_spend(completed)
        results: dict[ArtifactId, RolloutArtifact] = {}
        for cell in cells:
            if spent is None or spent >= ceiling:
                rollout = self._budget_failure_rollout(
                    spec,
                    cell,
                    world_model,
                    spec_input,
                    spec_digest,
                    spent,
                )
                results[cell.cell_id] = self._persist_rollout(rollout)
                continue
            rollout = self._execute_and_persist_cell(
                spec,
                cell,
                world_model,
                spec_input,
                spec_digest,
            )
            results[cell.cell_id] = rollout
            episode_spend = _rollout_spend(rollout)
            spent = None if spent is None or episode_spend is None else spent + episode_spend
        return results

    def _execute_and_persist_cell(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        world_model: ResolvedModel,
        spec_input: ArtifactInput,
        spec_digest: str,
    ) -> RolloutArtifact:
        """Run one selected cell and atomically persist its terminal evidence."""
        rollout = self._execute_cell(spec, cell, world_model, spec_input, spec_digest)
        return self._persist_rollout(rollout)

    def _execute_cell(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        world_model: ResolvedModel,
        spec_input: ArtifactInput,
        spec_digest: str,
    ) -> RolloutArtifact:
        """Execute one no-tools episode, converting every boundary issue into a rollout artifact."""
        task = self._tasks[cell.task_id]
        candidate = self._candidate_models[cell.candidate_alias]
        started_at = _timestamp(self._clock)
        started_monotonic = self._monotonic()
        if task.tools:
            return self._failure_rollout(
                spec,
                cell,
                candidate,
                world_model,
                spec_input,
                spec_digest,
                started_at,
                StopReason.FAILURE,
                StructuredFailure(
                    code=FailureCode.UNSUPPORTED,
                    message="text world-model simulation cannot run a task that declares tools",
                    attribution=FailureAttribution.TOOL,
                    details={"phase": "task_tools", "tool_count": len(task.tools)},
                ),
                duration_seconds=_elapsed_seconds(started_monotonic, self._monotonic()),
            )
        settings = spec.world_model
        if settings is None:  # pragma: no cover - validated before this execution path
            raise SimulationConfigurationError("world-model simulation settings are missing")
        recorder = RecordingCandidateClient(
            task=task,
            candidate=candidate,
            world_model=world_model,
            maximum_steps=spec.maximum_steps,
            maximum_output_tokens=settings.maximum_output_tokens,
            redacted_field_names=self._redacted_field_names,
            clock=self._clock,
            token_counter=self._token_counter,
        )
        try:
            agent = self._agent_factory()
            episode = execute_agent_episode(
                agent,
                TextOnlyEnvironmentRuntime(),
                task,
                recorder,
            )
        except Exception as exc:  # noqa: BLE001 - a construction fault belongs in the selected cell
            return self._failure_rollout(
                spec,
                cell,
                candidate,
                world_model,
                spec_input,
                spec_digest,
                started_at,
                StopReason.FAILURE,
                _internal_failure("agent construction or lifecycle", exc),
                duration_seconds=_elapsed_seconds(started_monotonic, self._monotonic()),
                recorder=recorder,
            )
        ended_at = _timestamp(self._clock, not_before=started_at)
        duration_seconds = _elapsed_seconds(started_monotonic, self._monotonic())
        text_error = recorder.terminal_error
        if text_error is not None:
            stop_reason = text_error.stop_reason
            failure = None if stop_reason == StopReason.COMPLETED else text_error.failure
        else:
            stop_reason = episode.stop_reason
            failure = _normalize_text_tool_failure(episode)
        final_output = episode.final_action or recorder.last_candidate_action
        return self._make_rollout(
            spec,
            cell,
            candidate,
            world_model,
            spec_input,
            spec_digest,
            started_at,
            ended_at,
            stop_reason,
            failure,
            final_output,
            _combine_spans(
                episode.events,
                recorder.recorded.candidate_spans,
                recorder.recorded.world_model_spans,
                self._redacted_field_names,
            ),
            recorder.recorded.candidate_economics,
            recorder.recorded.world_model_economics,
            _orchestration_economics(duration_seconds),
        )

    def _failure_rollout(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        candidate: ResolvedModel,
        world_model: ResolvedModel,
        spec_input: ArtifactInput,
        spec_digest: str,
        started_at: datetime,
        stop_reason: StopReason,
        failure: StructuredFailure,
        *,
        duration_seconds: float,
        recorder: RecordingCandidateClient | None = None,
    ) -> RolloutArtifact:
        """Build an artifact-safe failed cell, retaining calls recorded before the failure."""
        ended_at = _timestamp(self._clock, not_before=started_at)
        recorded = recorder.recorded if recorder is not None else None
        spans = _combine_spans(
            (),
            recorded.candidate_spans if recorded is not None else (),
            recorded.world_model_spans if recorded is not None else (),
            self._redacted_field_names,
        )
        if not spans:
            spans = (_failure_span(started_at, ended_at, failure),)
        return self._make_rollout(
            spec,
            cell,
            candidate,
            world_model,
            spec_input,
            spec_digest,
            started_at,
            ended_at,
            stop_reason,
            failure,
            recorder.last_candidate_action if recorder is not None else None,
            spans,
            recorded.candidate_economics if recorded is not None else OperationEconomics(),
            recorded.world_model_economics if recorded is not None else OperationEconomics(),
            _orchestration_economics(duration_seconds),
        )

    def _budget_failure_rollout(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        world_model: ResolvedModel,
        spec_input: ArtifactInput,
        spec_digest: str,
        spent: float | None,
    ) -> RolloutArtifact:
        """Represent a not-admitted selected cell as immutable structured budget evidence."""
        timestamp = _timestamp(self._clock)
        candidate = self._candidate_models[cell.candidate_alias]
        failure = StructuredFailure(
            code=FailureCode.BUDGET,
            message=(
                "simulation spend is unknown or the spend ceiling was reached before this "
                "selected cell was admitted"
            ),
            attribution=FailureAttribution.MODEL,
            details={"phase": "episode_admission", "observed_spend_usd": spent},
        )
        return self._failure_rollout(
            spec,
            cell,
            candidate,
            world_model,
            spec_input,
            spec_digest,
            timestamp,
            StopReason.MAXIMUM_COST,
            failure,
            duration_seconds=0.0,
        )

    def _make_rollout(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        candidate: ResolvedModel,
        world_model: ResolvedModel,
        spec_input: ArtifactInput,
        spec_digest: str,
        started_at: datetime,
        ended_at: datetime,
        stop_reason: StopReason,
        failure: StructuredFailure | None,
        final_output: AssistantAction | None,
        spans: tuple[RolloutSpan, ...],
        candidate_economics: OperationEconomics,
        world_model_economics: OperationEconomics,
        orchestration_economics: OperationEconomics,
    ) -> RolloutArtifact:
        """Compose one canonical ``RolloutArtifact`` from completed or failed evidence."""
        del started_at, ended_at
        rollout_id = _rollout_id(spec_digest, cell.cell_id)
        return RolloutArtifact(
            schema_version=1,
            created_at=_timestamp(self._clock),
            inputs=_sorted_inputs(self._plan_input, spec_input),
            code_revision=spec.code_revision,
            artifact_id=rollout_id,
            simulation_id=spec.simulation_id,
            cell_id=cell.cell_id,
            mode=SimulationMode.WORLD_MODEL,
            rollout_id=rollout_id,
            trace_id=sha256_json({"simulation": spec.simulation_id, "cell": cell.cell_id}),
            evidence_source="world_model",
            source_run_id=spec.simulation_id,
            task_id=cell.task_id,
            candidate=candidate.snapshot,
            agent_id=spec.agent_id,
            simulator=WorldModelSimulatorSnapshot(
                simulator_id=_SIMULATOR_ID,
                prompt_id=WORLD_MODEL_TEXT_PROMPT_ID,
                world_model=world_model.snapshot,
            ),
            world_model=world_model.snapshot,
            seed=spec.seed,
            repeat=cell.repeat,
            spans=spans,
            final_output=redact_action(final_output, self._redacted_field_names),
            stop_reason=stop_reason,
            failure=redact_failure(failure, self._redacted_field_names),
            candidate_economics=candidate_economics,
            world_model_economics=world_model_economics,
            orchestration_economics=orchestration_economics,
            simulation_spec_sha256=spec_digest,
        )

    def _persist_rollout(self, rollout: RolloutArtifact) -> RolloutArtifact:
        """Atomically store one rollout or return its already completed resume counterpart."""
        try:
            self._store.write_json(
                artifact_id=rollout.artifact_id,
                artifact_type="rollout",
                envelope=rollout,
                files={_ROLLOUT_FILE: rollout},
            )
            return rollout
        except ArtifactAlreadyExistsError as exc:
            existing = self._load_rollout(rollout.artifact_id)
            if (
                existing.simulation_id != rollout.simulation_id
                or existing.cell_id != rollout.cell_id
                or existing.simulation_spec_sha256 != rollout.simulation_spec_sha256
            ):
                raise SimulationResumeError(
                    f"rollout ID {rollout.artifact_id!r} is already bound to incompatible evidence"
                ) from exc
            return existing

    def _load_rollout(self, rollout_id: ArtifactId) -> RolloutArtifact:
        """Load a verified rollout or surface malformed immutable data to the caller."""
        stored = self._store.read(rollout_id)
        if stored.manifest.artifact_type != "rollout":
            raise ArtifactCorruptionError(f"artifact {rollout_id!r} is not a rollout")
        try:
            return RolloutArtifact.model_validate_json(
                self._store.read_bytes(rollout_id, _ROLLOUT_FILE)
            )
        except (ArtifactCorruptionError, ValueError) as exc:
            raise ArtifactCorruptionError(
                f"rollout {rollout_id!r} is not valid canonical evidence"
            ) from exc

    def _load_optional_rollout(self, rollout_id: ArtifactId) -> RolloutArtifact | None:
        """Load an existing rollout while distinguishing absence from immutable corruption."""
        if rollout_id not in self._store.list_ids():
            return None
        return self._load_rollout(rollout_id)

    def _validate_resume_rollout(
        self,
        rollout: RolloutArtifact,
        spec: SimulationSpec,
        cell: EvaluationCell,
        spec_digest: str,
    ) -> None:
        """Ensure a stable rollout ID has not been reused for unrelated immutable data."""
        if (
            rollout.simulation_id != spec.simulation_id
            or rollout.cell_id != cell.cell_id
            or rollout.mode != SimulationMode.WORLD_MODEL
            or rollout.simulation_spec_sha256 != spec_digest
            or rollout.rollout_id != _rollout_id(spec_digest, cell.cell_id)
        ):
            raise SimulationResumeError(
                f"stored rollout {rollout.artifact_id!r} does not match the requested "
                "simulation cell"
            )

    def _persist_artifact_set(
        self,
        spec: SimulationSpec,
        spec_input: ArtifactInput,
        rollouts: Sequence[RolloutArtifact],
    ) -> SimulationArtifactSet:
        """Write or verify the immutable terminal index after every selected cell has evidence."""
        artifact_ids = tuple(rollout.artifact_id for rollout in rollouts)
        records = tuple({"artifact_id": artifact_id} for artifact_id in artifact_ids)
        index_payload = _jsonl_bytes(records)
        artifact_set_id = stable_id(
            "simulation-artifact-set",
            {"simulation_id": spec.simulation_id, "artifact_ids": artifact_ids},
        )
        artifact_set = SimulationArtifactSet(
            schema_version=1,
            created_at=_timestamp(self._clock),
            inputs=_sorted_inputs(self._plan_input, spec_input),
            code_revision=spec.code_revision,
            artifact_set_id=artifact_set_id,
            simulation_id=spec.simulation_id,
            artifact_ids=artifact_ids,
            artifacts_path=_ARTIFACT_IDS_FILE,
            artifacts_sha256=hashlib.sha256(index_payload).hexdigest(),
        )
        try:
            self._store.write(
                artifact_id=artifact_set_id,
                artifact_type="simulation-artifact-set",
                envelope=artifact_set,
                files={
                    _ARTIFACT_SET_FILE: canonical_json_bytes(artifact_set),
                    _ARTIFACT_IDS_FILE: index_payload,
                },
            )
            return artifact_set
        except ArtifactAlreadyExistsError as exc:
            stored = self._store.read(artifact_set_id)
            if stored.manifest.artifact_type != "simulation-artifact-set":
                raise SimulationResumeError(
                    f"artifact set ID {artifact_set_id!r} is already bound to another artifact type"
                ) from exc
            try:
                existing = SimulationArtifactSet.model_validate_json(
                    self._store.read_bytes(artifact_set_id, _ARTIFACT_SET_FILE)
                )
            except (ArtifactCorruptionError, ValueError) as exc:
                raise SimulationResumeError(
                    f"simulation artifact set {artifact_set_id!r} cannot be read safely"
                ) from exc
            if (
                existing.simulation_id != artifact_set.simulation_id
                or existing.artifact_ids != artifact_set.artifact_ids
                or existing.artifacts_path != artifact_set.artifacts_path
                or existing.artifacts_sha256 != artifact_set.artifacts_sha256
                or existing.inputs != artifact_set.inputs
                or existing.code_revision != artifact_set.code_revision
            ):
                raise SimulationResumeError(
                    f"artifact set ID {artifact_set_id!r} already names different rollout evidence"
                ) from exc
            return existing


def _utc_now() -> datetime:
    """Return a timezone-aware default timestamp without importing provider or runtime state."""
    return datetime.now(UTC)


def _rollout_id(spec_digest: str, cell_id: ArtifactId) -> ArtifactId:
    """Build a deterministic per-cell artifact ID tied to one immutable simulation recipe."""
    return stable_id("rollout", {"spec": spec_digest, "cell_id": cell_id})


def _sorted_inputs(*inputs: ArtifactInput) -> tuple[ArtifactInput, ...]:
    """Return exactly one immutable input per ID in the artifact envelope's required order."""
    by_id = {item.artifact_id: item for item in inputs}
    if len(by_id) != len(inputs):
        raise SimulationConfigurationError("simulation artifact inputs must have distinct IDs")
    return tuple(by_id[artifact_id] for artifact_id in sorted(by_id))


def _combine_spans(
    agent_events: Sequence[RolloutSpan],
    candidate_spans: Sequence[RolloutSpan],
    world_model_spans: Sequence[RolloutSpan],
    redacted_field_names: frozenset[str],
) -> tuple[RolloutSpan, ...]:
    """Merge redacted evidence while keeping recorder-owned candidate calls canonical."""
    agent_id_map = {event.span_id: f"agent-{event.span_id}" for event in agent_events}
    copied_agent_events = []
    for event in agent_events:
        if event.kind == RolloutEventKind.AGENT_MODEL_CALL:
            continue
        copied = redact_span(event, redacted_field_names)
        copied_agent_events.append(
            copied.model_copy(
                update={
                    "span_id": agent_id_map[event.span_id],
                    "parent_span_id": agent_id_map.get(
                        copied.parent_span_id,
                        copied.parent_span_id,
                    ),
                }
            )
        )
    combined = (*copied_agent_events, *candidate_spans, *world_model_spans)
    return tuple(
        sorted(
            combined,
            key=lambda span: (span.started_at, span.ended_at, span.span_id),
        )
    )


def _failure_span(
    started_at: datetime,
    ended_at: datetime,
    failure: StructuredFailure,
) -> RolloutSpan:
    """Build a minimum lifecycle span for a failure that preceded any model call."""
    return RolloutSpan(
        span_id="lifecycle-failure",
        kind=RolloutEventKind.LIFECYCLE,
        started_at=started_at,
        ended_at=ended_at,
        payload={"phase": failure.details.get("phase", "simulation")},
        failure=failure,
    )


def _internal_failure(phase: str, exception: Exception) -> StructuredFailure:
    """Normalize a local orchestration exception without retaining arbitrary exception text."""
    return StructuredFailure(
        code=FailureCode.INTERNAL,
        message=f"{phase} failed with {type(exception).__name__}",
        exception_type=type(exception).__name__,
        attribution=FailureAttribution.AGENT,
        details={"phase": phase},
    )


def _normalize_text_tool_failure(episode: AgentEpisode) -> StructuredFailure | None:
    """Translate a tool attempt into the text mode's explicit unsupported-cell evidence."""
    failure = episode.failure
    if failure is None or failure.exception_type != TextOnlyToolUseError.__name__:
        return failure
    return StructuredFailure(
        code=FailureCode.UNSUPPORTED,
        message="text world-model simulation cannot execute a customer-agent tool call",
        exception_type=failure.exception_type,
        attribution=FailureAttribution.TOOL,
        details={"phase": "agent_tool_call"},
    )


def _orchestration_economics(duration_seconds: float) -> OperationEconomics:
    """Record simulator-owned elapsed time without attributing it to the candidate model."""
    return OperationEconomics(
        latency_seconds=NumericMeasurement(value=duration_seconds, provenance="observed")
    )


def _known_total_spend(rollouts: Sequence[RolloutArtifact]) -> float | None:
    """Return total observed provider spend, or ``None`` if any completed episode is unpriced."""
    values = tuple(_rollout_spend(rollout) for rollout in rollouts)
    if any(value is None for value in values):
        return None
    return sum(cast(float, value) for value in values)


def _rollout_spend(rollout: RolloutArtifact) -> float | None:
    """Return observed provider cost, never interpreting an unpriced provider call as zero."""
    roles = (
        (rollout.candidate_economics, RolloutEventKind.AGENT_MODEL_CALL),
        (rollout.world_model_economics, RolloutEventKind.SIMULATOR_WORLD_MODEL_CALL),
    )
    total = 0.0
    for economics, span_kind in roles:
        made_call = any(span.kind == span_kind for span in rollout.spans)
        if not made_call:
            continue
        if economics is None or economics.cost_usd is None:
            return None
        total += economics.cost_usd.value
    return total


def _elapsed_seconds(started_at: float, ended_at: float) -> float:
    """Return a nonnegative orchestration duration from an injected monotonic clock."""
    return max(0.0, ended_at - started_at)


def _timestamp(clock: Callable[[], datetime], *, not_before: datetime | None = None) -> datetime:
    """Return an aware timestamp and prevent a deterministic test clock from moving backwards."""
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise SimulationConfigurationError("simulation clock must return timezone-aware datetimes")
    if not_before is not None and value < not_before:
        return not_before
    return value


def _jsonl_bytes(records: Sequence[Mapping[str, object]]) -> bytes:
    """Render a deterministic JSONL file from small internal artifact-index records."""
    payload = b"\n".join(
        canonical_json_bytes(cast(dict[str, object], record)) for record in records
    )
    return payload + (b"\n" if payload else b"")
