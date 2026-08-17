"""Atomic, resumable text world-model simulation over explicit evaluation-plan cells."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
    sorted_unique_inputs,
)
from wmo.common.evaluations import EvaluationCell, EvaluationPlan
from wmo.common.models import OperationEconomics
from wmo.common.progress import ProgressHook, report
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ArtifactStore,
    artifact_input,
)
from wmo.common.rollouts import (
    RolloutArtifact,
    SimulationArtifactSet,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
)
from wmo.runtime.agents import AgentRuntime
from wmo.runtime.models import ResolvedModel
from wmo.simulation.engines.clock import timestamp, utc_now
from wmo.simulation.engines.text.artifact_set import persist_artifact_set
from wmo.simulation.engines.text.bindings import (
    SimulationResolution,
    binding_digest,
    lease_id_for_binding,
    make_cell_binding,
    make_resolution,
    rollout_id_for_binding,
)
from wmo.simulation.engines.text.episode_loop import execute_text_episode_loop
from wmo.simulation.engines.text.errors import (
    SimulationConfigurationError,
    SimulationContentionError,
    SimulationResumeError,
)
from wmo.simulation.engines.text.grounded_rollout import GroundedRolloutBuilder
from wmo.simulation.engines.text.grounding import (
    completion_reservations,
    episode_reservation_failure,
    load_completion_contract,
    load_simulation_task_set,
    require_grounding_settings,
    verify_fit_retriever,
)
from wmo.simulation.engines.text.leases import (
    TextCellLeaseError,
    TextCellLeaseState,
    TextCellLeaseStore,
)
from wmo.simulation.engines.text.prompt import WORLD_MODEL_TEXT_PROMPT_VERSION
from wmo.simulation.engines.text.recording import (
    RecordingCandidateClient,
    TokenCounter,
    Utf8UpperBoundTokenCounter,
    text_prompt_digest,
)
from wmo.simulation.engines.text.redaction import redact_rollout_secrets, redacted_field_set
from wmo.simulation.engines.text.rollout_support import (
    combine_spans,
    elapsed_seconds,
    failure_span,
    internal_failure,
    known_total_spend,
    normalize_text_tool_failure,
    orchestration_economics,
)
from wmo.simulation.orchestration import require_implemented_mode
from wmo.simulation.retrieval import TraceRAGRetriever
from wmo.simulation.specs import SimulationSpec

if TYPE_CHECKING:
    from wmo.simulation.world_model import GroundedWorldModel

_SPEC_FILE = "simulation-spec.json"
_ROLLOUT_FILE = "rollout.json"


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
        task_set_input: Verified full immutable task-set manifest reference.
        fit_rag_input: Exact completed fit-only RAG manifest reference.
        fit_retriever: Read-only retriever bound to ``fit_rag_input`` and its exact embedder.
        candidate_models: Independently resolved candidate models keyed by plan alias.
        world_models: Independently resolved world-model providers keyed by local alias.
        grounded_world_models: Artifact-bound fit-only executors keyed by world-model alias.
        agent_factory: Creates an isolated customer-agent runtime for each episode worker.
        completion_contract_input: Optional exact automatic-simulation reservation artifact.
        redacted_field_names: Project-configured labels removed before evidence persists.
        clock: Time source for artifact and span timestamps.
        monotonic: Monotonic time source for orchestration latency measurements.
        token_counter: Optional full-request preflight counter. The default never truncates input.
        progress: Optional observer of exact per-cell completion counts.
    """

    def __init__(
        self,
        *,
        store: ArtifactStore,
        evaluation_plan: EvaluationPlan,
        evaluation_plan_input: ArtifactInput,
        task_set_input: ArtifactInput,
        fit_rag_input: ArtifactInput,
        fit_retriever: TraceRAGRetriever,
        candidate_models: Mapping[str, ResolvedModel],
        world_models: Mapping[str, ResolvedModel],
        grounded_world_models: Mapping[str, GroundedWorldModel],
        agent_factory: Callable[[], AgentRuntime],
        completion_contract_input: ArtifactInput | None = None,
        redacted_field_names: tuple[str, ...] = (),
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        token_counter: TokenCounter | None = None,
        progress: ProgressHook | None = None,
    ) -> None:
        """Bind one immutable plan and all explicit runtime dependencies.

        Args:
            store: Immutable store scoped to one `.wmo` project.
            evaluation_plan: Exact plan selected by later simulation specifications.
            evaluation_plan_input: Manifest identity for ``evaluation_plan``.
            task_set_input: Verified full immutable task-set manifest reference.
            fit_rag_input: Exact completed fit-only RAG manifest reference.
            fit_retriever: Retriever that can query only the frozen fit lineage set.
            candidate_models: Candidate aliases resolved through the canonical runtime package.
            world_models: Remote world-model aliases resolved through the canonical runtime package.
            grounded_world_models: Persisted fit-bound executors for those world models.
            agent_factory: Per-episode customer-agent construction seam.
            completion_contract_input: Optional exact completion reservation manifest.
            redacted_field_names: Local field labels removed before artifact persistence.
            clock: Time source, injectable for deterministic tests.
            monotonic: Duration source, injectable for deterministic tests.
            token_counter: Full-request counter used before each provider call.
            progress: Optional observer of exact per-cell completion counts.

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
        self._task_set_input = task_set_input
        self._fit_rag_input = fit_rag_input
        self._fit_retriever = fit_retriever
        verify_fit_retriever(fit_retriever, fit_rag_input)
        loaded_task_set = load_simulation_task_set(self._store, task_set_input)
        self._task_set = loaded_task_set.task_set
        self._tasks = {task.task_id: task for task in loaded_task_set.tasks}
        self._candidate_models = dict(candidate_models)
        self._world_models = dict(world_models)
        self._grounded_world_models = dict(grounded_world_models)
        self._agent_factory = agent_factory
        self._completion_contract_input = completion_contract_input
        self._completion_contract = load_completion_contract(self._store, completion_contract_input)
        self._redacted_field_names = redacted_field_set(redacted_field_names)
        self._clock = clock or utc_now
        self._monotonic = monotonic
        self._token_counter = token_counter or Utf8UpperBoundTokenCounter()
        self._progress = progress
        self._rollout_builder = GroundedRolloutBuilder(
            plan_input=self._plan_input,
            task_set_input=self._task_set_input,
            fit_rag_input=self._fit_rag_input,
            redacted_field_names=self._redacted_field_names,
            clock=self._clock,
        )
        self._leases = TextCellLeaseStore(store.project_directory, clock=self._clock)
        self._verify_persisted_evaluation_plan()

    def _verify_persisted_evaluation_plan(self) -> None:
        """Reject a caller-provided plan object that differs from its immutable manifest input."""
        try:
            stored = self._store.read(self._plan_input.artifact_id)
            persisted = EvaluationPlan.model_validate_json(
                self._store.read_bytes(self._plan_input.artifact_id, "evaluation-plan.json")
            )
        except (ArtifactCorruptionError, ValueError) as exc:
            raise SimulationConfigurationError(
                f"simulation evaluation plan {self._plan_input.artifact_id!r} cannot be read safely"
            ) from exc
        if (
            stored.manifest.artifact_type != "evaluation-plan"
            or artifact_input(stored.manifest) != self._plan_input
        ):
            raise SimulationConfigurationError(
                "evaluation_plan_input must name the exact persisted evaluation-plan manifest"
            )
        if persisted != self._plan:
            raise SimulationConfigurationError(
                "supplied evaluation plan differs from its immutable persisted manifest"
            )
        if self._plan.task_set_id != self._task_set.task_set_id:
            raise SimulationConfigurationError(
                "evaluation plan task_set_id does not match the supplied immutable task set"
            )

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
        cells, world_model, grounded_world_model = self._validate_spec_and_bindings(spec)
        spec_input = self._persist_specification(spec)
        resolution, resolution_input, bindings = self._persist_resolution(
            spec,
            spec_input,
            cells,
            world_model,
            grounded_world_model,
        )
        completed = self._load_completed_rollouts(cells, bindings, resolution_input)
        pending = tuple(cell for cell in cells if cell.cell_id not in completed)

        detail = _purpose_detail(cells)
        report(
            self._progress,
            "evaluation cells",
            completed=len(completed),
            total=len(cells),
            detail=detail,
        )
        for cell in pending:
            completed[cell.cell_id] = self._execute_and_persist_cell(
                spec,
                cell,
                world_model,
                grounded_world_model,
                spec_input,
                resolution,
                resolution_input,
                bindings,
            )
            report(
                self._progress,
                "evaluation cells",
                completed=len(completed),
                total=len(cells),
                detail=detail,
            )
        ordered_rollouts = tuple(completed[cell.cell_id] for cell in cells)
        return persist_artifact_set(
            store=self._store,
            plan_input=self._plan_input,
            task_set_input=self._task_set_input,
            fit_rag_input=self._fit_rag_input,
            spec=spec,
            spec_input=spec_input,
            resolution_input=resolution_input,
            rollouts=ordered_rollouts,
            clock=self._clock,
        )

    def _validate_spec_and_bindings(
        self,
        spec: SimulationSpec,
    ) -> tuple[tuple[EvaluationCell, ...], ResolvedModel, GroundedWorldModel]:
        """Validate all local inputs before any artifact write or provider call.

        Args:
            spec: Sparse finite-cost specification selected for execution or replay.

        Returns:
            Ordered cells, exact resolved model, and its persisted fit-bound executor.

        Raises:
            SimulationConfigurationError: A plan, task, RAG, model, prompt, or cell pin differs.
        """
        if spec.evaluation_plan_id != self._plan.plan_id:
            raise SimulationConfigurationError(
                "simulation spec evaluation_plan_id does not match the supplied evaluation plan"
            )
        if self._plan_input not in spec.inputs:
            raise SimulationConfigurationError(
                "simulation spec inputs must include the exact evaluation-plan manifest reference"
            )
        if self._task_set_input not in spec.inputs:
            raise SimulationConfigurationError(
                "simulation spec inputs must include the exact task-set manifest reference"
            )
        if (
            self._completion_contract_input is not None
            and self._completion_contract_input not in spec.inputs
        ):
            raise SimulationConfigurationError(
                "simulation spec inputs omit the exact completion reservation contract"
            )
        settings = require_grounding_settings(
            spec,
            fit_rag_input=self._fit_rag_input,
            retriever=self._fit_retriever,
        )
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
        grounded_world_model = self._grounded_world_models.get(settings.world_model_alias)
        if grounded_world_model is None:
            raise SimulationConfigurationError(
                "no persisted grounded executor is configured for the selected world model"
            )
        artifact = grounded_world_model.artifact
        if (
            grounded_world_model.artifact_input != settings.grounded_world_model_input
            or grounded_world_model.retriever is not self._fit_retriever
            or grounded_world_model.client is not world_model.client
            or artifact.model_alias != settings.world_model_alias
            or artifact.model != world_model.snapshot
            or artifact.prompt_version != settings.prompt_version
            or artifact.prompt_sha256 != text_prompt_digest()
        ):
            raise SimulationConfigurationError(
                "grounded executor differs from the persisted model, prompt, or fit RAG binding"
            )
        plan_cells = {cell.cell_id: cell for cell in self._plan.cells}
        cells = []
        expected_candidates = {
            snapshot.alias: snapshot for snapshot in self._plan.candidate_snapshots
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
            if candidate.snapshot != expected.model:
                raise SimulationConfigurationError(
                    f"candidate {cell.candidate_alias!r} does not match the plan's pinned snapshot"
                )
            if candidate.client is world_model.client:
                raise SimulationConfigurationError(
                    "candidate and world model must be resolved to independent model-client objects"
                )
            completion_reservations(
                self._completion_contract,
                candidate_alias=cell.candidate_alias,
                candidate=candidate,
                world_model=world_model,
            )
            cells.append(cell)
        return tuple(cells), world_model, grounded_world_model

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

    def _persist_resolution(
        self,
        spec: SimulationSpec,
        spec_input: ArtifactInput,
        cells: Sequence[EvaluationCell],
        world_model: ResolvedModel,
        grounded_world_model: GroundedWorldModel,
    ) -> tuple[SimulationResolution, ArtifactInput, dict[ArtifactId, SimulationCellBinding]]:
        """Persist or verify the complete pre-dispatch simulation resolution.

        Args:
            spec: Validated simulation specification.
            spec_input: Exact persisted specification manifest pointer.
            cells: Ordered sparse cells selected for this run.
            world_model: Exact resolved world-model client and snapshot.
            grounded_world_model: Persisted prompt and fit-RAG executor for that model.

        Returns:
            Resolution envelope, its manifest pointer, and bindings keyed by cell ID.

        Raises:
            SimulationResumeError: Existing resolution evidence differs or is unreadable.
        """
        bindings = {
            cell.cell_id: make_cell_binding(
                spec=spec,
                simulation_spec_input=spec_input,
                evaluation_plan_input=self._plan_input,
                task_set_input=self._task_set_input,
                fit_rag_input=self._fit_rag_input,
                task_set=self._task_set,
                cell=cell,
                task=self._tasks[cell.task_id],
                candidate=self._candidate_models[cell.candidate_alias],
                world_model=world_model,
                grounded_world_model=grounded_world_model,
            )
            for cell in cells
        }
        resolution = make_resolution(
            spec=spec,
            simulation_spec_input=spec_input,
            evaluation_plan_input=self._plan_input,
            task_set_input=self._task_set_input,
            fit_rag_input=self._fit_rag_input,
            bindings=tuple(bindings[cell.cell_id] for cell in cells),
            created_at=timestamp(self._clock),
        )
        try:
            manifest = self._store.write_json(
                artifact_id=resolution.resolution_id,
                artifact_type="simulation-resolution",
                envelope=resolution,
                files={"simulation-resolution.json": resolution},
            )
            return resolution, artifact_input(manifest), bindings
        except ArtifactAlreadyExistsError as exc:
            stored = self._store.read(resolution.resolution_id)
            if stored.manifest.artifact_type != "simulation-resolution":
                raise SimulationResumeError(
                    f"resolution {resolution.resolution_id!r} has an incompatible artifact type"
                ) from exc
            try:
                existing = SimulationResolution.model_validate_json(
                    self._store.read_bytes(resolution.resolution_id, "simulation-resolution.json")
                )
            except (ArtifactCorruptionError, ValueError) as error:
                raise SimulationResumeError(
                    f"simulation resolution {resolution.resolution_id!r} cannot be read safely"
                ) from error
            if existing.model_copy(update={"created_at": resolution.created_at}) != resolution:
                raise SimulationResumeError(
                    "simulation aliases, prompt, task content, or immutable inputs changed after "
                    "this simulation specification was first resolved"
                ) from exc
            return existing, artifact_input(stored.manifest), bindings

    def _load_completed_rollouts(
        self,
        cells: Sequence[EvaluationCell],
        bindings: Mapping[ArtifactId, SimulationCellBinding],
        resolution_input: ArtifactInput,
    ) -> dict[ArtifactId, RolloutArtifact]:
        """Load only completed rollout artifacts whose full immutable bindings still match."""
        completed: dict[ArtifactId, RolloutArtifact] = {}
        for cell in cells:
            binding = bindings[cell.cell_id]
            rollout = self._load_optional_rollout(rollout_id_for_binding(binding))
            if rollout is not None:
                self._validate_resume_rollout(rollout, cell, binding, resolution_input)
                completed[cell.cell_id] = rollout
        return completed

    def _execute_and_persist_cell(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        world_model: ResolvedModel,
        grounded_world_model: GroundedWorldModel,
        spec_input: ArtifactInput,
        resolution: SimulationResolution,
        resolution_input: ArtifactInput,
        bindings: Mapping[ArtifactId, SimulationCellBinding],
    ) -> RolloutArtifact:
        """Claim, execute, and persist one cell within the reconciled budget remainder.

        Args:
            spec: Validated finite-cost simulation specification.
            cell: Exact selected evaluation cell.
            world_model: Resolved world model pinned by the specification.
            grounded_world_model: Persisted fit-bound executor for that resolved model.
            spec_input: Persisted specification pointer used by the durable lease.
            resolution: Immutable resolution owning the cell binding.
            resolution_input: Exact resolution manifest pointer.
            bindings: Complete bindings for every selected cell.

        Returns:
            Newly persisted or exactly replayed rollout evidence.

        Raises:
            SimulationContentionError: Another live runner owns the paid cell.
            SimulationResumeError: Lease or persisted evidence is inconsistent.
        """
        binding = bindings[cell.cell_id]
        rollout_id = rollout_id_for_binding(binding)
        existing = self._load_optional_rollout(rollout_id)
        if existing is not None:
            self._validate_resume_rollout(existing, cell, binding, resolution_input)
            return existing
        try:
            claim = self._leases.acquire(
                lease_id=lease_id_for_binding(resolution, binding),
                resolution_id=resolution.resolution_id,
                simulation_id=spec.simulation_id,
                rollout_id=rollout_id,
                binding_sha256=binding_digest(binding),
                maximum_cost_usd=spec.maximum_cost_usd,
                rollout_completed=lambda item: self._load_optional_rollout(item) is not None,
                observed_spend_usd=lambda: self._known_resolution_spend(bindings, resolution_input),
            )
        except TextCellLeaseError as exc:
            raise SimulationResumeError(
                f"text simulation cell {cell.cell_id!r} cannot be admitted"
            ) from exc
        if claim.retryable:
            raise SimulationContentionError("text simulation cell is contended; retry the run")
        if claim.state == TextCellLeaseState.COMPLETED:
            completed = self._load_optional_rollout(rollout_id)
            if completed is None:  # pragma: no cover - artifact check and read share one store
                raise SimulationResumeError("completed text-cell claim has no readable rollout")
            self._validate_resume_rollout(completed, cell, binding, resolution_input)
            return completed
        if claim.state == TextCellLeaseState.STALE:
            stale = claim.lease
            if stale is None:  # pragma: no cover - stale state always retains its durable record
                raise SimulationResumeError("stale text-cell claim omitted its recovery evidence")
            rollout = self._lease_failure_rollout(
                spec, cell, world_model, binding, resolution_input, stale.lease_id
            )
            return self._persist_rollout(rollout, cell, binding, resolution_input)
        if claim.state == TextCellLeaseState.BUDGET_BLOCKED:
            rollout = self._budget_failure_rollout(
                spec, cell, world_model, binding, resolution_input, claim.observed_spend_usd
            )
            return self._persist_rollout(rollout, cell, binding, resolution_input)
        if claim.lease is None:  # pragma: no cover - owned state always creates an exact lease
            raise SimulationResumeError("owned text-cell admission omitted its durable lease")
        try:
            observed_spend = claim.observed_spend_usd or 0.0
            maximum_cell_cost_usd = (spec.maximum_cost_usd or 0.0) - observed_spend
            rollout = self._execute_cell(
                spec,
                cell,
                world_model,
                grounded_world_model,
                binding,
                resolution_input,
                maximum_cell_cost_usd=maximum_cell_cost_usd,
            )
            persisted = self._persist_rollout(rollout, cell, binding, resolution_input)
        except BaseException:
            raise
        self._leases.release(claim.lease)
        return persisted

    def _known_resolution_spend(
        self,
        bindings: Mapping[ArtifactId, SimulationCellBinding],
        resolution_input: ArtifactInput,
    ) -> float | None:
        """Return completed provider spend or unknown when one bound cell is unpriced."""
        rollouts = []
        for cell_id, binding in bindings.items():
            rollout = self._load_optional_rollout(rollout_id_for_binding(binding))
            if rollout is None:
                continue
            cell = next(item for item in self._plan.cells if item.cell_id == cell_id)
            self._validate_resume_rollout(rollout, cell, binding, resolution_input)
            rollouts.append(rollout)
        return known_total_spend(rollouts)

    def _execute_cell(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        world_model: ResolvedModel,
        grounded_world_model: GroundedWorldModel,
        binding: SimulationCellBinding,
        resolution_input: ArtifactInput,
        *,
        maximum_cell_cost_usd: float,
    ) -> RolloutArtifact:
        """Execute one grounded no-tools episode or retain its structured failure.

        Args:
            spec: Validated finite-cost simulation specification.
            cell: Exact selected evaluation cell.
            world_model: Resolved world model pinned by the specification.
            grounded_world_model: Persisted fit-bound executor for that resolved model.
            binding: Complete immutable binding for this cell.
            resolution_input: Exact resolution pointer retained by the rollout.
            maximum_cell_cost_usd: Reconciled provider-spend remainder for the cell.

        Returns:
            Completed or failed rollout with separated candidate, retrieval, and simulator costs.

        Raises:
            SimulationConfigurationError: Required world-model or retrieval settings are absent.
        """
        task = self._tasks[cell.task_id]
        candidate = self._candidate_models[cell.candidate_alias]
        started_at = timestamp(self._clock)
        started_monotonic = self._monotonic()
        if task.tools:
            return self._failure_rollout(
                spec,
                cell,
                candidate,
                world_model,
                binding,
                resolution_input,
                started_at,
                StopReason.FAILURE,
                StructuredFailure(
                    code=FailureCode.UNSUPPORTED,
                    message="text world-model simulation cannot run a task that declares tools",
                    attribution=FailureAttribution.TOOL,
                    details={"phase": "task_tools", "tool_count": len(task.tools)},
                ),
                duration_seconds=elapsed_seconds(started_monotonic, self._monotonic()),
            )
        settings = spec.world_model
        if settings is None:  # pragma: no cover - validated before this execution path
            raise SimulationConfigurationError("world-model simulation settings are missing")
        if settings.query_embedding is None:  # pragma: no cover - validated before execution
            raise SimulationConfigurationError("query-embedding reservation is missing")
        reservation_failure = episode_reservation_failure(
            settings,
            completion_contract=self._completion_contract,
            candidate_alias=cell.candidate_alias,
            maximum_steps=spec.maximum_steps,
            remaining_cost_usd=maximum_cell_cost_usd,
        )
        if reservation_failure is not None:
            return self._failure_rollout(
                spec,
                cell,
                candidate,
                world_model,
                binding,
                resolution_input,
                started_at,
                StopReason.MAXIMUM_COST,
                reservation_failure,
                duration_seconds=elapsed_seconds(started_monotonic, self._monotonic()),
            )
        candidate_request, world_request = completion_reservations(
            self._completion_contract,
            candidate_alias=cell.candidate_alias,
            candidate=candidate,
            world_model=world_model,
        )
        recorder = RecordingCandidateClient(
            task=task,
            candidate=candidate,
            world_model=world_model,
            grounded_world_model=grounded_world_model,
            query_embedding=settings.query_embedding,
            candidate_request=candidate_request,
            world_model_request=world_request,
            completion_maximum_attempts=(
                self._completion_contract.maximum_attempts
                if self._completion_contract is not None
                else 1
            ),
            maximum_cost_usd=maximum_cell_cost_usd,
            maximum_steps=spec.maximum_steps,
            maximum_output_tokens=settings.maximum_output_tokens,
            redacted_field_names=self._redacted_field_names,
            clock=self._clock,
            token_counter=self._token_counter,
        )
        try:
            outcome = execute_text_episode_loop(
                agent_factory=self._agent_factory,
                task=task,
                recorder=recorder,
            )
        except Exception as exc:  # noqa: BLE001 - construction faults remain cell evidence
            return self._failure_rollout(
                spec,
                cell,
                candidate,
                world_model,
                binding,
                resolution_input,
                started_at,
                StopReason.FAILURE,
                internal_failure("agent construction or lifecycle", exc),
                duration_seconds=elapsed_seconds(started_monotonic, self._monotonic()),
                recorder=recorder,
            )
        failure = outcome.failure
        if outcome.episodes and failure == outcome.episodes[-1].failure:
            failure = normalize_text_tool_failure(outcome.episodes[-1])
        return self._rollout_builder.make(
            spec=spec,
            cell=cell,
            candidate=candidate,
            world_model=world_model,
            binding=binding,
            resolution_input=resolution_input,
            stop_reason=outcome.stop_reason,
            failure=failure,
            final_output=outcome.final_output,
            spans=combine_spans(
                tuple(event for episode in outcome.episodes for event in episode.events),
                recorder.recorded.candidate_spans,
                recorder.recorded.world_model_spans,
                self._redacted_field_names,
            ),
            candidate_economics=recorder.recorded.candidate_economics,
            world_model_economics=recorder.recorded.world_model_economics,
            retrieval_economics=recorder.recorded.retrieval_economics,
            orchestration_economics=orchestration_economics(
                elapsed_seconds(started_monotonic, self._monotonic())
            ),
        )

    def _failure_rollout(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        candidate: ResolvedModel,
        world_model: ResolvedModel,
        binding: SimulationCellBinding,
        resolution_input: ArtifactInput,
        started_at: datetime,
        stop_reason: StopReason,
        failure: StructuredFailure,
        *,
        duration_seconds: float,
        recorder: RecordingCandidateClient | None = None,
    ) -> RolloutArtifact:
        """Build an artifact-safe failed cell while retaining completed calls.

        Args:
            spec: Simulation specification owning the cell.
            cell: Failed evaluation cell.
            candidate: Resolved candidate identity.
            world_model: Resolved world-model identity.
            binding: Complete immutable cell binding.
            resolution_input: Exact resolution manifest pointer.
            started_at: Episode start timestamp.
            stop_reason: Canonical terminal reason.
            failure: Structured failure safe for persistence.
            duration_seconds: Observed orchestration duration.
            recorder: Optional completed-call recorder available at failure time.

        Returns:
            Canonical failed rollout with any known operation economics.
        """
        ended_at = timestamp(self._clock, not_before=started_at)
        recorded = recorder.recorded if recorder is not None else None
        spans = combine_spans(
            (),
            recorded.candidate_spans if recorded is not None else (),
            recorded.world_model_spans if recorded is not None else (),
            self._redacted_field_names,
        )
        if not spans:
            spans = (failure_span(started_at, ended_at, failure),)
        return self._rollout_builder.make(
            spec=spec,
            cell=cell,
            candidate=candidate,
            world_model=world_model,
            binding=binding,
            resolution_input=resolution_input,
            stop_reason=stop_reason,
            failure=failure,
            final_output=recorder.last_candidate_action if recorder is not None else None,
            spans=spans,
            candidate_economics=(
                recorded.candidate_economics if recorded is not None else OperationEconomics()
            ),
            world_model_economics=(
                recorded.world_model_economics if recorded is not None else OperationEconomics()
            ),
            retrieval_economics=(
                recorded.retrieval_economics if recorded is not None else OperationEconomics()
            ),
            orchestration_economics=orchestration_economics(duration_seconds),
        )

    def _budget_failure_rollout(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        world_model: ResolvedModel,
        binding: SimulationCellBinding,
        resolution_input: ArtifactInput,
        spent: float | None,
    ) -> RolloutArtifact:
        """Represent a cell rejected by the durable finite-spend admission reservation."""
        failure = StructuredFailure(
            code=FailureCode.BUDGET,
            message="simulation spend is unknown or its durable reservation exhausted the ceiling",
            attribution=FailureAttribution.MODEL,
            details={"phase": "paid_cell_admission", "observed_spend_usd": spent},
        )
        return self._failure_rollout(
            spec,
            cell,
            self._candidate_models[cell.candidate_alias],
            world_model,
            binding,
            resolution_input,
            timestamp(self._clock),
            StopReason.MAXIMUM_COST,
            failure,
            duration_seconds=0.0,
        )

    def _lease_failure_rollout(
        self,
        spec: SimulationSpec,
        cell: EvaluationCell,
        world_model: ResolvedModel,
        binding: SimulationCellBinding,
        resolution_input: ArtifactInput,
        lease_id: ArtifactId,
    ) -> RolloutArtifact:
        """Record a non-replayed crash recovery outcome for an ambiguous expired paid-cell claim."""
        failure = StructuredFailure(
            code=FailureCode.BUDGET,
            message=(
                "a prior paid-cell claim expired after its owner exited; WMO will not replay it"
            ),
            attribution=FailureAttribution.MODEL,
            details={"phase": "paid_cell_stale_lease", "lease_id": lease_id},
        )
        return self._failure_rollout(
            spec,
            cell,
            self._candidate_models[cell.candidate_alias],
            world_model,
            binding,
            resolution_input,
            timestamp(self._clock),
            StopReason.MAXIMUM_COST,
            failure,
            duration_seconds=0.0,
        )

    def _persist_rollout(
        self,
        rollout: RolloutArtifact,
        cell: EvaluationCell,
        binding: SimulationCellBinding,
        resolution_input: ArtifactInput,
    ) -> RolloutArtifact:
        """Atomically store a rollout, accepting only an exactly bound completed counterpart.

        Secret-like substrings generated by simulated models are replaced with the fixed
        placeholder before the write, so the persisted content, its manifest digests, and the
        returned rollout all carry the redacted value and its audit count.
        """
        rollout = redact_rollout_secrets(rollout)
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
            try:
                self._validate_resume_rollout(existing, cell, binding, resolution_input)
            except SimulationResumeError as error:
                raise SimulationResumeError(
                    f"rollout ID {rollout.artifact_id!r} is already bound to incompatible evidence"
                ) from error
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
        destination = self._store.project_directory / "artifacts" / rollout_id
        if not destination.exists():
            return None
        return self._load_rollout(rollout_id)

    def _validate_resume_rollout(
        self,
        rollout: RolloutArtifact,
        cell: EvaluationCell,
        binding: SimulationCellBinding,
        resolution_input: ArtifactInput,
    ) -> None:
        """Validate an existing rollout against every requested immutable pin.

        Args:
            rollout: Previously persisted rollout proposed for exact replay.
            cell: Requested evaluation cell.
            binding: Newly derived complete cell binding.
            resolution_input: Exact resolution pointer required by the replay.

        Raises:
            SimulationResumeError: Any stable ID, task, mode, RAG, input, or binding differs.
        """
        expected_inputs = sorted_unique_inputs(
            self._plan_input,
            self._task_set_input,
            self._fit_rag_input,
            binding.grounded_world_model_input,
            binding.simulation_spec_input,
            resolution_input,
        )
        if (
            rollout.cell_id != cell.cell_id
            or rollout.task_id != cell.task_id
            or rollout.mode != SimulationMode.WORLD_MODEL
            or rollout.rollout_id != rollout_id_for_binding(binding)
            or rollout.simulation_binding != binding
            or rollout.inputs != expected_inputs
        ):
            raise SimulationResumeError(
                f"stored rollout {rollout.artifact_id!r} does not match the requested "
                "simulation cell"
            )


def _purpose_detail(cells: Sequence[EvaluationCell]) -> str | None:
    """Name the one evaluation purpose shared by every selected cell.

    Args:
        cells: Exact evaluation-plan cells selected by the running specification.

    Returns:
        The shared purpose with underscores spelled as hyphens, or ``None`` when purposes differ.
    """
    purposes = {cell.purpose for cell in cells}
    if len(purposes) != 1:
        return None
    return purposes.pop().replace("_", "-")
