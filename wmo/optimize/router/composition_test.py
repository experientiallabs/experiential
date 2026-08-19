"""Deterministic full-artifact coverage for public router composition."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from wmo.cli.build_cmd import _lineage_bindings
from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
    stable_id,
)
from wmo.common.evaluations import (
    EvaluationPlan,
    EvaluationProtocol,
    ObservedProductionCell,
    load_evaluation_dataset,
)
from wmo.common.evaluations.build_test import (
    _persist_calibration,
    _persist_rollout,
    _production_rollout,
)
from wmo.common.evaluations.evidence import read_calibration
from wmo.common.judging import (
    DimensionJudgment,
    Judgment,
    Rubric,
    RubricDimension,
    ScoreAnchor,
)
from wmo.common.models import (
    CandidateTokenPrice,
    CompletionCostReservation,
    EmbeddingCostReservation,
    ModelCapabilities,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    PricingSnapshot,
    RoutedCandidateSnapshot,
    completion_cost_reservation,
)
from wmo.common.project import (
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectStore,
    artifact_input,
)
from wmo.common.rollouts import SimulationArtifactSet, StopReason
from wmo.common.routing import KnnGuard
from wmo.common.tasks import load_task_set
from wmo.optimize.router.composition import (
    ApprovedRouterReview,
    RouterCompositionBudget,
    RouterCompositionError,
    RouterEvaluationSetup,
    RouterWorkflowServices,
    compose_router,
)
from wmo.optimize.router.errors import (
    JudgeDispatchExhaustedError,
    JudgeTranscriptAdmissionError,
)
from wmo.optimize.router.evaluation.spend import observed_rollout_spend
from wmo.optimize.router.fit.workflow_test import _persist_embeddings, _persist_pricing
from wmo.optimize.router.judgment_budget import (
    JudgmentDispatchReceipt,
    JudgmentExclusionRecord,
)
from wmo.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from wmo.runtime.models.providers.transport import ProviderTransportError
from wmo.runtime.router.runtime_test import _Client, _request
from wmo.simulation.build import ProjectBuild, build_project, select_completed_build
from wmo.simulation.engines.text import simulator as text_simulator_module
from wmo.simulation.engines.text.resume import MAXIMUM_CELL_ATTEMPTS
from wmo.simulation.engines.text.simulator import WorldModelSimulator
from wmo.simulation.engines.text.simulator_test import (
    _OneTurnAgent,
    _response,
    _ScriptedClient,
)
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.orchestration import Simulator
from wmo.simulation.retrieval import (
    load_fit_rag_retriever,
    load_rag_index,
    persist_trace_rag,
)
from wmo.simulation.retrieval.retrieval_test import _message_trace as _trace
from wmo.simulation.specs import (
    CandidateCompletionReservation,
    SimulationCompletionContract,
    SimulationSpec,
    WorldModelSettings,
    persist_simulation_completion_contract,
)
from wmo.simulation.world_model import bind_fit_grounded_world_model, persist_grounded_world_model

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


def _bind_completed_build(
    project: ProjectStore,
    normalized: TraceNormalizationResult,
    *,
    revision: str,
) -> ProjectBuildArtifacts:
    """Persist and select the exact grounded build consumed by router composition.

    Args:
        project: Initialized test project receiving the immutable build graph.
        normalized: Canonical real traces used for tasks and retrieval.
        revision: Exact source revision bound to every artifact.

    Returns:
        Completed project build pointers selected in ``project.toml``.
    """
    built = build_project(
        normalized,
        project,
        created_at=_TIME,
        code_revision=revision,
    )
    trace_input = artifact_input(built.artifacts.trace_dataset.manifest)
    task_input = built.review.task_set
    bindings = _lineage_bindings(built)
    serving = persist_trace_rag(
        project.artifacts,
        (trace_input,),
        bindings,
        created_at=_TIME,
        code_revision=revision,
        included_partitions=frozenset({"fit", "held_out"}),
    )
    fit = persist_trace_rag(
        project.artifacts,
        (trace_input,),
        bindings,
        created_at=_TIME,
        code_revision=revision,
        included_partitions=frozenset({"fit"}),
    )
    serving_input = artifact_input(serving.manifest)
    world = persist_grounded_world_model(
        project.artifacts,
        serving_input,
        model_alias="world-model-a",
        model=_snapshot("world-model-a"),
        created_at=_TIME,
        code_revision=revision,
        top_k=5,
    )
    completed = ProjectBuildArtifacts(
        trace_dataset=trace_input,
        task_set=task_input,
        serving_rag=serving_input,
        fit_rag=artifact_input(fit.manifest),
        world_model=artifact_input(world.manifest),
    )
    select_completed_build(project, completed, built.review)
    return completed


def _capabilities(alias: str) -> ModelCapabilities:
    """Return the exact frozen capabilities used by each focused model."""
    if alias == "embedder":
        return ModelCapabilities(supports_embeddings=True)
    return ModelCapabilities(context_window_tokens=100_000, maximum_output_tokens=16_000)


def _snapshot(alias: str) -> ModelSnapshot:
    """Freeze one model identity over the capabilities needed by text simulation.

    Args:
        alias: Stable local model alias.

    Returns:
        Deterministic fixture snapshot with the exact capability digest.
    """
    return ModelSnapshot(
        provider="test",
        model_id=alias,
        revision="fixture",
        capabilities_sha256=_capabilities(alias).identity_sha256(),
        connection_sha256="b" * 64,
    )


def _resolved(alias: str, client: ModelClient) -> ResolvedModel:
    """Resolve one fake client with the same capabilities pinned by the test snapshot."""
    capabilities = _capabilities(alias)
    return ResolvedModel(
        alias=alias,
        snapshot=_snapshot(alias),
        capabilities=capabilities,
        client=client,
        embedding_client=None,
    )


def _priced_capabilities() -> ModelCapabilities:
    """Return completion capabilities carrying the explicit prices reservations verify."""
    return ModelCapabilities(
        supports_completions=True,
        context_window_tokens=100_000,
        maximum_output_tokens=16_000,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=1.0,
        cached_input_cost_per_million_tokens_usd=1.0,
        cache_write_cost_per_million_tokens_usd=1.0,
    )


def _priced_resolved(alias: str, client: ModelClient) -> ResolvedModel:
    """Resolve one fake completion client under reservation-verifiable priced capabilities."""
    return ResolvedModel(
        alias=alias,
        snapshot=_snapshot(alias),
        capabilities=_priced_capabilities(),
        client=client,
        embedding_client=None,
    )


def _completion_reservation(alias: str) -> CompletionCostReservation:
    """Freeze one conservative single-attempt completion reservation for ``alias``."""
    return completion_cost_reservation(
        model=_snapshot(alias),
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=1.0,
        cached_input_usd_per_million_tokens=1.0,
        cache_write_usd_per_million_tokens=1.0,
        maximum_attempts=1,
        maximum_input_tokens=84_000,
        maximum_output_tokens=16_000,
        estimated_input_tokens=1_000,
    )


class _Catalog:
    """Resolve the exact simulation-capable identities frozen by this composition test."""

    def __init__(self, snapshots: dict[str, ModelSnapshot], client: _Client) -> None:
        """Store exact model snapshots and their shared deterministic client."""
        self.snapshots = snapshots
        self.client = client

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        """Return one alias snapshot and its deterministic capabilities."""
        return self.snapshots[alias], _capabilities(alias)

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        del role
        """Resolve one alias to the shared deterministic client."""
        snapshot, capabilities = self.snapshot(alias)
        return ResolvedModel(
            alias=alias,
            snapshot=snapshot,
            capabilities=capabilities,
            client=self.client,
            embedding_client=self.client if capabilities.supports_embeddings else None,
        )


class _ReviewSupplier:
    """Persist one explicit approved rubric and sealed calibration, idempotently."""

    def __call__(
        self,
        project: ProjectStore,
        build: ProjectBuild,
        budget: RouterCompositionBudget,
    ) -> ApprovedRouterReview:
        """Persist or replay one exact approved rubric and calibration."""
        del budget
        if "rubric-a" not in project.artifacts.list_ids():
            task_input = build.review.task_set
            rubric = Rubric(
                schema_version=1,
                created_at=_TIME,
                inputs=(task_input,),
                code_revision="test-revision",
                rubric_id="rubric-a",
                dimensions=(
                    RubricDimension(
                        dimension_id="dimension-a",
                        name="Task success",
                        description="Whether the task was completed.",
                        min_score=0,
                        max_score=5,
                        anchors=(
                            ScoreAnchor(score=0, description="Score 0 outcome."),
                            ScoreAnchor(score=1, description="Score 1 outcome."),
                            ScoreAnchor(score=2, description="Score 2 outcome."),
                            ScoreAnchor(score=3, description="Score 3 outcome."),
                            ScoreAnchor(score=4, description="Score 4 outcome."),
                            ScoreAnchor(score=5, description="Score 5 outcome."),
                        ),
                    ),
                ),
                source_task_set_id=task_input.artifact_id,
                status="human_approved",
                approved_at=_TIME,
            )
            project.artifacts.write_json(
                artifact_id=rubric.rubric_id,
                artifact_type="rubric",
                envelope=rubric,
                files={"rubric.json": rubric},
            )
            tasks = load_task_set(project.artifacts, task_input.artifact_id).tasks
            _persist_calibration(
                project.artifacts,
                fit_lineages=tuple(
                    task.lineage_group_id for task in tasks if task.partition == "fit"
                ),
                held_out_lineages=tuple(
                    task.lineage_group_id for task in tasks if task.partition == "held_out"
                ),
            )
        return ApprovedRouterReview(rubric_id="rubric-a", calibration_id="calibration-a")


class _SetupSupplier:
    """Persist explicit production overlaps, frozen embeddings, and pricing."""

    def __call__(
        self,
        project: ProjectStore,
        build: ProjectBuild,
        review: ApprovedRouterReview,
        budget: RouterCompositionBudget,
    ) -> RouterEvaluationSetup:
        """Persist reviewed evaluation inputs bound to the selected completed build.

        Args:
            project: Project owning the completed build and evaluation artifacts.
            build: Deterministic trace and task-set build reused by composition.
            review: Approved rubric and manual calibration identifiers.
            budget: Finite composition budget already validated by the workflow.

        Returns:
            Complete evaluation setup using the project's exact fit-only RAG pointer.
        """
        del review, budget
        completed = project.load_project().build
        assert completed is not None
        fit_index = load_rag_index(project.artifacts, completed.fit_rag.artifact_id).index
        tasks = load_task_set(project.artifacts, build.artifacts.task_set.task_set_id).tasks
        candidate = RoutedCandidateSnapshot(alias="candidate-a", model=_snapshot("candidate-a"))
        observed = []
        fit = tuple(task for task in tasks if task.partition == "fit")
        for index, task in enumerate(fit[:10]):
            rollout_id = f"production-{index:02d}"
            if rollout_id not in project.artifacts.list_ids():
                rollout = _production_rollout(
                    rollout_id,
                    cell_id=f"import-{index}",
                    task=task,
                    candidate=candidate.model,
                ).model_copy(update={"trace_id": task.source_trace_ids[0]})
                _persist_rollout(project.artifacts, rollout)
            observed.append(
                ObservedProductionCell(
                    task_id=task.task_id,
                    candidate_alias=candidate.alias,
                    repeat=0,
                    rollout_artifact_id=rollout_id,
                )
            )
        if "pricing-a" not in project.artifacts.list_ids():
            _persist_pricing(project.artifacts)
            _persist_embeddings(project.artifacts, tasks)
        production = EvaluationProtocol(
            protocol_id="protocol-production",
            evidence_source="production",
            agent_id="agent-a",
            simulator_id="production-import-v1",
            rubric_id="rubric-a",
            judge_calibration_id="calibration-a",
            pricing_snapshot_id="pricing-a",
        )
        world = EvaluationProtocol(
            protocol_id="protocol-world",
            evidence_source="world_model",
            agent_id="agent-a",
            simulator_id="text-world-model-v1",
            world_model=_snapshot("world-model-a"),
            simulator_prompt_id="world-model-text-v1",
            rubric_id="rubric-a",
            judge_calibration_id="calibration-a",
            pricing_snapshot_id="pricing-a",
        )
        return RouterEvaluationSetup(
            candidates=(candidate,),
            observed_cells=tuple(observed),
            production_protocol=production,
            simulation_protocol=world,
            embedding_set_id="embeddings-a",
            fit_rag_input=completed.fit_rag,
            pricing_snapshot_id="pricing-a",
            incumbent_alias="candidate-a",
            guard=KnnGuard(
                maximum_neighbors=10,
                minimum_paired_observations=8,
                relative_similarity_threshold=0.0,
                uncertainty_multiplier=0.5,
                quality_tolerance=0.0,
            ),
            judgment_status="provisional",
            world_model_settings=WorldModelSettings(
                world_model_alias="world-model-a",
                grounded_world_model_input=completed.world_model,
                prompt_version="text-world-model-v1",
                query_embedding=EmbeddingCostReservation(
                    model=fit_index.embedder,
                    input_usd_per_million_tokens=0.0,
                    maximum_attempts=1,
                    maximum_input_tokens=10_000,
                ),
            ),
            agent_id="agent-a",
            seed=7,
            maximum_steps=2,
            maximum_concurrency=1,
        )


def _persist_two_candidate_pricing(project: ProjectStore) -> None:
    """Persist one pricing snapshot covering both routed candidate aliases.

    Args:
        project: Project store receiving the immutable pricing snapshot.
    """
    pricing = PricingSnapshot(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        pricing_snapshot_id="pricing-two",
        candidate_prices=tuple(
            CandidateTokenPrice(
                candidate_alias=alias,
                input_usd_per_million_tokens=1.0,
                output_usd_per_million_tokens=2.0,
            )
            for alias in ("candidate-a", "candidate-b")
        ),
    )
    project.artifacts.write_json(
        artifact_id=pricing.pricing_snapshot_id,
        artifact_type="pricing-snapshot",
        envelope=pricing,
        files={"pricing.json": pricing},
    )


class _ReservedSetupSupplier(_SetupSupplier):
    """Add a second routed candidate and a persisted exact completion reservation contract."""

    def __call__(
        self,
        project: ProjectStore,
        build: ProjectBuild,
        review: ApprovedRouterReview,
        budget: RouterCompositionBudget,
    ) -> RouterEvaluationSetup:
        """Persist the reviewed two-candidate setup and its frozen completion reservations.

        Args:
            project: Project owning the completed build and evaluation artifacts.
            build: Deterministic trace and task-set build reused by composition.
            review: Approved rubric and manual calibration identifiers.
            budget: Finite composition budget already validated by the workflow.

        Returns:
            Evaluation setup whose simulation binds a persisted completion contract.
        """
        setup = super().__call__(project, build, review, budget)
        if "pricing-two" not in project.artifacts.list_ids():
            _persist_two_candidate_pricing(project)
        _contract, contract_input = persist_simulation_completion_contract(
            project.artifacts,
            inputs=(),
            candidate_requests=(
                CandidateCompletionReservation(
                    candidate_alias="candidate-a",
                    request=_completion_reservation("candidate-a"),
                ),
                CandidateCompletionReservation(
                    candidate_alias="candidate-b",
                    request=_completion_reservation("candidate-b"),
                ),
            ),
            world_model_alias="world-model-a",
            world_model_request=_completion_reservation("world-model-a"),
            maximum_attempts=1,
            created_at=_TIME,
            code_revision="test-revision",
        )
        return setup.model_copy(
            update={
                "candidates": (
                    *setup.candidates,
                    RoutedCandidateSnapshot(alias="candidate-b", model=_snapshot("candidate-b")),
                ),
                "pricing_snapshot_id": "pricing-two",
                "production_protocol": setup.production_protocol.model_copy(
                    update={"pricing_snapshot_id": "pricing-two"}
                ),
                "simulation_protocol": setup.simulation_protocol.model_copy(
                    update={"pricing_snapshot_id": "pricing-two"}
                ),
                "simulation_completion_input": contract_input,
            }
        )


class _MismatchedSetupSupplier(_SetupSupplier):
    """Return a reviewed setup whose fit pointer is outside the completed build."""

    def __call__(
        self,
        project: ProjectStore,
        build: ProjectBuild,
        review: ApprovedRouterReview,
        budget: RouterCompositionBudget,
    ) -> RouterEvaluationSetup:
        """Replace the valid setup fit pointer with an unrelated immutable pointer.

        Args:
            project: Project owning the completed build and evaluation artifacts.
            build: Deterministic trace and task-set build reused by composition.
            review: Approved rubric and manual calibration identifiers.
            budget: Finite composition budget already validated by the workflow.

        Returns:
            Otherwise valid evaluation setup with a deliberately mismatched fit RAG.
        """
        setup = super().__call__(project, build, review, budget)
        return setup.model_copy(
            update={"fit_rag_input": ArtifactInput(artifact_id="other-fit-rag", sha256=_DIGEST)}
        )


class _Judge:
    """Return one stable score while counting actual dispatches."""

    def __init__(self) -> None:
        """Initialize dispatch counters and optional failure injection."""
        self.calls = 0
        self.log: list[tuple[str, bool]] = []
        self.fail_on_call: int | None = None
        self.raise_after_lock: list[Exception] = []

    def judge_persisted(
        self,
        store: ProjectStore,
        *,
        rollout_artifact_id: str,
        rubric_artifact_id: str,
        calibration_artifact_id: str,
    ) -> Judgment:
        """Persist one stable judgment unless the configured call must fail."""
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated judgment dispatch interruption")
        rollout_value = read_rollout(store.artifacts, rollout_artifact_id)[0]
        locked = any(
            artifact_id.startswith("router-policy-lock-")
            for artifact_id in store.artifacts.list_ids()
        )
        if locked and self.raise_after_lock:
            raise self.raise_after_lock.pop(0)
        self.log.append((rollout_value.cell_id or "observed", locked))
        rollout = store.artifacts.read(rollout_artifact_id)
        rubric = store.artifacts.read(rubric_artifact_id)
        calibration = store.artifacts.read(calibration_artifact_id)
        calibration_value = read_calibration(store.artifacts, calibration_artifact_id)[0]
        inputs = tuple(
            sorted(
                map(artifact_input, (rollout.manifest, rubric.manifest, calibration.manifest)),
                key=lambda item: item.artifact_id,
            )
        )
        return Judgment(
            schema_version=1,
            created_at=_TIME,
            inputs=inputs,
            code_revision="test-revision",
            judgment_id=stable_id(
                "judgment",
                {"rollout": rollout_artifact_id, "rubric": rubric_artifact_id},
            ),
            rollout_id=rollout_artifact_id,
            rubric_id=rubric_artifact_id,
            calibration_id=calibration_artifact_id,
            judge_model=calibration_value.judge_model,
            judge_prompt_id=calibration_value.judge_prompt_id,
            judge_prompt_sha256=calibration_value.judge_prompt_sha256,
            dimensions=(
                DimensionJudgment(
                    dimension_id="dimension-a",
                    raw_score=4,
                    calibrated_score=4.0,
                    rationale="Deterministic workflow score.",
                ),
            ),
            overall_score=0.8,
            judge_economics=OperationEconomics(),
        )


def read_rollout(store, rollout_id):  # noqa: ANN001, ANN201
    """Import the verified rollout helper lazily for the focused fake judge."""
    from wmo.common.evaluations.evidence import read_rollout as read

    return read(store, rollout_id)


class _SimulatorFactory:
    """Bind the real text simulator to deterministic explicit fake model clients."""

    def __init__(self) -> None:
        """Initialize deterministic candidate and world-model simulators."""
        self.log: list[tuple[tuple[str, ...], bool]] = []
        self.candidate = _ScriptedClient(
            [_response("Resolved.", snapshot=_snapshot("candidate-a"), cost=0.01)] * 80
        )
        self.world = _ScriptedClient(
            [
                _response(
                    '{"message":"Done.","terminal":true}',
                    snapshot=_snapshot("world-model-a"),
                    cost=0.01,
                )
            ]
            * 80
        )

    def __call__(self, project: ProjectStore, plan: EvaluationPlan) -> Simulator:
        """Bind the real text simulator to the project's exact completed fit RAG.

        Args:
            project: Project owning the completed immutable build graph.
            plan: Frozen evaluation plan selected for simulation.

        Returns:
            Logging adapter around the grounded text simulator.
        """
        completed = project.load_project().build
        assert completed is not None
        fit_retriever = load_fit_rag_retriever(project.artifacts, completed.fit_rag)
        world_model = _resolved("world-model-a", cast(ModelClient, self.world))
        simulator = WorldModelSimulator(
            store=project.artifacts,
            evaluation_plan=plan,
            evaluation_plan_input=artifact_input(project.artifacts.read(plan.plan_id).manifest),
            task_set_input=artifact_input(project.artifacts.read(plan.task_set_id).manifest),
            fit_rag_input=completed.fit_rag,
            fit_retriever=fit_retriever,
            candidate_models={
                "candidate-a": _resolved("candidate-a", cast(ModelClient, self.candidate))
            },
            world_models={"world-model-a": world_model},
            grounded_world_models={
                "world-model-a": bind_fit_grounded_world_model(
                    project.artifacts,
                    completed.world_model,
                    client=world_model.client,
                    fit_retriever=fit_retriever,
                )
            },
            agent_factory=_OneTurnAgent,
            clock=lambda: _TIME,
            monotonic=lambda: 1.0,
        )
        return _LoggingSimulator(project, simulator, self.log)


class _LoggingSimulator:
    """Record phase cell IDs and lock state before delegating to the real simulator."""

    def __init__(
        self,
        project: ProjectStore,
        simulator: WorldModelSimulator,
        log: list[tuple[tuple[str, ...], bool]],
    ) -> None:
        """Store the delegated simulator and phase observation log."""
        self.project = project
        self.simulator = simulator
        self.log = log

    def run(self, spec: SimulationSpec) -> SimulationArtifactSet:
        """Record policy-lock state and delegate one simulation phase."""
        locked = any(
            artifact_id.startswith("router-policy-lock-")
            for artifact_id in self.project.artifacts.list_ids()
        )
        self.log.append((spec.cell_ids, locked))
        return self.simulator.run(spec)


class _TargetedTransportFailureClient(_ScriptedClient):
    """Scripted second candidate whose dispatch for one exact task always fails in transport."""

    def __init__(self, responses: list[ModelResponse], failing_instruction: str) -> None:
        """Store scripted answers plus the instruction whose dispatch never succeeds.

        Args:
            responses: Responses served in order for every other task.
            failing_instruction: Exact task instruction whose dispatch always fails.
        """
        super().__init__(responses)
        self._failing_instruction = failing_instruction
        self.failures = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Fail the targeted task at the transport level and answer every other task.

        Args:
            request: Candidate request emitted by the recording boundary.

        Returns:
            The next scripted response for a non-targeted task.

        Raises:
            ProviderTransportError: Every dispatch that carries the targeted instruction.
        """
        if request.messages[0].content == self._failing_instruction:
            self.requests.append(request)
            self.failures += 1
            raise ProviderTransportError("connection reset by provider")
        return super().complete(request)


class _CapExhaustedSimulatorFactory(_SimulatorFactory):
    """Simulator factory whose second candidate persistently fails one task's dispatch."""

    def __init__(self, failing_instruction: str) -> None:
        """Target one task with a persistent second-candidate transport failure.

        Args:
            failing_instruction: Exact task instruction whose dispatch always fails.
        """
        super().__init__()
        self.candidate = _ScriptedClient(
            [_response("Resolved.", snapshot=_snapshot("candidate-a"), cost=0.01)] * 200
        )
        self.failing = _TargetedTransportFailureClient(
            [_response("Resolved.", snapshot=_snapshot("candidate-b"), cost=0.01)] * 200,
            failing_instruction,
        )
        self.world = _ScriptedClient(
            [
                _response(
                    '{"message":"Done.","terminal":true}',
                    snapshot=_snapshot("world-model-a"),
                    cost=0.01,
                )
            ]
            * 400
        )

    def __call__(self, project: ProjectStore, plan: EvaluationPlan) -> Simulator:
        """Bind the grounded text simulator to the project's frozen completion contract.

        Args:
            project: Project owning the completed immutable build graph.
            plan: Frozen evaluation plan selected for simulation.

        Returns:
            Logging adapter around the reservation-bound grounded text simulator.
        """
        completed = project.load_project().build
        assert completed is not None
        fit_retriever = load_fit_rag_retriever(project.artifacts, completed.fit_rag)
        world_model = _priced_resolved("world-model-a", cast(ModelClient, self.world))
        contract_id = next(
            artifact_id
            for artifact_id in project.artifacts.list_ids()
            if project.artifacts.read(artifact_id).manifest.artifact_type
            == "simulation-completion-contract"
        )
        simulator = WorldModelSimulator(
            store=project.artifacts,
            evaluation_plan=plan,
            evaluation_plan_input=artifact_input(project.artifacts.read(plan.plan_id).manifest),
            task_set_input=artifact_input(project.artifacts.read(plan.task_set_id).manifest),
            fit_rag_input=completed.fit_rag,
            fit_retriever=fit_retriever,
            candidate_models={
                "candidate-a": _priced_resolved("candidate-a", cast(ModelClient, self.candidate)),
                "candidate-b": _priced_resolved("candidate-b", cast(ModelClient, self.failing)),
            },
            world_models={"world-model-a": world_model},
            grounded_world_models={
                "world-model-a": bind_fit_grounded_world_model(
                    project.artifacts,
                    completed.world_model,
                    client=world_model.client,
                    fit_retriever=fit_retriever,
                )
            },
            agent_factory=_OneTurnAgent,
            completion_contract_input=artifact_input(project.artifacts.read(contract_id).manifest),
            clock=lambda: _TIME,
            monotonic=lambda: 1.0,
        )
        return _LoggingSimulator(project, simulator, self.log)


def test_composition_rejects_fit_rag_outside_completed_build_before_simulation(
    tmp_path: Path,
) -> None:
    """Reject an unrelated setup fit RAG before simulation or model dispatch.

    Args:
        tmp_path: Isolated project root for build and setup artifacts.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)), issues=()
    )
    _bind_completed_build(project, normalized, revision="test-revision")
    simulator = _SimulatorFactory()
    judge = _Judge()
    services = RouterWorkflowServices(
        review_supplier=_ReviewSupplier(),
        setup_supplier=_MismatchedSetupSupplier(),
        simulator_factory=simulator,
        judge=judge,
        runtime_catalog=cast(
            RuntimeModelCatalog,
            _Catalog(
                {
                    "candidate-a": _snapshot("candidate-a"),
                    "embedder": _snapshot("embedder"),
                },
                _Client(),
            ),
        ),
    )

    with pytest.raises(RouterCompositionError, match="differs from the completed project build"):
        compose_router(
            project,
            normalized,
            services=services,
            budget=RouterCompositionBudget(
                maximum_simulation_cost_usd=10.0,
                maximum_judgments=100,
            ),
            created_at=_TIME,
            code_revision="test-revision",
        )

    artifact_types = {
        project.artifacts.read(artifact_id).manifest.artifact_type
        for artifact_id in project.artifacts.list_ids()
    }
    assert not artifact_types.intersection(
        {"simulation-spec", "simulation-resolution", "simulation-artifact-set"}
    )
    assert simulator.log == []
    assert simulator.candidate.requests == []
    assert simulator.world.requests == []
    assert judge.calls == 0


def test_public_composition_runs_and_resumes_complete_frozen_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach sticky runtime and resume without duplicate provider dispatch.

    Args:
        tmp_path: Isolated project root for composed router artifacts.
        monkeypatch: Patch fixture used to bind the runtime loader.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)), issues=()
    )
    completed_build = _bind_completed_build(project, normalized, revision="test-revision")
    simulator = _SimulatorFactory()
    judge = _Judge()
    runtime_client = _Client()
    runtime_catalog = cast(
        RuntimeModelCatalog,
        _Catalog(
            {
                "candidate-a": _snapshot("candidate-a"),
                "embedder": _snapshot("embedder"),
            },
            runtime_client,
        ),
    )
    phases = []
    services = RouterWorkflowServices(
        review_supplier=_ReviewSupplier(),
        setup_supplier=_SetupSupplier(),
        simulator_factory=simulator,
        judge=judge,
        runtime_catalog=runtime_catalog,
    )
    budget = RouterCompositionBudget(
        maximum_simulation_cost_usd=10.0,
        maximum_judgments=100,
    )
    attempts = []
    emitted = []
    receipts: set[str] = set()

    def capture(event, completion_id, properties, *, root):  # noqa: ANN001, ANN202
        """Record idempotent telemetry attempts for composition replay."""
        attempt = (event, completion_id, properties, root)
        attempts.append(attempt)
        if completion_id in receipts:
            return False
        receipts.add(completion_id)
        emitted.append(attempt)
        return True

    monkeypatch.setattr("wmo.optimize.router.composition.capture_completion_once", capture)

    crash_after_policy = True
    crash_after_report = True

    def crash_after_lock(phase: str) -> None:
        """Crash once after each durable lock and report boundary."""
        nonlocal crash_after_policy, crash_after_report
        phases.append(phase)
        if phase == "policy_locked" and crash_after_policy:
            crash_after_policy = False
            raise RuntimeError("simulated crash after policy lock")
        if phase == "report_complete" and crash_after_report:
            crash_after_report = False
            raise RuntimeError("simulated crash after telemetry receipt")

    with pytest.raises(RuntimeError, match="after policy lock"):
        compose_router(
            project,
            normalized,
            services=services,
            budget=budget,
            created_at=_TIME,
            code_revision="test-revision",
            phase_hook=crash_after_lock,
        )
    from wmo.optimize.router import composition as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "fit_router",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fit repeated")),
    )
    with pytest.raises(RuntimeError, match="after telemetry receipt"):
        compose_router(
            project,
            normalized,
            services=services,
            budget=budget,
            created_at=_TIME + timedelta(hours=1),
            code_revision="test-revision",
            phase_hook=crash_after_lock,
        )
    assert len(emitted) == 1
    first = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=_TIME + timedelta(hours=2),
        code_revision="test-revision",
        phase_hook=crash_after_lock,
    )
    dispatched = (
        len(simulator.log),
        len(simulator.candidate.requests),
        len(simulator.world.requests),
        judge.calls,
    )
    second = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=_TIME + timedelta(hours=3),
        code_revision="test-revision",
    )

    assert second.optimization == first.optimization
    assert completed_build.fit_rag in first.simulation_spec.inputs
    assert completed_build.fit_rag in first.held_out_simulation_spec.inputs
    assert first.held_out_simulation_spec.maximum_cost_usd == pytest.approx(
        budget.maximum_simulation_cost_usd
    )
    assert first.total_simulation_spend_usd == pytest.approx(
        first.fit_simulation_spend_usd + first.held_out_simulation_spend_usd
    )
    assert first.total_simulation_spend_usd <= budget.maximum_simulation_cost_usd
    assert (
        len(simulator.log),
        len(simulator.candidate.requests),
        len(simulator.world.requests),
        judge.calls,
    ) == dispatched
    assert phases.index("fit_started") < phases.index("policy_locked")
    assert phases.index("policy_locked") < phases.index("heldout_opened")
    assert phases.index("heldout_opened") < phases.index("report_complete")
    assert len(attempts) == 3
    assert len(emitted) == 1
    assert attempts[0][0] == "wmo simulation completed"
    assert all(
        completion_id == first.optimization.optimization.report.report_id
        for _event, completion_id, _properties, _root in attempts
    )
    assert all(
        event_root == tmp_path for _event, _completion_id, _properties, event_root in attempts
    )
    assert all(
        properties["rollout_count"]
        == len(first.simulation_spec.cell_ids) + len(first.held_out_simulation_spec.cell_ids)
        for _event, _completion_id, properties, _root in attempts
    )
    assert all(
        properties["cost_usd"] == pytest.approx(first.total_simulation_spend_usd)
        for _event, _completion_id, properties, _root in attempts
    )
    purposes = {cell.cell_id: cell.purpose for cell in first.plan.cells}
    assert all(
        locked or all(purposes[cell_id] == "fit" for cell_id in cell_ids)
        for cell_ids, locked in simulator.log
    )
    assert all(locked or purposes.get(cell_id) != "held_out" for cell_id, locked in judge.log)
    dispatches = tuple(
        JudgmentDispatchReceipt.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "dispatch.json")
        )
        for artifact_id in project.artifacts.list_ids()
        if project.artifacts.read(artifact_id).manifest.artifact_type == "judgment-dispatch"
    )
    assert len(dispatches) == judge.calls <= budget.maximum_judgments
    assert {purposes[item.cell_id] for item in dispatches} >= {"fit", "held_out"}

    near_cap_simulator = _SimulatorFactory()
    near_cap_services = RouterWorkflowServices(
        review_supplier=services.review_supplier,
        setup_supplier=services.setup_supplier,
        simulator_factory=near_cap_simulator,
        judge=judge,
        runtime_catalog=runtime_catalog,
    )
    calls_before_near_cap = judge.calls

    def exact_cap_spend(
        phase_project: ProjectStore,
        artifact_set: SimulationArtifactSet,
        phase_setup: RouterEvaluationSetup,
    ) -> float:
        """Return spend that exactly exhausts the admitted phase budget."""
        del phase_project, artifact_set, phase_setup
        return 1.0

    monkeypatch.setattr(workflow_module, "_verified_simulation_spend", exact_cap_spend)
    with pytest.raises(RouterCompositionError, match="reached the shared ceiling"):
        compose_router(
            project,
            normalized,
            services=near_cap_services,
            budget=RouterCompositionBudget(
                maximum_simulation_cost_usd=1.0,
                maximum_judgments=100,
                stop_on_overspend=True,
            ),
            created_at=_TIME,
            code_revision="test-revision",
        )
    assert len(near_cap_simulator.log) == 1
    assert judge.calls == calls_before_near_cap

    fit_set = next(
        SimulationArtifactSet.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "artifact-set.json")
        )
        for artifact_id in project.artifacts.list_ids()
        if project.artifacts.read(artifact_id).manifest.artifact_type == "simulation-artifact-set"
        and SimulationArtifactSet.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "artifact-set.json")
        ).simulation_id
        == first.simulation_spec.simulation_id
    )
    rollout = read_rollout(project.artifacts, fit_set.artifact_ids[0])[0]
    unknown = rollout.model_copy(update={"candidate_economics": OperationEconomics()})
    with pytest.raises(RouterCompositionError, match="not fully observed"):
        observed_rollout_spend(unknown)
    assert rollout.candidate_economics.cost_usd is not None
    assert rollout.world_model_economics is not None
    assert rollout.world_model_economics.cost_usd is not None
    reservation_derived = rollout.model_copy(
        update={
            "candidate_economics": rollout.candidate_economics.model_copy(
                update={
                    "cost_usd": rollout.candidate_economics.cost_usd.model_copy(
                        update={"provenance": "estimated"}
                    )
                }
            ),
            "world_model_economics": rollout.world_model_economics.model_copy(
                update={
                    "cost_usd": rollout.world_model_economics.cost_usd.model_copy(
                        update={"provenance": "estimated"}
                    )
                }
            ),
        }
    )
    assert observed_rollout_spend(reservation_derived) == pytest.approx(
        observed_rollout_spend(rollout)
    )

    decision = first.runtime.select(_request(), episode_id="customer-episode")
    assert first.runtime.select(_request(), episode_id="customer-episode") == decision
    assert first.plan.cells


def test_failed_rollouts_skip_judging_and_rerun_replays_after_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed rollouts stay unjudged and a later-clock rerun resumes without redispatch.

    Args:
        tmp_path: Isolated project root for composed router artifacts.
        monkeypatch: Patch fixture used to inject admission failures and one crash.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)), issues=()
    )
    _bind_completed_build(project, normalized, revision="test-revision")
    simulator = _SimulatorFactory()
    judge = _Judge()
    services = RouterWorkflowServices(
        review_supplier=_ReviewSupplier(),
        setup_supplier=_SetupSupplier(),
        simulator_factory=simulator,
        judge=judge,
        runtime_catalog=cast(
            RuntimeModelCatalog,
            _Catalog(
                {
                    "candidate-a": _snapshot("candidate-a"),
                    "embedder": _snapshot("embedder"),
                },
                _Client(),
            ),
        ),
    )
    budget = RouterCompositionBudget(
        maximum_simulation_cost_usd=10.0,
        maximum_judgments=100,
    )
    real_admission = text_simulator_module.episode_reservation_failure
    admissions = {"count": 0}

    def failing_admission(
        settings: WorldModelSettings,
        *,
        completion_contract: SimulationCompletionContract | None,
        remaining_cost_usd: float,
        stop_on_overspend: bool = True,
    ) -> StructuredFailure | None:
        """Reject the first two held-out episode admissions before any provider dispatch."""
        locked = any(
            artifact_id.startswith("router-policy-lock-")
            for artifact_id in project.artifacts.list_ids()
        )
        if locked:
            admissions["count"] += 1
        if locked and admissions["count"] <= 2:
            return StructuredFailure(
                code=FailureCode.BUDGET,
                message="reconciled provider spend has exhausted the remaining simulation ceiling",
                attribution=FailureAttribution.MODEL,
                details={"phase": "episode_provider_spend"},
            )
        return real_admission(
            settings,
            completion_contract=completion_contract,
            remaining_cost_usd=remaining_cost_usd,
            stop_on_overspend=stop_on_overspend,
        )

    monkeypatch.setattr(text_simulator_module, "episode_reservation_failure", failing_admission)
    real_persist_set = text_simulator_module.persist_artifact_set

    def crash_before_artifact_set(*args: object, **kwargs: object) -> SimulationArtifactSet:
        """Simulate one crash after rollouts persist but before their set persists."""
        del args, kwargs
        raise RuntimeError("simulated crash before artifact-set persistence")

    monkeypatch.setattr(text_simulator_module, "persist_artifact_set", crash_before_artifact_set)
    with pytest.raises(RuntimeError, match="before artifact-set persistence"):
        compose_router(
            project,
            normalized,
            services=services,
            budget=budget,
            created_at=_TIME,
            code_revision="test-revision",
        )
    monkeypatch.setattr(text_simulator_module, "persist_artifact_set", real_persist_set)

    first = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        code_revision="test-revision",
    )

    simulated_cells = tuple(
        cell
        for cell in first.plan.cells
        if cell.execution == "simulate" and cell.purpose in {"fit", "held_out"}
    )
    assert len(simulator.candidate.requests) == len(simulated_cells) - 2
    assert len(simulator.world.requests) == len(simulated_cells) - 2
    held_set = next(
        SimulationArtifactSet.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "artifact-set.json")
        )
        for artifact_id in project.artifacts.list_ids()
        if project.artifacts.read(artifact_id).manifest.artifact_type == "simulation-artifact-set"
        and SimulationArtifactSet.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "artifact-set.json")
        ).simulation_id
        == first.held_out_simulation_spec.simulation_id
    )
    failed_cells = {
        rollout.cell_id
        for rollout in (
            read_rollout(project.artifacts, rollout_id)[0] for rollout_id in held_set.artifact_ids
        )
        if rollout.failure is not None
    }
    assert len(failed_cells) == 2
    judged_cells = {cell_id for cell_id, _locked in judge.log}
    assert failed_cells.isdisjoint(judged_cells)
    dispatches = tuple(
        JudgmentDispatchReceipt.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "dispatch.json")
        )
        for artifact_id in project.artifacts.list_ids()
        if project.artifacts.read(artifact_id).manifest.artifact_type == "judgment-dispatch"
    )
    assert failed_cells.isdisjoint({item.cell_id for item in dispatches})
    scored_cells = tuple(cell for cell in first.plan.cells if cell.purpose in {"fit", "held_out"})
    assert judge.calls == len(dispatches) == len(scored_cells) - 2

    dispatched = (
        len(simulator.log),
        len(simulator.candidate.requests),
        len(simulator.world.requests),
        judge.calls,
    )
    second = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=datetime(2026, 8, 14, tzinfo=UTC),
        code_revision="test-revision",
    )
    assert second.optimization == first.optimization
    assert (
        len(simulator.log),
        len(simulator.candidate.requests),
        len(simulator.world.requests),
        judge.calls,
    ) == dispatched


def test_retry_cap_exhausted_cell_binds_failed_evidence_and_replays_without_dispatch(
    tmp_path: Path,
) -> None:
    """A cap-exhausted transport-failed cell covers its plan entry as excluded evidence.

    Each resume below the retry cap re-executes the failing cell and still aborts at exact
    plan coverage. Once the final permitted generation fails, evidence assembly binds the
    terminal failed rollout, the fitter sees the cell only as a failed unjudged row, its
    unknown spend stays conservatively charged, and replay adds no provider dispatches.

    Args:
        tmp_path: Isolated project root for composed router artifacts.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)), issues=()
    )
    completed = _bind_completed_build(project, normalized, revision="test-revision")
    tasks = load_task_set(project.artifacts, completed.task_set.artifact_id).tasks
    failing_task = tuple(task for task in tasks if task.partition == "fit")[10]
    simulator = _CapExhaustedSimulatorFactory(failing_task.instruction)
    judge = _Judge()
    services = RouterWorkflowServices(
        review_supplier=_ReviewSupplier(),
        setup_supplier=_ReservedSetupSupplier(),
        simulator_factory=simulator,
        judge=judge,
        runtime_catalog=cast(
            RuntimeModelCatalog,
            _Catalog(
                {
                    "candidate-a": _snapshot("candidate-a"),
                    "candidate-b": _snapshot("candidate-b"),
                    "embedder": _snapshot("embedder"),
                },
                _Client(),
            ),
        ),
    )
    budget = RouterCompositionBudget(
        maximum_simulation_cost_usd=10.0,
        maximum_judgments=400,
    )

    for hour in range(MAXIMUM_CELL_ATTEMPTS - 1):
        with pytest.raises(ValueError, match="cover the plan exactly"):
            compose_router(
                project,
                normalized,
                services=services,
                budget=budget,
                created_at=_TIME + timedelta(hours=hour),
                code_revision="test-revision",
            )
    first = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=_TIME + timedelta(hours=MAXIMUM_CELL_ATTEMPTS),
        code_revision="test-revision",
    )

    failing_cell_ids = {
        cell.cell_id
        for cell in first.plan.cells
        if cell.task_id == failing_task.task_id
        and cell.candidate_alias == "candidate-b"
        and cell.purpose == "fit"
        and cell.execution == "simulate"
    }
    assert failing_cell_ids
    assert simulator.failing.failures == MAXIMUM_CELL_ATTEMPTS * len(failing_cell_ids)
    terminal = {
        rollout.cell_id: rollout
        for artifact_set in (
            SimulationArtifactSet.model_validate_json(
                project.artifacts.read_bytes(artifact_id, "artifact-set.json")
            )
            for artifact_id in project.artifacts.list_ids()
            if project.artifacts.read(artifact_id).manifest.artifact_type
            == "simulation-artifact-set"
        )
        for rollout in (
            read_rollout(project.artifacts, rollout_id)[0]
            for rollout_id in artifact_set.artifact_ids
        )
        if rollout.cell_id in failing_cell_ids
        and rollout.retry_attempt == MAXIMUM_CELL_ATTEMPTS - 1
    }
    assert set(terminal) == failing_cell_ids
    for rollout in terminal.values():
        assert rollout.stop_reason == StopReason.FAILURE
        assert rollout.failure is not None
        assert rollout.failure.retryable is True
        assert rollout.failure.details["provider_dispatch_unknown_spend"] is True
        assert observed_rollout_spend(rollout) > 0
    fit_dataset = load_evaluation_dataset(project.artifacts, first.optimization.fit_evaluation_id)
    failed_rows = tuple(row for row in fit_dataset.rows if row.cell_id in failing_cell_ids)
    assert len(failed_rows) == len(failing_cell_ids)
    assert all(row.status == "failed" and row.judgment_id is None for row in failed_rows)
    assert failing_cell_ids.isdisjoint({cell_id for cell_id, _locked in judge.log})

    dispatched = (
        len(simulator.log),
        len(simulator.candidate.requests),
        len(simulator.failing.requests),
        len(simulator.world.requests),
        judge.calls,
    )
    second = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=_TIME + timedelta(hours=MAXIMUM_CELL_ATTEMPTS + 1),
        code_revision="test-revision",
    )
    assert second.optimization == first.optimization
    assert (
        len(simulator.log),
        len(simulator.candidate.requests),
        len(simulator.failing.requests),
        len(simulator.world.requests),
        judge.calls,
    ) == dispatched


def test_rerun_completes_a_reserved_judgment_dispatch_left_without_a_judgment(
    tmp_path: Path,
) -> None:
    """A crash between dispatch reservation and judgment persistence resumes on rerun.

    The rerun uses a fresh judge, as a new process would, so it also proves judgment dispatch
    works when fit simulation replays without a simulator.

    Args:
        tmp_path: Isolated project root for composed router artifacts.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)), issues=()
    )
    _bind_completed_build(project, normalized, revision="test-revision")
    simulator = _SimulatorFactory()
    catalog = cast(
        RuntimeModelCatalog,
        _Catalog(
            {
                "candidate-a": _snapshot("candidate-a"),
                "embedder": _snapshot("embedder"),
            },
            _Client(),
        ),
    )

    def _services(judge: _Judge) -> RouterWorkflowServices:
        """Return workflow services for one composition attempt.

        Args:
            judge: Judge double for one composition attempt.

        Returns:
            Workflow services mirroring the automatic service wiring.
        """
        return RouterWorkflowServices(
            review_supplier=_ReviewSupplier(),
            setup_supplier=_SetupSupplier(),
            simulator_factory=simulator,
            judge=judge,
            runtime_catalog=catalog,
        )

    first_judge = _Judge()
    first_judge.fail_on_call = 1
    budget = RouterCompositionBudget(
        maximum_simulation_cost_usd=10.0,
        maximum_judgments=100,
    )
    with pytest.raises(RuntimeError, match="simulated judgment dispatch interruption"):
        compose_router(
            project,
            normalized,
            services=_services(first_judge),
            budget=budget,
            created_at=_TIME,
            code_revision="test-revision",
        )

    def _artifact_ids(artifact_type: str) -> tuple[str, ...]:
        """Return persisted artifact IDs of one exact manifest type.

        Args:
            artifact_type: Exact immutable manifest artifact type.

        Returns:
            Matching persisted artifact identifiers.
        """
        return tuple(
            artifact_id
            for artifact_id in project.artifacts.list_ids()
            if project.artifacts.read(artifact_id).manifest.artifact_type == artifact_type
        )

    orphaned = _artifact_ids("judgment-dispatch")
    assert len(orphaned) == 1
    assert not _artifact_ids("judgment")
    assert first_judge.calls == 1

    second_judge = _Judge()
    result = compose_router(
        project,
        normalized,
        services=_services(second_judge),
        budget=budget,
        created_at=datetime(2026, 8, 14, tzinfo=UTC),
        code_revision="test-revision",
    )

    scored_cells = tuple(cell for cell in result.plan.cells if cell.purpose in {"fit", "held_out"})
    receipts = _artifact_ids("judgment-dispatch")
    assert set(orphaned) <= set(receipts)
    assert len(receipts) == len(scored_cells)
    assert len(_artifact_ids("judgment")) == len(scored_cells)
    assert second_judge.calls == len(scored_cells)


def test_per_cell_judge_failures_exclude_cells_durably_and_replay_without_redispatch(
    tmp_path: Path,
) -> None:
    """Over-ceiling and exhausted-retry judge failures exclude single cells, never the run.

    Args:
        tmp_path: Isolated project root for composed router artifacts.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)), issues=()
    )
    _bind_completed_build(project, normalized, revision="test-revision")
    simulator = _SimulatorFactory()
    judge = _Judge()
    judge.raise_after_lock = [
        JudgeTranscriptAdmissionError("judge request exceeds its reserved input-token ceiling"),
        JudgeDispatchExhaustedError(
            "OpenAI Responses output has no text or tool call",
            conservative_cost_usd=0.25,
        ),
    ]
    services = RouterWorkflowServices(
        review_supplier=_ReviewSupplier(),
        setup_supplier=_SetupSupplier(),
        simulator_factory=simulator,
        judge=judge,
        runtime_catalog=cast(
            RuntimeModelCatalog,
            _Catalog(
                {
                    "candidate-a": _snapshot("candidate-a"),
                    "embedder": _snapshot("embedder"),
                },
                _Client(),
            ),
        ),
    )
    budget = RouterCompositionBudget(
        maximum_simulation_cost_usd=10.0,
        maximum_judgments=100,
    )

    first = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=_TIME,
        code_revision="test-revision",
    )

    def _artifact_ids(artifact_type: str) -> tuple[str, ...]:
        """Return persisted artifact IDs of one exact manifest type.

        Args:
            artifact_type: Exact immutable manifest artifact type.

        Returns:
            Matching persisted artifact identifiers.
        """
        return tuple(
            artifact_id
            for artifact_id in project.artifacts.list_ids()
            if project.artifacts.read(artifact_id).manifest.artifact_type == artifact_type
        )

    exclusions = tuple(
        JudgmentExclusionRecord.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "exclusion.json")
        )
        for artifact_id in _artifact_ids("judgment-exclusion")
    )
    assert {record.reason for record in exclusions} == {
        "transcript_exceeds_judge_admission_ceiling",
        "judge_dispatch_failed",
    }
    assert {record.cell_id for record in exclusions} <= {
        cell.cell_id for cell in first.plan.cells if cell.purpose == "held_out"
    }
    costs_by_reason = {record.reason: record.conservative_cost_usd for record in exclusions}
    assert costs_by_reason["transcript_exceeds_judge_admission_ceiling"] == 0.0
    assert costs_by_reason["judge_dispatch_failed"] == 0.25
    scored_cells = tuple(cell for cell in first.plan.cells if cell.purpose in {"fit", "held_out"})
    assert len(_artifact_ids("judgment")) == len(scored_cells) - 2
    assert len(_artifact_ids("judgment-dispatch")) == len(scored_cells)
    excluded_cells = {record.cell_id for record in exclusions}
    judged_cells = {cell_id for cell_id, _locked in judge.log}
    assert excluded_cells.isdisjoint(judged_cells)

    dispatched = judge.calls
    second = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=datetime(2026, 8, 14, tzinfo=UTC),
        code_revision="test-revision",
    )

    assert second.optimization == first.optimization
    assert judge.calls == dispatched
    assert len(_artifact_ids("judgment-exclusion")) == len(exclusions)
