"""W16 deterministic customer-path evidence over the public router composition API."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

import wmo
from wmo.common.evaluations import EvaluationPlan, EvaluationProtocol, ObservedProductionCell
from wmo.common.evaluations.build_test import _persist_rollout, _production_rollout
from wmo.common.judging import (
    DimensionScoreMap,
    JudgeCalibration,
    Judgment,
    Rubric,
    RubricDimension,
    ScoreAnchor,
)
from wmo.common.models import (
    AssistantAction,
    CandidateTokenPrice,
    EmbeddingCostReservation,
    ModelClient,
    ModelFinishReason,
    ModelRequest,
    ModelResponse,
    NumericMeasurement,
    OperationEconomics,
    PricingSnapshot,
    RoutedCandidateSnapshot,
    ToolCall,
    Usage,
)
from wmo.common.project import ProjectConfig, ProjectStore, artifact_input
from wmo.common.routing import KnnGuard
from wmo.common.tasks import load_task_set
from wmo.optimize.router.composition import (
    ApprovedRouterReview,
    RouterCompositionBudget,
    RouterEvaluationSetup,
    RouterWorkflowServices,
)
from wmo.optimize.router.composition_test import (
    _bind_completed_build,
    _capabilities,
    _Judge,
    _resolved,
    _snapshot,
)
from wmo.optimize.router.fit.workflow_test import _persist_embeddings
from wmo.optimize.router.judgment_budget import JudgmentDispatchReceipt
from wmo.release_revision_test import exact_checkout_revision, verify_release_evidence
from wmo.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from wmo.runtime.router.application import create_project_router_app
from wmo.simulation.engines.text.simulator import WorldModelSimulator
from wmo.simulation.engines.text.simulator_test import (
    _OneTurnAgent,
    _persist_completion_contract,
    _response,
    _ScriptedClient,
)
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.orchestration import Simulator
from wmo.simulation.retrieval import load_fit_rag_retriever, load_rag_index
from wmo.simulation.retrieval.retrieval_test import _message_trace as _trace
from wmo.simulation.specs import WorldModelSettings
from wmo.simulation.world_model import bind_fit_grounded_world_model

_TIME = datetime(2026, 8, 12, tzinfo=UTC)


class _EvidenceSetupSupplier:
    """Persist two measured candidates, reviewed overlaps, embeddings, and pricing."""

    def __init__(self, revision: str) -> None:
        """Store the exact release revision used by persisted evidence."""
        self.revision = revision

    def __call__(
        self,
        project: ProjectStore,
        build: wmo.ProjectBuild,
        review: ApprovedRouterReview,
        budget: RouterCompositionBudget,
    ) -> RouterEvaluationSetup:
        """Persist reviewed evidence bound to the project's completed fit RAG.

        Args:
            project: Project owning the completed build and release evidence.
            build: Deterministic trace and task-set build reused by composition.
            review: Approved rubric and calibration identifiers.
            budget: Finite composition budget already validated by the workflow.

        Returns:
            Complete release evaluation setup over the exact completed fit RAG.
        """
        del review, budget
        completed = project.load_project().build
        assert completed is not None
        fit_index = load_rag_index(project.artifacts, completed.fit_rag.artifact_id).index
        tasks = load_task_set(project.artifacts, build.artifacts.task_set.task_set_id).tasks
        candidates = tuple(
            RoutedCandidateSnapshot(alias=alias, model=_snapshot(alias))
            for alias in ("candidate-baseline", "candidate-economy")
        )
        fit_tasks = tuple(task for task in tasks if task.partition == "fit")
        observed = []
        for index, task in enumerate(fit_tasks[:10]):
            rollout_id = f"w16-production-{index:02d}"
            if rollout_id not in project.artifacts.list_ids():
                rollout = _production_rollout(
                    rollout_id,
                    cell_id=f"w16-import-{index}",
                    task=task,
                    candidate=candidates[0].model,
                ).model_copy(
                    update={"trace_id": task.source_trace_ids[0], "code_revision": self.revision}
                )
                _persist_rollout(project.artifacts, rollout)
            observed.append(
                ObservedProductionCell(
                    task_id=task.task_id,
                    candidate_alias="candidate-baseline",
                    repeat=0,
                    rollout_artifact_id=rollout_id,
                )
            )
        if "w16-pricing" not in project.artifacts.list_ids():
            pricing = PricingSnapshot(
                schema_version=1,
                created_at=_TIME,
                code_revision=self.revision,
                pricing_snapshot_id="w16-pricing",
                candidate_prices=(
                    CandidateTokenPrice(
                        candidate_alias="candidate-baseline",
                        input_usd_per_million_tokens=2.0,
                        output_usd_per_million_tokens=4.0,
                    ),
                    CandidateTokenPrice(
                        candidate_alias="candidate-economy",
                        input_usd_per_million_tokens=1.0,
                        output_usd_per_million_tokens=2.0,
                    ),
                ),
            )
            project.artifacts.write_json(
                artifact_id=pricing.pricing_snapshot_id,
                artifact_type="pricing-snapshot",
                envelope=pricing,
                files={"pricing.json": pricing},
            )
        embedding_set_id = _persist_embeddings(
            project.artifacts,
            tasks,
            task_set_input=build.review.task_set,
            embedder=_snapshot("embedder"),
            created_at=_TIME,
            code_revision=self.revision,
        )
        production = EvaluationProtocol(
            protocol_id="w16-production-protocol",
            evidence_source="production",
            agent_id="agent-a",
            simulator_id="production-import-v1",
            rubric_id="rubric-a",
            judge_calibration_id="calibration-a",
            pricing_snapshot_id="w16-pricing",
        )
        world = EvaluationProtocol(
            protocol_id="w16-world-protocol",
            evidence_source="world_model",
            agent_id="agent-a",
            simulator_id="text-world-model-v1",
            world_model=_snapshot("world-model-a"),
            simulator_prompt_id="world-model-text-v1",
            rubric_id="rubric-a",
            judge_calibration_id="calibration-a",
            pricing_snapshot_id="w16-pricing",
        )
        return RouterEvaluationSetup(
            candidates=candidates,
            observed_cells=tuple(observed),
            production_protocol=production,
            simulation_protocol=world,
            embedding_set_id=embedding_set_id,
            fit_rag_input=completed.fit_rag,
            pricing_snapshot_id="w16-pricing",
            incumbent_alias="candidate-baseline",
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
            simulation_completion_input=_persist_completion_contract(
                project.artifacts,
                candidate_aliases=("candidate-baseline", "candidate-economy"),
                snapshot=_snapshot,
                code_revision=self.revision,
            ),
            agent_id="agent-a",
            seed=16,
            maximum_steps=2,
            maximum_concurrency=1,
        )


class _EvidenceSimulatorFactory:
    """Bind the actual text simulator to deterministic local model-client fakes."""

    def __init__(self, revision: str) -> None:
        """Initialize deterministic candidate clients and a call counter.

        Args:
            revision: Exact checkout revision bound to persisted reservations.
        """
        self.revision = revision
        self.calls = 0
        self.candidates = {
            "candidate-baseline": _ScriptedClient(
                [
                    _response(
                        "Resolved by baseline.",
                        snapshot=_snapshot("candidate-baseline"),
                        cost=0.0,
                        usage=Usage(input_tokens=0, output_tokens=0),
                    )
                ]
                * 100
            ),
            "candidate-economy": _ScriptedClient(
                [
                    _response(
                        "Resolved by economy.",
                        snapshot=_snapshot("candidate-economy"),
                        cost=0.0,
                        usage=Usage(input_tokens=0, output_tokens=0),
                    )
                ]
                * 100
            ),
        }
        self.world = _ScriptedClient(
            [
                _response(
                    '{"message":"Done.","terminal":true}',
                    snapshot=_snapshot("world-model-a"),
                    cost=0.0,
                    usage=Usage(input_tokens=0, output_tokens=0),
                )
            ]
            * 200
        )

    def __call__(self, project: ProjectStore, plan: EvaluationPlan) -> Simulator:
        """Bind the release simulator to the project's exact completed fit RAG.

        Args:
            project: Project owning the completed immutable build graph.
            plan: Frozen evaluation plan selected for simulation.

        Returns:
            Grounded text simulator with deterministic local model clients.
        """
        self.calls += 1
        completed = project.load_project().build
        assert completed is not None
        fit_retriever = load_fit_rag_retriever(project.artifacts, completed.fit_rag)
        world_model = _resolved("world-model-a", cast(ModelClient, self.world))
        return WorldModelSimulator(
            store=project.artifacts,
            evaluation_plan=plan,
            evaluation_plan_input=artifact_input(project.artifacts.read(plan.plan_id).manifest),
            task_set_input=artifact_input(project.artifacts.read(plan.task_set_id).manifest),
            fit_rag_input=completed.fit_rag,
            fit_retriever=fit_retriever,
            candidate_models={
                alias: _resolved(alias, cast(ModelClient, client))
                for alias, client in self.candidates.items()
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
            completion_contract_input=_persist_completion_contract(
                project.artifacts,
                candidate_aliases=("candidate-baseline", "candidate-economy"),
                snapshot=_snapshot,
                code_revision=self.revision,
            ),
            clock=lambda: _TIME,
            monotonic=lambda: 1.0,
        )


class _EvidenceJudge(_Judge):
    """Persist the deterministic judgment with exact revision and observed zero-dollar cost."""

    def __init__(self, revision: str) -> None:
        """Initialize the exact-revision deterministic judge."""
        super().__init__()
        self.revision = revision

    def judge_persisted(
        self,
        store: ProjectStore,
        *,
        rollout_artifact_id: str,
        rubric_artifact_id: str,
        calibration_artifact_id: str,
    ) -> Judgment:
        """Persist one zero-cost judgment under the exact release revision."""
        judgment = super().judge_persisted(
            store,
            rollout_artifact_id=rollout_artifact_id,
            rubric_artifact_id=rubric_artifact_id,
            calibration_artifact_id=calibration_artifact_id,
        )
        return judgment.model_copy(
            update={
                "code_revision": self.revision,
                "judge_economics": OperationEconomics(
                    cost_usd=NumericMeasurement(value=0.0, provenance="observed")
                ),
            }
        )


class _RuntimeClient:
    """Return one exact routed candidate identity while counting local calls."""

    def __init__(self, alias: str) -> None:
        """Initialize counters for one exact routed alias."""
        self.alias = alias
        self.complete_calls = 0
        self.embed_calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one deterministic tool call and record the dispatch."""
        del request
        self.complete_calls += 1
        return ModelResponse(
            output=AssistantAction(
                tool_calls=(ToolCall(call_id="w16-call", name="resolve", arguments={}),)
            ),
            model=_snapshot(self.alias),
            economics=OperationEconomics(
                usage=Usage(input_tokens=7, output_tokens=3, cached_input_tokens=0),
                cost_usd=NumericMeasurement(value=0.0, provenance="observed"),
            ),
            finish_reason=ModelFinishReason.COMPLETED,
        )

    def embed(self, texts) -> tuple:  # noqa: ANN001, ANN201 - protocol fixture
        """Return one deterministic embedding for one input text."""
        self.embed_calls += 1
        assert len(texts) == 1
        from wmo.common.models import Embedding

        return (Embedding(values=(1.0, 0.0)),)


class _RuntimeCatalog:
    """Resolve exact candidate and embedder clients without a provider or environment."""

    def __init__(self) -> None:
        """Create exact local clients for every release-evidence model."""
        self.clients = {
            alias: _RuntimeClient(alias)
            for alias in ("candidate-baseline", "candidate-economy", "embedder")
        }

    def snapshot(self, alias: str) -> tuple:
        """Return the exact snapshot and capabilities for one alias."""
        return _snapshot(alias), _capabilities(alias)

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        """Resolve one alias without environment or provider access."""
        del role
        snapshot, capabilities = self.snapshot(alias)
        client = self.clients[alias]
        return ResolvedModel(
            alias=alias,
            snapshot=snapshot,
            capabilities=capabilities,
            client=client,
            embedding_client=client if alias == "embedder" else None,
        )


def test_w16_public_router_evidence_is_complete_replay_safe_and_openai_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove replay safety, budgets, and HTTP stickiness end to end.

    Crash-and-resume at the durable phase boundaries is owned by
    composition_test.py::test_public_composition_runs_and_resumes_complete_frozen_router.

    Args:
        tmp_path: Isolated project root for composed router evidence.
        monkeypatch: Patch fixture used to bind runtime loading.
    """
    revision = exact_checkout_revision()
    project = ProjectStore(tmp_path, "w16-project")
    project.initialize(ProjectConfig(project_id="w16-project"))
    list_ids_calls = 0
    list_ids = project.artifacts.list_ids

    def count_list_ids() -> tuple[str, ...]:
        """Count artifact-directory scans across the complete evidence workflow."""
        nonlocal list_ids_calls
        list_ids_calls += 1
        return list_ids()

    monkeypatch.setattr(project.artifacts, "list_ids", count_list_ids)
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)),
        issues=(),
    )
    completed_build = _bind_completed_build(project, normalized, revision=revision)
    simulator = _EvidenceSimulatorFactory(revision)
    judge = _EvidenceJudge(revision)
    runtime_catalog = _RuntimeCatalog()
    services = RouterWorkflowServices(
        review_supplier=_EvidenceReviewSupplier(revision),
        setup_supplier=_EvidenceSetupSupplier(revision),
        simulator_factory=simulator,
        judge=judge,
        runtime_catalog=cast(RuntimeModelCatalog, runtime_catalog),
    )
    budget = RouterCompositionBudget(
        maximum_simulation_cost_usd=2.0,
        maximum_judgments=200,
    )
    telemetry_attempts: list[str] = []
    telemetry_delivered: set[str] = set()

    def capture(_event, completion_id, _properties, *, root):  # noqa: ANN001, ANN202
        """Record one local telemetry attempt with idempotent delivery."""
        assert root == tmp_path
        telemetry_attempts.append(completion_id)
        if completion_id in telemetry_delivered:
            return False
        telemetry_delivered.add(completion_id)
        return True

    monkeypatch.setattr("wmo.optimize.router.composition.capture_completion_once", capture)
    result = wmo.compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=_TIME,
        code_revision=revision,
    )
    dispatches_after_completion = _dispatch_counts(simulator, judge)
    replay = wmo.compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=_TIME,
        code_revision=revision,
    )

    tasks = load_task_set(project.artifacts, result.plan.task_set_id).tasks
    assert len(normalized.traces) == 100
    assert len(tuple(task for task in tasks if task.partition == "fit")) == 50
    assert len(tuple(task for task in tasks if task.partition == "held_out")) == 20
    assert result.build.review.status == "proposals_pending"
    assert result.review.rubric_id == "rubric-a"
    assert result.review.calibration_id == "calibration-a"
    assert result.policy_lock_id
    report = result.optimization.optimization.report
    assert len(report.held_out_task_ids) == 20
    assert report.coverage.planned_row_count == 40
    assert report.coverage.failed_row_count == 0
    assert report.coverage.not_run_row_count == 0
    assert report.source_strata
    assert result.build.review.paid_calls_made == 0
    assert result.fit_simulation_spend_usd == 0.0
    assert result.held_out_simulation_spend_usd == 0.0
    assert result.total_simulation_spend_usd == 0.0
    assert result.held_out_simulation_spec.maximum_cost_usd == (budget.maximum_simulation_cost_usd)
    assert completed_build.fit_rag in result.simulation_spec.inputs
    assert completed_build.fit_rag in result.held_out_simulation_spec.inputs
    assert report.run_spend.candidate.known_total_usd == 0.0
    assert report.run_spend.world_model.known_total_usd == 0.0
    assert report.run_spend.judge.known_total_usd == 0.0
    assert report.run_spend.candidate.missing_row_count == 0
    assert report.run_spend.world_model.missing_row_count == 0
    assert report.run_spend.judge.missing_row_count == 0
    assert replay.optimization == result.optimization
    assert _dispatch_counts(simulator, judge) == dispatches_after_completion
    planned_phase_cells = len(result.simulation_spec.cell_ids) + len(
        result.held_out_simulation_spec.cell_ids
    )
    assert planned_phase_cells == 130
    assert dispatches_after_completion[:2] == (
        130,
        130,
    )
    reservations = tuple(
        JudgmentDispatchReceipt.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "dispatch.json")
        )
        for artifact_id in project.artifacts.list_ids()
        if project.artifacts.read(artifact_id).manifest.artifact_type == "judgment-dispatch"
    )
    assert len(reservations) == judge.calls == len(result.plan.cells) == 140
    assert judge.calls <= budget.maximum_judgments
    assert len(telemetry_delivered) == 1
    assert len(telemetry_attempts) == 2

    app = create_project_router_app("w16-router", result.runtime)
    http = TestClient(app)
    payload = {
        "model": "w16-router",
        "messages": [{"role": "user", "content": "Resolve customer case 17"}],
    }
    first_http = http.post("/v1/chat/completions", json=payload)
    second_http = http.post(
        "/v1/chat/completions",
        json={
            **payload,
            "messages": [
                *payload["messages"],
                {"role": "user", "content": "Continue with the same customer case"},
            ],
        },
    )
    assert first_http.status_code == second_http.status_code == 200
    assert first_http.headers["X-WMO-Routed-Model"] == second_http.headers["X-WMO-Routed-Model"]
    assert first_http.json()["object"] == second_http.json()["object"] == "chat.completion"
    assert runtime_catalog.clients["embedder"].embed_calls == 2
    assert sum(client.complete_calls for client in runtime_catalog.clients.values()) == 2
    assert "routing_decision" not in first_http.text
    provenance = verify_release_evidence(
        project.artifacts,
        expected_revision=revision,
        report_name="w16-router",
        claims={
            "normalized_trace_count": 100,
            "fit_task_count": 50,
            "held_out_task_count": 20,
            "planned_cell_count": 140,
            "simulated_cell_count": 130,
            "judgment_count": 140,
            "maximum_judgments": 200,
            "maximum_simulation_cost_usd": 2.0,
            "observed_spend_usd": 0.0,
            "hosted_call_count": 0,
        },
    )
    assert provenance["code_revision"] == revision
    assert list_ids_calls < 1_000


class _EvidenceReviewSupplier:
    """Persist one exact-revision approved rubric and sealed calibration."""

    def __init__(self, revision: str) -> None:
        """Store the exact release revision used by review artifacts."""
        self.revision = revision

    def __call__(
        self,
        project: ProjectStore,
        build: wmo.ProjectBuild,
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
                code_revision=self.revision,
                rubric_id="rubric-a",
                dimensions=(
                    RubricDimension(
                        dimension_id="dimension-a",
                        name="Task success",
                        description="Whether the task was completed.",
                        min_score=0,
                        max_score=5,
                        anchors=tuple(
                            ScoreAnchor(score=score, description=f"Score {score} outcome.")
                            for score in (0, 1, 2, 3, 4, 5)
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
            calibration = JudgeCalibration(
                schema_version=1,
                created_at=_TIME,
                code_revision=self.revision,
                calibration_id="calibration-a",
                rubric_id="rubric-a",
                judge_model=_snapshot("judge-model"),
                judge_prompt_id="judge-prompt-v1",
                judge_prompt_sha256="a" * 64,
                label_set_id="label-set-a",
                calibration_lineage_ids=tuple(
                    task.lineage_group_id for task in tasks if task.partition == "fit"
                ),
                excluded_router_held_out_lineage_ids=tuple(
                    task.lineage_group_id for task in tasks if task.partition == "held_out"
                ),
                validation_method="grouped_k_fold",
                out_of_fold_report_id="calibration-report-a",
                out_of_fold_report_sha256="a" * 64,
                score_maps=(
                    DimensionScoreMap(
                        dimension_id="dimension-a",
                        min_score=0,
                        max_score=5,
                        calibrated_scores=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                    ),
                ),
                status="provisional",
            )
            project.artifacts.write_json(
                artifact_id=calibration.calibration_id,
                artifact_type="judge-calibration",
                envelope=calibration,
                files={"calibration.json": calibration},
            )
        return ApprovedRouterReview(rubric_id="rubric-a", calibration_id="calibration-a")


def _dispatch_counts(
    simulator: _EvidenceSimulatorFactory,
    judge: _Judge,
) -> tuple[int, int, int]:
    """Return candidate, world-model, and judge dispatch totals."""
    candidate_calls = sum(len(client.requests) for client in simulator.candidates.values())
    return candidate_calls, len(simulator.world.requests), judge.calls
