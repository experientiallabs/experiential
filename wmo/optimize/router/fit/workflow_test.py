"""Provider-free Python coverage for the composed router workflow."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.evaluations import (
    EvaluationProtocol,
    ObservedProductionCell,
    build_evaluation_plan,
)
from wmo.common.evaluations.build_test import (
    _persist_calibration,
    _persist_rollout,
    _persist_task_set,
    _production_rollout,
    _task,
)
from wmo.common.evaluations.evidence import (
    EvaluationCellEvidence,
    read_calibration,
)
from wmo.common.judging import DimensionJudgment, Judgment
from wmo.common.models import (
    CandidateTokenPrice,
    Embedding,
    ModelSnapshot,
    OperationEconomics,
    PricingSnapshot,
    RoutedCandidateSnapshot,
)
from wmo.common.project import ArtifactStore, ProjectConfig, ProjectStore
from wmo.common.routing import (
    KnnGuard,
    RouterFeatureExtractor,
    persist_router_embeddings,
    router_embedding_reservation,
)
from wmo.common.tasks import TaskCase
from wmo.optimize.router.activation import load_project_router
from wmo.optimize.router.fit.workflow import (
    EvaluationInputs,
    RouterOptimizationConfig,
    optimize_router,
)
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.runtime_test import _Catalog, _Client, _request, _snapshot

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


def _artifact_input(value: object) -> ArtifactInput:
    """Normalize helper-return unions to the exact persisted input contract."""
    return ArtifactInput.model_validate(value)


def test_single_workflow_freezes_before_held_out_and_resumes_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove replay reuses one provider-free config and immutable artifact set.

    Args:
        tmp_path: Isolated project directory for persisted workflow artifacts.
        monkeypatch: Patch fixture used to observe evaluation dataset construction.
    """
    store, config = _workflow_fixture(tmp_path)
    from wmo.optimize.router.fit import workflow as workflow_module

    actual_build = workflow_module.build_evaluation_dataset
    opened: list[tuple[str, ...]] = []

    def checked_build(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - local seam
        """Record evaluation purposes and enforce policy-first held-out replay."""
        purposes = tuple(kwargs["purposes"])
        if purposes == ("held_out",):
            assert any(
                store.artifacts.read(artifact_id).manifest.artifact_type == "router-policy"
                for artifact_id in store.artifacts.list_ids()
            )
        opened.append(purposes)
        return actual_build(*args, **kwargs)

    monkeypatch.setattr(workflow_module, "build_evaluation_dataset", checked_build)
    first = optimize_router(store.artifacts, config)
    artifact_ids = store.artifacts.list_ids()
    replay = optimize_router(store.artifacts, config)

    assert opened == [("fit",), ("held_out",), ("fit",), ("held_out",)]
    assert replay == first
    assert store.artifacts.list_ids() == artifact_ids
    assert first.optimization.report.held_out_task_ids == (
        "task-held-00",
        "task-held-01",
    )
    client = _Client()
    runtime = load_project_router(
        "project-a",
        tmp_path,
        runtime_catalog=cast(
            RuntimeModelCatalog,
            _Catalog(
                {
                    "candidate-a": first.optimization.policy.candidates[0].model,
                    "embedder": first.optimization.policy.embedder,
                },
                client,
            ),
        ),
    )
    sticky = runtime.select(_request(), episode_id="customer-episode")
    assert runtime.select(_request(), episode_id="customer-episode") == sticky
    assert sticky.selected_alias == "candidate-a"
    client.embedding_values = RuntimeError("local embedding failed")
    fallback = runtime.select(_request(), episode_id="fallback-episode")
    assert fallback.selected_alias == "candidate-a"
    assert fallback.fallback_reason == "embedding_error"


def _workflow_fixture(root: Path) -> tuple[ProjectStore, RouterOptimizationConfig]:
    """Persist a deterministic completed-evidence project with no live client."""
    project = ProjectStore(root, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    store = project.artifacts
    fit = tuple(_task(f"task-fit-{index:02d}", partition="fit") for index in range(10))
    held_out = tuple(_task(f"task-held-{index:02d}", partition="held_out") for index in range(2))
    tasks = (*fit, *held_out)
    task_set_input = _persist_task_set(store, "task-set-workflow", tasks)
    calibration_input = _persist_calibration(
        store,
        fit_lineages=tuple(task.lineage_group_id for task in fit),
        held_out_lineages=tuple(task.lineage_group_id for task in held_out),
    )
    calibration = read_calibration(store, "calibration-a")[0]
    candidate = RoutedCandidateSnapshot(alias="candidate-a", model=_snapshot("candidate-a"))
    observed = []
    evidence_by_task: dict[str, EvaluationCellEvidence] = {}
    for task in tasks:
        cell_id = f"cell-{task.task_id}"
        rollout = _production_rollout(
            f"rollout-{task.task_id}",
            cell_id=cell_id,
            task=task,
            candidate=candidate.model,
            candidate_cost=0.2,
        )
        rollout_input = _persist_rollout(store, rollout)
        judgment = Judgment(
            schema_version=1,
            created_at=_TIME,
            inputs=(
                _artifact_input(calibration_input),
                _artifact_input(rollout_input),
            ),
            code_revision="test-revision",
            judgment_id=f"judgment-{task.task_id}",
            rollout_id=rollout.rollout_id,
            rubric_id="rubric-a",
            calibration_id="calibration-a",
            judge_model=calibration.judge_model,
            judge_prompt_id=calibration.judge_prompt_id,
            judge_prompt_sha256=calibration.judge_prompt_sha256,
            dimensions=(
                DimensionJudgment(
                    dimension_id="dimension-a",
                    raw_score=4,
                    calibrated_score=4.0,
                    rationale="Deterministic completed evidence.",
                ),
            ),
            overall_score=0.8,
            judge_economics=OperationEconomics(),
        )
        store.write_json(
            artifact_id=judgment.judgment_id,
            artifact_type="judgment",
            envelope=judgment,
            files={"judgment.json": judgment},
        )
        observed.append(
            ObservedProductionCell(
                task_id=task.task_id,
                candidate_alias=candidate.alias,
                repeat=0,
                rollout_artifact_id=rollout.rollout_id,
            )
        )
        evidence_by_task[task.task_id] = EvaluationCellEvidence(
            cell_id=cell_id,
            protocol_id="protocol-production",
            rollout_artifact_id=rollout.rollout_id,
            judgment_artifact_id=judgment.judgment_id,
            source_run_id=rollout.source_run_id,
        )
    _persist_pricing(store)
    plan = build_evaluation_plan(
        store,
        task_set_id="task-set-workflow",
        candidate_snapshots=(candidate,),
        pricing_snapshot_id="pricing-a",
        observed_cells=observed,
        created_at=_TIME,
        code_revision="test-revision",
    )
    production_protocol = EvaluationProtocol(
        protocol_id="protocol-production",
        evidence_source="production",
        agent_id="agent-a",
        simulator_id="production-import-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id="pricing-a",
    )
    main_evidence = tuple(
        evidence_by_task[cell.task_id].model_copy(update={"cell_id": cell.cell_id})
        for cell in plan.cells
    )
    embedding_set_id = _persist_embeddings(store, tasks, task_set_input=task_set_input)
    protocols = (production_protocol,)
    return project, RouterOptimizationConfig(
        fit=EvaluationInputs(
            evaluation_plan_id=plan.plan_id,
            protocols=protocols,
            cell_evidence=tuple(
                item
                for item in main_evidence
                if next(cell for cell in plan.cells if cell.cell_id == item.cell_id).purpose
                == "fit"
            ),
        ),
        held_out=EvaluationInputs(
            evaluation_plan_id=plan.plan_id,
            protocols=protocols,
            cell_evidence=tuple(
                item
                for item in main_evidence
                if next(cell for cell in plan.cells if cell.cell_id == item.cell_id).purpose
                == "held_out"
            ),
        ),
        embedding_set_id=embedding_set_id,
        incumbent_alias="candidate-a",
        pricing_snapshot_id="pricing-a",
        guard=KnnGuard(
            maximum_neighbors=10,
            minimum_paired_observations=8,
            relative_similarity_threshold=0.0,
            uncertainty_multiplier=0.5,
            quality_tolerance=0.0,
        ),
        judgment_status="provisional",
        created_at=_TIME,
        code_revision="test-revision",
    )


def _persist_pricing(store) -> None:  # noqa: ANN001 - focused fixture
    """Persist the exact single-candidate pricing identity."""
    pricing = PricingSnapshot(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        pricing_snapshot_id="pricing-a",
        candidate_prices=(
            CandidateTokenPrice(
                candidate_alias="candidate-a",
                input_usd_per_million_tokens=1.0,
                output_usd_per_million_tokens=2.0,
            ),
        ),
    )
    store.write_json(
        artifact_id=pricing.pricing_snapshot_id,
        artifact_type="pricing-snapshot",
        envelope=pricing,
        files={"pricing.json": pricing},
    )


class _UnitEmbeddingClient:
    """Return a fixed unit vector for every requested feature text."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return one deterministic unit vector per input text."""
        return tuple(Embedding(values=(1.0, 0.0)) for _ in texts)


def _persist_embeddings(
    store: ArtifactStore,
    tasks: Sequence[TaskCase],
    *,
    task_set_input: ArtifactInput,
    embedder: ModelSnapshot | None = None,
    created_at: datetime = _TIME,
    code_revision: str = "test-revision",
) -> str:
    """Persist reserved local vectors for every request-visible task feature.

    Args:
        store: Project-local artifact store.
        tasks: Fit and held-out tasks whose request-visible features are embedded.
        task_set_input: Exact task-set pointer bound into the reserved artifact.
        embedder: Exact embedder identity, or the shared workflow-test snapshot.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Content-addressed reserved embedding-set identity.
    """
    model = embedder or _snapshot("embedder")
    texts = tuple(dict.fromkeys(RouterFeatureExtractor().from_task(task) for task in tasks))
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=0.0,
        maximum_attempts_per_feature=1,
        maximum_input_tokens_per_feature=10_000,
        feature_count=len(texts),
    )
    artifact = persist_router_embeddings(
        store,
        task_set_input=task_set_input,
        tasks=tasks,
        embedder_alias="embedder",
        embedder=model,
        client=_UnitEmbeddingClient(),
        reservation=reservation,
        active_input_usd_per_million_tokens=0.0,
        active_maximum_attempts_per_feature=1,
        created_at=created_at,
        code_revision=code_revision,
    )
    return artifact.embedding_set_id
