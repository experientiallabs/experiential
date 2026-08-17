"""Adversarial tests for fit-only bank evidence, economics, and immutable bytes."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from wmo.common.core.artifacts import FailureCode, StructuredFailure, canonical_json_bytes
from wmo.common.evaluations import (
    EvaluationDataset,
    EvaluationDatasetManifest,
    EvaluationProtocol,
    EvaluationRow,
    FidelityFailure,
    FidelityPair,
    FidelityReport,
)
from wmo.common.evaluations.build_test import _candidate, _money, _snapshot, _task
from wmo.common.evaluations.evidence import evaluation_protocol_digest
from wmo.common.models import Embedding
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, ProjectPaths
from wmo.common.routing import RouterFeatureExtractor
from wmo.common.routing.bank import (
    CandidateEvidenceCount,
    KnnBankManifest,
    KnnEvidenceBank,
    bank_bytes,
    build_knn_bank,
    evidence_counts,
    load_knn_bank,
)
from wmo.common.tasks import TaskCase
from wmo.optimize.router.fit.optimizer import RouterOptimizationError, choose_baseline

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


class _DeterministicEmbedder:
    """No-network embedding fixture that records the exact fit batch."""

    def __init__(self) -> None:
        """Initialize an empty ordered embedding-call log."""
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Build deterministic unit vectors for the supplied texts.

        Args:
            texts: Request texts to embed in order.

        Returns:
            Deterministic unit vectors in the same order.
        """
        self.calls.append(tuple(texts))
        vectors = []
        for text in texts:
            parity = hashlib.sha256(text.encode("utf-8")).digest()[0] % 2
            vectors.append(Embedding(values=(1.0, 0.0) if parity else (0.0, 1.0)))
        return tuple(vectors)


def test_bank_excludes_failed_fidelity_and_uses_only_candidate_cost() -> None:
    """Rejected simulation and failed rows stay out, while direct sandbox evidence remains."""
    tasks = (_task("task-fit-0", partition="fit"), _task("task-fit-1", partition="fit"))
    production = _protocol("protocol-production", "production")
    sandbox = _protocol("protocol-sandbox", "sandbox")
    world = _world_protocol()
    report = _insufficient_report(world)
    rows = (
        _row(tasks[0], "candidate-a", production, score=0.8, candidate_cost=0.5),
        _row(tasks[1], "candidate-a", production, score=0.8, candidate_cost=0.5),
        _row(
            tasks[0],
            "candidate-b",
            sandbox,
            score=0.9,
            candidate_cost=0.1,
            world_cost=99.0,
            judge_cost=88.0,
        ),
        EvaluationRow(
            cell_id="cell-failed",
            task_id=tasks[1].task_id,
            candidate_alias="candidate-b",
            repeat=0,
            protocol_id=sandbox.protocol_id,
            source_run_id="run-failed",
            purpose="fit",
            status="failed",
            error=StructuredFailure(code=FailureCode.TIMEOUT, message="sandbox failed"),
        ),
        _row(
            tasks[1],
            "candidate-b",
            world,
            score=1.0,
            candidate_cost=0.01,
            repeat=1,
        ),
    )
    dataset = _dataset(tasks, rows, (production, sandbox, world))

    bank = build_knn_bank(
        dataset,
        tasks,
        {report.fidelity_report_id: report},
        embedder=_DeterministicEmbedder(),
        feature_extractor=RouterFeatureExtractor(),
    )

    assert evidence_counts(bank) == (
        CandidateEvidenceCount(
            candidate_alias="candidate-a", scored_task_count=2, costed_task_count=2
        ),
        CandidateEvidenceCount(
            candidate_alias="candidate-b", scored_task_count=1, costed_task_count=1
        ),
    )
    assert bank.candidate_costs[0, 1] == pytest.approx(0.1)
    assert math.isnan(float(bank.candidate_costs[1, 1]))
    assert math.isnan(float(bank.scores[1, 1]))
    assert choose_baseline(bank, incumbent_alias=None) == "candidate-a"
    with pytest.raises(RouterOptimizationError, match="lacks score evidence"):
        choose_baseline(bank, incumbent_alias="candidate-b")


def test_fifty_fit_tasks_are_bank_only_and_twenty_held_out_rows_cannot_change_it() -> None:
    """The default 50/20 split is sealed at embedding and bank serialization boundaries."""
    fit = tuple(_task(f"task-fit-{index:02d}", partition="fit") for index in range(50))
    held_out = tuple(_task(f"task-held-{index:02d}", partition="held_out") for index in range(20))
    tasks = (*fit, *held_out)
    protocol = _protocol("protocol-production", "production")
    rows = tuple(
        _row(
            task,
            alias,
            protocol,
            score=(0.8 if alias == "candidate-a" else 0.9),
            candidate_cost=(0.5 if alias == "candidate-a" else 0.1),
        )
        for task in tasks
        for alias in ("candidate-a", "candidate-b")
    )
    first_dataset = _dataset(tasks, rows, (protocol,))
    changed_rows = tuple(
        row.model_copy(update={"score": 0.0 if row.purpose == "held_out" else row.score})
        for row in rows
    )
    changed_dataset = first_dataset.model_copy(update={"rows": changed_rows})
    first_embedder = _DeterministicEmbedder()
    changed_embedder = _DeterministicEmbedder()

    first = build_knn_bank(
        first_dataset,
        tasks,
        {},
        embedder=first_embedder,
        feature_extractor=RouterFeatureExtractor(),
    )
    changed = build_knn_bank(
        changed_dataset,
        tasks,
        {},
        embedder=changed_embedder,
        feature_extractor=RouterFeatureExtractor(),
    )

    assert first.task_ids == tuple(task.task_id for task in fit)
    assert not set(first.task_ids).intersection(task.task_id for task in held_out)
    assert tuple(len(call) for call in first_embedder.calls) == (50,)
    assert tuple(len(call) for call in changed_embedder.calls) == (50,)
    assert bank_bytes(first) == bank_bytes(changed)


def test_persisted_bank_detects_sidecar_mutation(tmp_path: Path) -> None:
    """Raw numeric bytes cannot change behind either manifest or policy hash pins."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    bank = KnnEvidenceBank(
        task_ids=("task-a", "task-b"),
        candidate_aliases=("candidate-a",),
        embeddings=np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        scores=np.asarray(((0.8,), (0.9,)), dtype=np.float32),
        candidate_costs=np.asarray(((0.5,), (0.6,)), dtype=np.float64),
        score_counts=np.ones((2, 1), dtype=np.int32),
        cost_counts=np.ones((2, 1), dtype=np.int32),
        workload_weights=np.ones(2, dtype=np.float64),
        novelty_floor=0.0,
    )
    payload = bank_bytes(bank)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = KnnBankManifest(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        bank_artifact_id="knn-bank-a",
        fit_evaluation_id="evaluation-a",
        evaluation_plan_id="plan-a",
        evaluation_plan_sha256=_DIGEST,
        task_set_id="task-set-a",
        task_set_sha256=_DIGEST,
        task_ids=bank.task_ids,
        candidate_aliases=bank.candidate_aliases,
        evaluation_protocols_sha256=_DIGEST,
        embedder_alias="embedder",
        embedder=_snapshot("embedder"),
        feature_extractor_id="request-visible-v1",
        feature_schema_sha256=_DIGEST,
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        bank_sha256=digest,
        embedding_dimension=2,
        novelty_floor=0.0,
        evidence_counts=evidence_counts(bank),
    )
    store.write(
        artifact_id=manifest.bank_artifact_id,
        artifact_type="knn-bank",
        envelope=manifest,
        files={"bank.json": canonical_json_bytes(manifest), "bank.npz": payload},
    )

    loaded_manifest, loaded = load_knn_bank(
        store, manifest.bank_artifact_id, expected_sha256=digest
    )
    assert loaded_manifest == manifest
    assert np.array_equal(loaded.scores, bank.scores)
    stored = store.read(manifest.bank_artifact_id)
    (stored.directory / "bank.npz").write_bytes(payload + b"mutation")
    with pytest.raises(ArtifactCorruptionError, match="digest mismatch"):
        load_knn_bank(store, manifest.bank_artifact_id, expected_sha256=digest)


def _dataset(
    tasks: tuple[TaskCase, ...],
    rows: tuple[EvaluationRow, ...],
    protocols: tuple[EvaluationProtocol, ...],
) -> EvaluationDataset:
    """Create one direct immutable-contract fixture without touching a provider."""
    manifest = EvaluationDatasetManifest(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        evaluation_id="evaluation-a",
        evaluation_plan_id="plan-a",
        evaluation_plan_sha256=_DIGEST,
        task_set_id="task-set-a",
        fit_task_ids=tuple(task.task_id for task in tasks if task.partition == "fit"),
        held_out_task_ids=tuple(task.task_id for task in tasks if task.partition == "held_out"),
        candidate_snapshots=(_candidate("candidate-a"), _candidate("candidate-b")),
        protocols=protocols,
        fidelity_report_ids=tuple(
            sorted(
                protocol.fidelity_report_id
                for protocol in protocols
                if protocol.fidelity_report_id is not None
            )
        ),
        rows_path="rows.jsonl",
        rows_sha256=_DIGEST,
    )
    return EvaluationDataset(manifest=manifest, rows=rows)


def _row(
    task: TaskCase,
    alias: str,
    protocol: EvaluationProtocol,
    *,
    score: float,
    candidate_cost: float,
    repeat: int = 0,
    world_cost: float | None = None,
    judge_cost: float | None = None,
) -> EvaluationRow:
    """Create one completed sparse row with deliberately separable spend."""
    suffix = f"{task.task_id}-{alias}-{repeat}"
    return EvaluationRow(
        cell_id=f"cell-{suffix}",
        task_id=task.task_id,
        candidate_alias=alias,
        repeat=repeat,
        protocol_id=protocol.protocol_id,
        source_run_id=f"run-{suffix}",
        purpose=task.partition,
        status="observed" if protocol.evidence_source == "production" else "completed",
        rollout_id=f"rollout-{suffix}",
        judgment_id=f"judgment-{suffix}",
        score=score,
        candidate_cost_usd=_money(candidate_cost),
        world_model_cost_usd=_money(world_cost) if world_cost is not None else None,
        judge_cost_usd=_money(judge_cost) if judge_cost is not None else None,
    )


def _protocol(protocol_id: str, source: Literal["sandbox", "production"]) -> EvaluationProtocol:
    """Create one compatible direct-execution protocol."""
    return EvaluationProtocol(
        protocol_id=protocol_id,
        evidence_source=source,
        agent_id="agent-a",
        simulator_id=f"{source}-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id="pricing-a",
    )


def _world_protocol() -> EvaluationProtocol:
    """Create a world-model protocol linked to insufficient fidelity evidence."""
    return EvaluationProtocol(
        protocol_id="protocol-world",
        evidence_source="world_model",
        agent_id="agent-a",
        simulator_id="world-v1",
        world_model=_snapshot("world-model"),
        simulator_prompt_id="world-prompt-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id="pricing-a",
        fidelity_report_id="fidelity-insufficient",
    )


def _insufficient_report(protocol: EvaluationProtocol) -> FidelityReport:
    """Create a seven-of-ten fidelity report that cannot authorize fit rows."""
    overlap_ids = tuple(f"cell-overlap-{index}" for index in range(10))
    failures = tuple(
        FidelityFailure(
            cell_id=overlap_ids[index],
            failure=StructuredFailure(code=FailureCode.TIMEOUT, message="overlap failed"),
        )
        for index in range(7, 10)
    )
    return FidelityReport(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        fidelity_report_id="fidelity-insufficient",
        evaluation_plan_id="plan-a",
        evaluation_plan_sha256=_DIGEST,
        protocol_sha256=evaluation_protocol_digest(protocol),
        overlap_cell_ids=overlap_ids,
        planned_overlap_count=10,
        usable_overlap_count=7,
        failed_overlap_count=3,
        score_mae=0.01,
        failures=failures,
        pairs=tuple(
            FidelityPair(
                fidelity_cell_id=cell_id,
                observed_cell_id=f"observed-{cell_id}",
                observed_rollout_id=f"observed-rollout-{index}",
                simulated_rollout_id=(f"simulated-rollout-{index}" if index < 7 else None),
                observed_score=0.8 if index < 7 else None,
                simulated_score=0.8 if index < 7 else None,
                absolute_error=0.0 if index < 7 else None,
                status="usable" if index < 7 else "failed",
                error=None if index < 7 else failures[index - 7].failure,
            )
            for index, cell_id in enumerate(overlap_ids)
        ),
        gate_id="fidelity-gate-a",
        gate_sha256=_DIGEST,
        status="insufficient",
    )
