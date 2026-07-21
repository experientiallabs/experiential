"""Behavioral tests for crash-safe local population checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmh.harness.e2b_sandbox import SandboxUsage
from wmh.harness.live_session import SessionEvent
from wmh.harness.population import PopulationIteration, PopulationOptimizationResult
from wmh.harness.population_checkpoint import (
    PopulationCheckpointError,
    PopulationCheckpointIdentity,
    PopulationCheckpointLockError,
    PopulationCheckpointStateError,
    PopulationCheckpointStore,
)
from wmh.harness.project_proposer import (
    CandidateProposal,
    CandidateProposalError,
    EvaluatedCandidate,
)
from wmh.harness.runtime import TokenUsage
from wmh.harness.scoring import (
    EvaluationArtifact,
    HarnessScore,
    HarnessScoreReport,
    ScoreCell,
    ScoreContext,
    ScoreRequest,
)
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmh.providers.base import ProviderConfig, ProviderKind

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64


class _Reader:
    def __init__(self, content: dict[str, bytes]) -> None:
        self.content = content

    def read_bytes(self, path: str) -> bytes:
        return self.content[path]


def _source(prompt: str) -> HarnessSourceTree:
    return HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content=prompt),
            HarnessSourceFile(
                path="config.toml",
                content='[harness]\ntools = ["bash", "submit"]\nruntime_kind = "pi-node"\n',
            ),
        )
    )


def _request() -> ScoreRequest:
    return ScoreRequest(
        context=ScoreContext(
            task_set_digest=_DIGEST_A,
            evaluator_digest=_DIGEST_B,
            execution_config_digest=_DIGEST_C,
        ),
        task_ids=("task-a", "task-b"),
        attempts=1,
    )


def _evaluated(
    candidate_id: str,
    source: HarnessSourceTree,
    value: float,
    *,
    artifact_path: str | None = None,
) -> EvaluatedCandidate:
    path = artifact_path or f"raw/{candidate_id}.json"
    content = json.dumps({"candidate_id": candidate_id}).encode()
    artifact = EvaluationArtifact.from_bytes(path=path, content=content)
    request = _request()
    report = HarnessScoreReport(
        source_run_id=f"run-{candidate_id}",
        candidate_doc_hash=source.to_doc(candidate_id).doc_hash,
        request=request,
        cells=tuple(
            ScoreCell(
                task_id=task_id,
                attempt=1,
                score=value,
                passed=value >= 0.5,
                summary="official result",
                artifact_paths=(path,),
            )
            for task_id in request.task_ids
        ),
        artifacts=(artifact,),
    )
    return EvaluatedCandidate(
        candidate_id=candidate_id,
        source=source,
        score=HarnessScore(report=report, artifacts=_Reader({path: content})),
    )


def _status(
    candidate_id: str,
    source: HarnessSourceTree | None,
    *,
    valid: bool,
) -> str:
    document_hash: str | None = None
    if source is not None:
        try:
            document_hash = source.to_doc(candidate_id).doc_hash
        except ValueError:
            pass
    return json.dumps(
        {
            "agent_error": None,
            "candidate_doc_hash": document_hash,
            "candidate_id": candidate_id,
            "source_tree_hash": source.tree_hash if source is not None else None,
            "valid": valid,
            "validation_error": None if valid else "incomplete",
        },
        indent=2,
        sort_keys=True,
    )


def _results() -> tuple[
    PopulationOptimizationResult,
    PopulationOptimizationResult,
    PopulationOptimizationResult,
]:
    seed = _evaluated("candidate-0000", _source("seed"), 0.2)
    seed_result = PopulationOptimizationResult(population=(seed,), iterations=(), best=seed)

    first = _evaluated("candidate-0001", _source("first"), 0.8)
    proposal = CandidateProposal(
        candidate_id=first.candidate_id,
        source=first.source,
        candidate=first.candidate,
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(input_tokens=10, output_tokens=5, calls=1),
        request="produce candidate-0001",
        status_json=_status(first.candidate_id, first.source, valid=True),
    )
    first_iteration = PopulationIteration(index=1, proposal=proposal, evaluation=first)
    first_result = PopulationOptimizationResult(
        population=(seed, first),
        iterations=(first_iteration,),
        best=first,
    )

    invalid_source = HarnessSourceTree(
        files=(HarnessSourceFile(path="notes.txt", content="unfinished"),)
    )
    error = CandidateProposalError(
        "candidate-0002",
        "incomplete candidate",
        source=invalid_source,
        events=(SessionEvent(kind="error", payload={"message": "incomplete"}),),
        worker_usage=None,
        request="produce candidate-0002",
        status_json=_status("candidate-0002", invalid_source, valid=False),
    )
    final_result = PopulationOptimizationResult(
        population=(seed, first),
        iterations=(first_iteration, PopulationIteration(index=2, error=error)),
        best=first,
    )
    return seed_result, first_result, final_result


def _identity(root: Path, seed: HarnessSourceTree) -> PopulationCheckpointIdentity:
    return PopulationCheckpointIdentity(
        output_name="selected",
        artifact_root=str(root.resolve()),
        seed_reference=None,
        seed_source_tree_hash=seed.tree_hash,
        score_request=_request(),
        iterations=2,
        planned_score_cells=6,
        max_score_cells=8,
        harbor_job_template={"job_name": "template", "jobs_dir": "run/harbor"},
        meta_provider=ProviderConfig(
            kind=ProviderKind.OPENAI_RESPONSES,
            model="meta-model",
            reasoning_effort="high",
        ),
        agent_provider=ProviderConfig(kind=ProviderKind.ANTHROPIC, model="agent-model"),
        optimizer_document_hash="optimizer-doc-hash",
        harness_backend="e2b",
        e2b_template="template",
        environment_command_timeout_sec=300,
        episode_timeout_sec=300,
        project_timeout_sec=900,
        max_source_files=100,
        max_source_bytes=1_000_000,
        max_history_candidates=20,
        max_history_bytes=2_000_000,
    )


def _commit_all(store: PopulationCheckpointStore) -> PopulationOptimizationResult:
    seed_result, first_result, final_result = _results()
    store.begin_setup()
    store.before_step(0)
    store.commit_boundary(seed_result)
    store.finish_project_segment(SandboxUsage())
    store.begin_setup()
    store.before_step(1)
    store.commit_boundary(first_result)
    store.finish_project_segment(SandboxUsage(count=1, seconds=1.5))
    store.begin_setup()
    store.before_step(2)
    store.commit_boundary(final_result)
    store.finish_project_segment(SandboxUsage(count=1, seconds=2.0))
    return final_result


def test_checkpoint_appends_exact_boundaries_and_restores_without_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    seed = _source("seed")
    identity = _identity(tmp_path / "root", seed)
    with PopulationCheckpointStore.create(run_dir, identity=identity, seed=seed) as store:
        final_result = _commit_all(store)
        assert store.control.state == "ready"
        assert store.control.committed_step == 2
        assert store.control.known_score_cells == 4
        assert store.control.project_sandbox_usage == SandboxUsage(count=2, seconds=3.5)

    with PopulationCheckpointStore.open(run_dir) as resumed:
        assert resumed.identity == identity
        assert resumed.seed == seed
        assert resumed.result is not None
        assert [item.candidate_id for item in resumed.result.population] == [
            "candidate-0000",
            "candidate-0001",
        ]
        assert [item.candidate_id for item in resumed.result.iterations] == [
            "candidate-0001",
            "candidate-0002",
        ]
        assert resumed.result.best.candidate_id == final_result.best.candidate_id
        assert resumed.result.iterations[1].error is not None
        assert resumed.result.iterations[1].error.reason == "incomplete candidate"
        assert resumed.result.iterations[1].error.worker_usage is None


def test_checkpoint_lock_has_one_local_owner(tmp_path: Path) -> None:
    seed = _source("seed")
    store = PopulationCheckpointStore.create(
        tmp_path / "run",
        identity=_identity(tmp_path / "root", seed),
        seed=seed,
    )
    try:
        with pytest.raises(PopulationCheckpointLockError, match="another process"):
            PopulationCheckpointStore.open(tmp_path / "run")
    finally:
        store.close()


def test_only_verified_cleanup_and_finalization_states_resume(tmp_path: Path) -> None:
    seed = _source("seed")
    in_progress_dir = tmp_path / "in-progress"
    store = PopulationCheckpointStore.create(
        in_progress_dir,
        identity=_identity(tmp_path / "root", seed),
        seed=seed,
    )
    store.begin_setup()
    store.close()
    with pytest.raises(PopulationCheckpointStateError, match="in_progress"):
        PopulationCheckpointStore.open(in_progress_dir)

    cleanup_dir = tmp_path / "cleanup"
    store = PopulationCheckpointStore.create(
        cleanup_dir,
        identity=_identity(tmp_path / "root", seed),
        seed=seed,
    )
    seed_result, _first, _final = _results()
    store.begin_setup()
    store.before_step(0)
    store.commit_boundary(seed_result)
    store.close()
    with PopulationCheckpointStore.open(cleanup_dir) as recovered:
        assert recovered.control.state == "ready"
        assert recovered.control.committed_step == 0
        assert recovered.control.project_sandbox_usage is None
        assert recovered.result is not None
        assert recovered.result.iterations == ()

    complete_dir = tmp_path / "complete"
    store = PopulationCheckpointStore.create(
        complete_dir,
        identity=_identity(tmp_path / "root", seed),
        seed=seed,
    )
    final_result = _commit_all(store)
    publication_id = store.publication_id
    store.begin_finalization(
        publication_id=publication_id,
        harness_version=1,
        document_hash=final_result.best.candidate.doc_hash,
        prior_champion_version=None,
        archive_manifest="population/manifest.json",
        outcome_path="outcome.json",
    )
    store.close()
    finalization_tail = complete_dir / f"checkpoint/control.json.tmp-{'e' * 32}"
    finalization_tail.write_text("{}", encoding="utf-8")
    with PopulationCheckpointStore.open(complete_dir) as finalizing:
        assert finalizing.control.state == "in_progress"
        assert finalizing.control.active_kind == "finalize"
        assert finalizing.control.publication_intent is not None
        assert finalizing.control.publication_intent.publication_id == publication_id
    assert not finalization_tail.exists()

    store = PopulationCheckpointStore.open(complete_dir)
    archive_manifest = complete_dir / "population/manifest.json"
    archive_manifest.parent.mkdir()
    archive_manifest.write_text("{}\n", encoding="utf-8")
    outcome_path = complete_dir / "outcome.json"
    outcome_path.write_text("{}\n", encoding="utf-8")
    saved = final_result.best.candidate.model_copy(update={"name": "selected", "version": 1})
    store.mark_complete(
        saved=saved,
        archive_manifest=archive_manifest,
        outcome_path=outcome_path,
    )
    store.close()
    with pytest.raises(PopulationCheckpointStateError, match="complete"):
        PopulationCheckpointStore.open(complete_dir)


def test_checkpoint_identity_drift_rejects_without_mutation(tmp_path: Path) -> None:
    seed = _source("seed")
    run_dir = tmp_path / "run"
    identity = _identity(tmp_path / "root", seed)
    with PopulationCheckpointStore.create(run_dir, identity=identity, seed=seed) as store:
        control_before = (run_dir / "checkpoint/control.json").read_bytes()
        changed = identity.model_copy(update={"max_score_cells": 9})
        with pytest.raises(PopulationCheckpointError, match="max_score_cells"):
            store.assert_identity(changed)
        assert (run_dir / "checkpoint/control.json").read_bytes() == control_before

        changed_optimizer = identity.model_copy(
            update={"optimizer_document_hash": "updated-optimizer-doc-hash"}
        )
        with pytest.raises(PopulationCheckpointError, match="optimizer_document_hash"):
            store.assert_identity(changed_optimizer)
        assert (run_dir / "checkpoint/control.json").read_bytes() == control_before


def test_checkpoint_tamper_fails_closed_and_atomic_tail_recovers(tmp_path: Path) -> None:
    seed = _source("seed")
    tampered_dir = tmp_path / "tampered"
    store = PopulationCheckpointStore.create(
        tampered_dir,
        identity=_identity(tmp_path / "root", seed),
        seed=seed,
    )
    seed_result, _first, _final = _results()
    store.begin_setup()
    store.before_step(0)
    store.commit_boundary(seed_result)
    store.finish_project_segment(SandboxUsage(count=1, seconds=1.5))
    store.close()
    artifact = next((tampered_dir / "checkpoint/steps/0000/artifacts").rglob("*"))
    while artifact.is_dir():
        artifact = next(artifact.rglob("*"))
    artifact.write_bytes(b"changed")
    with pytest.raises(PopulationCheckpointError, match="differs from its manifest"):
        PopulationCheckpointStore.open(tampered_dir)

    torn_dir = tmp_path / "torn"
    store = PopulationCheckpointStore.create(
        torn_dir,
        identity=_identity(tmp_path / "root", seed),
        seed=seed,
    )
    store.close()
    temporary = torn_dir / f"checkpoint/control.json.tmp-{'f' * 32}"
    temporary.write_text("{}")
    with PopulationCheckpointStore.open(torn_dir) as recovered:
        assert recovered.control.state == "ready"
        assert recovered.control.committed_step == -1
    assert not temporary.exists()


def test_checkpoint_preserves_manifested_artifact_that_looks_like_atomic_tail(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    seed = _source("seed")
    artifact_path = f"raw/trace.tmp-{'a' * 32}"
    evaluated = _evaluated(
        "candidate-0000",
        seed,
        0.5,
        artifact_path=artifact_path,
    )
    result = PopulationOptimizationResult(
        population=(evaluated,),
        iterations=(),
        best=evaluated,
    )
    with PopulationCheckpointStore.create(
        run_dir,
        identity=_identity(tmp_path / "root", seed),
        seed=seed,
    ) as store:
        store.begin_setup()
        store.before_step(0)
        store.commit_boundary(result)
        store.finish_project_segment(SandboxUsage())

    artifact = run_dir / "checkpoint/steps/0000/artifacts" / artifact_path
    before = artifact.read_bytes()
    with PopulationCheckpointStore.open(run_dir) as recovered:
        assert recovered.result is not None
        assert recovered.result.population[0].score.report.artifacts[0].path == artifact_path
    assert artifact.read_bytes() == before


def test_score_cell_plan_must_fit_explicit_ceiling(tmp_path: Path) -> None:
    seed = _source("seed")
    values = _identity(tmp_path / "root", seed).model_dump(mode="python")
    values["max_score_cells"] = 5
    with pytest.raises(ValueError, match="exceed"):
        PopulationCheckpointIdentity.model_validate(values)
