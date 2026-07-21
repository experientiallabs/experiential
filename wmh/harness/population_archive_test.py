"""Tests for complete, append-only population result archives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmh.harness.live_session import SessionEvent
from wmh.harness.population import PopulationIteration, PopulationOptimizationResult
from wmh.harness.population_archive import write_population_archive
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

_DIGEST = "sha256:" + "a" * 64
_ARTIFACT_PATH = "raw/evidence.bin"


class _Reader:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def read_bytes(self, path: str) -> bytes:
        assert path == _ARTIFACT_PATH
        return self._content


def _source(prompt: str) -> HarnessSourceTree:
    return HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content=prompt),
            HarnessSourceFile(
                path="config.toml",
                content='[harness]\ntools = ["bash", "submit"]\n',
            ),
        )
    )


def _score(candidate_id: str, source: HarnessSourceTree, value: float) -> HarnessScore:
    request = ScoreRequest(
        context=ScoreContext(
            task_set_digest=_DIGEST,
            evaluator_digest=_DIGEST,
            execution_config_digest=_DIGEST,
        ),
        task_ids=("task-a",),
        attempts=1,
    )
    content = f"evidence for {candidate_id}".encode()
    artifact = EvaluationArtifact.from_bytes(path=_ARTIFACT_PATH, content=content)
    report = HarnessScoreReport(
        source_run_id=f"run-{candidate_id}",
        candidate_doc_hash=source.to_doc(candidate_id).doc_hash,
        request=request,
        cells=(
            ScoreCell(
                task_id="task-a",
                attempt=1,
                score=value,
                passed=value == 1.0,
                summary="official result",
                artifact_paths=(_ARTIFACT_PATH,),
            ),
        ),
        artifacts=(artifact,),
    )
    return HarnessScore(report=report, artifacts=_Reader(content))


def _evaluated(candidate_id: str, prompt: str, score: float) -> EvaluatedCandidate:
    source = _source(prompt)
    return EvaluatedCandidate(
        candidate_id=candidate_id,
        source=source,
        score=_score(candidate_id, source, score),
    )


def test_archive_copies_sources_reports_artifacts_and_proposal_events(tmp_path: Path) -> None:
    seed = _evaluated("candidate-0000", "seed", 0.0)
    candidate = _evaluated("candidate-0001", "candidate", 1.0)
    proposal = CandidateProposal(
        candidate_id=candidate.candidate_id,
        source=candidate.source,
        candidate=candidate.candidate,
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(input_tokens=10, output_tokens=4, calls=1),
    )
    result = PopulationOptimizationResult(
        population=(seed, candidate),
        iterations=(PopulationIteration(index=1, proposal=proposal, evaluation=candidate),),
        best=candidate,
    )

    destination = tmp_path / "archive"
    manifest_path = write_population_archive(destination, result)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["best_candidate_id"] == "candidate-0001"
    assert manifest["best_score"] == 1.0
    assert [entry["candidate_id"] for entry in manifest["population"]] == [
        "candidate-0000",
        "candidate-0001",
    ]
    assert (destination / "population/0001/source/SYSTEM.md").read_text() == "candidate"
    assert (destination / "population/0001/artifacts/raw/evidence.bin").read_bytes() == (
        b"evidence for candidate-0001"
    )
    events = json.loads((destination / "iterations/0001/events.json").read_text(encoding="utf-8"))
    assert events == [{"kind": "submit", "payload": {"answer": "done"}}]
    assert manifest["iterations"][0]["worker_usage"]["calls"] == 1


def test_archive_preserves_invalid_iteration_source_and_unknown_usage(tmp_path: Path) -> None:
    seed = _evaluated("candidate-0000", "seed", 0.5)
    invalid_source = _source("invalid but captured")
    error = CandidateProposalError(
        "candidate-0001",
        "did not submit",
        source=invalid_source,
        events=(SessionEvent(kind="error", payload={"message": "stopped"}),),
        worker_usage=None,
    )
    result = PopulationOptimizationResult(
        population=(seed,),
        iterations=(PopulationIteration(index=1, error=error),),
        best=seed,
    )

    destination = tmp_path / "archive"
    manifest_path = write_population_archive(destination, result)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    [iteration] = manifest["iterations"]
    assert iteration["outcome"] == "invalid"
    assert iteration["worker_usage"] is None
    assert "did not submit" in iteration["error"]
    assert (destination / "iterations/0001/source/SYSTEM.md").read_text() == (
        "invalid but captured"
    )


def test_archive_materializes_an_empty_invalid_source_directory(tmp_path: Path) -> None:
    seed = _evaluated("candidate-0000", "seed", 0.5)
    error = CandidateProposalError(
        "candidate-0001",
        "empty snapshot",
        source=HarnessSourceTree(files=()),
    )
    result = PopulationOptimizationResult(
        population=(seed,),
        iterations=(PopulationIteration(index=1, error=error),),
        best=seed,
    )

    destination = tmp_path / "archive"
    manifest_path = write_population_archive(destination, result)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    [iteration] = manifest["iterations"]
    assert iteration["source_path"] == "iterations/0001/source"
    assert (destination / "iterations/0001/source").is_dir()
    assert list((destination / "iterations/0001/source").iterdir()) == []


def test_archive_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "archive"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("owned", encoding="utf-8")
    seed = _evaluated("candidate-0000", "seed", 0.5)
    result = PopulationOptimizationResult(population=(seed,), iterations=(), best=seed)

    with pytest.raises(FileExistsError, match="already exists"):
        write_population_archive(destination, result)

    assert sentinel.read_text(encoding="utf-8") == "owned"


def test_archive_manifest_is_absent_when_atomic_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "archive"
    seed = _evaluated("candidate-0000", "seed", 0.5)
    result = PopulationOptimizationResult(population=(seed,), iterations=(), best=seed)

    def fail_replace(self: Path, target: Path) -> Path:
        del self, target
        raise OSError("publication interrupted")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="publication interrupted"):
        write_population_archive(destination, result)

    assert not (destination / "manifest.json").exists()
    assert (destination / "manifest.json.tmp").exists()


def test_archive_rejects_artifact_bytes_that_differ_from_manifest(tmp_path: Path) -> None:
    seed = _evaluated("candidate-0000", "seed", 0.5)
    broken_score = HarnessScore(
        report=seed.score.report,
        artifacts=_Reader(b"different bytes"),
    )
    broken = EvaluatedCandidate(
        candidate_id=seed.candidate_id,
        source=seed.source,
        score=broken_score,
    )
    result = PopulationOptimizationResult(population=(broken,), iterations=(), best=broken)

    with pytest.raises(ValueError, match="differs from its score manifest"):
        write_population_archive(tmp_path / "archive", result)

    assert not (tmp_path / "archive/manifest.json").exists()
