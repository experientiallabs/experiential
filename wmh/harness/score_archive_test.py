"""Tests for neutral, self-contained multi-harness score archives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from wmh.harness.doc import HarnessDoc
from wmh.harness.score_archive import write_score_archive
from wmh.harness.score_batch import HarnessScoreBatch, HarnessScoreTarget, ScoredHarness
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


@dataclass(frozen=True)
class _Reader:
    content: bytes

    def read_bytes(self, path: str) -> bytes:
        assert path == _ARTIFACT_PATH
        return self.content


def _request() -> ScoreRequest:
    return ScoreRequest(
        context=ScoreContext(
            task_set_digest=_DIGEST,
            evaluator_digest=_DIGEST,
            execution_config_digest=_DIGEST,
        ),
        task_ids=("task-a",),
        attempts=1,
    )


def _entry(label: str, prompt: str, value: float, request: ScoreRequest) -> ScoredHarness:
    source = HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content=prompt),
            HarnessSourceFile(
                path="config.toml",
                content='[harness]\ntools = ["bash", "submit"]\n',
            ),
        )
    )
    harness = source.to_doc(label.replace("@", "-"))
    content = f"evidence:{label}".encode()
    artifact = EvaluationArtifact.from_bytes(path=_ARTIFACT_PATH, content=content)
    report = HarnessScoreReport(
        source_run_id=f"run-{label}",
        candidate_doc_hash=harness.doc_hash,
        request=request,
        cells=(
            ScoreCell(
                task_id="task-a",
                attempt=1,
                score=value,
                passed=value == 1.0,
                artifact_paths=(_ARTIFACT_PATH,),
            ),
        ),
        artifacts=(artifact,),
    )
    return ScoredHarness(
        target=HarnessScoreTarget(label=label, harness=harness),
        score=HarnessScore(report=report, artifacts=_Reader(content)),
    )


def _batch() -> HarnessScoreBatch:
    request = _request()
    return HarnessScoreBatch(
        request=request,
        entries=(
            _entry("default", "base", 0.0, request),
            _entry("candidate@v2", "changed", 1.0, request),
        ),
    )


def test_archive_copies_ordered_sources_reports_and_verified_artifacts(tmp_path: Path) -> None:
    destination = tmp_path / "scores"

    manifest_path = write_score_archive(destination, _batch())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "best" not in manifest
    assert "winner" not in manifest
    assert [entry["label"] for entry in manifest["entries"]] == [
        "default",
        "candidate@v2",
    ]
    assert [entry["score"] for entry in manifest["entries"]] == [0.0, 1.0]
    assert (destination / "entries/0001/source/SYSTEM.md").read_text() == "changed"
    document = json.loads((destination / "entries/0001/document.json").read_text(encoding="utf-8"))
    report = json.loads((destination / "entries/0001/score.json").read_text(encoding="utf-8"))
    assert document["name"] == "candidate-v2"
    authoritative = HarnessDoc.model_validate(document)
    assert authoritative.doc_hash == report["candidate_doc_hash"]
    assert manifest["entries"][1]["document_path"] == "entries/0001/document.json"
    exported = HarnessSourceTree(
        files=tuple(
            HarnessSourceFile(
                path=path.relative_to(destination / "entries/0001/source").as_posix(),
                content=path.read_text(encoding="utf-8"),
            )
            for path in sorted((destination / "entries/0001/source").rglob("*"))
            if path.is_file()
        )
    )
    assert exported.to_doc(authoritative.name).doc_hash != authoritative.doc_hash
    assert (destination / "entries/0001/artifacts/raw/evidence.bin").read_bytes() == (
        b"evidence:candidate@v2"
    )
    assert report["request"] == manifest["score_request"]


def test_archive_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "scores"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("owned", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_score_archive(destination, _batch())

    assert sentinel.read_text(encoding="utf-8") == "owned"


def test_archive_manifest_is_absent_when_atomic_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "scores"

    def fail_replace(self: Path, target: Path) -> Path:
        del self, target
        raise OSError("publication interrupted")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="publication interrupted"):
        write_score_archive(destination, _batch())

    assert not (destination / "manifest.json").exists()
    assert (destination / "manifest.json.tmp").exists()


def test_archive_rejects_artifact_bytes_that_differ_from_manifest(tmp_path: Path) -> None:
    batch = _batch()
    first = batch.entries[0]
    broken = ScoredHarness(
        target=first.target,
        score=HarnessScore(report=first.score.report, artifacts=_Reader(b"different")),
    )
    broken_batch = HarnessScoreBatch(
        request=batch.request,
        entries=(broken, *batch.entries[1:]),
    )

    with pytest.raises(ValueError, match="differs from its score manifest"):
        write_score_archive(tmp_path / "scores", broken_batch)

    assert not (tmp_path / "scores/manifest.json").exists()
