"""Tests for benchmark-neutral harness scoring contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from wmh.harness.doc import HarnessDoc, SurfaceKind
from wmh.harness.scoring import (
    EvaluationArtifact,
    HarnessScore,
    HarnessScoreReport,
    ScoreCell,
    ScoreContext,
    ScoreRequest,
    score_harness,
)

_TASK_SET_DIGEST = "sha256:" + "a" * 64
_EVALUATOR_DIGEST = "sha256:" + "b" * 64
_EXECUTION_CONFIG_DIGEST = "sha256:" + "c" * 64
_RAW_PATH = "raw/evidence.txt"


class _ArtifactReader:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def read_bytes(self, path: str) -> bytes:
        return self.files[path]


def _context() -> ScoreContext:
    return ScoreContext(
        task_set_digest=_TASK_SET_DIGEST,
        evaluator_digest=_EVALUATOR_DIGEST,
        execution_config_digest=_EXECUTION_CONFIG_DIGEST,
    )


def _request() -> ScoreRequest:
    return ScoreRequest(context=_context(), task_ids=("task-a", "task-b"), attempts=2)


def _cells() -> tuple[ScoreCell, ...]:
    return (
        ScoreCell(
            task_id="task-a",
            attempt=1,
            score=1.0,
            passed=True,
            summary="completed",
            artifact_paths=(_RAW_PATH,),
        ),
        ScoreCell(
            task_id="task-a",
            attempt=2,
            score=0.0,
            passed=False,
            artifact_paths=(_RAW_PATH,),
        ),
        ScoreCell(
            task_id="task-b",
            attempt=1,
            score=0.5,
            passed=False,
            artifact_paths=(_RAW_PATH,),
        ),
        ScoreCell(
            task_id="task-b",
            attempt=2,
            score=1.0,
            passed=True,
            artifact_paths=(_RAW_PATH,),
        ),
    )


def _score(candidate: HarnessDoc, *, content: str = "raw trace\n") -> HarnessScore:
    artifact = EvaluationArtifact.from_text(path=_RAW_PATH, content=content)
    report = HarnessScoreReport(
        source_run_id="run-1",
        candidate_execution_hash=candidate.execution_hash,
        request=_request(),
        cells=_cells(),
        artifacts=(artifact,),
    )
    return HarnessScore(
        report=report,
        artifacts=_ArtifactReader({_RAW_PATH: content.encode("utf-8")}),
    )


@pytest.mark.parametrize("task_ids", [(), ("task-a", "task-a")])
def test_score_request_rejects_empty_or_duplicate_tasks(task_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="task_ids"):
        ScoreRequest(context=_context(), task_ids=task_ids, attempts=1)


def test_score_request_rejects_task_ids_that_cells_cannot_represent() -> None:
    with pytest.raises(ValidationError, match="512"):
        ScoreRequest(context=_context(), task_ids=("x" * 513,), attempts=1)


@pytest.mark.parametrize(
    "model,value",
    [
        (ScoreContext, _context().model_dump()),
        (ScoreRequest, _request().model_dump()),
        (
            EvaluationArtifact,
            EvaluationArtifact.from_text(path=_RAW_PATH, content="raw trace\n").model_dump(),
        ),
        (ScoreCell, _cells()[0].model_dump()),
    ],
)
def test_public_score_models_reject_unknown_fields(
    model: type[ScoreContext | ScoreRequest | EvaluationArtifact | ScoreCell],
    value: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate({**value, "unexpected": "value"})


def test_score_report_rejects_unknown_fields_and_invalid_candidate_hash() -> None:
    candidate = HarnessDoc.baseline("candidate")
    report = _score(candidate).report

    with pytest.raises(ValidationError, match="extra_forbidden"):
        HarnessScoreReport.model_validate({**report.model_dump(), "unexpected": "value"})
    with pytest.raises(ValidationError, match="candidate_execution_hash"):
        HarnessScoreReport.model_validate(
            {**report.model_dump(), "candidate_execution_hash": "not-a-hash"}
        )


def test_score_models_reject_stringified_or_integer_typed_measurements() -> None:
    request = _request().model_dump()
    cell = _cells()[0].model_dump()
    artifact = EvaluationArtifact.from_text(path=_RAW_PATH, content="raw trace\n").model_dump()

    with pytest.raises(ValidationError, match="attempts"):
        ScoreRequest.model_validate({**request, "attempts": "2"})
    with pytest.raises(ValidationError, match="attempt"):
        ScoreCell.model_validate({**cell, "attempt": "1"})
    with pytest.raises(ValidationError, match="score"):
        ScoreCell.model_validate({**cell, "score": "1.0"})
    with pytest.raises(ValidationError, match="passed"):
        ScoreCell.model_validate({**cell, "passed": 1})
    with pytest.raises(ValidationError, match="size_bytes"):
        EvaluationArtifact.model_validate({**artifact, "size_bytes": "10"})


@pytest.mark.parametrize(
    "cells,match",
    [
        (_cells()[:-1], "missing"),
        (
            (
                *_cells(),
                ScoreCell(
                    task_id="task-c",
                    attempt=1,
                    score=0.0,
                    passed=False,
                    artifact_paths=(_RAW_PATH,),
                ),
            ),
            "extra",
        ),
        ((_cells()[0], *_cells()), "duplicate"),
    ],
)
def test_score_report_requires_the_exact_task_attempt_matrix(
    cells: tuple[ScoreCell, ...], match: str
) -> None:
    candidate = HarnessDoc.baseline("candidate")
    artifact = EvaluationArtifact.from_text(path=_RAW_PATH, content="raw trace\n")

    with pytest.raises(ValidationError, match=match):
        HarnessScoreReport(
            source_run_id="run-1",
            candidate_execution_hash=candidate.execution_hash,
            request=_request(),
            cells=cells,
            artifacts=(artifact,),
        )


@pytest.mark.parametrize("path", ["", ".", "/absolute.txt", "../escape.txt", "a/../b"])
def test_artifact_manifest_rejects_unsafe_or_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="path"):
        EvaluationArtifact.from_text(path=path, content="raw trace")


def test_score_report_requires_raw_evidence_and_unique_artifact_paths() -> None:
    candidate = HarnessDoc.baseline("candidate")
    artifact = EvaluationArtifact.from_text(path=_RAW_PATH, content="raw trace\n")
    unreferenced = _cells()[0].model_copy(update={"artifact_paths": ("missing.txt",)})
    no_evidence = _cells()[0].model_copy(update={"artifact_paths": ()})

    with pytest.raises(ValidationError, match="artifact_paths"):
        ScoreCell.model_validate(no_evidence.model_dump())

    with pytest.raises(ValidationError, match="missing artifact"):
        HarnessScoreReport(
            source_run_id="run-1",
            candidate_execution_hash=candidate.execution_hash,
            request=_request(),
            cells=(unreferenced, *_cells()[1:]),
            artifacts=(artifact,),
        )

    with pytest.raises(ValidationError, match="duplicate artifact"):
        HarnessScoreReport(
            source_run_id="run-1",
            candidate_execution_hash=candidate.execution_hash,
            request=_request(),
            cells=_cells(),
            artifacts=(artifact, artifact),
        )


@pytest.mark.parametrize("score", [True, -0.01, 1.01, float("nan"), float("inf")])
def test_score_cell_rejects_boolean_nonfinite_or_out_of_range_scores(
    score: float | bool,
) -> None:
    with pytest.raises(ValidationError, match="score"):
        ScoreCell(
            task_id="task",
            attempt=1,
            score=score,
            passed=False,
            artifact_paths=(_RAW_PATH,),
        )


def test_raw_artifacts_are_hash_bound_and_read_without_truncation() -> None:
    candidate = HarnessDoc.baseline("candidate")
    content = "event\n" * 100_000

    class Scorer:
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            del request
            return _score(candidate, content=content)

    scored = score_harness(
        Scorer(),
        candidate,
        request=_request(),
    )

    assert scored.artifacts.read_bytes(_RAW_PATH) == content.encode("utf-8")
    assert scored.report.artifacts[0].content_hash.startswith("sha256:")
    assert scored.report.artifacts[0].size_bytes == len(content.encode("utf-8"))
    assert content not in scored.report.model_dump_json()


def test_artifact_manifest_and_reader_preserve_arbitrary_bytes() -> None:
    candidate = HarnessDoc.baseline("candidate")
    path = "raw/blob.bin"
    content = b"\x00\xffbinary\x00"
    artifact = EvaluationArtifact.from_bytes(path=path, content=content)
    cells = tuple(cell.model_copy(update={"artifact_paths": (path,)}) for cell in _cells())
    report = HarnessScoreReport(
        source_run_id="run-1",
        candidate_execution_hash=candidate.execution_hash,
        request=_request(),
        cells=cells,
        artifacts=(artifact,),
    )

    class Scorer:
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            del candidate, request
            return HarnessScore(report=report, artifacts=_ArtifactReader({path: content}))

    scored = score_harness(Scorer(), candidate, request=_request())

    assert scored.artifacts.read_bytes(path) == content


def test_report_identity_is_canonical_and_commits_to_all_evidence() -> None:
    candidate = HarnessDoc.baseline("candidate")
    scored = _score(candidate)
    report = scored.report
    reordered = HarnessScoreReport(
        source_run_id=report.source_run_id,
        candidate_execution_hash=report.candidate_execution_hash,
        request=report.request,
        cells=tuple(reversed(report.cells)),
        artifacts=tuple(reversed(report.artifacts)),
    )
    changed_cell = report.model_copy(
        update={
            "cells": (
                report.cells[0].model_copy(update={"score": 0.75}),
                *report.cells[1:],
            )
        }
    )
    changed_content = "different\n"
    changed_artifact = report.model_copy(
        update={
            "artifacts": (EvaluationArtifact.from_text(path=_RAW_PATH, content=changed_content),)
        }
    )
    changed_run = report.model_copy(update={"source_run_id": "run-2"})

    assert reordered.cells == report.cells
    assert reordered.artifacts == report.artifacts
    assert reordered.report_hash == report.report_hash
    assert changed_cell.report_hash != report.report_hash
    assert changed_artifact.report_hash != report.report_hash
    assert changed_run.report_hash != report.report_hash
    assert report.score == pytest.approx(0.625)
    # `passed` is evaluator-authoritative diagnostic evidence. Search ranks by `score`, not this.
    assert report.pass_rate == pytest.approx(0.5)


def test_score_harness_rejects_candidate_request_or_artifact_identity_drift() -> None:
    candidate = HarnessDoc.baseline("candidate")
    other = HarnessDoc(
        name="other",
        surfaces=[
            surface.model_copy(update={"content": "changed"})
            if surface.kind is SurfaceKind.PROMPT
            else surface
            for surface in candidate.surfaces
        ],
    )

    class WrongCandidateScorer:
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            del candidate, request
            return _score(other)

    with pytest.raises(ValueError, match="candidate execution hash"):
        score_harness(WrongCandidateScorer(), candidate, request=_request())

    class WrongRequestScorer:
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            del request
            scored = _score(candidate)
            wrong = scored.report.model_copy(
                update={
                    "request": ScoreRequest(context=_context(), task_ids=("task-a",), attempts=1)
                }
            )
            return HarnessScore(report=wrong, artifacts=scored.artifacts)

    with pytest.raises(ValueError, match="score request"):
        score_harness(WrongRequestScorer(), candidate, request=_request())

    class MutatedArtifactScorer:
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            del request
            scored = _score(candidate)
            source = _ArtifactReader({_RAW_PATH: b"mutated!!\n"})
            return HarnessScore(report=scored.report, artifacts=source)

    with pytest.raises(ValueError, match="artifact.*digest"):
        score_harness(MutatedArtifactScorer(), candidate, request=_request())


def test_score_harness_returns_a_detached_deeply_immutable_snapshot() -> None:
    candidate = HarnessDoc.baseline("candidate")
    original = _score(candidate)

    class Scorer:
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            assert candidate.execution_hash == original.report.candidate_execution_hash
            assert request == _request()
            return original

    snapshot = score_harness(Scorer(), candidate, request=_request())

    assert snapshot.report == original.report
    assert snapshot.report is not original.report
    assert snapshot.report.cells is not original.report.cells
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.report.source_run_id = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.report.cells[0].summary = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.report.artifacts[0].path = "changed.txt"
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.report.request.attempts = 3
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.report.request.context.task_set_digest = _EVALUATOR_DIGEST
    with pytest.raises(FrozenInstanceError):
        snapshot.report = original.report  # ty: ignore[invalid-assignment]


def test_score_harness_detects_artifact_mutation_after_snapshot() -> None:
    candidate = HarnessDoc.baseline("candidate")
    content = "raw trace\n"
    artifact = EvaluationArtifact.from_text(path=_RAW_PATH, content=content)
    source = _ArtifactReader({_RAW_PATH: content.encode("utf-8")})
    report = HarnessScoreReport(
        source_run_id="run-1",
        candidate_execution_hash=candidate.execution_hash,
        request=_request(),
        cells=_cells(),
        artifacts=(artifact,),
    )

    class Scorer:
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            del candidate, request
            return HarnessScore(report=report, artifacts=source)

    snapshot = score_harness(Scorer(), candidate, request=_request())
    source.files[_RAW_PATH] = b"mutated!!\n"

    with pytest.raises(ValueError, match="artifact.*digest"):
        snapshot.artifacts.read_bytes(_RAW_PATH)
