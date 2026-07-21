"""Tests for ordered scoring of complete harnesses."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from wmh.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmh.harness.score_batch import HarnessScoreTarget, score_harnesses
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


@dataclass(frozen=True)
class _Reader:
    content: bytes

    def read_bytes(self, path: str) -> bytes:
        assert path == "raw/evidence.json"
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


def _target(label: str, prompt: str) -> HarnessScoreTarget:
    source = HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content=prompt),
            HarnessSourceFile(
                path="config.toml",
                content='[harness]\ntools = ["bash", "submit"]\n',
            ),
        )
    )
    return HarnessScoreTarget(
        label=label,
        harness=source.to_doc(label.replace("@", "-")),
    )


class _Scorer:
    def __init__(self, request: ScoreRequest) -> None:
        self.request = request
        self.calls: list[str] = []

    def score(self, candidate: object, *, request: ScoreRequest) -> HarnessScore:
        from wmh.harness.doc import HarnessDoc

        assert isinstance(candidate, HarnessDoc)
        assert request is self.request
        self.calls.append(candidate.name)
        content = candidate.name.encode()
        artifact = EvaluationArtifact.from_bytes(
            path="raw/evidence.json",
            content=content,
            media_type="application/json",
        )
        report = HarnessScoreReport(
            source_run_id=f"run-{candidate.name}",
            candidate_doc_hash=candidate.doc_hash,
            request=request,
            cells=(
                ScoreCell(
                    task_id="task-a",
                    attempt=1,
                    score=float(len(self.calls) - 1),
                    passed=len(self.calls) == 2,
                    artifact_paths=("raw/evidence.json",),
                ),
            ),
            artifacts=(artifact,),
        )
        return HarnessScore(report=report, artifacts=_Reader(content))


def test_score_harnesses_preserves_declared_order_and_one_request() -> None:
    request = _request()
    scorer = _Scorer(request)
    targets = (_target("first", "one"), _target("second@v2", "two"))

    result = score_harnesses(scorer, targets, request=request)

    assert scorer.calls == ["first", "second-v2"]
    assert [entry.target.label for entry in result.entries] == ["first", "second@v2"]
    assert [entry.score.report.score for entry in result.entries] == [0.0, 1.0]
    assert all(entry.score.report.request is not request for entry in result.entries)
    assert all(entry.score.report.request == request for entry in result.entries)


def test_score_harnesses_rejects_duplicate_labels_before_spend() -> None:
    request = _request()
    scorer = _Scorer(request)

    with pytest.raises(ValueError, match="duplicate score target label"):
        score_harnesses(
            scorer,
            (_target("same", "one"), _target("same", "two")),
            request=request,
        )

    assert scorer.calls == []


def test_score_target_exports_source_from_the_authoritative_document() -> None:
    target = _target("source", "one")

    assert target.source == HarnessSourceTree.from_doc(target.harness)


def test_score_target_rejects_unexportable_document_during_resolution() -> None:
    harness = HarnessDoc(
        name="invalid-export",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(
                id="code:reserved",
                kind=SurfaceKind.CODE,
                path="SYSTEM.md",
                content="conflict",
            ),
        ],
    )

    with pytest.raises(ValueError, match="collides with a reserved file"):
        HarnessScoreTarget(label="invalid-export", harness=harness)


@pytest.mark.parametrize("label", ["", "bad\nlabel"])
def test_score_target_rejects_unsafe_labels(label: str) -> None:
    source = _target("valid", "one").source

    with pytest.raises(ValueError, match="label"):
        HarnessScoreTarget(label=label, harness=source.to_doc("valid"))
