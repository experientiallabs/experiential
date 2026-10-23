"""Task-quality approval cannot be manufactured from model votes or verifier success."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.factory.candidates import run_candidate_batch
from exp.optimize.evaluation.factory.candidates_test import batch_fixture
from exp.optimize.evaluation.factory.quality import (
    QUALITY_AXES,
    QualityFinding,
    TaskQualityReview,
    validate_quality_reviews,
)
from exp.optimize.evaluation.factory.quality_proxy_test import proxy_transport


def review() -> TaskQualityReview:
    """Build a software-only review with synthetic reference identities."""
    source = ArtifactInput(artifact_id="source", sha256="a" * 64)
    return TaskQualityReview(
        task=source,
        proposals=source,
        reviewer="software-fixture",
        authority="human",
        approved=True,
        findings=tuple(
            QualityFinding(
                axis=axis, verdict="pass", rationale="Fixture evidence", sources=(source,)
            )
            for axis in QUALITY_AXES
        ),
    )


def test_all_axes_and_independent_attestation_are_required() -> None:
    """Neither incomplete criteria nor a model's approval can satisfy the quality gate."""
    original = review()
    assert original.disposition() == "accepted"
    for findings in (original.findings[:-1], (*original.findings, original.findings[0])):
        with pytest.raises(ValidationError):
            TaskQualityReview.model_validate({**original.model_dump(), "findings": findings})
    for authority in ("generated_hypothesis", "native_execution", "recorded_reference"):
        with pytest.raises(ValidationError, match="human review"):
            TaskQualityReview.model_validate({**original.model_dump(), "authority": authority})
    assert original.model_copy(update={"approved": False}).disposition() == "needs_review"
    assert (
        original.model_copy(
            update={"authority": "generated_hypothesis", "approved": False}
        ).disposition()
        == "needs_review"
    )
    for verdict, expected in (("drop", "rejected"), ("revise", "needs_review")):
        changed = original.model_copy(
            update={
                "findings": (
                    original.findings[0].model_copy(update={"verdict": verdict}),
                    *original.findings[1:],
                )
            }
        )
        assert changed.disposition() == expected


def test_successful_actors_and_controls_do_not_replace_quality_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review can be added without repurchasing the already saved actors or author draw."""
    project, spec, transport = batch_fixture(tmp_path, monkeypatch)
    pending = spec.model_copy(update={"quality_reviews": ()})
    proxy_transport(
        monkeypatch, tuple(row.task for row in spec.quality_reviews), verdict="unassessable"
    )
    measured = run_candidate_batch(project, pending, code_revision="test")
    assert measured.counts["needs_review"] == 2
    assert all(
        row.qualification and row.panel and row.package is None for row in measured.candidates
    )
    assert spec.destination is not None and not spec.destination.exists()
    attempts = len(transport.requests)
    assert attempts == 12
    approved = run_candidate_batch(project, spec, code_revision="test")
    assert approved.counts["packaged"] == 2
    assert len(transport.requests) == attempts
    assert run_candidate_batch(project, pending, code_revision="test") == measured
    assert len(transport.requests) == attempts
    rejected = spec.model_copy(
        update={
            "quality_reviews": tuple(
                row.model_copy(
                    update={
                        "findings": (
                            row.findings[0].model_copy(update={"verdict": "drop"}),
                            *row.findings[1:],
                        )
                    }
                )
                for row in spec.quality_reviews
            )
        }
    )
    dropped = run_candidate_batch(project, rejected, code_revision="test")
    assert dropped.counts["rejected_quality"] == 2
    assert all(row.package is None for row in dropped.candidates)
    assert len(transport.requests) == attempts


def test_review_cannot_transfer_to_changed_prompt_or_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact task/recipe binding and source hashes are checked before provider calls."""
    project, spec, transport = batch_fixture(tmp_path, monkeypatch)
    original = spec.quality_reviews[0]
    for changed in (
        original.model_copy(update={"task": spec.proposals}),
        original.model_copy(update={"proposals": original.task}),
    ):
        with pytest.raises(ValueError, match="another task or generation recipe"):
            run_candidate_batch(
                project,
                spec.model_copy(update={"quality_reviews": (changed,)}),
                code_revision="test",
            )
    with pytest.raises(ValueError, match="distinct candidates"):
        validate_quality_reviews(
            project,
            (original, original),
            proposals=spec.proposals,
            tasks=frozenset(row.task for row in spec.quality_reviews),
        )
    source = original.findings[0].sources[0]
    changed = original.model_copy(
        update={
            "findings": (
                original.findings[0].model_copy(
                    update={"sources": (source.model_copy(update={"sha256": "0" * 64}),)}
                ),
                *original.findings[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="exact saved evidence"):
        validate_quality_reviews(
            project,
            (changed,),
            proposals=spec.proposals,
            tasks=frozenset(row.task for row in spec.quality_reviews),
        )
    assert not transport.requests


def test_factory_retains_missing_quality_as_pending_instead_of_rejecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A good mechanical result cannot fill a family quota without quality evidence."""
    from exp.optimize.evaluation.factory.runner import run_task_factory
    from exp.optimize.evaluation.factory.runner_test import factory_fixture

    project, spec, transport = factory_fixture(tmp_path, monkeypatch)
    proxy_transport(
        monkeypatch,
        tuple(row.task for row in spec.families[0].quality_reviews),
        verdict="unassessable",
    )
    pending = spec.model_copy(
        update={"families": (spec.families[0].model_copy(update={"quality_reviews": ()}),)}
    )
    report = run_task_factory(project, pending, code_revision="test")
    assert report.status == "needs_review" and report.cells[0].status == "needs_review"
    assert report.counts["qualified_seed_families"] == 0
    assert report.counts["packaged_candidates"] == 0
    assert len(transport.requests) == 12
