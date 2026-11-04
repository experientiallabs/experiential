"""Offline tests for the family admission policy: coverage, rubric, compatibility and freeze."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.variance.admission import (
    FamilyAdmissionPolicy,
    VariantJudging,
    coverage_problems,
    judging_problems,
    variant_problems,
)
from exp.optimize.evaluation.variance.judged import JudgeCoverage
from exp.optimize.evaluation.variance.judged_test import WORLD, episode_judge, judge_settings
from exp.optimize.evaluation.variance.world_freeze import (
    FREEZE_TYPE,
    CompatibilityReview,
    Disagreement,
    VariantFreeze,
    WorldFreeze,
)
from exp.simulation.rules.properties import DEFAULT_REFUSAL, PropertyReport


def save_freeze(
    project: ProjectStore,
    *,
    world_sha256: str = WORLD,
    status: str = "frozen",
    trainable: bool = True,
) -> ArtifactInput:
    """Save a minimal world-freeze report with one variant ``v``."""
    freeze = WorldFreeze(
        identity="freeze",
        packet_sha256="1" * 64,
        world=ArtifactInput(artifact_id="world", sha256="2" * 64),
        world_sha256=world_sha256,
        properties=PropertyReport(transitions=0, refusal=DEFAULT_REFUSAL, results=()),
        workloads=(),
        probes=(),
        diagnostic_rules=(),
        problems=(),
        variants=(VariantFreeze.model_validate({"variant_id": "v", "status": status}),),
        trainable=trainable,
    )
    artifact_id = f"freeze-{world_sha256[:6]}-{status}-{str(trainable).lower()}"
    manifest = project.artifacts.write_json(
        artifact_id=artifact_id,
        artifact_type=FREEZE_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
        ),
        files={"freeze.json": freeze},
    )
    return artifact_input(manifest)


def bound_judging(
    project: ProjectStore,
    *,
    world_sha256: str = WORLD,
    status: str = "frozen",
    trainable: bool = True,
) -> VariantJudging:
    """A judge binding with a compatible review and a frozen world, unless overridden."""
    judge = episode_judge(project, judge_settings(project))
    definition = judge.definition
    review = CompatibilityReview(
        packet_sha256="1" * 64,
        world_sha256=definition.world_sha256,
        rubric_sha256=definition.rubric_sha256,
        disagreements=(),
        compatible=True,
    )
    return VariantJudging(
        judge=judge,
        world_freeze=save_freeze(
            project, world_sha256=world_sha256, status=status, trainable=trainable
        ),
        freeze_variant="v",
        compatibility=review,
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    """A project with a judge alias."""
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    return ProjectStore(tmp_path, "case")


def test_coverage_limits_count_operational_gaps_and_unscored_share() -> None:
    """Negatives are fine; operational gaps, too many unscored and rubric problems are not."""
    policy = FamilyAdmissionPolicy()
    negatives = JudgeCoverage(judge="0" * 64, planned=16, passed=2, failed=14)
    assert coverage_problems(negatives, policy) == ((), ())
    gaps = JudgeCoverage(judge="0" * 64, planned=16, passed=8, failed=4, malformed=1, gaps=3)
    covered, rubric = coverage_problems(gaps, policy)
    assert "4 episodes are operationally unscored" in covered[0]
    assert "4 of 16 episodes are unscored" in covered[1] and rubric == ()
    noted = JudgeCoverage(judge="0" * 64, planned=8, passed=8, rubric_problems=2)
    assert coverage_problems(noted, policy)[1] == ("2 judged episodes reported a rubric problem",)
    lenient = FamilyAdmissionPolicy(allow_operational_gaps=True, maximum_unscored_fraction=0.5)
    assert coverage_problems(gaps, lenient) == ((), ())


def test_a_compatible_frozen_variant_passes_and_each_defect_is_named(
    project: ProjectStore,
) -> None:
    """Missing review, disagreements, another world, untrainable or unfrozen freezes all fail."""
    good = bound_judging(project)
    assert variant_problems(project, good) == ((), ())
    review = good.compatibility
    assert review is not None
    disagreeing = review.model_copy(
        update={
            "compatible": False,
            "disagreements": (
                Disagreement(
                    kind="unobservable_evidence", subject="c", world="-", rubric="needs x"
                ),
            ),
        }
    )
    compatibility, _ = variant_problems(project, good.model_copy(update={"compatibility": None}))
    assert compatibility == ("no compatibility review of the rubric against the world",)
    compatibility, _ = variant_problems(
        project, good.model_copy(update={"compatibility": disagreeing})
    )
    assert compatibility == ("1 unreconciled world/rubric disagreements",)
    _, frozen = variant_problems(project, bound_judging(project, world_sha256="3" * 64))
    assert frozen == ("the freeze froze a different world than the judge's",)
    _, frozen = variant_problems(project, bound_judging(project, status="diagnostic"))
    assert frozen == ("freeze variant v is diagnostic",)
    _, frozen = variant_problems(project, bound_judging(project, trainable=False))
    assert frozen == ("the world freeze is not trainable",)
    _, frozen = variant_problems(project, good.model_copy(update={"world_freeze": None}))
    assert frozen == ("the judge's world has no freeze report for this variant",)


def test_judging_problems_cover_every_confirmed_variant(project: ProjectStore) -> None:
    """A confirmed variant without a binding, or judged by another judge, is recorded."""
    good = bound_judging(project)
    coverage = JudgeCoverage(judge=good.judge.identity, planned=4, passed=2, failed=2)
    other = coverage.model_copy(update={"judge": "0" * 64})
    found = judging_problems(
        project,
        {"x": good, "z": good},
        {"x": coverage, "y": coverage, "z": other},
        FamilyAdmissionPolicy(),
    )
    assert found["judge_coverage"] == {"z": ("confirmation was judged by another judge",)}
    assert found["rubric_compatibility"] == {"y": ("no judge binding",)}
    assert found["world_freeze"] == {"y": ("no judge binding",)}
    assert found["rubric_problems"] == {}
