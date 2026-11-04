"""Offline tests for the family admission policy: coverage, rubric, freeze and provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.variance.admission import (
    FamilyAdmissionPolicy,
    VariantJudging,
    coverage_problems,
    frozen_run_digests,
    frozen_start,
    judging_problems,
    provenance_problems,
    variant_problems,
)
from exp.optimize.evaluation.variance.judged import JudgeCoverage
from exp.optimize.evaluation.variance.judged_test import (
    WORLD,
    episode_judge,
    judge_settings,
    run_digests,
)
from exp.optimize.evaluation.variance.world_freeze import (
    FREEZE_TYPE,
    CompatibilityReview,
    Disagreement,
    VariantFreeze,
    WorldFreeze,
)
from exp.simulation.continuations import persist_continuation_checkpoint
from exp.simulation.continuations_test import capsule
from exp.simulation.rules.properties import DEFAULT_REFUSAL, PropertyReport

Status = Literal["frozen", "diagnostic", "rejected"]
"""A freeze variant's status."""


def save_freeze(
    project: ProjectStore,
    *,
    world_sha256: str = WORLD,
    status: Status = "frozen",
    trainable: bool = True,
    variant: str = "v",
    start: ArtifactInput | None = None,
    runtime: JsonObject | None = None,
) -> ArtifactInput:
    """Save a minimal world-freeze report with one variant, its saved start and runtime digest."""
    row = VariantFreeze(
        variant_id=variant,
        status=status,
        checkpoint=start,
        definition_sha256=None if runtime is None else sha256_json(runtime),
        restored=None if start is None else True,
    )
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
        variants=(row,),
        trainable=trainable,
    )
    artifact_id = stable_id("freeze", freeze)
    if not project.artifacts.exists(artifact_id):
        project.artifacts.write_json(
            artifact_id=artifact_id,
            artifact_type=FREEZE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
            ),
            files={"freeze.json": freeze},
        )
    return artifact_input(project.artifacts.read(artifact_id).manifest)


def bound_judging(
    project: ProjectStore,
    start: ArtifactInput,
    runtime: JsonObject,
    *,
    world_sha256: str = WORLD,
    status: Status = "frozen",
    trainable: bool = True,
    variant: str = "v",
) -> VariantJudging:
    """A judge bound to a run from ``start`` under ``runtime``, frozen there by a world freeze.

    The freeze records ``start`` as the variant's saved checkpoint and ``runtime``'s digest as
    its definition, the review covers the frozen world, and the judge binds the run's digests.
    """
    judge = episode_judge(project, judge_settings(project), **run_digests(project, start, runtime))
    review = CompatibilityReview(
        packet_sha256="1" * 64,
        world_sha256=WORLD,
        rubric_sha256=judge.definition.rubric_sha256,
        disagreements=(),
        compatible=True,
    )
    freeze = save_freeze(
        project,
        world_sha256=world_sha256,
        status=status,
        trainable=trainable,
        variant=variant,
        start=start,
        runtime=runtime,
    )
    return VariantJudging(
        judge=judge,
        world_freeze=freeze,
        freeze_variant=variant,
        runtime=runtime,
        compatibility=review,
    )


def unfrozen_judging(
    project: ProjectStore, start: ArtifactInput, runtime: JsonObject
) -> VariantJudging:
    """A judge bound to a run from ``start`` under ``runtime`` with no world freeze."""
    frozen = bound_judging(project, start, runtime)
    return frozen.model_copy(update={"world_freeze": None, "freeze_variant": None})


RUNTIME: JsonObject = {"runtime": "frozen-variant"}
"""Runtime configuration standing in for a frozen variant's grounded runtime definition."""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    """A project with a judge alias."""
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    return ProjectStore(tmp_path, "case")


@pytest.fixture
def start(project: ProjectStore) -> ArtifactInput:
    """A saved starting checkpoint standing in for a frozen variant's start."""
    checkpoint, _ = capsule(project.artifacts)
    return persist_continuation_checkpoint(
        project.artifacts, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )


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
    project: ProjectStore, start: ArtifactInput
) -> None:
    """Missing review, disagreements, another world, untrainable or unfrozen freezes all fail."""
    good = bound_judging(project, start, RUNTIME)
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
    other_world = bound_judging(project, start, RUNTIME, world_sha256="3" * 64)
    compatibility, _ = variant_problems(project, other_world)
    assert compatibility == ("the compatibility review covers another rubric or world",)
    _, frozen = variant_problems(
        project, bound_judging(project, start, RUNTIME, status="diagnostic")
    )
    assert frozen == ("freeze variant v is diagnostic",)
    _, frozen = variant_problems(project, bound_judging(project, start, RUNTIME, trainable=False))
    assert frozen == ("the world freeze is not trainable",)
    _, frozen = variant_problems(project, good.model_copy(update={"world_freeze": None}))
    assert frozen == ("the judge's world has no freeze report for this variant",)


def test_contrasts_must_start_from_the_frozen_checkpoint_under_the_frozen_runtime(
    project: ProjectStore, start: ArtifactInput
) -> None:
    """Matching digests are not enough: the start, the runtime and the judge must all be frozen."""
    good = bound_judging(project, start, RUNTIME)
    assert frozen_start(project, good) == (start, RUNTIME)
    assert good.world_freeze is not None
    assert frozen_run_digests(project, good.world_freeze, "v") == (
        good.judge.definition.task_sha256,
        good.judge.definition.world_sha256,
    )
    assert provenance_problems(project, good, (start, start)) == ()

    elsewhere = ArtifactInput(artifact_id="another-start", sha256="4" * 64)
    assert provenance_problems(project, good, (start, elsewhere)) == (
        f"a contrast started from another-start, not the frozen start {start.artifact_id}",
    )
    assert provenance_problems(project, good, ()) == ("no contrast of this variant was recorded",)
    swapped = good.model_copy(update={"runtime": {"runtime": "another"}})
    assert provenance_problems(project, swapped, (start,)) == (
        "the variant's runtime is not the freeze's recorded definition",
    )
    with pytest.raises(ValueError, match="frozen definition"):
        frozen_start(project, swapped)
    other_judge = good.model_copy(
        update={"judge": unfrozen_judging(project, start, {"runtime": "another"}).judge}
    )
    assert provenance_problems(project, other_judge, (start,)) == (
        "the judge binds another runtime than the frozen definition",
    )
    unfrozen = unfrozen_judging(project, start, RUNTIME)
    assert frozen_start(project, unfrozen) is None
    assert provenance_problems(project, unfrozen, (start,)) == (
        "no world-freeze report binds this variant, so no frozen start was used",
    )
    unstarted = good.model_copy(update={"world_freeze": save_freeze(project)})
    assert provenance_problems(project, unstarted, (start,)) == (
        "freeze variant v has no saved frozen start",
    )
    with pytest.raises(ValueError, match="no saved start"):
        frozen_start(project, unstarted)


def test_judging_problems_cover_every_confirmed_variant(
    project: ProjectStore, start: ArtifactInput
) -> None:
    """A confirmed variant without a binding, judged by another judge, or run elsewhere is named."""
    good = bound_judging(project, start, RUNTIME)
    coverage = JudgeCoverage(judge=good.judge.identity, planned=4, passed=2, failed=2)
    other = coverage.model_copy(update={"judge": "0" * 64})
    elsewhere = ArtifactInput(artifact_id="another-start", sha256="4" * 64)
    found = judging_problems(
        project,
        {"x": good, "z": good},
        {"x": coverage, "y": coverage, "z": other},
        FamilyAdmissionPolicy(),
        {"x": (start, start), "z": (elsewhere,)},
    )
    assert found["judge_coverage"] == {"z": ("confirmation was judged by another judge",)}
    assert found["rubric_compatibility"] == {"y": ("no judge binding",)}
    assert found["world_freeze"] == {"y": ("no judge binding",)}
    assert found["world_provenance"] == {
        "y": ("no judge binding",),
        "z": (f"a contrast started from another-start, not the frozen start {start.artifact_id}",),
    }
    assert found["rubric_problems"] == {}
