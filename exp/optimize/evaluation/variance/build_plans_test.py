"""A reused variance build plan is the plan: same identity, same front-door digest, fewer reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.optimize.evaluation.variance import build_plans
from exp.optimize.evaluation.variance.bootstrap_selection_test import preparation_fixture
from exp.optimize.evaluation.variance.build import plan_variance_build, run_variance_build
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger
from exp.optimize.evaluation.variance.frontdoor import (
    FrontDoorApproval,
    plan_front_door,
    run_front_door,
)
from exp.optimize.evaluation.variance.frontdoor_test import front_door


def test_a_reused_plan_equals_a_fresh_one_and_the_build_is_prepared_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Intent: reuse never changes what the plan says, and ``saved`` follows the store."""
    build_plans.clear_planned_variance_builds()
    project, preparation, reviewer = preparation_fixture(tmp_path, monkeypatch)
    spec = preparation.build.spec
    assert preparation.build.reviewer is not None
    args = {
        "code_revision": "test",
        "reviewer_model": reviewer.model,
        "reviewer_configuration_sha256": reviewer.configuration_sha256,
    }
    calls: list[str] = []

    def counted(*a: object, **k: object):  # noqa: ANN202
        calls.append("plan")
        return plan_variance_build(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(build_plans, "plan_variance_build", counted)
    fresh = plan_variance_build(project, spec, **args)
    assert fresh.embedding_inputs_saved == fresh.embedding_inputs
    assert build_plans.planned_variance_build(project, spec, **args) == fresh
    assert build_plans.planned_variance_build(project, spec, **args) == fresh
    assert calls == ["plan"] and not fresh.saved

    tagger = _tagger(project, FakeDecisions())
    report = run_variance_build(
        project, spec, code_revision="test", tagger=tagger, reviewer=reviewer
    )
    after = plan_variance_build(project, spec, **args)
    assert after.saved and after == fresh.model_copy(update={"saved": True})
    assert build_plans.planned_variance_build(project, spec, **args) == after
    replayed = build_plans.replay_or_run_variance_build(
        project, spec, code_revision="test", tagger=tagger, reviewer=reviewer
    )
    assert replayed == report and calls == ["plan"]
    other = {**args, "reviewer_configuration_sha256": sha256_json("another reviewer")}
    assert build_plans.planned_variance_build(project, spec, **other).identity != after.identity
    assert calls == ["plan", "plan"]


def test_the_front_door_plan_digest_is_the_same_with_and_without_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Intent: the plan digest a run consented to stays valid; reuse only skips reads."""
    run = front_door(tmp_path, monkeypatch)
    project, spec, deps = run.project, run.spec, run.dependencies

    def plans() -> tuple[str, str]:
        build_plans.clear_planned_variance_builds()
        with monkeypatch.context() as off:
            off.setattr(build_plans, "_key", lambda *a: None)  # no reuse
            cold = plan_front_door(project, spec, code_revision="test", dependencies=deps)
        warm = plan_front_door(project, spec, code_revision="test", dependencies=deps)
        again = plan_front_door(project, spec, code_revision="test", dependencies=deps)
        assert cold.model_dump() == warm.model_dump() == again.model_dump()
        return cold.digest, warm.digest

    cold, warm = plans()
    assert cold == warm
    approval = FrontDoorApproval(plan_sha256=warm, ceiling_usd=10.0, unpriced_attempts=1_000)
    run_front_door(project, spec, code_revision="test", approval=approval, dependencies=deps)
    replay_cold, replay_warm = plans()
    assert replay_cold == replay_warm
