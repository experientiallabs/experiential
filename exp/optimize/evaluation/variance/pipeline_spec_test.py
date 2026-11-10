"""Offline tests of the decision-variance stage graph contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.family_stages import MEASURE_PROFILE
from exp.optimize.evaluation.variance.pipeline import plan_decision_variance
from exp.optimize.evaluation.variance.pipeline_spec import (
    COUNCIL_ACTORS,
    STAGES,
    AuthoringSettings,
    CostLine,
    DecisionVarianceSpec,
    MiningSettings,
    PipelinePlan,
    StagePlan,
    TraceMLSettings,
    stage_identity,
)
from exp.optimize.evaluation.variance.pipeline_test import make_spec, wired


def test_explicit_request_capped_profile_reaches_every_derived_evidence_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A validated pipeline retains opt-in budgets without changing ordinary role defaults."""
    run = wired(tmp_path, monkeypatch)
    base = make_spec(run.fixture, run.rubric, run.bank)
    profile = MEASURE_PROFILE.model_copy(
        update={"budget": MEASURE_PROFILE.budget.model_copy(update={"maximum_usd": None})}
    )
    data = base.model_dump(mode="json")
    data["authoring"].update(
        decision_variance=True, decision_profile=profile.model_dump(mode="json")
    )
    data["records"]["option_profile"] = profile.model_dump(mode="json")
    spec = DecisionVarianceSpec.model_validate(data)
    measure, search, ladder = (
        spec.authoring.decision_measure(),
        spec.authoring.search_spec(),
        spec.authoring.ladder_spec(),
    )
    options = spec.point_options()
    assert measure is not None and search is not None and ladder is not None and options is not None
    assert all(
        bound == profile
        for bound in (
            measure.classifier_profile,
            measure.reviewer_profile,
            search.profile,
            ladder.profile,
            options.profile,
        )
    )
    assert base.authoring.decision_profile.budget.maximum_usd is not None
    assert base.records.option_profile.budget.maximum_usd is not None
    assert spec.authoring.decision_profile.budget.model_calls == MEASURE_PROFILE.budget.model_calls
    assert spec.slice("families") != base.slice("families")
    assert spec.slice("records") != base.slice("records")


def test_the_council_is_five_open_weight_tiers_in_stage_order() -> None:
    """The default actors are the five council actors, and the graph has six ordered stages."""
    assert COUNCIL_ACTORS == (
        "gemma-4-26b-a4b-it",
        "glm-5.3-flash",
        "glm-5.3",
        "kimi-k3",
        "mimo-v2.6-pro",
    )
    assert STAGES[0] == "variance-build" and STAGES[-1] == "records" and len(STAGES) == 6


def test_identities_chain_so_a_changed_slice_rekeys_only_later_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the records slice keeps every earlier identity; changing mining re-keys three."""
    run = wired(tmp_path, monkeypatch)
    base = make_spec(run.fixture, run.rubric, run.bank)

    def chain(spec: DecisionVarianceSpec) -> list[str]:
        """Every stage identity of a spec, chained in order."""
        identities: list[str] = []
        for stage in STAGES:
            identities.append(stage_identity(stage, spec, tuple(identities), "test"))
        return identities

    original = chain(base)
    records = chain(
        base.model_copy(update={"records": base.records.model_copy(update={"harbor": False})})
    )
    mining = chain(base.model_copy(update={"mining": base.mining.model_copy(update={"seed": 1})}))
    assert records[:5] == original[:5] and records[5] != original[5]
    assert mining[:3] == original[:3]
    assert all(a != b for a, b in zip(mining[3:], original[3:], strict=True))


def test_duplicate_or_invalid_tiers_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each actor is its own tier, named by a lowercase tier label."""
    run = wired(tmp_path, monkeypatch)
    base = make_spec(run.fixture, run.rubric, run.bank)
    data = base.mining.model_dump(mode="json")
    with pytest.raises(ValueError, match="distinct"):
        MiningSettings.model_validate({**data, "actors": ["actor", "actor"]})
    with pytest.raises(ValueError, match="tier labels"):
        MiningSettings.model_validate({**data, "actors": ["Actor"]})


def test_a_plan_totals_only_unsaved_stages_and_keeps_unpriced_lines_visible() -> None:
    """Saved stages cost nothing; unpriced components stay listed beside the priced bound."""
    plan = PipelinePlan(
        spec_sha256="a" * 64,
        code_revision="test",
        stages=(
            StagePlan(
                stage="rules",
                identity="r",
                saved=True,
                costs=(CostLine(name="author", attempts=3, usd=9.0),),
            ),
            StagePlan(
                stage="records",
                identity="s",
                saved=False,
                costs=(
                    CostLine(name="judge", attempts=2, usd=1.5),
                    CostLine(name="actor", attempts=4, usd=None),
                ),
            ),
        ),
    )
    assert plan.known_usd == 1.5 and not plan.complete
    assert [row.name for row in plan.unpriced] == ["actor"]


def test_an_enabled_traceml_stage_names_its_move_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No rule stands in for the pivot and return judgment, so the judge is required."""
    run = wired(tmp_path, monkeypatch)
    settings = run.fixture.settings
    with pytest.raises(ValueError, match="move_judge"):
        TraceMLSettings()
    with pytest.raises(ValueError, match="unknown traceml move variants"):
        TraceMLSettings(move_judge=settings, move_variants=("tool-change-rules",))
    assert TraceMLSettings(enabled=False).move_judge is None
    narrowed = TraceMLSettings(move_judge=settings, move_variants=("evidence-first",))
    spec = make_spec(run.fixture, run.rubric, run.bank)
    changed = spec.model_copy(update={"traceml": narrowed})
    assert spec.slice("traceml") != changed.slice("traceml")


def test_the_library_design_critic_and_decision_variance_are_on_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain authoring registration labels, curates, critiques and measures every family."""
    run = wired(tmp_path, monkeypatch)
    settings = run.fixture.settings
    authoring = AuthoringSettings(
        decision_author=settings,
        scenario_author=settings,
        world_author=settings,
        rubric_author=settings,
    )
    assert authoring.library_id is not None and authoring.labeler_variants
    assert authoring.rubric_profile.critic is not None
    measure = authoring.decision_measure()
    assert measure is not None and measure.classifier == measure.reviewer == settings
    off = authoring.model_copy(update={"decision_variance": False})
    assert off.decision_measure() is None


def test_search_ladder_and_point_decisions_are_on_by_default_on_the_authoring_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The search, the ladder and the option proposer reuse the open-weight authoring roles."""
    run = wired(tmp_path, monkeypatch)
    base = make_spec(run.fixture, run.rubric, run.bank)
    settings = run.fixture.settings
    authoring = base.authoring.model_copy(update={"decision_variance": True})
    search, ladder = authoring.search_spec(), authoring.ladder_spec()
    assert search is not None and search.author == authoring.scenario_author
    assert search.maximum_rounds == authoring.search_rounds == 3
    assert ladder is not None and ladder.author == authoring.scenario_author
    assert ladder.leak_reviewer == ladder.realism_reviewer == authoring.rubric_author
    assert ladder.strengths == ("strong", "medium", "weak") and ladder.maximum_rounds == 2
    off = authoring.model_copy(update={"decision_variance": False})
    assert off.search_spec() is None and off.ladder_spec() is None
    assert authoring.model_copy(update={"nudge_ladder": False}).ladder_spec() is None
    options = base.point_options()
    assert options is not None and options.proposer == base.authoring.decision_author == settings
    assert options.classifier == base.authoring.scenario_author
    quiet = base.records.model_copy(update={"point_decisions": False})
    assert base.model_copy(update={"records": quiet}).point_options() is None

    spec = base.model_copy(update={"authoring": authoring})
    plan = plan_decision_variance(run.project, spec, code_revision="test", executors=run.executors)
    families = next(row for row in plan.stages if row.stage == "families")
    records = next(row for row in plan.stages if row.stage == "records")
    assert any(line.startswith("Variant search: up to 3 rounds") for line in families.lines)
    assert any(line.startswith("Nudge ladder: up to 2 rounds") for line in families.lines)
    assert any(line.startswith("Point decisions:") for line in records.lines)
    bare = plan_decision_variance(
        run.project,
        spec.model_copy(
            update={
                "authoring": authoring.model_copy(
                    update={"variant_search": False, "nudge_ladder": False}
                )
            }
        ),
        code_revision="test",
        executors=run.executors,
    )
    priced = next(row for row in bare.stages if row.stage == "families")
    assert sum(row.attempts for row in families.costs) > sum(row.attempts for row in priced.costs)


def test_mining_draws_on_its_own_finite_simulator_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mining job's simulator ceiling is sized from the mining shape or set explicitly;
    the world's frozen semantics, and so the captured start, are unchanged."""
    from exp.common.core.artifacts import sha256_json
    from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition

    run = wired(tmp_path, monkeypatch)
    mining = make_spec(run.fixture, run.rubric, run.bank).mining
    job = mining.job
    assert job is not None and "maximum_physical_attempts" in job.configuration
    sized = mining.registered.configuration["maximum_physical_attempts"]
    rollouts = len(mining.actors) * (mining.attempts + mining.points * mining.continuations)
    reasks = job.configuration.get("maximum_reasks", 0)
    assert sized == min(100_000, rollouts * (job.maximum_model_calls or 1) * (1 + reasks) * 3)
    explicit = mining.model_copy(update={"simulator_physical_attempts": 12_345})
    assert explicit.registered.configuration["maximum_physical_attempts"] == 12_345
    before = GroundedRuntimeDefinition.model_validate(job.configuration)
    after = GroundedRuntimeDefinition.model_validate(explicit.registered.configuration)
    assert before.probe_sha256 == after.probe_sha256
    assert sha256_json(before.model_dump(mode="json", exclude={"maximum_physical_attempts"})) == (
        sha256_json(after.model_dump(mode="json", exclude={"maximum_physical_attempts"}))
    )
