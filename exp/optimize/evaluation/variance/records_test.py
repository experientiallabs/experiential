"""Offline tests for audit dispositions, computed eligibility and confirmation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import BillingSource, ModelSnapshot, ToolCall
from exp.common.project import ProjectStore
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.variance import actor_critic_test as mining
from exp.optimize.evaluation.variance.actor_critic import (
    ActorCriticReport,
    ActorCriticSpec,
    run_actor_critic,
)
from exp.optimize.evaluation.variance.contrast import (
    FlipVerdict,
    OptionContrast,
    OptionOutcome,
    VariantVerdict,
)
from exp.optimize.evaluation.variance.episodes import JudgedEpisode
from exp.optimize.evaluation.variance.exposure import EstimateMeaning
from exp.optimize.evaluation.variance.judge import judge_episode
from exp.optimize.evaluation.variance.judge_test import _evidence
from exp.optimize.evaluation.variance.judged import EpisodeJudge, JudgedCell, JudgedSlot
from exp.optimize.evaluation.variance.judged_test import episode_judge, judge_settings, route_judge
from exp.optimize.evaluation.variance.records import (
    CONDITIONS,
    AuditDisposition,
    AuditFinding,
    EligibilityCondition,
    JudgeScreening,
    TrainingEligibility,
    TrainingRecords,
    audit_finding,
    pending_findings,
    policy_key,
    selector_records,
    training_eligibility,
    training_records,
    unreproduced,
)
from exp.optimize.evaluation.variance.rewards import (
    DEFAULT_SELECTOR_OBJECTIVE,
    JudgeScore,
    PointDecision,
    SelectorObjective,
    SelectorObjectiveV2,
    group_advantages,
    score_selector,
    selector_score,
)

_SNAPSHOT = ModelSnapshot(
    provider="fixture",
    model_id="actor-fixture",
    billing_source=BillingSource.CUSTOMER_MANAGED,
    capabilities_sha256="a" * 64,
    connection_sha256="b" * 64,
)


def _meaning(
    checkpoint: str = "c-0", *, temperature: float = 1.0, judge: str = "9" * 64
) -> EstimateMeaning:
    """Build a meaning at one checkpoint under the fixture policy."""
    return EstimateMeaning(
        checkpoint=ArtifactInput(artifact_id=checkpoint, sha256="0" * 64),
        actor_model=_SNAPSHOT,
        harness_sha256="h",
        system_prompt_sha256="s",
        maximum_model_calls=4,
        remaining_calls=4,
        temperature=temperature,
        output_tokens=1024,
        reasoning_effort=None,
        world_sha256="w",
        judge_sha256=judge,
        probe_sha256="p",
    )


def _verdict(scenario: str, status: str, leader: str = "a") -> VariantVerdict:
    """Build one variant verdict."""
    return VariantVerdict.model_validate(
        {"scenario_id": scenario, "correct_option": leader, "status": status, "leader": leader}
    )


def _flip(*variants: VariantVerdict) -> FlipVerdict:
    """Build a family verdict from variant verdicts, qualifying on two matching winners."""
    winners = tuple(sorted({v.leader for v in variants if v.status == "separated_matches"}))
    return FlipVerdict(
        qualifies=len(winners) >= 2, winners=tuple(w for w in winners if w), variants=variants
    )


def test_finding_identity_hashes_content_and_refuses_a_forged_identity() -> None:
    """Equal findings share an identity; another identity for the same content is refused."""
    first = audit_finding("actor_critic", {"category": "guess_required", "draw": 0})
    assert first == audit_finding("actor_critic", {"draw": 0, "category": "guess_required"})
    assert first.finding_id != audit_finding("decision_state", first.detail).finding_id
    with pytest.raises(ValidationError, match="hash"):
        AuditFinding(finding_id="0" * 64, source="actor_critic", detail=first.detail)


def test_pending_findings_lists_each_unresolved_finding_once() -> None:
    """Only findings without a disposition remain, repeats collapsed, in input order."""
    one = audit_finding("scenario", {"reason": "one"})
    two = audit_finding("variant_review", {"reason": "two"})
    resolved = AuditDisposition(
        finding_id=one.finding_id,
        resolution="accepted_with_reason",
        rationale="Simulated clock is not part of this decision.",
        resolver="reviewer",
    )
    assert pending_findings((one, two, two), (resolved,)) == (two,)
    assert pending_findings((one,), (resolved,)) == ()


def test_eligibility_reports_every_condition_and_cannot_be_asserted() -> None:
    """All failing conditions are listed; omitting a condition is refused."""
    finding = audit_finding("decision_state", {"category": "answer_leakage"})
    eligibility = training_eligibility(
        selection=None,
        confirmation=None,
        findings=(finding,),
        dispositions=(),
        critics=(),
        uncovered_tools=("lookup",),
        relied_rules=("rule.a", "rule.b"),
        contradicted_rules=("rule.a",),
        unwitnessed_rules=("rule.a", "rule.b"),
        meanings=(_meaning(), _meaning(temperature=0.0)),
        judging={},
    )
    assert not eligibility.eligible
    failed = [c.name for c in eligibility.conditions if not c.holds]
    assert failed == [
        "family_flip",
        "fresh_confirmation",
        "audit_dispositions",
        "tool_coverage",
        "rule_witnesses",
        "policy_consistency",
        "judge_coverage",
        "rubric_problems",
        "rubric_compatibility",
        "world_freeze",
        "world_provenance",
    ]
    assert any("rule.a was contradicted" in r and "rule.b has no" in r for r in eligibility.reasons)
    assert eligibility.reasons[-1] == (
        "world_provenance: no confirmed variant was checked against the policy"
    )
    with pytest.raises(ValidationError, match="every condition"):
        TrainingEligibility(
            conditions=(EligibilityCondition(name="family_flip", holds=True, detail="ok"),)
        )
    assert len(CONDITIONS) == len(eligibility.conditions)


def test_policy_key_ignores_only_where_the_estimate_started() -> None:
    """Checkpoint and remaining calls differ within a policy; sampling settings do not."""
    moved = _meaning("c-1", judge="8" * 64).model_copy(update={"remaining_calls": 2})
    assert policy_key(_meaning()) == policy_key(moved)
    assert policy_key(_meaning()) != policy_key(_meaning(temperature=0.0))


def test_confirmation_must_reproduce_every_selected_winner() -> None:
    """A variant whose fresh contrast separates differently, or not at all, is unreproduced."""
    selection = _flip(
        _verdict("x", "separated_matches", "a"),
        _verdict("y", "separated_matches", "b"),
        _verdict("z", "unseparated"),
    )
    same = _flip(_verdict("x", "separated_matches", "a"), _verdict("y", "separated_matches", "b"))
    assert unreproduced(selection, same) == ()
    weaker = _flip(_verdict("x", "separated_matches", "a"), _verdict("y", "unseparated", "b"))
    assert unreproduced(selection, weaker) == ("y",)
    assert unreproduced(selection, None) == ("x", "y")


def test_each_failed_admission_check_is_recorded_per_variant() -> None:
    """Admission problems name the failing variant; an empty map means every check passed."""
    selection = _flip(
        _verdict("x", "separated_matches", "a"), _verdict("y", "separated_matches", "b")
    )

    def eligibility(problems: dict[str, dict[str, tuple[str, ...]]]) -> TrainingEligibility:
        """Compute eligibility for a confirmed family with the given admission problems."""
        return training_eligibility(
            selection=selection,
            confirmation=selection,
            findings=(),
            dispositions=(),
            critics=(),
            uncovered_tools=(),
            relied_rules=(),
            contradicted_rules=(),
            unwitnessed_rules=(),
            meanings=(_meaning(), _meaning("c-1", judge="7" * 64)),
            judging=problems,
        )

    names = (
        "judge_coverage",
        "rubric_problems",
        "rubric_compatibility",
        "world_freeze",
        "world_provenance",
    )
    assert eligibility({name: {} for name in names}).eligible
    blocked = eligibility(
        {
            **{name: {} for name in names},
            "judge_coverage": {"y": ("2 episodes are operationally unscored",)},
            "world_freeze": {"x": ("the world freeze is not trainable",)},
        }
    )
    assert [c.name for c in blocked.conditions if not c.holds] == ["judge_coverage", "world_freeze"]
    assert blocked.reasons == (
        "judge_coverage: y: 2 episodes are operationally unscored",
        "world_freeze: x: the world freeze is not trainable",
    )
    with pytest.raises(ValidationError, match="every condition"):
        TrainingEligibility(conditions=blocked.conditions[:-1])


def _judged_option(
    project: ProjectStore, judge: EpisodeJudge, option: str, prefixes: tuple[str, ...]
) -> OptionOutcome:
    """Judge one synthetic episode per prefix and pool them as one option's cells."""
    cells = []
    for position, prefix in enumerate(prefixes):
        evidence = _evidence(prefix)
        reference, _ = judge_episode(
            project,
            judge.definition,
            evidence,
            settings=judge.settings,
            code_revision="test",
        )
        slot = JudgedSlot(
            position=position,
            episode=JudgedEpisode(
                episode_id=evidence.episode_id, assessment=reference, judge=judge.identity
            ),
        )
        cells.append(JudgedCell(contrast=reference, pair=position, arm="baseline", slot=slot))
    return OptionOutcome(
        option_id=option,
        action=ToolCall(call_id=option, name="act", arguments={}),
        planned=len(cells),
        assessable=len(cells),
        successes=0,
        unknown=0,
        cells=tuple(cells),
        bounds=(0.0, 1.0),
    )


def test_exported_advantages_are_read_from_the_saved_verdicts_and_follow_a_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same episodes under opposite scripted verdicts export opposite advantages."""
    selection = _flip(
        _verdict("x", "separated_matches", "a"), _verdict("y", "separated_matches", "b")
    )
    eligible = training_eligibility(
        selection=selection,
        confirmation=selection,
        findings=(),
        dispositions=(),
        critics=(),
        uncovered_tools=(),
        relied_rules=(),
        contradicted_rules=(),
        unwitnessed_rules=(),
        meanings=(_meaning(),),
        judging={
            name: {}
            for name in (
                "judge_coverage",
                "rubric_problems",
                "rubric_compatibility",
                "world_freeze",
                "world_provenance",
            )
        },
    )
    exported = []
    for run, flip in enumerate((False, True)):
        root = tmp_path / f"run-{run}"
        root.mkdir()
        project = ProjectStore(root, "case")
        write_catalog(root)
        monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
        judge = episode_judge(
            project, judge_settings(project), task_sha256="c" * 64, world_sha256="c" * 64
        )
        verdicts = iter(["pass", "fail", "unassessable"])
        route_judge(
            monkeypatch,
            verdict=lambda evidence, flip=flip, verdicts=verdicts: {
                "pass": "fail" if flip else "pass",
                "fail": "pass" if flip else "fail",
            }.get(next(verdicts), "unassessable"),
        )
        option = _judged_option(project, judge, "a", ("one", "two", "three"))
        contrast = OptionContrast(
            step=0,
            decision="d",
            checkpoint=ArtifactInput(artifact_id="c", sha256="0" * 64),
            reference_option="a",
            repeats=3,
            options=(option, option.model_copy(update={"option_id": "b"})),
            judge=judge.identity,
        )
        discovery = ActorCriticReport.model_construct(
            spec=ActorCriticSpec.model_construct(episode_judge=judge),
            group=ArtifactInput(artifact_id="group", sha256="0" * 64),
            picked=0,
            points=(),
            critics=(),
            judges=(),
        )
        records = training_records(project, eligible, discovery, {"x": (contrast, _meaning())})
        assert isinstance(records, TrainingRecords) and records.critics_refused is None
        other = episode_judge(
            project, judge_settings(project), task_sha256="d" * 64, world_sha256="d" * 64
        )
        foreign = discovery.model_copy(
            update={"spec": ActorCriticSpec.model_construct(episode_judge=other)}
        )
        refused = training_records(project, eligible, foreign, {"x": (contrast, _meaning())})
        assert isinstance(refused, TrainingRecords) and refused.critics == ()
        assert refused.critics_refused is not None and "variants ['x']" in refused.critics_refused
        assert records.selectors is not None and records.selectors.optimizer == "external"
        assert refused.selectors is not None
        assert refused.critics_refused in refused.selectors.refused
        group = records.groups["untiered"][0]
        assert group.episodes == tuple(cell.slot for cell in option.cells)
        assert not group.imposed_action_trainable and group.judge == judge.identity
        exported.append(group.advantages)
    assert exported[0] == (1.0, -1.0, None) and exported[1] == (-1.0, 1.0, None)


def _mining_run(
    root: Path, monkeypatch: pytest.MonkeyPatch, verdict: mining.Verdict
) -> tuple[ProjectStore, ActorCriticReport]:
    """Run a scripted error-mining run whose two selectors choose overlapping points."""
    project, job, _ = job_fixture(root, monkeypatch)
    spec = mining._spec(project, job, judges=2)
    group = [reply for row in (2, 3, 4) for reply in (mining._edit(row), mining._done())]
    continuations = [
        mining._edit(2),
        mining._done(),
        mining._edit(3),
        mining._done(),
        mining._done(),
        mining._done(),
    ]
    roles = [
        mining._tool_reply("choose_points", {"points": [mining._point(0)], "limitations": ["x"]}),
        mining._tool_reply(
            "choose_points",
            {"points": [mining._point(0), mining._point(1)], "limitations": ["x"]},
        ),
        mining._tool_reply("estimate_points", mining._estimates([0, 1])),
    ]
    mining._wire(monkeypatch, group + continuations, roles, verdict)
    exposure = mining._exposure(project, job, "mining_probes")
    return project, run_actor_critic(project, spec, exposure=exposure, code_revision="test")


def test_selector_records_reward_decision_relevance_and_reference_saved_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each draw's reward is read back from judged slots, normalized across the run's draws."""
    project, report = _mining_run(tmp_path, monkeypatch, mining.deletes_row(2))
    objective = SelectorObjective(minimum_scored=2)
    judge = report.spec.episode_judge.identity

    records = selector_records(project, report, judge=judge, objective=objective)

    assert records.refused == () and records.optimizer == "external"
    assert [point.point for point in records.points] == [0, 1]
    outcomes = {point.outcome.point_id: point.outcome for point in records.points}
    for point, row in zip(records.points, report.points, strict=True):
        assert point.judged == row.judged and len(point.slots) == 2
        assert point.outcome == row.outcome
    expected = [
        selector_score(picks, outcomes, objective).reward
        for picks in (["point-0"], ["point-0", "point-1"])
    ]
    assert [draw.score.reward for draw in records.draws if draw.score] == expected
    assert tuple(draw.advantage for draw in records.draws) == (
        group_advantages(records.group, expected).advantages
    )
    for draw, saved in zip(records.draws, report.judges, strict=True):
        assert draw.trainable and draw.refused == ()
        assert draw.transcript == saved.transcript and draw.response == saved.response
        assert draw.settings == report.spec.judges[draw.draw]
    assert records.draws[0].points == (0,) and records.draws[1].points == (0, 1)
    assert records.coverage.planned == 4 and records.coverage.scored == 4

    other = selector_records(project, report, judge="0" * 64, objective=objective)
    assert not any(draw.trainable for draw in other.draws)
    assert "is not the selector judge" in other.refused[0]
    thin = selector_records(project, report, judge=judge, objective=SelectorObjective())
    assert all(draw.score and draw.score.reward is None for draw in thin.draws)
    assert any("fewer than two" in reason for reason in thin.refused)


def test_selector_records_are_refused_when_judge_coverage_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unscored continuations fail the coverage policy, so no selector draw is trainable."""
    project, report = _mining_run(tmp_path, monkeypatch, lambda evidence: "malformed")

    records = selector_records(
        project,
        report,
        judge=report.spec.episode_judge.identity,
        objective=SelectorObjective(minimum_scored=1),
    )

    assert records.coverage.malformed == 4
    assert any("operationally unscored" in reason for reason in records.refused)
    assert not any(draw.trainable for draw in records.draws)


def test_screening_trainability_is_computed_from_its_selector_record() -> None:
    """A screening entry cannot claim trainability its selector record does not have."""
    score = JudgeScore(reward=0.1, observed_variance=0.2, noise_variance=0.1, scored_points=2)
    assert not JudgeScreening(draw=0, score=score).trainable
    with pytest.raises(ValidationError, match="exactly when"):
        JudgeScreening(draw=0, score=score, trainable=True)


def test_selector_records_default_to_version_two_and_read_point_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Version 2 scores picks by joint score or explained variance; without decisions, unscored."""
    project, report = _mining_run(tmp_path, monkeypatch, mining.deletes_row(2))
    judge = report.spec.episode_judge.identity
    decisions = {
        "point-0": PointDecision(point_id="point-0", joint_score=0.6),
        "point-1": PointDecision(
            point_id="point-1", choices=("a", "b", "a", "b"), rewards=(1.0, 0.0, 1.0, 0.0)
        ),
    }

    bare = selector_records(project, report, judge=judge)
    records = selector_records(project, report, judge=judge, decisions=decisions)

    assert bare.objective == DEFAULT_SELECTOR_OBJECTIVE
    assert isinstance(records.objective, SelectorObjectiveV2)
    assert all(draw.score and draw.score.status == "unscored" for draw in bare.draws)
    assert any("fewer than two" in reason for reason in bare.refused)
    outcomes = {point.outcome.point_id: point.outcome for point in records.points}
    expected = [
        score_selector(picks, outcomes, DEFAULT_SELECTOR_OBJECTIVE, decisions).reward
        for picks in (["point-0"], ["point-0", "point-1"])
    ]
    assert [draw.score.reward for draw in records.draws if draw.score] == expected
    assert expected[0] == pytest.approx(0.6) and expected[1] == pytest.approx(0.675)
    assert all(
        draw.score and draw.score.objective == "selector-decision-variance-v2"
        for draw in records.draws
    )
    assert all(draw.trainable for draw in records.draws)
    legacy = selector_records(
        project, report, judge=judge, objective=SelectorObjective(minimum_scored=2)
    )
    assert isinstance(legacy.objective, SelectorObjective)
    assert legacy.model_validate_json(legacy.model_dump_json()) == legacy
    assert records.model_validate_json(records.model_dump_json()) == records


def test_a_family_blocked_by_the_joint_measure_names_the_decision_variance_condition() -> None:
    """The decision_variance condition carries the deficits; confirmation says it was blocked."""
    selection = _flip(
        _verdict("x", "separated_matches", "a"), _verdict("y", "separated_matches", "b")
    )

    def eligibility(
        measured: tuple[bool, str] | None, confirmation: FlipVerdict | None
    ) -> TrainingEligibility:
        """Compute eligibility for a qualifying family with every admission check passing."""
        return training_eligibility(
            selection=selection,
            confirmation=confirmation,
            findings=(),
            dispositions=(),
            critics=(),
            uncovered_tools=(),
            relied_rules=(),
            contradicted_rules=(),
            unwitnessed_rules=(),
            meanings=(_meaning(),),
            judging={
                name: {}
                for name in (
                    "judge_coverage",
                    "rubric_problems",
                    "rubric_compatibility",
                    "world_freeze",
                    "world_provenance",
                )
            },
            decision_variance=measured,
        )

    blocked = eligibility((False, "choices do not change across states"), None)
    assert CONDITIONS[:3] == ("family_flip", "decision_variance", "fresh_confirmation")
    assert blocked.reasons[0] == "decision_variance: choices do not change across states"
    assert "blocked confirmation" in blocked.reasons[1]
    assert eligibility((True, "admitted, joint score 0.400"), selection).eligible
    assert eligibility(None, selection).eligible
    unrequired = eligibility(None, selection).conditions[1]
    assert unrequired.name == "decision_variance" and unrequired.detail.startswith("not required")
