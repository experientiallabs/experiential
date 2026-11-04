"""Offline tests for audit dispositions, computed eligibility and confirmation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import BillingSource, ModelSnapshot, ToolCall
from exp.common.project import ProjectStore
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
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
    TrainingEligibility,
    TrainingRecords,
    audit_finding,
    pending_findings,
    policy_key,
    training_eligibility,
    training_records,
    unreproduced,
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
    ]
    assert any("rule.a was contradicted" in r and "rule.b has no" in r for r in eligibility.reasons)
    assert eligibility.reasons[-1] == (
        "world_freeze: no confirmed variant was checked against the policy"
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

    names = ("judge_coverage", "rubric_problems", "rubric_compatibility", "world_freeze")
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
            )
        },
    )
    discovery = ActorCriticReport.model_construct(points=(), critics=(), judges=())
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
        records = training_records(project, eligible, discovery, {"x": (contrast, _meaning())})
        assert isinstance(records, TrainingRecords)
        group = records.groups["untiered"][0]
        assert group.episodes == tuple(cell.slot for cell in option.cells)
        assert not group.imposed_action_trainable and group.judge == judge.identity
        exported.append(group.advantages)
    assert exported[0] == (1.0, -1.0, None) and exported[1] == (-1.0, 1.0, None)
