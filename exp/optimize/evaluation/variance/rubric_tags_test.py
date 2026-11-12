"""Rubric-authored tags: structured step evidence, conjunction scoring and the unscored policy."""

from __future__ import annotations

import json

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.optimize.evaluation.variance.behavior_differences import (
    BehaviorDifferenceSpec,
    PairAudit,
    PairDifference,
    behavior_delta,
    difference_groups,
)
from exp.optimize.evaluation.variance.behavior_differences_test import aligned, attempt, settings
from exp.optimize.evaluation.variance.episodes import EpisodeEvent, EpisodeEvidence
from exp.optimize.evaluation.variance.rubric_author import RubricCriterion
from exp.optimize.evaluation.variance.rubric_tag_settings import RubricTagSettings
from exp.optimize.evaluation.variance.rubric_tags import (
    RubricStepTagger,
    RubricTag,
    RubricTagSet,
    TagAnswer,
    TagPart,
    agreement,
    check_tags,
    fire_screen,
    step_evidence,
)
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.mining.tagging import TagMatrix, TagRow, TraceTaggingError
from exp.simulation.mining.traceml import TraceStep, TraceTrajectory, step_units

BEFORE = "def rep(pos):\n    for k in ends(pos):\n        yield k\n"
AFTER = "def rep(pos):\n    for k in ends(pos):\n        if k == pos:\n            continue\n        yield k\n"


def event(seq: int, event_id: str, kind: str, owner: str, content: object, scope: str = "scored") -> EpisodeEvent:
    return EpisodeEvent(
        event_id=event_id,
        sequence=seq,
        kind=kind,
        owner=owner,
        scope=scope,
        content=content,
        source=f"log#{seq}",
    )


def episode() -> tuple[TraceTrajectory, EpisodeEvidence]:
    """Read, edit (through a shell, so only the world state shows the change), run, claim."""
    events = (
        event(0, "state.initial", "state_snapshot", "world", {"files": {"m.py": BEFORE}}, "context"),
        event(1, "call.0.tool.0", "tool_call", "actor", {"call_id": "a", "name": "bash", "arguments": {"command": "cat m.py"}}),
        event(2, "call.0.tool.0.reply", "tool_result", "world", {"tool_call_id": "a", "content": BEFORE}),
        event(3, "call.1.tool.0", "tool_call", "actor", {"call_id": "b", "name": "bash", "arguments": {"command": "python fix.py"}}),
        event(4, "call.1.tool.0.reply", "tool_result", "world", {"tool_call_id": "b", "content": ""}),
        event(5, "call.1.tool.0.state", "state_snapshot", "world", {"files": {"m.py": AFTER}}),
        event(6, "call.2.tool.0", "tool_call", "actor", {"call_id": "c", "name": "bash", "arguments": {"command": "pytest | tail -3"}}),
        event(7, "call.2.tool.0.reply", "tool_result", "world", {"tool_call_id": "c", "content": "3 passed"}),
        event(8, "call.3.final", "final_answer", "actor", "All tests pass."),
    )
    evidence = EpisodeEvidence(
        episode_id="ep", task_sha256="1" * 64, world_sha256="2" * 64, events=events, complete=True
    )
    steps = (
        TraceStep(index=0, locator="call.0.tool.0", boundary=0, kind="tool", tool="bash", arguments={"command": "cat m.py"}, result={"content": BEFORE}),
        TraceStep(index=1, locator="call.1.tool.0", boundary=1, kind="tool", tool="bash", arguments={"command": "python fix.py"}, result={"content": ""}),
        TraceStep(index=2, locator="call.2.tool.0", boundary=2, kind="tool", tool="bash", arguments={"command": "pytest | tail -3"}, result={"content": "3 passed"}),
        TraceStep(index=3, locator="call.3.final", boundary=3, kind="message", message="All tests pass."),
    )
    return TraceTrajectory(trace_id="actor/attempt:0", tier="actor", task="Fix rep", steps=steps), evidence


def tag(tag_id: str, slot: str, polarity: str = "negative", applies: str | None = None) -> RubricTag:
    path = {"edit": "`source`", "action": "`step.arguments`", "claim": "`message`"}[slot]
    return RubricTag(
        tag_id=tag_id,
        polarity=polarity,
        criterion_id="rep_semantics",
        derived_from="Discarding a zero-width child match",
        slot=slot,
        applies=applies,
        parts=(
            TagPart(
                question=f"In {path}, is a zero-width match skipped ({tag_id})?",
                statement=f"{path} skips a zero-width match ({tag_id}).",
                true="It is skipped without counting",
                false="It is counted once toward the minimum before recursion stops",
            ),
        ),
    )


def tag_set(*tags: RubricTag) -> RubricTagSet:
    return RubricTagSet(rubric=ArtifactInput(artifact_id="episode-rubric-x", sha256="6" * 64), tags=tags)


def test_step_evidence_reads_edits_from_world_states_not_from_tool_arguments() -> None:
    trajectory, evidence = episode()
    states = step_evidence(trajectory, evidence, RubricTagSettings())
    assert [s.edit is not None for s in states] == [False, True, False, False]
    edit = states[1].edit
    assert edit is not None and edit["edit"]["files"] == ["m.py"]
    assert "+        if k == pos:" in edit["edit"]["diff"] and edit["source"] == {"m.py": AFTER}
    assert states[2].action is not None and states[2].action["step"]["output"] == "3 passed"
    claim = states[3].claim
    assert claim is not None and states[3].action is None
    assert [run["arguments"]["command"] for run in claim["runs_since_last_edit"]] == ["pytest | tail -3"]
    assert claim["edited_files"] == ["m.py"]


def test_values_are_the_conjunction_of_applicability_and_agreeing_wordings() -> None:
    assert agreement(0.9, 0.8, 0.4) == pytest.approx(0.85)
    assert agreement(0.9, 0.2, 0.4) == 0.5
    trajectory, evidence = episode()
    tags = tag_set(
        tag("drops_zero_width", "edit", applies="`edit.diff` changes the repetition code"),
        tag("pipes_tests", "action"),
        tag("claims_green", "claim", "positive"),
    )
    asked: list[DecisionRequest] = []

    def decide(request: DecisionRequest) -> dict[str, object]:
        asked.append(request)
        answers = {}
        for name, question in request.questions.items():
            text = json.dumps(question.instructions)
            value = 0.95 if "drops_zero_width" in text or "repetition" in text else 0.1
            if "pipes_tests" in text:
                value = 0.6
            answers[name] = {"type": "noul", "noul": value}
        return {"answers": answers}

    tagger = RubricStepTagger(decide, {"model": "fixture"}, tags, RubricTagSettings())
    units = step_units(trajectory)
    matrix = tagger.tag_steps(trajectory, evidence, units, tags.bank())
    columns = {q: i for i, q in enumerate(matrix.question_ids)}
    rows = [row.values for row in matrix.rows]
    assert rows[1][columns["drops_zero_width"]] == pytest.approx(0.95)
    assert rows[0][columns["drops_zero_width"]] == 0.0  # no file changed: not asked
    assert rows[2][columns["pipes_tests"]] == pytest.approx(0.6)  # stays uncertain
    assert rows[3][columns["pipes_tests"]] == 0.0  # a message is not a tool call
    # One request per applicable slot per step: 3 action, 1 edit, 1 claim.
    assert len(asked) == 5 and all(isinstance(r.state, dict) for r in asked)
    assert tagger.last_answers is not None and tagger.last_answers.steps[1].tags["drops_zero_width"].asked
    with pytest.raises(TraceTaggingError):
        tagger.tag(units, tags.bank())


def test_the_bank_carries_polarity_and_changes_with_any_tag_text() -> None:
    tags = tag_set(tag("drops_zero_width", "edit"), tag("counts_once", "edit", "positive"))
    bank = tags.bank()
    assert [q.family for q in bank.questions] == ["behavior.negative", "behavior.positive"]
    reworded = tag_set(
        tag("drops_zero_width", "edit", applies="`edit.diff` changes rep"),
        tag("counts_once", "edit", "positive"),
    )
    assert reworded.bank_id != tags.bank_id
    with pytest.raises(ValueError, match="negative"):
        tag_set(tag("counts_once", "edit", "positive"))


def test_structure_checks_cite_criteria_slot_fields_and_coverage() -> None:
    from exp.optimize.evaluation.variance.rubric_author import Rubric, RubricBody

    criterion = RubricCriterion(
        criterion_id="rep_semantics",
        question="Is rep correct?",
        role="required_correctness",
        applies_when="always",
        scenario_ids=("base",),
        supported_by=("claim:x",),
        evidence_needed=({"kind": "state_snapshot", "detail": "final source"},),
        met_when="m",
        not_met_when="n",
        insufficient_when="i",
        failure_patterns=("drops zero-width",),
    )
    rubric = Rubric.model_construct(
        family_id="f",
        packet_sha256="7" * 64,
        body=RubricBody.model_construct(criteria=(criterion,)),
    )
    good = TagAnswer(tags=(tag("drops_zero_width", "edit"),))
    assert check_tags(good, rubric) == []
    wrong_slot = tag("drops_zero_width", "action").model_copy(
        update={"applies": "`source` changes the repetition code"}
    )
    problems = check_tags(TagAnswer(tags=(wrong_slot,)), rubric)
    assert any("does not have" in p for p in problems)
    positive_only = TagAnswer(tags=(tag("counts_once", "edit", "positive"),))
    assert any("no negative tag" in p for p in check_tags(positive_only, rubric))
    other = tag("drops_zero_width", "edit").model_copy(update={"criterion_id": "missing"})
    assert any("not in the rubric" in p for p in check_tags(TagAnswer(tags=(other,)), rubric))


def test_fire_screen_drops_tags_that_fire_on_nearly_every_attempt() -> None:
    def matrix(values: tuple[float, float]) -> TagMatrix:
        return TagMatrix(
            bank_id="b",
            bank_version=1,
            question_ids=("always", "rare"),
            view="step",
            rows=(TagRow(trace_id="t", unit_hash="h", values=values),),
        )

    rows = [matrix((0.9, 0.1)) for _ in range(5)] + [matrix((0.8, 0.9))]
    screened = fire_screen(rows, ("always", "rare"), confidence=0.7, maximum=0.9)
    assert set(screened) == {"always"}
    assert fire_screen(rows[:2], ("always",), confidence=0.7, maximum=0.9) == {}


def rubric_spec() -> BehaviorDifferenceSpec:
    from exp.optimize.evaluation.variance.behavior_differences_test import NEGATIVE, POSITIVE

    return BehaviorDifferenceSpec(
        bank_id="behavior",
        author=settings(),
        reviewer=settings(),
        positive_tags=POSITIVE,
        negative_tags=NEGATIVE,
        rubric_tags=RubricTagSettings(),
    )


def test_uncertain_tags_stay_unscored_without_blocking_a_confident_signature() -> None:
    config = rubric_spec()
    good = attempt(config, 0, [{"verified_result"}])
    poor = attempt(config, 1, [{"unsupported_success_claim"}])
    values = list(poor.tags.rows[0].values)
    values[poor.tags.question_ids.index("hid_failure")] = 0.5
    poor = poor.model_copy(
        update={
            "tags": poor.tags.model_copy(
                update={"rows": (poor.tags.rows[0].model_copy(update={"values": tuple(values)}),)}
            )
        }
    )
    delta = behavior_delta(good, poor, aligned(), config, ("skipped_required_step",))
    assert delta.added == ("unsupported_success_claim",) and delta.removed == ("verified_result",)
    assert delta.uncertain == () and delta.unscored == ("hid_failure",)
    assert delta.screened == ("skipped_required_step",)
    pair = PairDifference(
        pair_id="pair",
        good=0,
        poor=1,
        alignment=aligned(),
        delta=delta,
        audit=PairAudit(verdict="accept", good_events=("e0",), poor_events=("e0",), rationale="ok"),
        point=0,
        checkpoint=poor.checkpoints[0],
    )
    assert pair.accepted
    (group,) = difference_groups([pair], bank_sha256="0" * 64)
    assert group.added == ("unsupported_success_claim",)
    # The default policy still blocks the pair on the same evidence.
    default = config.model_copy(update={"rubric_tags": None})
    assert behavior_delta(good, poor, aligned(), default).uncertain == ("hid_failure",)


def test_the_setting_is_off_by_default_and_leaves_serialized_identities_unchanged() -> None:
    from exp.optimize.evaluation.variance.behavior_differences import BehaviorDelta

    off = BehaviorDifferenceSpec(bank_id="behavior")
    assert off.rubric_tags is None and "rubric_tags" not in off.model_dump(mode="json")
    delta = BehaviorDelta(added=("a",), removed=(), uncertain=())
    assert delta.model_dump(mode="json") == {"added": ["a"], "removed": [], "uncertain": []}
    assert sha256_json(off) == sha256_json(BehaviorDifferenceSpec.model_validate(off.model_dump()))


def test_each_question_group_sees_only_the_fields_it_cites_and_structured_criteria_go_on_the_wire() -> None:
    from exp.optimize.evaluation.variance.rubric_tags import Situation, cited_fields

    trajectory, evidence = episode()
    structured = tag("drops_zero_width", "edit").model_copy(
        update={
            "parts": (
                TagPart(
                    question="In `source`, is a zero-width child end skipped before counting?",
                    statement="`source` skips a zero-width child end before counting it.",
                    true=Situation(what="It is skipped", examples=("if k == pos: continue",)),
                    false=Situation(what="It is counted once", examples=("if k == pos: yield pos",)),
                ),
            )
        }
    )
    diff_tag = tag("edits_rep", "edit").model_copy(
        update={"applies": "`edit.diff` changes the repetition code"}
    )
    assert cited_fields(structured) == ("source",)
    assert cited_fields(diff_tag) == ("edit", "source")
    tags = tag_set(structured, diff_tag)
    asked: list[DecisionRequest] = []

    def decide(request: DecisionRequest) -> dict[str, object]:
        asked.append(request)
        return {"answers": {n: {"type": "noul", "noul": 0.9} for n in request.questions}}

    RubricStepTagger(decide, {"m": 1}, tags, RubricTagSettings()).tag_steps(
        trajectory, evidence, step_units(trajectory), tags.bank()
    )
    assert sorted(sorted(r.state) for r in asked) == [["edit", "source"], ["source"]]
    wired = [q.criteria for r in asked for q in r.questions.values() if q.criteria is not None]
    assert any(isinstance(c.true, dict) and c.true["examples"] for c in wired)
