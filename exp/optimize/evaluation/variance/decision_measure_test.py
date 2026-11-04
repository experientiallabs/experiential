"""Tests for decision-variance measurement: choices, consequence, clarity and the joint verdict."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    stable_id,
)
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.contracts import (
    Criterion,
    DecisionOption,
    DecisionState,
    Scenario,
    StateEdit,
    VariationAxis,
    WitnessStep,
)
from exp.optimize.evaluation.variance.contrast_test import _contrast
from exp.optimize.evaluation.variance.decision_measure import (
    MEASURE_TYPE,
    DecisionMeasureSpec,
    StateEvidence,
    StateInput,
    classify_episode,
    decision_admission,
    joint_verdict,
    measure_states,
    outcome_consequence,
    point_decision,
)
from exp.optimize.evaluation.variance.episodes import (
    EpisodeAssessment,
    EpisodeEvent,
    EpisodeEvidence,
    EventKind,
    EventOwner,
    EventScope,
    JudgedEpisode,
)
from exp.optimize.evaluation.variance.evidence_tools import EvidenceBudget, EvidenceToolProfile
from exp.optimize.evaluation.variance.evidence_tools_test import price_judge
from exp.optimize.evaluation.variance.judge import ASSESSMENT_TYPE, save_evidence
from exp.optimize.evaluation.variance.judged import JUDGED_STUDY_TYPE, JudgedSlot, JudgedStudy
from exp.optimize.evaluation.variance.rewards import DecisionVarianceObjective, OptionCounts
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog

JUDGE = "7" * 64
"""Judge identity every fixture assessment carries."""

FORCED = EvidenceToolProfile()
"""One forced answer per draw, so each draw is one scripted reply."""

OBJECTIVE = DecisionVarianceObjective(resamples=400)
"""The joint objective with fewer resamples to keep tests quick."""


def decision() -> DecisionState:
    """Return a two-option refund decision deciding on the account tier."""
    return DecisionState(
        step=0,
        decision="Whether to refund the order.",
        options=(
            DecisionOption(option_id="refund", action="Issue a full refund"),
            DecisionOption(option_id="decline", action="Decline the refund request"),
        ),
        relevant_paths=("/account/tier",),
        axes=(VariationAxis(path="/account/tier", values=("gold", "basic"), effect="tier"),),
        criteria=(Criterion(criterion_id="right-call", text="The right call was made."),),
        evidence_refs=("run:a:step:0",),
    )


def scenario(scenario_id: str, correct: str, tier: str) -> Scenario:
    """Return one tier variant whose deciding tier one lookup reveals."""
    return Scenario(
        scenario_id=scenario_id,
        edits=(StateEdit(path="/account/tier", value=tier),),
        correct_option=correct,
        deciding_paths=("/account/tier",),
        witness=(WitnessStep(tool="lookup", arguments={"account": 7}, reveals=("/account/tier",)),),
        rationale=f"{tier} accounts get {correct}.",
        criteria=("right-call",),
    )


def episode(name: str, action: str) -> EpisodeEvidence:
    """Return one episode: a context prompt, a read, then the consequential call."""
    rows: tuple[tuple[str, EventKind, EventOwner, EventScope, JsonValue], ...] = (
        ("e0", "user_message", "user", "context", "Please refund order 7."),
        ("e1", "tool_call", "actor", "scored", {"name": "lookup", "arguments": {"account": 7}}),
        ("e2", "tool_result", "world", "scored", {"tier": "gold"}),
        ("e3", "tool_call", "actor", "scored", {"name": action, "arguments": {"order": 7}}),
    )
    return EpisodeEvidence(
        episode_id=name,
        task_sha256="1" * 64,
        world_sha256="2" * 64,
        events=tuple(
            EpisodeEvent(
                event_id=event_id,
                sequence=index,
                kind=kind,
                owner=owner,
                scope=scope,
                content=content,
                source=f"{name}:{index}",
            )
            for index, (event_id, kind, owner, scope, content) in enumerate(rows)
        ),
        complete=True,
    )


def judged_study(
    project: ProjectStore, name: str, episodes: list[tuple[str, str]]
) -> ArtifactInput:
    """Save one judged free study of (action, verdict) episodes, with a gap for a None action."""
    store = project.artifacts
    slots: list[JudgedSlot] = []
    for position, (action, verdict) in enumerate(episodes):
        if not action:
            slots.append(JudgedSlot(position=position, gap="no evidence was recorded"))
            continue
        evidence = episode(f"{name}-{position}", action)
        evidence_ref = save_evidence(project, evidence, code_revision="test")
        assessment = EpisodeAssessment(
            episode_id=evidence.episode_id,
            evidence_sha256=evidence.identity,
            judge=JUDGE,
            status="judged",
            verdict="pass" if verdict == "pass" else "fail",
            decisive_refs=("e3",),
        )
        manifest = store.write_json(
            artifact_id=stable_id("fixture-assessment", {"episode": evidence.episode_id}),
            artifact_type=ASSESSMENT_TYPE,
            envelope=_envelope((evidence_ref,)),
            files={"assessment.json": assessment},
        )
        reference = JudgedEpisode(
            episode_id=evidence.episode_id, assessment=artifact_input(manifest), judge=JUDGE
        )
        slots.append(JudgedSlot(position=position, episode=reference))
    study = JudgedStudy(
        study=ArtifactInput(artifact_id=f"study-{name}", sha256="3" * 64),
        judge=JUDGE,
        branches=tuple(slots),
    )
    manifest = store.write_json(
        artifact_id=stable_id("fixture-judged", {"name": name}),
        artifact_type=JUDGED_STUDY_TYPE,
        envelope=_envelope(()),
        files={"judged.json": study},
    )
    return artifact_input(manifest)


def _envelope(inputs: tuple[ArtifactInput, ...]) -> ArtifactEnvelope:
    """Return a fixture artifact envelope."""
    return ArtifactEnvelope(
        schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=inputs
    )


def reply(calls: list[tuple[str, JsonObject]]) -> JsonHttpResponse:
    """One scripted role reply making the given tool calls."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "id": "reply",
            "model": "judge-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"call-{i}",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                            for i, (name, arguments) in enumerate(calls)
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        },
    )


def role_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[JsonHttpResponse]
) -> tuple[ProjectStore, AuthorSettings, ScriptedJsonTransport]:
    """Return a project, role settings and the scripted role transport."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    digest = price_judge(project)
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    settings = AuthorSettings(
        model_alias="judge",
        model=catalog.snapshot("judge")[0],
        catalog_sha256=digest,
        output_tokens=1024,
        deadline=job.deadline,
    )
    transport = ScriptedJsonTransport(tuple(replies))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    return project, settings, transport


def choice(label: str, action: str = "e3") -> JsonHttpResponse:
    """A forced classification reply."""
    refs = [action] if action else ["e0"]
    return reply(
        [
            (
                "classify_choice",
                {
                    "option_id": label,
                    "action_event": action,
                    "evidence_refs": refs,
                    "rationale": "The first state-changing call.",
                },
            )
        ]
    )


def review(best: str, *, observable: bool = True, issues: tuple[str, ...] = ()) -> JsonHttpResponse:
    """A forced clarity review reply."""
    return reply(
        [
            (
                "review_clarity",
                {
                    "best_option": best,
                    "deciding_facts_observable": observable,
                    "solvable_without_guessing": observable,
                    "issues": list(issues),
                    "evidence_refs": ["witness", "scenario"],
                    "rationale": "The lookup shows the tier.",
                },
            )
        ]
    )


def spec(settings: AuthorSettings) -> DecisionMeasureSpec:
    """Forced-profile measurement settings with two review stances."""
    return DecisionMeasureSpec(
        classifier=settings,
        reviewer=settings,
        classifier_profile=FORCED,
        reviewer_profile=FORCED,
        reviewer_stances=("observable_solver", "ambiguity_hunter"),
        objective=OBJECTIVE.model_copy(update={"minimum_classified": 3}),
    )


def _evidence(
    state: str, correct: str, choices: str, refund: int, decline: int, *, review: bool = True
) -> StateEvidence:
    """Build one state's evidence: choices as 'r'/'d'/'o'/'u' and forced successes of 20."""
    names = {"r": "refund", "d": "decline", "o": "other", "u": "unclear"}
    return StateEvidence(
        state_id=state,
        correct_option=correct,
        choices=tuple(names[c] for c in choices),
        options=(
            OptionCounts(option_id="refund", successes=refund, planned=20),
            OptionCounts(option_id="decline", successes=decline, planned=20),
        ),
        clarity_review=review,
        clarity_notes=() if review else ("the tier is never shown",),
    )


def test_a_decision_whose_choices_and_outcomes_both_change_is_admitted() -> None:
    """Choices shift across clear states and the better forced option flips with them."""
    verdict = joint_verdict(
        [
            _evidence("gold", "refund", "rrrrrrrd", 18, 3),
            _evidence("basic", "decline", "dddddddr", 2, 17),
        ],
        ["refund", "decline"],
        OBJECTIVE,
    )

    assert verdict.admitted and verdict.choice_shift and verdict.consequence
    assert verdict.clarity and verdict.flip and 0.0 < verdict.score <= 1.0
    assert verdict.deficits == ()
    assert [state.counts for state in verdict.states] == [
        {"refund": 7, "decline": 1, "other": 0},
        {"refund": 1, "decline": 7, "other": 0},
    ]


def test_a_confusing_high_entropy_variant_is_refused_as_pure_uncertainty() -> None:
    """Agents split evenly and no forced option is better: maximal outcome variance, refused."""
    verdict = joint_verdict(
        [
            _evidence("gold", "refund", "rdrdrdrd", 10, 10),
            _evidence("basic", "decline", "drdrdrdr", 9, 11),
        ],
        ["refund", "decline"],
        OBJECTIVE,
    )

    assert not verdict.admitted and verdict.score == 0.0
    assert not verdict.choice_shift and not verdict.consequence and not verdict.clarity
    assert any("pure uncertainty" in deficit for deficit in verdict.deficits)
    assert all(state.entropy and state.entropy >= 0.8 for state in verdict.states)


def test_choices_that_change_without_changing_outcomes_are_refused() -> None:
    """The actor's choice tracks the state, but forcing either option succeeds equally often."""
    verdict = joint_verdict(
        [
            _evidence("gold", "refund", "rrrrrrrr", 10, 10),
            _evidence("basic", "decline", "dddddddd", 10, 10),
        ],
        ["refund", "decline"],
        OBJECTIVE,
    )

    assert verdict.choice_shift and not verdict.consequence and not verdict.admitted
    assert any("does not change success" in deficit for deficit in verdict.deficits)


def test_clarity_needs_the_review_and_a_matching_separated_best() -> None:
    """A refused review or a contradicting best option blocks admission on clarity alone."""
    refused = joint_verdict(
        [
            _evidence("gold", "refund", "rrrrrrrd", 18, 3, review=False),
            _evidence("basic", "decline", "dddddddr", 2, 17),
        ],
        ["refund", "decline"],
        OBJECTIVE,
    )
    contradicted = joint_verdict(
        [
            _evidence("gold", "decline", "rrrrrrrd", 18, 3),
            _evidence("basic", "decline", "dddddddr", 2, 17),
        ],
        ["refund", "decline"],
        OBJECTIVE,
    )

    assert refused.consequence and refused.choice_shift and not refused.clarity
    assert any("tier is never shown" in deficit for deficit in refused.deficits)
    assert not contradicted.admitted and not contradicted.flip
    assert any("contradicts the declared decline" in d for d in contradicted.deficits)
    with pytest.raises(ValueError, match="may not be"):
        joint_verdict([], ["refund", "other"], OBJECTIVE)


def test_outcome_consequence_reads_the_judged_contrast() -> None:
    """A judged contrast's forced counts give the effect and the separated best option."""
    consequence = outcome_consequence(_contrast(18, 2), OBJECTIVE)

    assert consequence.holds and consequence.separated_best == "refund"
    assert not outcome_consequence(_contrast(10, 10), OBJECTIVE).holds


def test_measure_classifies_reviews_and_judges_every_state_then_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free choices are classified with citations, states reviewed, and the verdict saved."""
    replies = [
        choice("refund"),
        choice("refund"),
        choice("decline", action="e2"),
        choice("refund", action="e9"),
        review("refund"),
        review("refund"),
        choice("decline"),
        choice("decline"),
        choice("decline"),
        review("decline"),
        review("decline"),
    ]
    project, settings, transport = role_fixture(tmp_path, monkeypatch, replies)
    gold = judged_study(
        project,
        "gold",
        [("refund", "pass"), ("refund", "pass"), ("decline", "fail"), ("refund", "pass")],
    )
    basic = judged_study(
        project, "basic", [("decline", "pass"), ("decline", "pass"), ("decline", "pass"), ("", "")]
    )
    states = [
        StateInput(
            scenario=scenario("gold", "refund", "gold"), free=gold, contrast=_contrast(18, 2)
        ),
        StateInput(
            scenario=scenario("basic", "decline", "basic"), free=basic, contrast=_contrast(2, 18)
        ),
    ]
    measure_spec = spec(settings)

    reference, measure = measure_states(
        project, measure_spec, decision(), states, code_revision="test"
    )

    gold_state, basic_state = measure.states
    assert [c.label for c in gold_state.choices] == ["refund", "refund", "unclear", "unclear"]
    assert "scored event owned by the actor" in str(gold_state.choices[2].failure)
    assert "not in the episode" in str(gold_state.choices[3].failure)
    assert [c.stance for c in gold_state.choices] == [
        "action_matching",
        "intent_tracing",
        "skeptical_other",
        "action_matching",
    ]
    assert basic_state.choices[3].label == "unclear" and basic_state.choices[3].transcript is None
    assert basic_state.choices[0].stance == "intent_tracing"
    assert all(r.passed for r in (*gold_state.reviews, *basic_state.reviews))
    assert gold_state.evidence.rewards == (1.0, 1.0, 0.0, 1.0)
    assert measure.verdict.consequence and measure.verdict.clarity
    assert not measure.verdict.admitted
    assert any("free choices were unclear" in d for d in measure.verdict.deficits)
    assert decision_admission(measure) == (False, measure.verdict.deficits)
    assert project.artifacts.read(reference.artifact_id).manifest.artifact_type == MEASURE_TYPE
    sent = json.dumps(transport.requests[0].payload)
    assert "STANCE (action_matching)" in sent and "classify_choice" in sent
    assert point_decision("p", gold_state.choices).choices[2] == "unclear"

    assert measure_states(project, measure_spec, decision(), states, code_revision="test") == (
        reference,
        measure,
    )
    assert len(transport.requests) == len(replies)


def test_a_review_naming_another_option_or_hidden_facts_fails_the_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clarity fails when a reviewer names another option or reports unobservable facts."""
    replies = [
        *(choice("refund") for _ in range(3)),
        review("decline"),
        review("refund", observable=False, issues=("the tier is hidden",)),
        *(choice("decline") for _ in range(3)),
        review("decline"),
        review("decline"),
    ]
    project, settings, _ = role_fixture(tmp_path, monkeypatch, replies)
    states = [
        StateInput(
            scenario=scenario(name, correct, name),
            free=judged_study(project, name, [(correct, "pass")] * 3),
            contrast=_contrast(18, 2) if correct == "refund" else _contrast(2, 18),
        )
        for name, correct in (("gold", "refund"), ("basic", "decline"))
    ]

    _, measure = measure_states(project, spec(settings), decision(), states, code_revision="test")

    gold = measure.verdict.states[0]
    assert not gold.clear and measure.verdict.states[1].clear
    assert any("the tier is hidden" in problem for problem in gold.problems)
    assert not measure.verdict.admitted


def test_the_agentic_classifier_plans_reads_drafts_checks_and_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host's draft validation reaches the self check; the final answer may correct it."""
    replies = [
        reply([("record_plan", {"hypotheses": ["e3 commits"], "planned_reads": ["event:e3"]})]),
        reply([("read_evidence", {"locator": "event:e3"})]),
        reply(
            [
                (
                    "submit_draft",
                    {
                        "option_id": "refund",
                        "action_event": "e2",
                        "evidence_refs": ["e2"],
                        "rationale": "draft",
                    },
                )
            ]
        ),
        reply([("record_self_check", {"issues": ["e2 is a world event"], "revise": True})]),
        choice("refund"),
    ]
    project, settings, transport = role_fixture(tmp_path, monkeypatch, replies)
    profile = EvidenceToolProfile(
        mode="agentic", budget=EvidenceBudget(model_calls=6, tool_calls=2)
    )
    measure_spec = spec(settings).model_copy(update={"classifier_profile": profile})
    evidence = episode("agentic", "refund")
    save_evidence(project, evidence, code_revision="test")

    result = classify_episode(
        project,
        measure_spec,
        decision(),
        evidence,
        position=0,
        reward=1.0,
        stance="skeptical_other",
        sources=(),
        code_revision="test",
    )

    assert result.label == "refund" and result.action_event == "e3"
    assert result.evidence_refs == ("e3",) and result.failure is None
    assert "scored event owned by the actor" in json.dumps(transport.requests[3].payload)
    assert "STANCE (skeptical_other)" in json.dumps(transport.requests[0].payload)
    assert len(transport.requests) == 5


def test_specs_refuse_unknown_stances_and_an_unreachable_quorum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stances must be known and distinct, and the quorum reachable."""
    _, settings, _ = role_fixture(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError, match="distinct members"):
        DecisionMeasureSpec(classifier=settings, reviewer=settings, classifier_stances=("x",))
    with pytest.raises(ValueError, match="minimum_reviews"):
        DecisionMeasureSpec(
            classifier=settings,
            reviewer=settings,
            reviewer_stances=("ambiguity_hunter",),
            minimum_reviews=2,
        )
