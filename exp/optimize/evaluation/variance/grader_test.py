"""Offline tests for the rubric decision grader over a fake typed decision boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, ModelSnapshot
from exp.optimize.evaluation.variance.contracts import Criterion
from exp.optimize.evaluation.variance.grader import (
    CRITERION_FAMILY,
    EVIDENCE_FAMILY,
    Band,
    EpisodeView,
    GradedCriterion,
    GraderSettings,
    ReferenceVerdict,
    criterion_bank,
    grade_episodes,
    references_from,
)
from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS, DecisionRequest
from exp.simulation.mining.decisions import DecisionsTraceTagger

_SNAPSHOT = ModelSnapshot(
    provider="fixture",
    model_id="decisions-fixture",
    billing_source=BillingSource.CUSTOMER_MANAGED,
    capabilities_sha256="a" * 64,
    connection_sha256="b" * 64,
)

Answer = Callable[[str, str], float]
"""Probability for one question identity about one episode text."""


@dataclass
class _Decider:
    """A decision boundary answering each question from a rule and recording every request."""

    answer: Answer
    requests: list[DecisionRequest] = field(default_factory=list)

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Record one request and answer every question in it."""
        self.requests.append(request)
        state = str(request.state)
        return {
            "id": "decision",
            "answers": {
                name: {"type": "noul", "noul": self.answer(name, state)}
                for name in request.questions
            },
            "usage": {"input_tokens": 10, "output_tokens": 0, "cost": 0.0},
        }


def _criterion(cid: str, *paraphrases: str) -> GradedCriterion:
    """One criterion with optional paraphrases."""
    return GradedCriterion(
        criterion=Criterion(criterion_id=cid, text=f"Did the agent satisfy {cid}?"),
        paraphrases=paraphrases,
    )


def _grade(
    answer: Answer,
    criteria: tuple[GradedCriterion, ...],
    episodes: tuple[str, ...] = ("e1",),
    *,
    settings: GraderSettings | None = None,
) -> tuple[_Decider, list[str], dict[tuple[str, str], str]]:
    """Grade episodes whose text is their identity; return calls and classifications."""
    decider = _Decider(answer)
    report = grade_episodes(
        DecisionsTraceTagger(decider, _SNAPSHOT),
        tuple(EpisodeView(episode_id=e, text=f"episode {e}") for e in episodes),
        criteria,
        settings=settings,
    )
    reasons = [reason for grade in report.grades for reason in grade.reasons]
    return (
        decider,
        reasons,
        {(g.episode_id, g.criterion_id): g.classification for g in report.grades},
    )


def _evidence(value: float, criterion: float, *, draw: float | None = None) -> Answer:
    """Answer evidence questions with one value and criterion questions with another."""

    def answer(name: str, state: str) -> float:
        """Route by question family suffix."""
        del state
        if name.endswith("#evidence"):
            return value
        if draw is not None and name.endswith("#draw1"):
            return draw
        return criterion

    return answer


def test_the_bank_asks_draws_paraphrases_and_evidence_in_few_calls() -> None:
    """Twelve criteria with two draws and evidence are 36 questions: two calls per episode."""
    criteria = tuple(_criterion(f"c{index}") for index in range(12))
    bank = criterion_bank(criteria, GraderSettings())
    families = {question.family for question in bank.questions}

    decider, _, classes = _grade(_evidence(0.95, 0.9), criteria)

    assert families == {CRITERION_FAMILY, EVIDENCE_FAMILY} and len(bank.questions) == 36
    assert len(decider.requests) == 2
    assert all(len(request.questions) <= MAX_DECISION_QUESTIONS for request in decider.requests)
    assert set(classes.values()) == {"pass"}
    with pytest.raises(ValueError, match="distinct"):
        criterion_bank((_criterion("c0"), _criterion("c0")), GraderSettings())


def test_confident_answers_pass_or_fail_and_a_lone_half_is_only_uncertain() -> None:
    """A single 0.5 with agreeing draws and evidence present is uncertain, not ambiguous."""
    for value, expected in ((0.9, "pass"), (0.1, "fail"), (0.5, "uncertain")):
        _, reasons, classes = _grade(_evidence(0.9, value), (_criterion("c"),))
        assert classes[("e1", "c")] == expected
        assert not any(reason.startswith("ambiguous") for reason in reasons)


def test_missing_evidence_is_separated_from_a_failing_criterion() -> None:
    """A low evidence-presence probability wins over a confident criterion answer."""
    _, reasons, classes = _grade(_evidence(0.1, 0.05), (_criterion("c"),))

    assert classes[("e1", "c")] == "insufficient_evidence"
    assert any(reason.startswith("insufficient_evidence:") for reason in reasons)


def test_disagreeing_draws_or_paraphrases_make_a_criterion_ambiguous() -> None:
    """Draws straddling the band, or a paraphrase answering the other way, are ambiguity."""
    _, _, draws = _grade(_evidence(0.9, 0.9, draw=0.1), (_criterion("c"),))

    def paraphrased(name: str, state: str) -> float:
        """Answer the paraphrase opposite to the exact text."""
        del state
        return 0.1 if "#paraphrase" in name else 0.9

    _, reasons, paraphrases = _grade(paraphrased, (_criterion("c", "Was c done?"),))

    assert draws[("e1", "c")] == "ambiguous_criterion"
    assert paraphrases[("e1", "c")] == "ambiguous_criterion"
    assert any("paraphrases" in reason for reason in reasons)


def test_uncertainty_across_independent_episodes_makes_a_criterion_ambiguous() -> None:
    """A criterion in the band on most evidence-bearing episodes goes back to authoring."""

    def answer(name: str, state: str) -> float:
        """Episode e4 is confident; the rest are in the band."""
        if name.endswith("#evidence"):
            return 0.9
        return 0.95 if state.endswith("e4") else 0.5

    decider = _Decider(answer)
    report = grade_episodes(
        DecisionsTraceTagger(decider, _SNAPSHOT),
        tuple(EpisodeView(episode_id=e, text=f"episode {e}") for e in ("e1", "e2", "e3", "e4")),
        (_criterion("c"),),
    )
    classes = {grade.episode_id: grade.classification for grade in report.grades}

    assert classes == {
        "e1": "ambiguous_criterion",
        "e2": "ambiguous_criterion",
        "e3": "ambiguous_criterion",
        "e4": "pass",
    }
    assert report.summaries[0].ambiguous
    assert "3 of 4 independent episodes" in report.summaries[0].reasons[0]

    _, _, few = _grade(answer, (_criterion("c"),), ("e1", "e2"))
    assert set(few.values()) == {"uncertain"}


def test_a_confident_reference_the_other_way_is_a_disagreement() -> None:
    """A judge or second grader that disagrees is recorded by source, not averaged away."""
    judge = (
        ReferenceVerdict(source="llm-judge", episode_id="e1", criterion_id="c", passed=False),
        ReferenceVerdict(source="llm-judge", episode_id="e1", criterion_id="d", passed=True),
    )
    decider = _Decider(_evidence(0.9, 0.9))
    report = grade_episodes(
        DecisionsTraceTagger(decider, _SNAPSHOT),
        (EpisodeView(episode_id="e1", text="episode e1"),),
        (_criterion("c"), _criterion("d")),
        references=judge,
    )
    by_id = {grade.criterion_id: grade for grade in report.grades}

    assert by_id["c"].classification == "disagreement"
    assert by_id["c"].disagreeing_sources == ("llm-judge",)
    assert by_id["d"].classification == "pass"
    assert references_from(report, "second")[0].passed is True


def test_the_band_is_configurable_and_must_be_ordered() -> None:
    """A narrower band turns a 0.65 into a pass; an inverted band is refused."""
    narrow = GraderSettings(band=Band(low=0.4, high=0.6))
    _, _, classes = _grade(_evidence(0.9, 0.65), (_criterion("c"),), settings=narrow)

    assert classes[("e1", "c")] == "pass"
    with pytest.raises(ValueError, match="below"):
        Band(low=0.7, high=0.3)
