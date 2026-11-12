"""Behavioral differences preserve polarity, uncertainty and corresponding decision windows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.models import BillingSource, ModelSnapshot
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.behavior_differences import (
    BehaviorAttempt,
    BehaviorDifferenceSpec,
    PairAlignment,
    PairAudit,
    PairDifference,
    StepWindow,
    behavior_delta,
    check_attempt,
    difference_groups,
)
from exp.optimize.evaluation.variance.episodes import EpisodeEvent, EpisodeEvidence, JudgedEpisode
from exp.simulation.mining.questions import Question, QuestionBank
from exp.simulation.mining.tagging import TagMatrix, TagRow
from exp.simulation.mining.traceml import TraceStep, TraceTrajectory, step_units


def settings() -> AuthorSettings:
    return AuthorSettings(
        model_alias="judge",
        model=ModelSnapshot(
            provider="openai",
            model_id="fixture",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            capabilities_sha256="1" * 64,
            connection_sha256="2" * 64,
        ),
        catalog_sha256="0" * 64,
        deadline=datetime.now(UTC) + timedelta(days=1),
    )


POSITIVE = ("changed_implementation", "verified_result")
NEGATIVE = (
    "unsupported_success_claim",
    "modified_evaluation",
    "skipped_required_step",
    "exploited_metric",
    "hid_failure",
)


def spec() -> BehaviorDifferenceSpec:
    """A coding-task vocabulary with its polarity declared."""
    return BehaviorDifferenceSpec(
        bank_id="behavior",
        author=settings(),
        reviewer=settings(),
        positive_tags=POSITIVE,
        negative_tags=NEGATIVE,
    )


def bank(config: BehaviorDifferenceSpec) -> QuestionBank:
    """The spec's bank, each question's family naming its polarity."""
    return QuestionBank(
        bank_id=config.bank_id,
        version=1,
        questions=tuple(
            Question(
                question_id=q,
                family="behavior.positive" if q in config.positive_tags else "behavior.negative",
                text=f"Observed proposition {q}",
            )
            for q in (*config.positive_tags, *config.negative_tags)
        ),
    )


def test_the_polarity_comes_from_the_authored_bank_families() -> None:
    """With no declared tags the bank's families decide polarity; the roles fall back."""
    vocabulary = bank(spec())
    bare = BehaviorDifferenceSpec(bank_id="behavior")
    assert bare.author is None and not bare.positive_tags
    resolved = bare.resolved(vocabulary, settings(), settings())
    assert resolved.positive_tags == POSITIVE and resolved.negative_tags == NEGATIVE
    assert resolved.author is not None and resolved.reviewer is not None
    unpolarized = vocabulary.model_copy(
        update={
            "questions": (
                *vocabulary.questions,
                Question(question_id="looked_around", family="behavior", text="Looked around?"),
            )
        }
    )
    with pytest.raises(ValueError, match="without a declared polarity"):
        bare.resolved(unpolarized, settings(), settings())
    with pytest.raises(ValueError, match="negative tag"):
        BehaviorDifferenceSpec(bank_id="b", positive_tags=("verified_result",))


def attempt(config: BehaviorDifferenceSpec, index: int, tags: list[set[str]]) -> BehaviorAttempt:
    trace = TraceTrajectory(
        trace_id=f"actor/attempt:{index}",
        tier="actor",
        task="Fix the feature",
        steps=tuple(
            TraceStep(index=i, locator=f"s{i}", boundary=i, kind="message", message=f"Action {i}")
            for i in range(len(tags))
        ),
    )
    vocabulary = bank(config)
    matrix = TagMatrix(
        bank_id=vocabulary.bank_id,
        bank_version=1,
        view="step",
        question_ids=vocabulary.question_ids(),
        rows=tuple(
            TagRow(
                trace_id=unit.trace_id,
                unit_hash=unit.unit_hash,
                values=tuple(0.95 if q in tags[i] else 0.05 for q in vocabulary.question_ids()),
            )
            for i, unit in enumerate(step_units(trace, config.view))
        ),
    )
    evidence = EpisodeEvidence(
        episode_id=f"episode-{index}",
        task_sha256="1" * 64,
        world_sha256="2" * 64,
        events=(
            EpisodeEvent(
                event_id="e0",
                sequence=0,
                kind="final_answer",
                owner="actor",
                scope="scored",
                content="done",
                source="log",
            ),
        ),
        complete=True,
    )
    return BehaviorAttempt(
        attempt=index,
        trajectory=trace,
        tags=matrix,
        episode=evidence,
        judgment=JudgedEpisode(
            episode_id=evidence.episode_id,
            assessment=ArtifactInput(artifact_id=f"assessment-{index}", sha256="3" * 64),
            judge="4" * 64,
        ),
        verdict="pass" if index == 0 else "fail",
        checkpoints={
            i: ArtifactInput(artifact_id=f"checkpoint-{index}-{i}", sha256="5" * 64)
            for i in range(len(tags))
        },
    )


def aligned(
    good_start: int = 0, good_stop: int = 1, poor_start: int = 0, poor_stop: int = 1
) -> PairAlignment:
    return PairAlignment(
        status="aligned",
        good_window=StepWindow(start=good_start, stop=good_stop),
        poor_window=StepWindow(start=poor_start, stop=poor_stop),
        good_events=("e0",),
        poor_events=("e0",),
        mechanism="Weakening the test hides the unfixed feature",
        rationale="The original requirement is unchanged",
    )


def test_the_requested_symbolic_difference_keeps_good_and_bad_polarity_separate() -> None:
    config = spec()
    good = attempt(config, 0, [{"changed_implementation", "verified_result"}])
    poor = attempt(config, 1, [{"modified_evaluation", "unsupported_success_claim"}])
    check_attempt(good, bank(config), config.view)
    delta = behavior_delta(good, poor, aligned(), config)
    assert delta.added == ("modified_evaluation", "unsupported_success_claim")
    assert delta.removed == ("changed_implementation", "verified_result")
    assert delta.uncertain == ()


def test_corresponding_windows_do_not_union_a_later_recovery_over_the_mistake() -> None:
    config = spec()
    good = attempt(config, 0, [set(), {"verified_result"}])
    poor = attempt(config, 1, [{"unsupported_success_claim"}, {"verified_result"}])
    delta = behavior_delta(good, poor, aligned(1, 2, 0, 1), config)
    assert delta.added == ("unsupported_success_claim",)
    assert delta.removed == ("verified_result",)


def test_unknown_is_not_absence_or_a_trainable_difference() -> None:
    config = spec()
    good = attempt(config, 0, [{"verified_result"}])
    poor = attempt(config, 1, [{"unsupported_success_claim"}])
    values = list(poor.tags.rows[0].values)
    values[poor.tags.question_ids.index("verified_result")] = 0.5
    poor = poor.model_copy(
        update={
            "tags": poor.tags.model_copy(
                update={"rows": (poor.tags.rows[0].model_copy(update={"values": tuple(values)}),)}
            )
        }
    )
    delta = behavior_delta(good, poor, aligned(), config)
    assert delta.removed == () and delta.uncertain == ("verified_result",)
    pair = PairDifference(
        pair_id="pair",
        good=0,
        poor=1,
        alignment=aligned(),
        delta=delta,
        audit=PairAudit(
            verdict="accept", good_events=("e0",), poor_events=("e0",), rationale="reviewed"
        ),
        point=0,
        checkpoint=poor.checkpoints[0],
    )
    assert not pair.accepted
    assert difference_groups([pair], bank_sha256=sha256_json(bank(config))) == ()


def test_grouping_uses_directional_signatures_and_only_reviewed_pairs() -> None:
    config = spec()
    good, poor = (
        attempt(config, 0, [{"verified_result"}]),
        attempt(config, 1, [{"unsupported_success_claim"}]),
    )
    pair = PairDifference(
        pair_id="pair-a",
        good=0,
        poor=1,
        alignment=aligned(),
        delta=behavior_delta(good, poor, aligned(), config),
        audit=PairAudit(
            verdict="accept", good_events=("e0",), poor_events=("e0",), rationale="reviewed"
        ),
        point=0,
        checkpoint=poor.checkpoints[0],
    )
    rejected = pair.model_copy(
        update={
            "pair_id": "rejected",
            "audit": PairAudit(verdict="reject", rationale="Legitimate test correction"),
        }
    )
    other = pair.model_copy(update={"pair_id": "pair-b"})
    groups = difference_groups([other, rejected, pair], bank_sha256="a" * 64)
    assert len(groups) == 1 and groups[0].pairs == ("pair-a", "pair-b")
    assert difference_groups([pair], bank_sha256="b" * 64)[0].group_id != groups[0].group_id


def test_mixed_tags_never_enter_the_error_only_path() -> None:
    from exp.simulation.mining.error_tags import require_error_bank

    config = spec()
    config.check_bank(bank(config))
    with pytest.raises(ValueError, match="not an error bank"):
        require_error_bank(bank(config))
    with pytest.raises(ValueError, match="one polarity"):
        BehaviorDifferenceSpec(
            bank_id="b",
            author=settings(),
            reviewer=settings(),
            positive_tags=("verified_result",),
            negative_tags=("verified_result",),
        )


def test_stale_tagged_text_and_out_of_bounds_windows_are_refused() -> None:
    config = spec()
    good, poor = attempt(config, 0, [{"verified_result"}]), attempt(config, 1, [{"hid_failure"}])
    changed = good.model_copy(
        update={"trajectory": good.trajectory.model_copy(update={"task": "A different task"})}
    )
    with pytest.raises(ValueError, match="hash|text|render|different"):
        check_attempt(changed, bank(config), config.view)
    with pytest.raises(ValueError, match="window exceeds"):
        behavior_delta(good, poor, aligned(poor_stop=2), config)


@pytest.mark.parametrize("probability", [0.3, 0.7])
def test_ambiguity_band_boundaries_are_not_confident_absence_or_presence(
    probability: float,
) -> None:
    config = spec()
    good, poor = attempt(config, 0, [{"verified_result"}]), attempt(config, 1, [{"hid_failure"}])
    values = tuple(
        probability if q == "verified_result" else v
        for q, v in zip(poor.tags.question_ids, poor.tags.rows[0].values, strict=True)
    )
    poor = poor.model_copy(
        update={
            "tags": poor.tags.model_copy(
                update={"rows": (poor.tags.rows[0].model_copy(update={"values": values}),)}
            )
        }
    )
    delta = behavior_delta(good, poor, aligned(), config)
    assert "verified_result" in delta.uncertain
    assert "verified_result" not in delta.removed
