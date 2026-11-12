"""The dense finding and adversarial family machines refuse and record illegal moves."""

from __future__ import annotations

from enum import StrEnum

import pytest

from exp.common.state_machine import StateMachine, Transition, target_table
from exp.optimize.evaluation.variance.adversarial_rubric import resume
from exp.optimize.evaluation.variance.behavior_adversarial import mining_lifecycle
from exp.optimize.evaluation.variance.behavior_adversarial_test import report
from exp.optimize.evaluation.variance.dense_hybrid import finding_lifecycle
from exp.optimize.evaluation.variance.dense_reward import DenseFinding
from exp.optimize.evaluation.variance.reward_states import (
    FAMILY_MACHINE,
    FINDING_MACHINE,
    FamilyState,
    FindingState,
)

F = FindingState
A = FamilyState


@pytest.mark.parametrize(
    ("path", "illegal"),
    [
        ((F.TAGGED, F.ROUTED, F.UNCERTAIN, F.UNSCORED), F.SCORED),
        ((F.TAGGED, F.ROUTED, F.UNCERTAIN), F.SCORED),
        ((F.TAGGED, F.CLEARED), F.ESTABLISHED),
        ((F.TAGGED, F.ROUTED, F.ESTABLISHED, F.SCORED), F.UNSCORED),
        ((), F.SCORED),
        ((F.TAGGED,), F.ABSENT),
    ],
)
def test_illegal_finding_moves_are_refused_and_recorded(
    path: tuple[FindingState, ...], illegal: FindingState
) -> None:
    """An unscored finding never becomes scored; tags alone never establish; terminals hold."""
    states = FINDING_MACHINE.lifecycle().start("f")
    for target in path:
        assert states.move("f", target, "legal")
    before = states.state("f")

    assert not states.move("f", illegal, "attempted")

    assert states.state("f") == before
    (refusal,) = states.refusals
    assert (refusal.source, refusal.target) == (before.value, illegal.value)
    assert len(states.transitions) == len(path)


@pytest.mark.parametrize(
    ("path", "illegal"),
    [
        ((A.NOT_ADVERSARIAL,), A.MINED),
        ((), A.LINTER_ACCEPTED),
        ((A.ADVERSARIAL,), A.LINTER_DRAFTED),
        ((A.ADVERSARIAL, A.MINED, A.LINTER_DRAFTED, A.LINTER_REFUSED), A.LINTER_ACCEPTED),
        ((A.ADVERSARIAL, A.MINED, A.LINTER_FAILED), A.LINTER_DRAFTED),
    ],
)
def test_illegal_family_moves_are_refused_and_recorded(
    path: tuple[FamilyState, ...], illegal: FamilyState
) -> None:
    """A non-adversarial group is never mined; a refused linter is never accepted later."""
    states = FAMILY_MACHINE.lifecycle().start("g")
    for target in path:
        assert states.move("g", target)

    assert not states.move("g", illegal)
    assert states.refusals[0].target == illegal.value and len(states.transitions) == len(path)


def test_statecharts_are_generated_from_the_tables() -> None:
    """Every edge and terminal of the table appears in the Mermaid chart, nothing else."""
    for machine in (FINDING_MACHINE, FAMILY_MACHINE):
        chart = machine.mermaid().splitlines()
        assert chart[0] == "stateDiagram-v2" and chart[1] == f"    [*] --> {machine.start}"
        edges = {(s.value, f"{t.value}: {e}") for s, e, t in machine.transitions}
        drawn = {tuple(line.strip().split(" --> ")) for line in chart[2:]}
        assert drawn == edges | {(state.value, "[*]") for state in machine.final}
    assert "uncertain --> scored" not in FINDING_MACHINE.mermaid()
    assert set(FINDING_MACHINE.final) == {F.SCORED, F.UNSCORED}


def test_a_table_with_undeclared_states_is_refused() -> None:
    """The table is the only source of legal moves and every declared state must be reachable."""

    class Two(StrEnum):
        A = "a"
        B = "b"

    class Three(StrEnum):
        A = "a"
        B = "b"
        C = "c"

    with pytest.raises(ValueError, match="unreachable"):
        StateMachine(
            name="bad",
            states=Three,
            initial=frozenset({Three.A}),
            transitions=target_table({Three.A: frozenset({Three.B})}),
        )
    assert StateMachine(
        name="ok", states=Two, initial=frozenset({Two.A}), transitions=((Two.A, "b", Two.B),)
    ).final == frozenset({Two.B})


def test_an_aggregation_that_scores_an_uncertain_finding_is_refused() -> None:
    """Replaying an impossible scored-uncertain finding records the refusal, not the move."""
    bad = DenseFinding(
        finding_id="tag:x",
        tag_id="x",
        candidate_ids=("tag:x",),
        decision="uncertain",
        rationale="split",
        scored=True,
        reward=-0.5,
    )
    from exp.optimize.evaluation.variance.dense_hybrid import TagRoute

    states = finding_lifecycle(
        tagged=True,
        routes=(TagRoute(tag_id="x", route="judge"),),
        candidates=(),
        findings=(bad,),
    )

    assert states.state("tag:x") == F.UNCERTAIN
    assert [(r.source, r.target) for r in states.refusals] == [("uncertain", "scored")]


def test_family_mining_persists_its_moves_and_rechecks_them_on_resume() -> None:
    """Groups move to mined or not adversarial; a tampered persisted move is refused.

    The vocabulary here counts a skipped step as neither hack nor failure mode, so its group is
    not adversarial.
    """
    saved = report()
    lenient = saved.spec.model_copy(
        update={
            "positive_tags": (*saved.spec.positive_tags, "skipped_required_step"),
            "negative_tags": tuple(
                tag for tag in saved.spec.negative_tags if tag != "skipped_required_step"
            ),
        }
    )
    states = mining_lifecycle(saved.model_copy(update={"spec": lenient}))
    finals = sorted(state.value for state in states.current.values())
    assert finals == ["mined", "mined", "not_adversarial"]

    subject = next(s for s, state in states.current.items() if state == A.MINED)
    moves = [t for t in states.transitions if t.subject == subject]
    tampered = [
        Transition(
            machine="adversarial-family-v1", subject=subject, source="group", target="mined"
        ),
        *moves,
    ]
    resumed = resume(FAMILY_MACHINE.lifecycle(), subject, tampered)
    assert resumed.state(subject) == A.MINED
    assert resumed.refusals and resumed.refusals[0].target == "mined"
