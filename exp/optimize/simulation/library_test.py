"""Library growth requires held-out progress, not success on the repaired example."""

import pytest

from exp.optimize.simulation.library import LibraryCase, LibraryEvaluation, select_library_edit
from exp.simulation.engines.text.knowledge import SimulationKnowledge


def test_net_gain_cannot_hide_a_protected_regression() -> None:
    """A higher held-out average does not permit forgetting established behavior."""
    parent = SimulationKnowledge(rules="parent")
    child = SimulationKnowledge(rules="child")

    def evaluate(knowledge: SimulationKnowledge, case: LibraryCase) -> LibraryEvaluation:
        """Expose a held-out gain accompanied by a protected-case regression."""
        scores = {
            ("parent", "new"): 0.3,
            ("child", "new"): 0.8,
            ("parent", "protected"): 1.0,
            ("child", "protected"): 0.5,
        }
        return LibraryEvaluation(
            score=scores[knowledge.rules, case.case_id],
            status="completed",
            output={},
            feedback="Registered progress score",
        )

    result = select_library_edit(
        parent,
        child,
        training=(LibraryCase(case_id="repair-example", lineage_id="training-world", input={}),),
        held_out=(LibraryCase(case_id="new", lineage_id="new-world", input={}),),
        protected=(LibraryCase(case_id="protected", lineage_id="known-world", input={}),),
        evaluator_sha256="a" * 64,
        evaluate=evaluate,
        record=lambda value: None,
    )
    assert not result.accepted and result.active == parent
    assert result.protected_regressions == ("protected",)


def test_improvement_on_assessed_new_logs_selects_child() -> None:
    """A selected child leaves the old immutable knowledge available for pinned sessions."""
    parent, child = SimulationKnowledge(rules="parent"), SimulationKnowledge(rules="child")
    receipts = []

    def evaluate(knowledge: SimulationKnowledge, case: LibraryCase) -> LibraryEvaluation:
        """Return equal protection and a real paired held-out improvement."""
        score = 1 if case.case_id == "protected" else (0.75 if knowledge == child else 0.5)
        return LibraryEvaluation(
            score=score, status="completed", output={}, feedback="Observed progress"
        )

    result = select_library_edit(
        parent,
        child,
        training=(LibraryCase(case_id="repair-example", lineage_id="training-world", input={}),),
        held_out=(LibraryCase(case_id="new", lineage_id="new-world", input={}),),
        protected=(LibraryCase(case_id="protected", lineage_id="known-world", input={}),),
        evaluator_sha256="a" * 64,
        evaluate=evaluate,
        record=receipts.append,
    )
    assert result.accepted and result.active == child and result.held_out_gain == 0.25
    assert parent.rules == "parent" and len(receipts) == 4


@pytest.mark.parametrize("same_lineage", [False, True])
def test_protected_cases_cannot_be_exposed_as_training(same_lineage: bool) -> None:
    """Reject both repeated identities and a different case from the protected world."""
    protected = LibraryCase(case_id="protected", lineage_id="protected-world", input={})
    training = (
        protected if not same_lineage else protected.model_copy(update={"case_id": "another"})
    )

    def forbidden(knowledge: SimulationKnowledge, case: LibraryCase) -> LibraryEvaluation:
        """Any evaluation here would expose a bad registration to paid work."""
        raise AssertionError("invalid split reached evaluation")

    with pytest.raises(ValueError, match="training and protected"):
        select_library_edit(
            SimulationKnowledge(rules="parent"),
            SimulationKnowledge(rules="child"),
            training=(training,),
            held_out=(LibraryCase(case_id="selection", lineage_id="new-world", input={}),),
            protected=(protected,),
            evaluator_sha256="a" * 64,
            evaluate=forbidden,
            record=lambda value: None,
        )


def test_unchanged_knowledge_cannot_gain_acceptance_from_generation_noise() -> None:
    """Do not purchase paired draws to promote the identical artifact by chance."""
    parent = SimulationKnowledge(rules="unchanged")

    def forbidden(knowledge: SimulationKnowledge, case: LibraryCase) -> LibraryEvaluation:
        """Identical knowledge has no proposed edit to evaluate."""
        raise AssertionError("unchanged proposal reached evaluation")

    result = select_library_edit(
        parent,
        parent,
        training=(LibraryCase(case_id="train", lineage_id="train-world", input={}),),
        held_out=(LibraryCase(case_id="held", lineage_id="held-world", input={}),),
        protected=(LibraryCase(case_id="protected", lineage_id="protected-world", input={}),),
        evaluator_sha256="a" * 64,
        evaluate=forbidden,
        record=lambda value: None,
    )
    assert not result.accepted and result.active == parent and result.held_out_gain is None
