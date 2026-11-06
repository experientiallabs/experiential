"""Checkpoint 1 bank kinds, and checkpoint 3: dead rule tags and state tags' neighbours."""

from __future__ import annotations

from exp.optimize.evaluation.variance.tag_monitor_contracts import MonitorSpec
from exp.optimize.evaluation.variance.tag_monitor_stats import (
    CorpusInputs,
    RuleEvidence,
    RuleInputs,
    StateStep,
    corpus_stats,
    rule_stats,
)
from exp.simulation.mining.tagging import TagMatrix, TagRow
from exp.simulation.rules.admission import AdmissionVerdict, RuleTagMatrix, TrajectoryTags


def _matrix(bank: str, columns: dict[str, list[float]]) -> TagMatrix:
    """An observations matrix over traces t0, t1, ... from per-question columns."""
    names = list(columns)
    return TagMatrix(
        bank_id=bank,
        bank_version=1,
        question_ids=tuple(names),
        view="observations",
        rows=tuple(
            TagRow(trace_id=f"t{i}", unit_hash=f"h{i}", values=tuple(columns[q][i] for q in names))
            for i in range(len(columns[names[0]]))
        ),
    )


def test_only_descriptive_banks_halt_on_their_always_no_share() -> None:
    """Two banks with 2 of 3 questions always-no: the declared detector is flagged, not halted."""
    columns = {"rare.a": [0.05] * 6, "rare.b": [0.02] * 6, "live": [0.9, 0.1] * 3}
    inputs = CorpusInputs(matrices=(_matrix("shape", columns), _matrix("errors", columns)))
    subjects, halts, _ = corpus_stats(inputs, MonitorSpec(bank_kinds={"errors": "detector"}))
    rows = {row.subject: row for row in subjects}
    assert [(h.subject, h.condition) for h in halts] == [
        ("bank:shape.observations", "bank_uninformative")
    ]
    assert rows["bank:shape.observations"].flags == ("uninformative",)
    detector = rows["bank:errors.observations"]
    assert detector.flags == ("detector_rare",)
    assert detector.values["detector_rare_questions"] == 2.0
    assert "always_no" in rows["question:errors.observations/rare.a"].flags
    undeclared, _, _ = corpus_stats(inputs, MonitorSpec())
    assert "detector_rare_questions" not in {
        key for row in undeclared if row.kind == "bank" for key in row.values
    }


def _rule(rule: str, value: float, contradicting: tuple[str, ...] = ()) -> RuleEvidence:
    """One rule over three trajectories answering ``value`` to its one question."""
    return RuleEvidence(
        verdict=AdmissionVerdict(
            rule_id=rule,
            admitted=not contradicting,
            status="contradicted" if contradicting else "supported",
            supporting_lineages=("L1", "L2"),
            contradicting=contradicting,
            reason="fixture",
        ),
        matrix=RuleTagMatrix(
            rule_id=rule,
            question_ids=(f"{rule}.applies.q",),
            rows=tuple(
                TrajectoryTags(trace_id=f"t{i}", lineage_id=f"L{i}", values=(value,))
                for i in range(3)
            ),
        ),
    )


def test_rule_support_contradiction_and_dead_tags() -> None:
    """A never-firing rule is flagged; a contradicted one is flagged; one of three does not halt."""
    inputs = RuleInputs(
        rules=(_rule("r.live", 0.9), _rule("r.dead", 0.1), _rule("r.bad", 0.9, ("t9",)))
    )
    subjects, halts, _ = rule_stats(inputs, MonitorSpec())
    flags = {row.subject: row.flags for row in subjects}
    assert flags["rule:r.dead"] == ("tags_never_fire",)
    assert flags["rule_question:r.dead/r.dead.applies.q"] == ("never_fires",)
    assert flags["rule:r.bad"] == ("contradicted",) and flags["rule:r.live"] == ()
    assert halts == []


def test_a_state_tag_whose_neighbours_take_another_action_is_flagged() -> None:
    """Steps sharing tag ``a`` act differently across lineages; tag ``b`` steps act alike."""
    steps = (
        StateStep(step_id="s1", lineage_id="L1", probabilities=(0.9, 0.1), next_action="edit"),
        StateStep(step_id="s2", lineage_id="L2", probabilities=(0.9, 0.1), next_action="run"),
        StateStep(step_id="s3", lineage_id="L3", probabilities=(0.1, 0.9), next_action="read"),
        StateStep(step_id="s4", lineage_id="L4", probabilities=(0.1, 0.9), next_action="read"),
    )
    inputs = RuleInputs(state_bank="state.f", state_questions=("a", "b"), state_steps=steps)
    subjects, halts, evidence = rule_stats(inputs, MonitorSpec(recall_neighbours=1))
    rows = {row.subject: row for row in subjects}
    assert rows["state_tag:state.f/a"].values["neighbour_disagreement"] == 1.0
    assert "recall_disagrees" in rows["state_tag:state.f/a"].flags
    assert rows["state_tag:state.f/b"].values["neighbour_disagreement"] == 0.0
    assert "evidence:state_tag:state.f/a" in evidence and halts == []
