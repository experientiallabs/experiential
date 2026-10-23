"""Development selection of behavioral-library edits on fixed held-out logs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json
from exp.simulation.engines.text.knowledge import SimulationKnowledge


class LibraryCase(ContractModel):
    """One development example; confirmation data is absent from this selection interface.

    Attributes:
        case_id: Stable example identity.
        lineage_id: Shared source or fixture family, disjoint across selection splits.
        input: Permitted development inputs supplied to the declared evaluator.
    """

    case_id: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    input: JsonObject


class LibraryEvaluation(ContractModel):
    """Actual execution outcome and grounded feedback, including failed examples.

    Attributes:
        score: Fixed evaluator metric in [0, 1], with unavailable execution scoring zero.
        status: Actual execution/evaluation status, retained separately from score.
        output: Exact outputs or references retained by the evaluator.
        feedback: Development-only observations for proposing edits.
    """

    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    status: Literal["completed", "failed", "unassessable"]
    output: JsonObject
    feedback: str

    @model_validator(mode="after")
    def unavailable(self) -> LibraryEvaluation:
        """Do not reward failed evaluations by omitting them from the selection metric."""
        if self.status != "completed" and self.score != 0:
            raise ValueError("unavailable evaluations must retain zero selection credit")
        return self


class LibrarySelection(ContractModel):
    """An edit decision, distinct from publication or deployment authority.

    Attributes:
        parent_sha256: Previously active immutable behavioral knowledge.
        candidate_sha256: Attempted edit, retained even when rejected.
        accepted: True only for an assessed held-out gain without protected regressions.
        active: Selected knowledge for future sessions; running sessions remain pinned.
        held_out_gain: Mean paired progress difference, absent if any assessment is unavailable.
        protected_regressions: Previously assessed protected cases harmed by the edit.
        evaluator_sha256: Frozen progress metric and judge/calibration configuration.
    """

    parent_sha256: Sha256
    candidate_sha256: Sha256
    accepted: bool
    active: SimulationKnowledge
    held_out_gain: float | None
    protected_regressions: tuple[str, ...]
    evaluator_sha256: Sha256


def select_library_edit(
    parent: SimulationKnowledge,
    candidate: SimulationKnowledge,
    *,
    training: Sequence[LibraryCase],
    held_out: Sequence[LibraryCase],
    protected: Sequence[LibraryCase],
    evaluator_sha256: Sha256,
    evaluate: Callable[[SimulationKnowledge, LibraryCase], LibraryEvaluation],
    record: Callable[[JsonObject], None],
    minimum_gain: float = 0,
) -> LibrarySelection:
    """Keep a proposed library only if fixed held-out progress improves without regressions.

    Args:
        parent: Active pre-edit knowledge, frozen before examining held-out logs.
        candidate: One already frozen proposal, made from training errors only.
        training: Cases whose failures were available to the proposer.
        held_out: Distinct development logs for selection, not the final confirmation panel.
        protected: Previously established behavior that must not regress.
        evaluator_sha256: Caller-declared registration of progress and judge configuration.
            The callback is trusted application code, not attested by this label.
        evaluate: Actual paired execution with shared evidence, actor, budgets and metric.
        record: Durable sink for the frozen proposal, every outcome and acceptance decision.
        minimum_gain: Prospectively specified strict improvement margin in [0, 1].

    Returns:
        Parent on a tie, unassessable comparison or protected regression; child only on a gain.
    """
    if not held_out or not protected or not 0 <= minimum_gain <= 1:
        raise ValueError("nonempty selection/protection logs and a finite gain margin required")
    hold_ids = {case.case_id for case in held_out}
    protected_ids = {case.case_id for case in protected}
    if len(hold_ids) != len(held_out) or len(protected_ids) != len(protected):
        raise ValueError("selection or protection cases repeat")
    train_ids = {case.case_id for case in training}
    if len(train_ids) != len(training) or train_ids & protected_ids:
        raise ValueError("training and protected case inventories overlap or repeat")
    if {c.lineage_id for c in training} & {c.lineage_id for c in protected}:
        raise ValueError("training and protected logs share a source lineage")
    if hold_ids & (train_ids | protected_ids):
        raise ValueError("held-out selection logs were exposed as training or protected examples")
    if {c.lineage_id for c in held_out} & {c.lineage_id for c in (*training, *protected)}:
        raise ValueError("selection and protected logs share a source lineage")
    record(
        {
            "parent": parent.model_dump(mode="json"),
            "candidate": candidate.model_dump(mode="json"),
            "held_out": [c.model_dump(mode="json") for c in held_out],
            "protected": [c.model_dump(mode="json") for c in protected],
            "training": [c.model_dump(mode="json") for c in training],
            "minimum_gain": minimum_gain,
            "evaluator_sha256": evaluator_sha256,
        }
    )
    if parent.identity == candidate.identity:
        result = LibrarySelection(
            parent_sha256=parent.identity,
            candidate_sha256=candidate.identity,
            accepted=False,
            active=parent,
            held_out_gain=None,
            protected_regressions=(),
            evaluator_sha256=evaluator_sha256,
        )
        record(
            {
                "decision": result.model_dump(mode="json"),
                "skipped_reason": "unchanged knowledge",
                "authority": "development selection only",
            }
        )
        return result
    differences = []
    lost = []
    fully_assessed = True
    for label, cases in (("held_out", held_out), ("protected", protected)):
        for case in cases:
            before, after = evaluate(parent, case), evaluate(candidate, case)
            record(
                {
                    "set": label,
                    "case_id": case.case_id,
                    "before": before.model_dump(mode="json"),
                    "after": after.model_dump(mode="json"),
                }
            )
            assessed = before.status == after.status == "completed"
            fully_assessed &= assessed
            if label == "held_out":
                differences.append(after.score - before.score)
            elif before.status == "completed" and (not assessed or after.score < before.score):
                lost.append(case.case_id)
    gain = sum(differences) / len(differences) if fully_assessed else None
    accepted = gain is not None and gain > minimum_gain and not lost
    result = LibrarySelection(
        parent_sha256=parent.identity,
        candidate_sha256=candidate.identity,
        accepted=accepted,
        active=candidate if accepted else parent,
        held_out_gain=gain,
        protected_regressions=tuple(lost),
        evaluator_sha256=evaluator_sha256,
    )
    record(
        {
            "decision": result.model_dump(mode="json"),
            "decision_sha256": sha256_json(result),
            "authority": "development selection only",
        }
    )
    return result
