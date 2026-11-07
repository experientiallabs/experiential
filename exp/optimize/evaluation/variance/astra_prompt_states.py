"""The Astra prompt refinement's declared state machine: one target's loop and its human review.

One table, checked by :class:`~exp.common.state_machine.StateMachine` on every move. Each move a
refinement makes is persisted in its journal; an undeclared move is refused and journaled.

- ``opened``: the target, its calibration set and settings are fixed; the whole loop's bound is
  reserved on the cumulative ledger (``budget_refused`` when it does not fit, before any call).
- ``referenced``: Astra answered every calibration input under the role's own contract.
- ``evaluated``: the code prompt ran on both sides of the split and was judged blind.
- ``rewritten``: Astra proposed a rewrite from the cited rewrite-set failures.
- ``candidate_evaluated``: the rewrite passed the mechanical leakage check and was evaluated
  (and, for a judge target, run on the labeled acceptance cases).
- ``accepted_round`` / ``rejected_round``: held-out parity rose (and the gate held) or not; a
  leaking rewrite is rejected before any evaluation.
- ``target_met`` / ``plateaued``: the loop stops; the best prompt is a candidate (``plateaued``
  straight from ``evaluated`` when the code prompt leaves no rewrite-set failure to cite).
- ``approved`` / ``declined``: the human review; only an approved prompt is frozen.
"""

from __future__ import annotations

from enum import StrEnum

from exp.common.state_machine import StateMachine


class RefinementState(StrEnum):
    """One target's refinement state."""

    OPENED = "opened"
    BUDGET_REFUSED = "budget_refused"
    REFERENCED = "referenced"
    EVALUATED = "evaluated"
    REWRITTEN = "rewritten"
    CANDIDATE_EVALUATED = "candidate_evaluated"
    ACCEPTED_ROUND = "accepted_round"
    REJECTED_ROUND = "rejected_round"
    TARGET_MET = "target_met"
    PLATEAUED = "plateaued"
    APPROVED = "approved"
    DECLINED = "declined"


S = RefinementState

REFINEMENT_TRANSITIONS: tuple[tuple[RefinementState, str, RefinementState], ...] = (
    (S.OPENED, "budget_refused", S.BUDGET_REFUSED),
    (S.OPENED, "referenced", S.REFERENCED),
    (S.REFERENCED, "evaluated", S.EVALUATED),
    (S.EVALUATED, "target_met", S.TARGET_MET),
    (S.EVALUATED, "rewritten", S.REWRITTEN),
    (S.EVALUATED, "plateaued", S.PLATEAUED),
    (S.REWRITTEN, "leak_refused", S.REJECTED_ROUND),
    (S.REWRITTEN, "rewrite_unusable", S.REJECTED_ROUND),
    (S.REWRITTEN, "candidate_evaluated", S.CANDIDATE_EVALUATED),
    (S.CANDIDATE_EVALUATED, "accepted_round", S.ACCEPTED_ROUND),
    (S.CANDIDATE_EVALUATED, "rejected_round", S.REJECTED_ROUND),
    (S.ACCEPTED_ROUND, "rewritten", S.REWRITTEN),
    (S.ACCEPTED_ROUND, "target_met", S.TARGET_MET),
    (S.ACCEPTED_ROUND, "plateaued", S.PLATEAUED),
    (S.REJECTED_ROUND, "rewritten", S.REWRITTEN),
    (S.REJECTED_ROUND, "plateaued", S.PLATEAUED),
    (S.TARGET_MET, "approved", S.APPROVED),
    (S.TARGET_MET, "declined", S.DECLINED),
    (S.PLATEAUED, "approved", S.APPROVED),
    (S.PLATEAUED, "declined", S.DECLINED),
)
"""Every legal move."""

REFINEMENT_MACHINE = StateMachine(
    name="astra_prompt_refinement",
    states=RefinementState,
    initial=frozenset({S.OPENED}),
    transitions=REFINEMENT_TRANSITIONS,
)
"""One target's refinement loop and review."""


def statechart() -> str:
    """Render the refinement machine as Mermaid."""
    return REFINEMENT_MACHINE.mermaid()
