"""Feedback-directed one-step harness improvement behind an explicit promotion gate.

One piece of user feedback ("you should have access to my GitHub") drives one
:class:`~wmh.harness.population.HarnessPopulationOptimizer` run: the feedback text becomes the
proposer directive, and the must-pass verification tasks synthesized from it
(`wmh.scenarios.feedback.synthesize_verification_tasks`) join the standard suite in one exact
score request. Promotion is decided here, never by the optimizer's blended ``result.best``
(which averages suite and verification cells together): a candidate is promotable only when its
standard-suite mean stays within :class:`ImproveGate.suite_margin` of the seed AND a strict
majority of attempts passes on every verification task. Among gate-passing candidates the
highest suite score wins, with the earliest candidate breaking ties.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from wmh.evals.tasks import TaskSpec
from wmh.harness.doc import HarnessDoc
from wmh.harness.population import HarnessPopulationOptimizer, PopulationOptimizationResult
from wmh.harness.project_proposer import CandidateProposer, EvaluatedCandidate
from wmh.harness.scoring import HarnessScore, HarnessScoreReport, ScoreRequest
from wmh.harness.source_tree import HarnessSourceTree

DEFAULT_SUITE_MARGIN = 0.10


class ImproveGate(BaseModel):
    """Relative regression budget on the standard suite (verification tasks get none).

    A candidate's suite mean must be at least ``(1 - suite_margin) * seed_suite_mean``. The
    margin is a fraction in ``[0, 1)``: 0 demands no regression at all, and values at or above
    1 would accept a zero-scoring candidate, so they are rejected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite_margin: float = Field(default=DEFAULT_SUITE_MARGIN, ge=0.0, lt=1.0)


_DEFAULT_GATE = ImproveGate()


class VerificationResult(BaseModel):
    """Majority-rule outcome for one verification task across all its attempts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    passed: bool
    pass_count: int
    attempts: int


class ImproveScorer(Protocol):
    """The scorer seam ``improve_harness`` needs: request minting plus candidate scoring.

    ``wmh.evals.world_model_scorer.WorldModelScorer`` satisfies this protocol directly.
    """

    def request(self, *, attempts: int) -> ScoreRequest: ...

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore: ...


@dataclass(frozen=True)
class ImproveOutcome:
    """The gate's promotion decision plus the complete evaluated population evidence.

    ``candidate_suite_score`` and ``verification`` describe the selected candidate; they are
    ``None`` and empty when no candidate passed the gate (the rejection reason still carries the
    best candidate's numbers).
    """

    accepted: bool
    reason: str
    seed_suite_score: float
    candidate_suite_score: float | None
    verification: tuple[VerificationResult, ...]
    selected: EvaluatedCandidate | None
    result: PopulationOptimizationResult


def suite_score(report: HarnessScoreReport, suite_task_ids: Collection[str]) -> float:
    """Return the mean cell score restricted to the standard-suite tasks.

    Args:
        report: One complete candidate scorecard.
        suite_task_ids: The task ids that constitute the standard suite.

    Returns:
        The mean of ``cell.score`` over every cell whose task id is in ``suite_task_ids``.

    Raises:
        ValueError: When no cell in the report matches ``suite_task_ids``.
    """
    wanted = set(suite_task_ids)
    scores = [cell.score for cell in report.cells if cell.task_id in wanted]
    if not scores:
        raise ValueError("score report contains no cells for the given suite task ids")
    return fmean(scores)


def verification_results(
    report: HarnessScoreReport,
    verification_task_ids: Sequence[str],
) -> tuple[VerificationResult, ...]:
    """Apply the strict-majority pass rule to each verification task, in the given order.

    A task passes only when strictly more than half of its attempts have ``cell.passed`` True:
    2 of 3 passes, while exactly half (for example 1 of 2) does not.

    Args:
        report: One complete candidate scorecard containing the verification cells.
        verification_task_ids: The verification task ids, in reporting order.

    Returns:
        One :class:`VerificationResult` per requested task id.

    Raises:
        ValueError: When a named task has no cells in the report.
    """
    results: list[VerificationResult] = []
    for task_id in verification_task_ids:
        cells = [cell for cell in report.cells if cell.task_id == task_id]
        if not cells:
            raise ValueError(f"score report contains no cells for verification task {task_id!r}")
        pass_count = sum(1 for cell in cells if cell.passed)
        results.append(
            VerificationResult(
                task_id=task_id,
                passed=pass_count * 2 > len(cells),
                pass_count=pass_count,
                attempts=len(cells),
            )
        )
    return tuple(results)


def improve_harness(
    *,
    seed: HarnessSourceTree,
    feedback: str,
    suite_tasks: Sequence[TaskSpec],
    verification_tasks: Sequence[TaskSpec],
    scorer: ImproveScorer,
    proposer_factory: Callable[[str], CandidateProposer],
    iterations: int = 1,
    attempts: int = 3,
    gate: ImproveGate = _DEFAULT_GATE,
    should_cancel: Callable[[], bool] | None = None,
    on_boundary: Callable[[PopulationOptimizationResult], None] | None = None,
) -> ImproveOutcome:
    """Run one feedback-directed population step and gate promotion explicitly.

    The scorer must already be configured with the suite tasks followed by the verification
    tasks, in that exact order; the minted request is asserted against that contract before any
    spend. The feedback text is handed verbatim to ``proposer_factory`` as the proposal
    directive.

    Args:
        seed: The champion source tree every proposal slot starts from.
        feedback: The user's feedback; becomes the proposer directive.
        suite_tasks: The standard suite, gated on the relative margin.
        verification_tasks: The synthesized must-pass tasks, gated on strict majority.
        scorer: Scorer configured over ``suite_tasks + verification_tasks`` in order.
        proposer_factory: Builds the candidate proposer from the feedback directive.
        iterations: Fixed number of proposal slots.
        attempts: Attempts per task and candidate.
        gate: The suite regression budget.
        should_cancel: Optional cooperative cancellation callback.
        on_boundary: Optional durable-boundary callback forwarded to the optimizer.

    Returns:
        The promotion decision with the full population evidence. ``result.best`` (the
        optimizer's blended winner) is deliberately ignored for promotion.

    Raises:
        ValueError: On blank feedback, an empty task set, non-disjoint or duplicate task ids
            across the two sets, or a scorer request that does not match the declared task
            order.
    """
    if not feedback.strip():
        raise ValueError("feedback must be a nonempty description of the desired improvement")
    suite_ids = tuple(task.task_id for task in suite_tasks)
    verification_ids = tuple(task.task_id for task in verification_tasks)
    if not suite_ids:
        raise ValueError("suite_tasks must be nonempty")
    if not verification_ids:
        raise ValueError("verification_tasks must be nonempty")
    combined = suite_ids + verification_ids
    if len(set(combined)) != len(combined):
        raise ValueError("suite and verification task ids must be unique and disjoint")

    request = scorer.request(attempts=attempts)
    if request.task_ids != combined:
        raise ValueError(
            "scorer request task ids must be exactly the suite tasks followed by the "
            "verification tasks, in order"
        )

    result = HarnessPopulationOptimizer(proposer_factory(feedback), scorer).optimize(
        seed=seed,
        request=request,
        iterations=iterations,
        should_cancel=should_cancel,
        on_boundary=on_boundary,
    )
    seed_suite = suite_score(result.population[0].score.report, suite_ids)
    floor = (1.0 - gate.suite_margin) * seed_suite
    candidates = result.population[1:]
    if not candidates:
        return ImproveOutcome(
            accepted=False,
            reason=(
                "no valid proposals: every proposal slot failed to produce a scoreable candidate"
            ),
            seed_suite_score=seed_suite,
            candidate_suite_score=None,
            verification=(),
            selected=None,
            result=result,
        )

    graded = [
        (
            candidate,
            suite_score(candidate.score.report, suite_ids),
            verification_results(candidate.score.report, verification_ids),
        )
        for candidate in candidates
    ]
    # The population is ordered by ascending candidate_id, so replacing only on a strictly
    # higher suite score keeps the earliest candidate on ties.
    selected: tuple[EvaluatedCandidate, float, tuple[VerificationResult, ...]] | None = None
    for candidate, candidate_suite, verification in graded:
        if candidate_suite < floor or not all(item.passed for item in verification):
            continue
        if selected is None or candidate_suite > selected[1]:
            selected = (candidate, candidate_suite, verification)
    if selected is not None:
        candidate, candidate_suite, verification = selected
        return ImproveOutcome(
            accepted=True,
            reason=(
                f"candidate {candidate.candidate_id} passed the gate: suite "
                f"{candidate_suite:.6f} vs seed {seed_suite:.6f} (floor {floor:.6f} at margin "
                f"{gate.suite_margin:.0%}); all {len(verification)} verification task(s) passed"
            ),
            seed_suite_score=seed_suite,
            candidate_suite_score=candidate_suite,
            verification=verification,
            selected=candidate,
            result=result,
        )

    # No candidate passed. Diagnose using the best candidate by suite score (ties keep the
    # earliest, because max returns the first maximal element).
    best_candidate, best_suite, best_verification = max(graded, key=lambda entry: entry[1])
    if best_suite < floor:
        reason = (
            f"suite regression beyond margin: best candidate {best_candidate.candidate_id} "
            f"scored {best_suite:.6f} on the suite, below the floor {floor:.6f} "
            f"({1.0 - gate.suite_margin:.2f} x seed {seed_suite:.6f}, margin "
            f"{gate.suite_margin:.0%})"
        )
    else:
        failing = ", ".join(
            f"{item.task_id} ({item.pass_count}/{item.attempts} passed)"
            for item in best_verification
            if not item.passed
        )
        reason = (
            f"verification tasks failed: candidate {best_candidate.candidate_id} (suite "
            f"{best_suite:.6f} vs seed {seed_suite:.6f}) did not reach a strict majority on "
            f"{failing}"
        )
    return ImproveOutcome(
        accepted=False,
        reason=reason,
        seed_suite_score=seed_suite,
        candidate_suite_score=None,
        verification=(),
        selected=None,
        result=result,
    )
