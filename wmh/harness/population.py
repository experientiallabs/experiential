"""Fixed-iteration optimization over complete scored harness candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from wmh.harness.project_proposer import (
    CandidateProposal,
    CandidateProposalError,
    CandidateProposer,
    EvaluatedCandidate,
)
from wmh.harness.runtime import HarnessSearchCancelled
from wmh.harness.scoring import HarnessScorer, ScoreRequest, score_harness
from wmh.harness.source_tree import HarnessSourceTree


@dataclass(frozen=True)
class PopulationIteration:
    """One consumed proposal slot and its scored or invalid outcome."""

    index: int
    proposal: CandidateProposal | None = None
    evaluation: EvaluatedCandidate | None = None
    error: CandidateProposalError | None = None

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("population iteration index must be positive")
        if self.error is not None:
            if self.proposal is not None or self.evaluation is not None:
                raise ValueError("an invalid population iteration cannot contain an evaluation")
            return
        if self.proposal is None or self.evaluation is None:
            raise ValueError("a valid population iteration needs its proposal and evaluation")
        if self.proposal.candidate_id != self.evaluation.candidate_id:
            raise ValueError("population iteration proposal and evaluation identities differ")

    @property
    def candidate_id(self) -> str:
        """Return the proposed identity for either a valid or invalid iteration."""
        if self.error is not None:
            return self.error.candidate_id
        assert self.proposal is not None
        return self.proposal.candidate_id


@dataclass(frozen=True)
class PopulationOptimizationResult:
    """The full evaluated population, every consumed slot, and its score winner."""

    population: tuple[EvaluatedCandidate, ...]
    iterations: tuple[PopulationIteration, ...]
    best: EvaluatedCandidate

    @property
    def best_score(self) -> float:
        """Return the winner's primary score."""
        return self.best.score.report.score


class HarnessPopulationOptimizer:
    """Score one seed and a fixed number of singular complete-source proposals."""

    def __init__(self, proposer: CandidateProposer, scorer: HarnessScorer) -> None:
        self._proposer = proposer
        self._scorer = scorer

    def optimize(
        self,
        *,
        seed: HarnessSourceTree,
        request: ScoreRequest,
        iterations: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> PopulationOptimizationResult:
        """Evaluate an append-only population and select by primary score only.

        A :class:`CandidateProposalError` consumes its configured slot and leaves the evaluated
        population unchanged. Scorer and infrastructure errors propagate because the optimizer
        cannot safely reinterpret an incomplete evaluation as a candidate score.
        """
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        _check_cancelled(should_cancel)
        seed_id = "candidate-0000"
        seed_candidate = seed.to_doc(seed_id)
        seed_score = score_harness(self._scorer, seed_candidate, request=request)
        population = [
            EvaluatedCandidate(
                candidate_id=seed_id,
                source=seed,
                score=seed_score,
            )
        ]
        outcomes: list[PopulationIteration] = []

        for index in range(1, iterations + 1):
            _check_cancelled(should_cancel)
            try:
                proposal = self._proposer.propose(
                    tuple(population),
                    should_cancel=should_cancel,
                )
            except CandidateProposalError as error:
                outcomes.append(PopulationIteration(index=index, error=error))
                continue

            known_ids = {item.candidate_id for item in population}
            if proposal.candidate_id in known_ids:
                raise ValueError(
                    f"proposer reused evaluated candidate_id {proposal.candidate_id!r}"
                )
            if proposal.candidate.name != proposal.candidate_id:
                raise ValueError("proposal document name does not match its candidate_id")
            parsed = proposal.source.to_doc(proposal.candidate_id)
            if parsed.doc_hash != proposal.candidate.doc_hash:
                raise ValueError("proposal source does not match its candidate document")
            _check_cancelled(should_cancel)
            score = score_harness(self._scorer, parsed, request=request)
            evaluated = EvaluatedCandidate(
                candidate_id=proposal.candidate_id,
                source=proposal.source,
                score=score,
            )
            population.append(evaluated)
            outcomes.append(
                PopulationIteration(
                    index=index,
                    proposal=proposal,
                    evaluation=evaluated,
                )
            )

        _check_cancelled(should_cancel)
        best = max(population, key=lambda item: item.score.report.score)
        return PopulationOptimizationResult(
            population=tuple(population),
            iterations=tuple(outcomes),
            best=best,
        )


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness search cancelled")
