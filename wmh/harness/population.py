"""Fixed-iteration optimization over complete scored harness candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from wmh.harness.project_proposer import (
    CandidateProposal,
    CandidateProposalError,
    CandidateProposer,
    EvaluatedCandidate,
    ResumableCandidateProposer,
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

    def __post_init__(self) -> None:
        if not self.population:
            raise ValueError("population must include its evaluated seed")
        seed = self.population[0]
        if seed.candidate_id != "candidate-0000":
            raise ValueError("population seed must use candidate_id 'candidate-0000'")

        expected_population = [seed]
        request = seed.score.report.request
        for expected_index, iteration in enumerate(self.iterations, start=1):
            if iteration.index != expected_index:
                raise ValueError("population iteration indices must be contiguous and one-based")
            expected_id = f"candidate-{expected_index:04d}"
            if iteration.candidate_id != expected_id:
                raise ValueError(
                    "population iteration candidate_id does not match its consumed slot"
                )
            if iteration.error is not None:
                continue
            assert iteration.proposal is not None
            assert iteration.evaluation is not None
            if iteration.proposal.candidate.name != iteration.proposal.candidate_id:
                raise ValueError("proposal document name does not match its candidate_id")
            parsed = iteration.proposal.source.to_doc(iteration.proposal.candidate_id)
            if parsed.doc_hash != iteration.proposal.candidate.doc_hash:
                raise ValueError("proposal source does not match its candidate document")
            if iteration.proposal.source != iteration.evaluation.source:
                raise ValueError("proposal source does not match its evaluation source")
            if iteration.proposal.candidate.doc_hash != iteration.evaluation.candidate.doc_hash:
                raise ValueError("proposal document does not match its evaluation document")
            if iteration.evaluation.score.report.request != request:
                raise ValueError("all population evaluations must use the same score request")
            expected_population.append(iteration.evaluation)

        if tuple(map(_evaluation_identity, expected_population)) != tuple(
            map(_evaluation_identity, self.population)
        ):
            raise ValueError(
                "population must contain the seed followed by each valid evaluation in order"
            )
        candidate_ids = [candidate.candidate_id for candidate in self.population]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("population candidate_id values must be unique")
        expected_best = max(self.population, key=lambda item: item.score.report.score)
        if _evaluation_identity(self.best) != _evaluation_identity(expected_best):
            raise ValueError(
                "population best must be the earliest candidate with the maximum score"
            )

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
        resume: PopulationOptimizationResult | None = None,
        before_step: Callable[[int], None] | None = None,
        on_boundary: Callable[[PopulationOptimizationResult], None] | None = None,
    ) -> PopulationOptimizationResult:
        """Evaluate an append-only population and select by primary score only.

        A :class:`CandidateProposalError` consumes its configured slot and leaves the evaluated
        population unchanged. Scorer and infrastructure errors propagate because the optimizer
        cannot safely reinterpret an incomplete evaluation as a candidate score.
        """
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        _check_cancelled(should_cancel)
        if resume is None:
            if before_step is not None:
                before_step(0)
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
            result = _result(population, outcomes)
            if on_boundary is not None:
                on_boundary(result)
        else:
            _validate_resume(resume, seed=seed, request=request, iterations=iterations)
            population = list(resume.population)
            outcomes = list(resume.iterations)
            if len(outcomes) < iterations:
                if not isinstance(self._proposer, ResumableCandidateProposer):
                    raise TypeError("candidate proposer does not support resume")
                self._proposer.restore(
                    tuple(population),
                    tuple(_turn(iteration) for iteration in outcomes),
                )
                _check_cancelled(should_cancel)

        for index in range(len(outcomes) + 1, iterations + 1):
            _check_cancelled(should_cancel)
            if before_step is not None:
                before_step(index)
            try:
                proposal = self._proposer.propose(
                    tuple(population),
                    should_cancel=should_cancel,
                )
            except CandidateProposalError as error:
                outcomes.append(PopulationIteration(index=index, error=error))
                result = _result(population, outcomes)
                if on_boundary is not None:
                    on_boundary(result)
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
            result = _result(population, outcomes)
            if on_boundary is not None:
                on_boundary(result)

        _check_cancelled(should_cancel)
        return _result(population, outcomes)


def _result(
    population: list[EvaluatedCandidate],
    outcomes: list[PopulationIteration],
) -> PopulationOptimizationResult:
    best = max(population, key=lambda item: item.score.report.score)
    return PopulationOptimizationResult(
        population=tuple(population),
        iterations=tuple(outcomes),
        best=best,
    )


def _evaluation_identity(candidate: EvaluatedCandidate) -> tuple[str, str, str]:
    return (
        candidate.candidate_id,
        candidate.source.tree_hash,
        candidate.score.report.report_hash,
    )


def _turn(iteration: PopulationIteration) -> CandidateProposal | CandidateProposalError:
    if iteration.error is not None:
        return iteration.error
    assert iteration.proposal is not None
    return iteration.proposal


def _validate_resume(
    resume: PopulationOptimizationResult,
    *,
    seed: HarnessSourceTree,
    request: ScoreRequest,
    iterations: int,
) -> None:
    if len(resume.iterations) > iterations:
        raise ValueError("resume has more consumed slots than the requested iterations")
    evaluated_seed = resume.population[0]
    if evaluated_seed.source != seed:
        raise ValueError("resume seed source does not match the requested seed")
    if evaluated_seed.score.report.request != request:
        raise ValueError("resume score request does not match the requested score request")


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness search cancelled")
