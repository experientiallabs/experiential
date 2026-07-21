"""Behavioral tests for fixed-iteration harness population optimization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest

from wmh.agents.default import default_agent
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import SessionEvent
from wmh.harness.population import HarnessPopulationOptimizer, PopulationOptimizationResult
from wmh.harness.project_proposer import (
    CandidateProposal,
    CandidateProposalError,
    EvaluatedCandidate,
)
from wmh.harness.runtime import HarnessSearchCancelled, TokenUsage
from wmh.harness.scoring import (
    EvaluationArtifact,
    HarnessScore,
    HarnessScoreReport,
    ScoreCell,
    ScoreContext,
    ScoreRequest,
)
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64
_RAW_PATH = "raw/trace.json"


class _Reader:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def read_bytes(self, path: str) -> bytes:
        assert path == _RAW_PATH
        return self.content


class _Scorer:
    def __init__(self, outcomes: Sequence[tuple[float, bool] | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[HarnessDoc, ScoreRequest]] = []

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
        self.calls.append((candidate, request))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        score, passed = outcome
        raw = f'{{"candidate":"{candidate.doc_hash}"}}\n'.encode()
        artifact = EvaluationArtifact.from_bytes(
            path=_RAW_PATH,
            content=raw,
            media_type="application/json",
        )
        cells = tuple(
            ScoreCell(
                task_id=task_id,
                attempt=attempt,
                score=score,
                passed=passed,
                summary="official result",
                artifact_paths=(_RAW_PATH,),
            )
            for task_id in request.task_ids
            for attempt in range(1, request.attempts + 1)
        )
        report = HarnessScoreReport(
            source_run_id=f"run-{len(self.calls)}",
            candidate_doc_hash=candidate.doc_hash,
            request=request,
            cells=cells,
            artifacts=(artifact,),
        )
        return HarnessScore(report=report, artifacts=_Reader(raw))


class _Proposer:
    def __init__(self, outcomes: Sequence[CandidateProposal | CandidateProposalError]) -> None:
        self.outcomes = list(outcomes)
        self.histories: list[tuple[EvaluatedCandidate, ...]] = []
        self.cancel_callbacks: list[Callable[[], bool] | None] = []

    def propose(
        self,
        history: Sequence[EvaluatedCandidate],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal:
        self.histories.append(tuple(history))
        self.cancel_callbacks.append(should_cancel)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, CandidateProposalError):
            raise outcome
        return outcome


class _ResumableProposer(_Proposer):
    def __init__(self, outcomes: Sequence[CandidateProposal | CandidateProposalError]) -> None:
        super().__init__(outcomes)
        self.restore_calls: list[
            tuple[
                tuple[EvaluatedCandidate, ...],
                tuple[CandidateProposal | CandidateProposalError, ...],
            ]
        ] = []

    def restore(
        self,
        history: Sequence[EvaluatedCandidate],
        turns: Sequence[CandidateProposal | CandidateProposalError],
    ) -> None:
        self.restore_calls.append((tuple(history), tuple(turns)))


def _request() -> ScoreRequest:
    return ScoreRequest(
        context=ScoreContext(
            task_set_digest=_DIGEST_A,
            evaluator_digest=_DIGEST_B,
            execution_config_digest=_DIGEST_C,
        ),
        task_ids=("task-a", "task-b"),
        attempts=1,
    )


def _source(prompt: str) -> HarnessSourceTree:
    return HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content=prompt),
            HarnessSourceFile(
                path="config.toml",
                content=('[harness]\ntools = ["bash", "submit"]\nruntime_kind = "pi-node"\n'),
            ),
        )
    )


def _proposal(candidate_id: str, prompt: str) -> CandidateProposal:
    source = _source(prompt)
    return CandidateProposal(
        candidate_id=candidate_id,
        source=source,
        candidate=source.to_doc(candidate_id),
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(input_tokens=10, output_tokens=5, calls=1),
    )


def test_optimizer_scores_seed_then_each_valid_candidate_on_one_request() -> None:
    invalid = CandidateProposalError(
        "candidate-0002",
        "source tree is incomplete",
        events=(SessionEvent(kind="error", payload={"message": "incomplete"}),),
        worker_usage=TokenUsage(input_tokens=4, output_tokens=2, calls=1),
    )
    proposer = _Proposer(
        [
            _proposal("candidate-0001", "first"),
            invalid,
            _proposal("candidate-0003", "third"),
        ]
    )
    scorer = _Scorer([(0.2, False), (0.8, True), (0.4, False)])
    request = _request()

    result = HarnessPopulationOptimizer(proposer, scorer).optimize(
        seed=_source("seed"),
        request=request,
        iterations=3,
    )

    assert len(proposer.histories) == 3
    assert [[item.candidate_id for item in history] for history in proposer.histories] == [
        ["candidate-0000"],
        ["candidate-0000", "candidate-0001"],
        ["candidate-0000", "candidate-0001"],
    ]
    assert [candidate.name for candidate, _request_arg in scorer.calls] == [
        "candidate-0000",
        "candidate-0001",
        "candidate-0003",
    ]
    assert all(request_arg == request for _candidate, request_arg in scorer.calls)
    assert [item.candidate_id for item in result.population] == [
        "candidate-0000",
        "candidate-0001",
        "candidate-0003",
    ]
    assert len(result.iterations) == 3
    assert result.iterations[0].evaluation is result.population[1]
    assert result.iterations[1].error is invalid
    assert result.iterations[1].evaluation is None
    assert result.iterations[2].evaluation is result.population[2]
    assert result.best.candidate_id == "candidate-0001"
    assert result.best_score == pytest.approx(0.8)


def test_optimizer_uses_primary_score_only_and_keeps_earliest_tie() -> None:
    proposer = _Proposer(
        [
            _proposal("candidate-0001", "first"),
            _proposal("candidate-0002", "second"),
        ]
    )
    scorer = _Scorer([(0.1, False), (0.8, False), (0.8, True)])

    result = HarnessPopulationOptimizer(proposer, scorer).optimize(
        seed=_source("seed"),
        request=_request(),
        iterations=2,
    )

    assert result.population[1].score.report.pass_rate == 0.0
    assert result.population[2].score.report.pass_rate == 1.0
    assert result.best.candidate_id == "candidate-0001"


def test_complete_duplicate_candidate_is_still_scored_and_retained() -> None:
    seed = _source("same")
    duplicate = CandidateProposal(
        candidate_id="candidate-0001",
        source=seed,
        candidate=seed.to_doc("candidate-0001"),
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(calls=1),
    )
    proposer = _Proposer([duplicate])
    scorer = _Scorer([(0.3, False), (0.6, True)])

    result = HarnessPopulationOptimizer(proposer, scorer).optimize(
        seed=seed,
        request=_request(),
        iterations=1,
    )

    assert len(scorer.calls) == 2
    assert scorer.calls[0][0].doc_hash == scorer.calls[1][0].doc_hash
    assert [item.candidate_id for item in result.population] == [
        "candidate-0000",
        "candidate-0001",
    ]
    assert result.best.candidate_id == "candidate-0001"


def test_scorer_error_aborts_without_another_proposal_or_fabricated_score() -> None:
    proposer = _Proposer(
        [
            _proposal("candidate-0001", "first"),
            _proposal("candidate-0002", "unreached"),
        ]
    )
    scorer = _Scorer([(0.2, False), RuntimeError("evaluator unavailable")])

    with pytest.raises(RuntimeError, match="evaluator unavailable"):
        HarnessPopulationOptimizer(proposer, scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=2,
        )

    assert len(proposer.histories) == 1
    assert len(scorer.calls) == 2


def test_optimizer_rejects_proposal_identity_drift_before_scoring() -> None:
    source = _source("source")
    mismatched = CandidateProposal(
        candidate_id="candidate-0001",
        source=source,
        candidate=_source("different").to_doc("candidate-0001"),
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(calls=1),
    )
    proposer = _Proposer([mismatched])
    scorer = _Scorer([(0.2, False)])

    with pytest.raises(ValueError, match="source does not match"):
        HarnessPopulationOptimizer(proposer, scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=1,
        )

    assert len(scorer.calls) == 1


def test_optimizer_rejects_proposal_document_name_drift_before_scoring() -> None:
    source = _source("source")
    mismatched = CandidateProposal(
        candidate_id="candidate-0001",
        source=source,
        candidate=source.to_doc("different-name"),
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(calls=1),
    )
    proposer = _Proposer([mismatched])
    scorer = _Scorer([(0.2, False)])

    with pytest.raises(ValueError, match="name does not match"):
        HarnessPopulationOptimizer(proposer, scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=1,
        )

    assert len(scorer.calls) == 1


def test_optimizer_rejects_reused_candidate_id_before_scoring() -> None:
    proposer = _Proposer([_proposal("candidate-0000", "new")])
    scorer = _Scorer([(0.2, False)])

    with pytest.raises(ValueError, match="candidate_id"):
        HarnessPopulationOptimizer(proposer, scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=1,
        )

    assert len(scorer.calls) == 1


def test_optimizer_propagates_non_candidate_proposer_errors() -> None:
    class BrokenProposer:
        def propose(
            self,
            history: Sequence[EvaluatedCandidate],
            *,
            should_cancel: Callable[[], bool] | None = None,
        ) -> CandidateProposal:
            del history, should_cancel
            raise RuntimeError("provider unavailable")

    scorer = _Scorer([(0.2, False)])

    with pytest.raises(RuntimeError, match="provider unavailable"):
        HarnessPopulationOptimizer(BrokenProposer(), scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=1,
        )

    assert len(scorer.calls) == 1


def test_optimizer_honors_cancellation_after_proposal_before_scoring() -> None:
    checks = iter([False, False, True])
    proposer = _Proposer([_proposal("candidate-0001", "first")])
    scorer = _Scorer([(0.2, False)])

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        HarnessPopulationOptimizer(proposer, scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=1,
            should_cancel=lambda: next(checks),
        )

    assert len(proposer.histories) == 1
    assert len(scorer.calls) == 1


def test_optimizer_preserves_an_explicit_complete_default_pi_seed() -> None:
    seed_doc = default_agent()
    proposer = _Proposer([CandidateProposalError("candidate-0001", "invalid")])
    scorer = _Scorer([(0.2, False)])

    result = HarnessPopulationOptimizer(proposer, scorer).optimize(
        seed=HarnessSourceTree.from_doc(seed_doc),
        request=_request(),
        iterations=1,
    )

    assert result.population[0].candidate.doc_hash == seed_doc.doc_hash
    assert len(result.population[0].candidate.code_files()) == len(seed_doc.code_files())


@pytest.mark.parametrize("iterations", [0, -1, True])
def test_optimizer_requires_a_positive_fixed_iteration_count(iterations: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        HarnessPopulationOptimizer(_Proposer([]), _Scorer([])).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=iterations,
        )


def test_optimizer_honors_cancellation_before_scoring_seed() -> None:
    proposer = _Proposer([])
    scorer = _Scorer([])

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        HarnessPopulationOptimizer(proposer, scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=1,
            should_cancel=lambda: True,
        )

    assert scorer.calls == []
    assert proposer.histories == []


def test_optimizer_commits_seed_and_each_consumed_slot_at_pre_spend_boundaries() -> None:
    proposer = _Proposer(
        [
            _proposal("candidate-0001", "first"),
            CandidateProposalError("candidate-0002", "invalid"),
        ]
    )
    scorer = _Scorer([(0.2, False), (0.8, True)])
    started: list[int] = []
    committed: list[PopulationOptimizationResult] = []

    result = HarnessPopulationOptimizer(proposer, scorer).optimize(
        seed=_source("seed"),
        request=_request(),
        iterations=2,
        before_step=started.append,
        on_boundary=committed.append,
    )

    assert started == [0, 1, 2]
    assert [len(boundary.iterations) for boundary in committed] == [0, 1, 2]
    assert [len(boundary.population) for boundary in committed] == [1, 2, 2]
    assert committed[-1] == result


def test_optimizer_resumes_without_replaying_seed_or_consumed_slots() -> None:
    first_proposer = _Proposer(
        [
            _proposal("candidate-0001", "first"),
            CandidateProposalError("candidate-0002", "invalid"),
        ]
    )
    partial = HarnessPopulationOptimizer(
        first_proposer,
        _Scorer([(0.2, False), (0.8, True)]),
    ).optimize(
        seed=_source("seed"),
        request=_request(),
        iterations=2,
    )
    resumed_proposer = _ResumableProposer([_proposal("candidate-0003", "third")])
    resumed_scorer = _Scorer([(0.6, True)])
    started: list[int] = []
    committed: list[PopulationOptimizationResult] = []

    result = HarnessPopulationOptimizer(resumed_proposer, resumed_scorer).optimize(
        seed=_source("seed"),
        request=_request(),
        iterations=3,
        resume=partial,
        before_step=started.append,
        on_boundary=committed.append,
    )

    [restore] = resumed_proposer.restore_calls
    assert [candidate.candidate_id for candidate in restore[0]] == [
        "candidate-0000",
        "candidate-0001",
    ]
    assert [turn.candidate_id for turn in restore[1]] == [
        "candidate-0001",
        "candidate-0002",
    ]
    assert started == [3]
    assert len(resumed_scorer.calls) == 1
    assert [item.candidate_id for item in result.population] == [
        "candidate-0000",
        "candidate-0001",
        "candidate-0003",
    ]
    assert [len(boundary.iterations) for boundary in committed] == [3]


def test_optimizer_rejects_resume_with_changed_seed_or_request_before_spend() -> None:
    partial = HarnessPopulationOptimizer(
        _Proposer([_proposal("candidate-0001", "first")]),
        _Scorer([(0.2, False), (0.8, True)]),
    ).optimize(seed=_source("seed"), request=_request(), iterations=1)

    for seed, request, match in (
        (_source("changed"), _request(), "seed"),
        (
            _source("seed"),
            _request().model_copy(
                update={
                    "context": _request().context.model_copy(
                        update={"evaluator_digest": "sha256:" + "d" * 64}
                    )
                }
            ),
            "request",
        ),
    ):
        proposer = _ResumableProposer([])
        scorer = _Scorer([])
        with pytest.raises(ValueError, match=match):
            HarnessPopulationOptimizer(proposer, scorer).optimize(
                seed=seed,
                request=request,
                iterations=2,
                resume=partial,
            )
        assert proposer.restore_calls == []
        assert scorer.calls == []


def test_optimizer_requires_restore_capability_before_resuming() -> None:
    partial = HarnessPopulationOptimizer(
        _Proposer([_proposal("candidate-0001", "first")]),
        _Scorer([(0.2, False), (0.8, True)]),
    ).optimize(seed=_source("seed"), request=_request(), iterations=1)
    scorer = _Scorer([])

    with pytest.raises(TypeError, match="does not support resume"):
        HarnessPopulationOptimizer(_Proposer([]), scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=2,
            resume=partial,
        )

    assert scorer.calls == []


def test_optimizer_rejects_resume_with_too_many_consumed_slots_before_spend() -> None:
    partial = HarnessPopulationOptimizer(
        _Proposer(
            [
                _proposal("candidate-0001", "first"),
                CandidateProposalError("candidate-0002", "invalid"),
            ]
        ),
        _Scorer([(0.2, False), (0.8, True)]),
    ).optimize(seed=_source("seed"), request=_request(), iterations=2)
    proposer = _ResumableProposer([])
    scorer = _Scorer([])

    with pytest.raises(ValueError, match="more consumed slots"):
        HarnessPopulationOptimizer(proposer, scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=1,
            resume=partial,
        )

    assert proposer.restore_calls == []
    assert scorer.calls == []


def test_population_result_rejects_malformed_checkpoint_structure() -> None:
    result = HarnessPopulationOptimizer(
        _Proposer([_proposal("candidate-0001", "first")]),
        _Scorer([(0.2, False), (0.8, True)]),
    ).optimize(seed=_source("seed"), request=_request(), iterations=1)
    [iteration] = result.iterations
    assert iteration.proposal is not None

    with pytest.raises(ValueError, match="contiguous"):
        replace(result, iterations=(replace(iteration, index=2),))

    renamed = replace(
        iteration.proposal,
        candidate=iteration.proposal.source.to_doc("renamed"),
    )
    with pytest.raises(ValueError, match="document name"):
        replace(result, iterations=(replace(iteration, proposal=renamed),))

    with pytest.raises(ValueError, match="earliest candidate"):
        replace(result, best=result.population[0])


def test_boundary_failure_stops_before_the_next_spend_step() -> None:
    proposer = _Proposer([_proposal("candidate-0001", "unreached")])
    scorer = _Scorer([(0.2, False)])

    def fail_boundary(_result: PopulationOptimizationResult) -> None:
        raise OSError("checkpoint unavailable")

    with pytest.raises(OSError, match="checkpoint unavailable"):
        HarnessPopulationOptimizer(proposer, scorer).optimize(
            seed=_source("seed"),
            request=_request(),
            iterations=1,
            on_boundary=fail_boundary,
        )

    assert len(scorer.calls) == 1
    assert proposer.histories == []
