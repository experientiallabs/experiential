"""Behavioral tests for the feedback-directed one-step improvement gate."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pytest

from wmh.evals.tasks import TaskSpec
from wmh.harness.doc import HarnessDoc
from wmh.harness.improve import (
    ImproveGate,
    improve_harness,
    suite_score,
    verification_results,
)
from wmh.harness.live_session import SessionEvent
from wmh.harness.project_proposer import (
    CandidateProposal,
    CandidateProposalError,
    CandidateProposer,
    EvaluatedCandidate,
)
from wmh.harness.runtime import TokenUsage
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

# One preplanned score call: task_id -> one (score, passed) pair per attempt.
_CellPlan = Mapping[str, Sequence[tuple[float, bool]]]


class _Reader:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def read_bytes(self, path: str) -> bytes:
        assert path == _RAW_PATH
        return self.content


class _Scorer:
    """Yields one preplanned per-task cell matrix per score call (seed first)."""

    def __init__(
        self,
        task_ids: Sequence[str],
        attempts: int,
        outcomes: Sequence[_CellPlan],
    ) -> None:
        self._task_ids = tuple(task_ids)
        self._attempts = attempts
        self.outcomes = list(outcomes)
        self.calls: list[HarnessDoc] = []

    def request(self, *, attempts: int) -> ScoreRequest:
        return ScoreRequest(
            context=ScoreContext(
                task_set_digest=_DIGEST_A,
                evaluator_digest=_DIGEST_B,
                execution_config_digest=_DIGEST_C,
            ),
            task_ids=self._task_ids,
            attempts=attempts,
        )

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
        self.calls.append(candidate)
        per_task = self.outcomes.pop(0)
        raw = f'{{"candidate":"{candidate.doc_hash}"}}\n'.encode()
        artifact = EvaluationArtifact.from_bytes(
            path=_RAW_PATH,
            content=raw,
            media_type="application/json",
        )
        cells: list[ScoreCell] = []
        for task_id in request.task_ids:
            attempt_outcomes = per_task[task_id]
            assert len(attempt_outcomes) == request.attempts
            for attempt, (score, passed) in enumerate(attempt_outcomes, start=1):
                cells.append(
                    ScoreCell(
                        task_id=task_id,
                        attempt=attempt,
                        score=score,
                        passed=passed,
                        summary="planned cell",
                        artifact_paths=(_RAW_PATH,),
                    )
                )
        report = HarnessScoreReport(
            source_run_id=f"run-{len(self.calls)}",
            candidate_doc_hash=candidate.doc_hash,
            request=request,
            cells=tuple(cells),
            artifacts=(artifact,),
        )
        return HarnessScore(report=report, artifacts=_Reader(raw))


class _Proposer:
    def __init__(self, outcomes: Sequence[CandidateProposal | CandidateProposalError]) -> None:
        self.outcomes = list(outcomes)
        self.histories: list[tuple[EvaluatedCandidate, ...]] = []

    def propose(
        self,
        history: Sequence[EvaluatedCandidate],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal:
        del should_cancel
        self.histories.append(tuple(history))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, CandidateProposalError):
            raise outcome
        return outcome


def _factory(proposer: CandidateProposer) -> tuple[Callable[[str], CandidateProposer], list[str]]:
    directives: list[str] = []

    def factory(directive: str) -> CandidateProposer:
        directives.append(directive)
        return proposer

    return factory, directives


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


def _tasks(ids: Sequence[str]) -> tuple[TaskSpec, ...]:
    return tuple(
        TaskSpec(task_id=task_id, instruction=f"do {task_id}", gold=["g"]) for task_id in ids
    )


def _uniform(task_outcomes: Mapping[str, tuple[float, bool]], attempts: int) -> _CellPlan:
    return {task_id: [outcome] * attempts for task_id, outcome in task_outcomes.items()}


_SUITE = _tasks(["suite-a", "suite-b"])
_VERIFY = _tasks(["verify-1"])
_ALL_IDS = ("suite-a", "suite-b", "verify-1")


def test_gate_accepts_candidate_within_margin_with_all_verification_passing() -> None:
    scorer = _Scorer(
        _ALL_IDS,
        3,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 3),
            {
                "suite-a": [(0.75, True)] * 3,
                "suite-b": [(0.75, True)] * 3,
                "verify-1": [(1.0, True), (1.0, True), (0.0, False)],  # 2 of 3: strict majority
            },
        ],
    )
    factory, directives = _factory(_Proposer([_proposal("candidate-0001", "improved")]))

    outcome = improve_harness(
        seed=_source("seed"),
        feedback="the agent should have access to my GitHub",
        suite_tasks=_SUITE,
        verification_tasks=_VERIFY,
        scorer=scorer,
        proposer_factory=factory,
        iterations=1,
        attempts=3,
    )

    assert directives == ["the agent should have access to my GitHub"]
    assert outcome.accepted is True
    assert outcome.selected is not None
    assert outcome.selected.candidate_id == "candidate-0001"
    assert outcome.seed_suite_score == pytest.approx(0.8)
    assert outcome.candidate_suite_score == pytest.approx(0.75)
    [verification] = outcome.verification
    assert verification.task_id == "verify-1"
    assert verification.passed is True
    assert verification.pass_count == 2
    assert verification.attempts == 3
    assert "candidate-0001" in outcome.reason


def test_gate_rejects_suite_regression_beyond_margin() -> None:
    scorer = _Scorer(
        _ALL_IDS,
        3,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 3),
            # Suite mean 0.6 is below the floor 0.72 = 0.9 * 0.8 even though verification passes.
            _uniform({"suite-a": (0.6, True), "suite-b": (0.6, True), "verify-1": (1.0, True)}, 3),
        ],
    )
    factory, _directives = _factory(_Proposer([_proposal("candidate-0001", "regressed")]))

    outcome = improve_harness(
        seed=_source("seed"),
        feedback="add GitHub access",
        suite_tasks=_SUITE,
        verification_tasks=_VERIFY,
        scorer=scorer,
        proposer_factory=factory,
        iterations=1,
        attempts=3,
    )

    assert outcome.accepted is False
    assert outcome.selected is None
    assert outcome.candidate_suite_score is None
    assert outcome.verification == ()
    assert "suite regression beyond margin" in outcome.reason
    assert "margin" in outcome.reason
    assert "0.600000" in outcome.reason
    assert "0.720000" in outcome.reason


def test_gate_rejects_when_a_verification_task_fails_majority() -> None:
    scorer = _Scorer(
        _ALL_IDS,
        3,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 3),
            {
                "suite-a": [(0.9, True)] * 3,
                "suite-b": [(0.9, True)] * 3,
                "verify-1": [(1.0, True), (0.0, False), (0.0, False)],  # 1 of 3: no majority
            },
        ],
    )
    factory, _directives = _factory(_Proposer([_proposal("candidate-0001", "half done")]))

    outcome = improve_harness(
        seed=_source("seed"),
        feedback="add GitHub access",
        suite_tasks=_SUITE,
        verification_tasks=_VERIFY,
        scorer=scorer,
        proposer_factory=factory,
        iterations=1,
        attempts=3,
    )

    assert outcome.accepted is False
    assert outcome.selected is None
    assert "verification tasks failed" in outcome.reason
    assert "verify-1" in outcome.reason
    assert "1/3" in outcome.reason


def test_selection_prefers_highest_suite_score_over_optimizer_blended_best() -> None:
    scorer = _Scorer(
        _ALL_IDS,
        1,
        [
            _uniform({"suite-a": (0.5, True), "suite-b": (0.5, True), "verify-1": (0.0, False)}, 1),
            # Blended mean 0.666 but the best suite score (0.9).
            _uniform({"suite-a": (0.9, True), "suite-b": (0.9, True), "verify-1": (0.2, True)}, 1),
            # Blended mean 0.8 (the optimizer's best) but a lower suite score (0.7).
            _uniform({"suite-a": (0.7, True), "suite-b": (0.7, True), "verify-1": (1.0, True)}, 1),
        ],
    )
    factory, _directives = _factory(
        _Proposer(
            [
                _proposal("candidate-0001", "high suite"),
                _proposal("candidate-0002", "high verification"),
            ]
        )
    )

    outcome = improve_harness(
        seed=_source("seed"),
        feedback="add GitHub access",
        suite_tasks=_SUITE,
        verification_tasks=_VERIFY,
        scorer=scorer,
        proposer_factory=factory,
        iterations=2,
        attempts=1,
    )

    assert outcome.result.best.candidate_id == "candidate-0002"
    assert outcome.accepted is True
    assert outcome.selected is not None
    assert outcome.selected.candidate_id == "candidate-0001"
    assert outcome.candidate_suite_score == pytest.approx(0.9)


def test_selection_tie_on_suite_score_keeps_the_earliest_candidate() -> None:
    plan = _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (1.0, True)}, 1)
    scorer = _Scorer(
        _ALL_IDS,
        1,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 1),
            plan,
            plan,
        ],
    )
    factory, _directives = _factory(
        _Proposer(
            [
                _proposal("candidate-0001", "first tie"),
                _proposal("candidate-0002", "second tie"),
            ]
        )
    )

    outcome = improve_harness(
        seed=_source("seed"),
        feedback="add GitHub access",
        suite_tasks=_SUITE,
        verification_tasks=_VERIFY,
        scorer=scorer,
        proposer_factory=factory,
        iterations=2,
        attempts=1,
    )

    assert outcome.accepted is True
    assert outcome.selected is not None
    assert outcome.selected.candidate_id == "candidate-0001"


def test_seed_only_population_reports_no_valid_proposals() -> None:
    scorer = _Scorer(
        _ALL_IDS,
        3,
        [
            _uniform({"suite-a": (0.8, True), "suite-b": (0.8, True), "verify-1": (0.0, False)}, 3),
        ],
    )
    factory, _directives = _factory(
        _Proposer([CandidateProposalError("candidate-0001", "agent turn did not submit")])
    )

    outcome = improve_harness(
        seed=_source("seed"),
        feedback="add GitHub access",
        suite_tasks=_SUITE,
        verification_tasks=_VERIFY,
        scorer=scorer,
        proposer_factory=factory,
        iterations=1,
        attempts=3,
    )

    assert outcome.accepted is False
    assert "no valid proposals" in outcome.reason
    assert outcome.selected is None
    assert outcome.candidate_suite_score is None
    assert outcome.verification == ()
    assert [item.candidate_id for item in outcome.result.population] == ["candidate-0000"]
    assert outcome.seed_suite_score == pytest.approx(0.8)


def _report(task_ids: Sequence[str], attempts: int, plan: _CellPlan) -> HarnessScoreReport:
    scorer = _Scorer(task_ids, attempts, [plan])
    candidate = _source("unit").to_doc("candidate-0000")
    return scorer.score(candidate, request=scorer.request(attempts=attempts)).report


def test_suite_score_means_only_matching_cells_and_rejects_empty_selection() -> None:
    report = _report(
        ("suite-a", "suite-b", "verify-1"),
        2,
        {
            "suite-a": [(1.0, True), (0.0, False)],
            "suite-b": [(0.5, True), (0.5, True)],
            "verify-1": [(0.0, False), (0.0, False)],
        },
    )

    assert suite_score(report, ("suite-a", "suite-b")) == pytest.approx(0.5)
    assert suite_score(report, ("suite-a",)) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no cells"):
        suite_score(report, ("absent-task",))
    with pytest.raises(ValueError, match="no cells"):
        suite_score(report, ())


def test_verification_results_apply_the_strict_majority_rule() -> None:
    half = _report(("verify-1",), 2, {"verify-1": [(1.0, True), (0.0, False)]})
    [result] = verification_results(half, ("verify-1",))
    assert result.passed is False  # exactly half (1 of 2) is not a strict majority
    assert result.pass_count == 1
    assert result.attempts == 2

    majority = _report(
        ("verify-1",),
        3,
        {"verify-1": [(1.0, True), (1.0, True), (0.0, False)]},
    )
    [result] = verification_results(majority, ("verify-1",))
    assert result.passed is True
    assert result.pass_count == 2
    assert result.attempts == 3

    with pytest.raises(ValueError, match="absent-task"):
        verification_results(majority, ("absent-task",))


def test_improve_validates_task_sets_and_request_before_any_proposal() -> None:
    factory, directives = _factory(_Proposer([]))
    valid_scorer = _Scorer(_ALL_IDS, 1, [])

    with pytest.raises(ValueError, match="disjoint"):
        improve_harness(
            seed=_source("seed"),
            feedback="add GitHub access",
            suite_tasks=_SUITE,
            verification_tasks=_tasks(["suite-a"]),
            scorer=valid_scorer,
            proposer_factory=factory,
            attempts=1,
        )
    with pytest.raises(ValueError, match="suite_tasks must be nonempty"):
        improve_harness(
            seed=_source("seed"),
            feedback="add GitHub access",
            suite_tasks=(),
            verification_tasks=_VERIFY,
            scorer=valid_scorer,
            proposer_factory=factory,
            attempts=1,
        )
    with pytest.raises(ValueError, match="verification_tasks must be nonempty"):
        improve_harness(
            seed=_source("seed"),
            feedback="add GitHub access",
            suite_tasks=_SUITE,
            verification_tasks=(),
            scorer=valid_scorer,
            proposer_factory=factory,
            attempts=1,
        )
    with pytest.raises(ValueError, match="feedback"):
        improve_harness(
            seed=_source("seed"),
            feedback="   ",
            suite_tasks=_SUITE,
            verification_tasks=_VERIFY,
            scorer=valid_scorer,
            proposer_factory=factory,
            attempts=1,
        )

    reordered_scorer = _Scorer(("verify-1", "suite-a", "suite-b"), 1, [])
    with pytest.raises(ValueError, match="suite tasks followed by the verification tasks"):
        improve_harness(
            seed=_source("seed"),
            feedback="add GitHub access",
            suite_tasks=_SUITE,
            verification_tasks=_VERIFY,
            scorer=reordered_scorer,
            proposer_factory=factory,
            attempts=1,
        )
    assert directives == []


def test_margin_gate_validates_its_fraction() -> None:
    assert ImproveGate().suite_margin == pytest.approx(0.10)
    assert ImproveGate(suite_margin=0.0).suite_margin == 0.0
    for invalid in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            ImproveGate(suite_margin=invalid)
