"""Tests for the durable population loop: slot semantics, resume, and selection."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from wmh.harness.doc import HarnessDoc
from wmh.harness.population import (
    CandidateProposal,
    CandidateProposalError,
    EvaluatedCandidate,
    PopulationRunState,
    candidate_slot_id,
    optimize,
)
from wmh.harness.scoring import ScoreCell, ScoreReport, ScoreRequest
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree

_REQUEST = ScoreRequest(task_ids=("t1", "t2"), attempts=1)


def _tree(prompt: str) -> HarnessSourceTree:
    return HarnessSourceTree(files=(HarnessSourceFile(path="SYSTEM.md", content=prompt),))


class _FakeScorer:
    """Scores a doc by its system prompt via a fixed per-task reward table."""

    def __init__(
        self,
        rewards: dict[str, tuple[float, ...]],
        request: ScoreRequest = _REQUEST,
    ) -> None:
        self.rewards = rewards
        self._request = request
        self.scored: list[str] = []

    @property
    def request(self) -> ScoreRequest:
        return self._request

    def score(
        self,
        doc: HarnessDoc,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScoreReport:
        del should_cancel
        self.scored.append(doc.system_prompt())
        cells = tuple(
            ScoreCell(task_id=task_id, attempt=1, reward=reward, passed=reward > 0)
            for task_id, reward in zip(
                self._request.task_ids, self.rewards[doc.system_prompt()], strict=True
            )
        )
        return ScoreReport(
            doc_hash=doc.doc_hash,
            request=self._request,
            reward_mode="positive-binary",
            cells=cells,
        )


class _ScriptedProposer:
    """Pops one scripted item per slot: a source tree, "invalid", or an exception to raise."""

    def __init__(self, script: list[object]) -> None:
        self.script = script
        self.slots: list[int] = []

    def propose(
        self,
        population: Sequence[EvaluatedCandidate],
        *,
        slot: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal:
        del population, should_cancel
        self.slots.append(slot)
        item = self.script.pop(0)
        candidate_id = candidate_slot_id(slot)
        if item == "invalid":
            raise CandidateProposalError(candidate_id, "scripted invalid", evidence_dir="x")
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, HarnessSourceTree)
        return CandidateProposal(
            candidate_id=candidate_id, source=item, candidate=item.to_doc(candidate_id)
        )


def test_seed_and_proposals_scored_with_earliest_tie_selection(tmp_path: Path) -> None:
    scorer = _FakeScorer({"seed": (1.0, 0.0), "better": (1.0, 1.0), "tie": (1.0, 1.0)})
    proposer = _ScriptedProposer([_tree("better"), _tree("tie")])

    result = optimize(_tree("seed"), scorer, proposer, 2, run_dir=tmp_path)

    assert result.completed is True
    assert [o.candidate_id for o in result.outcomes] == [
        "candidate-0000",
        "candidate-0001",
        "candidate-0002",
    ]
    assert proposer.slots == [1, 2]
    # candidate-0001 and candidate-0002 both score 1.0: the EARLIEST wins the tie.
    assert result.best.candidate_id == "candidate-0001"
    assert result.best_score == 1.0


def test_invalid_proposal_consumes_slot_and_records_evidence(tmp_path: Path) -> None:
    scorer = _FakeScorer({"seed": (1.0, 0.0), "fix": (0.0, 1.0)})
    proposer = _ScriptedProposer(["invalid", _tree("fix")])

    result = optimize(_tree("seed"), scorer, proposer, 2, run_dir=tmp_path)

    assert result.completed is True
    assert result.outcomes[1].evaluated is None
    assert result.outcomes[1].reason == "scripted invalid"
    assert len(result.population) == 2  # the invalid slot never joins the population
    error = json.loads((tmp_path / "candidates" / "candidate-0001" / "error.json").read_text())
    assert error["reason"] == "scripted invalid"
    assert error["evidence_dir"] == "x"
    # seed (0.5) ties candidate-0002 (0.5): earliest wins, so the seed stays champion.
    assert result.best.candidate_id == "candidate-0000"


def test_infrastructure_errors_propagate_and_leave_committed_boundaries(tmp_path: Path) -> None:
    scorer = _FakeScorer({"seed": (1.0, 1.0)})
    proposer = _ScriptedProposer([RuntimeError("sandbox exploded")])

    with pytest.raises(RuntimeError, match="sandbox exploded"):
        optimize(_tree("seed"), scorer, proposer, 1, run_dir=tmp_path)

    committed = PopulationRunState(tmp_path).load()
    assert [o.candidate_id for o in committed] == ["candidate-0000"]


def test_resume_continues_at_first_missing_slot_without_rescoring(tmp_path: Path) -> None:
    scorer = _FakeScorer({"seed": (1.0, 0.0), "v2": (1.0, 1.0)})
    first = optimize(
        _tree("seed"), scorer, _ScriptedProposer([]), 1, run_dir=tmp_path, max_new_boundaries=1
    )
    assert first.completed is False
    assert len(first.outcomes) == 1

    proposer = _ScriptedProposer([_tree("v2")])
    second = optimize(_tree("seed"), scorer, proposer, 1, run_dir=tmp_path)

    assert second.completed is True
    assert proposer.slots == [1]
    assert scorer.scored == ["seed", "v2"]  # the committed seed boundary was never re-paid
    assert second.best.candidate_id == "candidate-0001"


def test_resume_rejects_a_different_seed_or_request(tmp_path: Path) -> None:
    scorer = _FakeScorer({"seed": (1.0, 0.0)})
    optimize(
        _tree("seed"), scorer, _ScriptedProposer([]), 1, run_dir=tmp_path, max_new_boundaries=1
    )

    with pytest.raises(ValueError, match="different seed"):
        optimize(_tree("other"), scorer, _ScriptedProposer([]), 1, run_dir=tmp_path)
    changed = _FakeScorer({"seed": (1.0,)}, request=ScoreRequest(task_ids=("t1",), attempts=1))
    with pytest.raises(ValueError, match="different task-by-attempt request"):
        optimize(_tree("seed"), changed, _ScriptedProposer([]), 1, run_dir=tmp_path)


def test_resume_rejects_more_recorded_slots_than_requested_iterations(tmp_path: Path) -> None:
    scorer = _FakeScorer({"seed": (1.0, 0.0)})
    proposer = _ScriptedProposer(["invalid", "invalid"])
    optimize(_tree("seed"), scorer, proposer, 2, run_dir=tmp_path)

    with pytest.raises(ValueError, match="rerun with the recorded iterations"):
        optimize(_tree("seed"), scorer, _ScriptedProposer([]), 1, run_dir=tmp_path)


def test_proposer_returning_the_wrong_slot_identity_is_a_bug(tmp_path: Path) -> None:
    class _WrongIdProposer:
        def propose(
            self,
            population: Sequence[EvaluatedCandidate],
            *,
            slot: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> CandidateProposal:
            del population, slot, should_cancel
            tree = _tree("v2")
            return CandidateProposal(
                candidate_id="candidate-0009", source=tree, candidate=tree.to_doc("candidate-0009")
            )

    scorer = _FakeScorer({"seed": (1.0, 0.0), "v2": (1.0, 1.0)})
    with pytest.raises(ValueError, match="expected 'candidate-0001'"):
        optimize(_tree("seed"), scorer, _WrongIdProposer(), 1, run_dir=tmp_path)
