"""Sequential population optimization of complete harness source trees.

`optimize` is the paper's outer loop: score the fixed seed once, then consume a fixed number of
sequential proposal slots. Each slot asks the proposer for one complete candidate source tree.
An invalid proposal (`CandidateProposalError`) consumes its slot as recorded evidence and the
loop continues; scorer and infrastructure exceptions PROPAGATE, because a missing evaluation can
never be reinterpreted as a reward. Selection is the maximum mean per-task score with the
earliest candidate winning ties.

Durable state is the run directory. After every boundary (the scored seed, one scored proposal,
or one consumed invalid slot) the outcome's evidence lands under `candidates/candidate-NNNN/`
(`source/`, `report.json`, and `proposal.json` or `error.json`) and the ordered index is
committed by an atomic tmp+rename of `state.json`. An ACCEPTED proposal is additionally
checkpointed (`source/` + `proposal.json` + `pending.json`) BEFORE its evaluation starts, so a
crash mid-score resumes by rescoring the exact same candidate instead of paying a fresh proposer
turn whose different doc hash would also orphan the part-paid evaluator job. Resuming reloads
state and continues at the first missing slot, so an interrupted run re-pays at most one
boundary (and the harbor scorer's own trial-level resume usually far less).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from wmh.harness.doc import HarnessDoc
from wmh.harness.runtime import HarnessSearchCancelled
from wmh.harness.scoring import Scorer, ScoreReport
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree

logger = logging.getLogger(__name__)

_STATE_FILE = "state.json"
_CANDIDATES_DIR = "candidates"
_PENDING_FILE = "pending.json"


def candidate_slot_id(slot: int) -> str:
    """The fixed candidate identity for one slot (`candidate-0000` is the seed)."""
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        raise ValueError("slot must be a nonnegative integer")
    return f"candidate-{slot:04d}"


@dataclass(frozen=True)
class EvaluatedCandidate:
    """One complete source candidate paired with its immutable score report."""

    candidate_id: str
    source: HarnessSourceTree
    report: ScoreReport

    def __post_init__(self) -> None:
        if self.source.to_doc(self.candidate_id).doc_hash != self.report.doc_hash:
            raise ValueError(
                f"score report for {self.candidate_id!r} does not match its source tree"
            )

    @property
    def candidate(self) -> HarnessDoc:
        """Reparse the complete source into its validated harness document."""
        return self.source.to_doc(self.candidate_id)

    @property
    def score(self) -> float:
        """The selection objective: mean of per-task pass rates."""
        return self.report.score


@dataclass(frozen=True)
class CandidateProposal:
    """One complete, host-captured and reparsed candidate proposal."""

    candidate_id: str
    source: HarnessSourceTree
    candidate: HarnessDoc

    def __post_init__(self) -> None:
        if self.candidate.name != self.candidate_id:
            raise ValueError("proposal document name does not match its candidate_id")
        if self.source.to_doc(self.candidate_id).doc_hash != self.candidate.doc_hash:
            raise ValueError("proposal source does not match its candidate document")


class CandidateProposalError(RuntimeError):
    """One proposal turn that did not publish a valid complete candidate.

    This is a CANDIDATE outcome: the loop records it and the slot is consumed. Raw evidence
    (the request, events, raw snapshot, and error) lives under ``evidence_dir`` when set.
    """

    def __init__(self, candidate_id: str, reason: str, *, evidence_dir: str = "") -> None:
        super().__init__(f"{candidate_id}: {reason}")
        self.candidate_id = candidate_id
        self.reason = reason
        self.evidence_dir = evidence_dir


class CandidateProposer(Protocol):
    """Produce exactly one complete candidate for one slot from the evaluated population."""

    def propose(
        self,
        population: Sequence[EvaluatedCandidate],
        *,
        slot: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal: ...


@dataclass(frozen=True)
class SlotOutcome:
    """One consumed slot: either a scored candidate or a recorded invalid proposal."""

    slot: int
    candidate_id: str
    evaluated: EvaluatedCandidate | None = None
    reason: str = ""
    evidence_dir: str = ""

    def __post_init__(self) -> None:
        if self.candidate_id != candidate_slot_id(self.slot):
            raise ValueError("slot outcome candidate_id does not match its slot")
        if (self.evaluated is None) == (not self.reason):
            raise ValueError("a slot outcome is either evaluated or carries an invalid reason")


@dataclass(frozen=True)
class PopulationResult:
    """Every consumed slot, the evaluated population, and the current score winner."""

    outcomes: tuple[SlotOutcome, ...]
    population: tuple[EvaluatedCandidate, ...]
    best: EvaluatedCandidate
    completed: bool

    @property
    def best_score(self) -> float:
        return self.best.score


def write_json_atomic(path: Path, value: JsonValue) -> None:
    """Write deterministic JSON through a same-directory fsynced tmp+rename.

    The temp file is fsynced before the rename and the directory after it: without both, a
    power loss can persist the rename while the data blocks are lost, leaving a truncated or
    empty state file behind an apparently successful commit.
    """
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class PopulationRunState:
    """Durable per-boundary population state under one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.run_dir / _CANDIDATES_DIR / candidate_id

    def load(self) -> tuple[SlotOutcome, ...]:
        """Reload every committed slot outcome, re-verifying candidate evidence integrity."""
        path = self.run_dir / _STATE_FILE
        if not path.exists():
            return ()
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw.get("outcomes")
        if not isinstance(entries, list):
            raise ValueError(f"{path} does not contain an outcomes list")
        outcomes: list[SlotOutcome] = []
        for index, entry in enumerate(entries):
            candidate_id = candidate_slot_id(index)
            if not isinstance(entry, dict):
                raise ValueError(f"{path} outcome at slot {index} is not an object")
            if entry.get("slot") != index or entry.get("candidate_id") != candidate_id:
                raise ValueError(f"{path} outcomes are not contiguous at slot {index}")
            if entry.get("kind") == "invalid":
                outcomes.append(
                    SlotOutcome(
                        slot=index,
                        candidate_id=candidate_id,
                        reason=str(entry.get("reason") or "invalid proposal"),
                        evidence_dir=str(entry.get("evidence_dir") or ""),
                    )
                )
                continue
            directory = self.candidate_dir(candidate_id)
            report_path = directory / "report.json"
            try:
                report = ScoreReport.model_validate_json(report_path.read_text(encoding="utf-8"))
                source = _read_source_tree(directory / "source")
                evaluated = EvaluatedCandidate(candidate_id, source, report)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"run dir {self.run_dir} holds corrupt or missing evidence for "
                    f"{candidate_id}: {error}"
                ) from error
            outcomes.append(
                SlotOutcome(
                    slot=index,
                    candidate_id=candidate_id,
                    evaluated=evaluated,
                    evidence_dir=str(entry.get("evidence_dir") or ""),
                )
            )
        return tuple(outcomes)

    def record_pending(self, proposal: CandidateProposal, *, evidence_dir: str = "") -> None:
        """Durably checkpoint one ACCEPTED proposal before its paid evaluation starts.

        The source tree and proposal record land first; the atomic `pending.json` marker is
        written last, so a marker's presence implies complete candidate bytes. A crash between
        this checkpoint and the boundary commit resumes by rescoring this exact candidate.
        """
        directory = self.candidate_dir(proposal.candidate_id)
        _write_candidate_source(directory / "source", proposal.source)
        write_json_atomic(
            directory / "proposal.json",
            {
                "candidate_id": proposal.candidate_id,
                "doc_hash": proposal.candidate.doc_hash,
                "tree_hash": proposal.source.tree_hash,
                "evidence_dir": evidence_dir,
            },
        )
        write_json_atomic(
            directory / _PENDING_FILE,
            {
                "candidate_id": proposal.candidate_id,
                "doc_hash": proposal.candidate.doc_hash,
                "tree_hash": proposal.source.tree_hash,
            },
        )

    def load_pending(self, slot: int) -> HarnessSourceTree | None:
        """The checkpointed-but-unscored candidate for `slot`, or None when there is none."""
        candidate_id = candidate_slot_id(slot)
        directory = self.candidate_dir(candidate_id)
        marker_path = directory / _PENDING_FILE
        if not marker_path.is_file():
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            source = _read_source_tree(directory / "source")
        except (OSError, ValueError) as error:
            raise ValueError(
                f"run dir {self.run_dir} holds a corrupt pending candidate {candidate_id}: "
                f"{error}; delete {directory} to redo the proposal"
            ) from error
        if not isinstance(marker, dict) or marker.get("tree_hash") != source.tree_hash:
            raise ValueError(
                f"pending candidate {candidate_id} in {self.run_dir} does not match its "
                f"recorded tree hash; delete {directory} to redo the proposal"
            )
        return source

    def commit(self, outcomes: Sequence[SlotOutcome]) -> None:
        """Persist the newest outcome's evidence, then atomically commit the ordered index."""
        latest = outcomes[-1]
        directory = self.candidate_dir(latest.candidate_id)
        directory.mkdir(parents=True, exist_ok=True)
        if latest.evaluated is None:
            write_json_atomic(
                directory / "error.json",
                {
                    "candidate_id": latest.candidate_id,
                    "reason": latest.reason,
                    "evidence_dir": latest.evidence_dir,
                },
            )
        else:
            _write_candidate_source(directory / "source", latest.evaluated.source)
            write_json_atomic(
                directory / "report.json", latest.evaluated.report.model_dump(mode="json")
            )
            if latest.slot > 0:
                write_json_atomic(
                    directory / "proposal.json",
                    {
                        "candidate_id": latest.candidate_id,
                        "doc_hash": latest.evaluated.report.doc_hash,
                        "tree_hash": latest.evaluated.source.tree_hash,
                        "evidence_dir": latest.evidence_dir,
                    },
                )
        write_json_atomic(
            self.run_dir / _STATE_FILE,
            {"outcomes": [_state_entry(outcome) for outcome in outcomes]},
        )
        # Cleared only AFTER the state commit: a crash in between leaves a stale marker for an
        # already-committed slot, which load_pending never consults again.
        (directory / _PENDING_FILE).unlink(missing_ok=True)


def optimize(
    seed: HarnessSourceTree,
    scorer: Scorer,
    proposer: CandidateProposer,
    iterations: int,
    *,
    run_dir: str | Path,
    should_cancel: Callable[[], bool] | None = None,
    max_new_boundaries: int | None = None,
    on_boundary: Callable[[SlotOutcome], None] | None = None,
) -> PopulationResult:
    """Score the seed and `iterations` sequential proposal slots with durable resume.

    `iterations == 0` is score-only: the seed is scored and committed and no proposal runs,
    which is how a fixed harness (a baseline or a frozen champion) is scored on a task set.
    `max_new_boundaries` stops this invocation after that many NEW boundaries (already
    committed slots do not count), leaving the fixed total plan resumable. The result's
    `completed` flag says whether every slot has been consumed.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    if max_new_boundaries is not None and (
        isinstance(max_new_boundaries, bool)
        or not isinstance(max_new_boundaries, int)
        or max_new_boundaries < 1
    ):
        raise ValueError("max_new_boundaries must be a positive integer")
    state = PopulationRunState(run_dir)
    outcomes = list(state.load())
    if len(outcomes) > iterations + 1:
        raise ValueError(
            f"run dir {state.run_dir} already holds {len(outcomes)} slot outcomes, more than "
            f"the requested {iterations} iterations allow; rerun with the recorded iterations"
        )
    _validate_resumed(outcomes, seed=seed, scorer=scorer)
    new_boundaries = 0

    def record(outcome: SlotOutcome) -> None:
        nonlocal new_boundaries
        outcomes.append(outcome)
        state.commit(outcomes)
        new_boundaries += 1
        if on_boundary is not None:
            on_boundary(outcome)

    _check_cancelled(should_cancel)
    if not outcomes:
        seed_id = candidate_slot_id(0)
        report = scorer.score(seed.to_doc(seed_id), should_cancel=should_cancel)
        record(
            SlotOutcome(
                slot=0,
                candidate_id=seed_id,
                evaluated=EvaluatedCandidate(seed_id, seed, report),
            )
        )

    while len(outcomes) <= iterations:
        if max_new_boundaries is not None and new_boundaries >= max_new_boundaries:
            break
        _check_cancelled(should_cancel)
        slot = len(outcomes)
        candidate_id = candidate_slot_id(slot)
        # A pending checkpoint means this slot's proposal was already accepted and paid for;
        # skip straight to (re)scoring it instead of buying a fresh proposer turn whose new
        # doc hash would also orphan the part-paid evaluator job.
        source = state.load_pending(slot)
        if source is None:
            population = tuple(
                outcome.evaluated for outcome in outcomes if outcome.evaluated is not None
            )
            try:
                proposal = proposer.propose(population, slot=slot, should_cancel=should_cancel)
            except CandidateProposalError as error:
                logger.info("slot %d consumed by an invalid proposal: %s", slot, error.reason)
                record(
                    SlotOutcome(
                        slot=slot,
                        candidate_id=candidate_id,
                        reason=error.reason,
                        evidence_dir=error.evidence_dir,
                    )
                )
                continue
            if proposal.candidate_id != candidate_id:
                raise ValueError(
                    f"proposer returned {proposal.candidate_id!r} for slot {slot}; "
                    f"expected {candidate_id!r}"
                )
            state.record_pending(proposal)
            source = proposal.source
        else:
            logger.info("slot %d resumes its checkpointed pending candidate", slot)
        _check_cancelled(should_cancel)
        report = scorer.score(source.to_doc(candidate_id), should_cancel=should_cancel)
        record(
            SlotOutcome(
                slot=slot,
                candidate_id=candidate_id,
                evaluated=EvaluatedCandidate(candidate_id, source, report),
            )
        )

    population = tuple(outcome.evaluated for outcome in outcomes if outcome.evaluated is not None)
    return PopulationResult(
        outcomes=tuple(outcomes),
        population=population,
        best=_earliest_best(population),
        completed=len(outcomes) == iterations + 1,
    )


def _earliest_best(population: Sequence[EvaluatedCandidate]) -> EvaluatedCandidate:
    """The maximum-score candidate; the EARLIEST one wins ties (seed beats equal successors)."""
    best = population[0]
    for candidate in population[1:]:
        if candidate.score > best.score:
            best = candidate
    return best


def _validate_resumed(
    outcomes: Sequence[SlotOutcome],
    *,
    seed: HarnessSourceTree,
    scorer: Scorer,
) -> None:
    if not outcomes:
        return
    recorded_seed = outcomes[0].evaluated
    if recorded_seed is None:
        raise ValueError("run dir state is corrupt: the seed slot is recorded as invalid")
    if recorded_seed.source.tree_hash != seed.tree_hash:
        raise ValueError(
            "run dir belongs to a different seed source tree; start a fresh run dir for a new seed"
        )
    for outcome in outcomes:
        if outcome.evaluated is not None and outcome.evaluated.report.request != scorer.request:
            raise ValueError(
                "run dir was scored under a different task-by-attempt request; "
                "rerun with the recorded tasks and attempts or start a fresh run dir"
            )


def _state_entry(outcome: SlotOutcome) -> dict[str, JsonValue]:
    if outcome.evaluated is None:
        return {
            "slot": outcome.slot,
            "candidate_id": outcome.candidate_id,
            "kind": "invalid",
            "reason": outcome.reason,
            "evidence_dir": outcome.evidence_dir,
        }
    return {
        "slot": outcome.slot,
        "candidate_id": outcome.candidate_id,
        "kind": "scored",
        "score": outcome.evaluated.score,
        "doc_hash": outcome.evaluated.report.doc_hash,
        "evidence_dir": outcome.evidence_dir,
    }


def _write_candidate_source(directory: Path, source: HarnessSourceTree) -> None:
    """Replace one candidate's source dir with exact bytes (no leftovers, no newline drift).

    Clearing first matters: a crashed prior attempt's leftover files would otherwise merge into
    a redone slot's tree and fail doc-hash re-verification on every later load. Bytes are
    written and read without newline translation so content hashes round-trip exactly.
    """
    if directory.exists():
        shutil.rmtree(directory)
    for item in source.files:
        target = directory / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.content.encode("utf-8"))


def _read_source_tree(directory: Path) -> HarnessSourceTree:
    # read_bytes + exact decode: Path.read_text's universal newlines would fold \r into \n and
    # silently change content hashes on every resume.
    files = [
        HarnessSourceFile(
            path=path.relative_to(directory).as_posix(),
            content=path.read_bytes().decode("utf-8"),
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    return HarnessSourceTree(files=tuple(files))


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness search cancelled")
