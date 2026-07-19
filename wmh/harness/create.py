"""`wmh harness create`: budgeted search over harness deltas, gated by non-regression.

Each iteration freezes the current champion, clusters its failures into mechanisms, and asks the
proposer for a sibling batch of `HarnessDelta` objects against one size-weighted,
expansion-discounted cluster. Every sibling is applied and evaluated against that same frozen
champion. After the full batch resolves, at most one gate-eligible sibling becomes the next
iteration's champion:

- **Tier 1 — regression suite**: the child's score on the suite (tasks the search has already
  mastered) must not drop below the champion's. Newly-passing tasks promote into the suite on
  accept, so wins are locked in and later deltas cannot quietly trade them away.
- **Tier 2 — full split**: the child's overall success rate must be at least the best seen.
- **Tier 3 — held-out (optional)**: with a holdout task file, the child must also be no worse than
  the champion on tasks the proposer never saw evidence from.

Ties pass every gate tier: with k passes per task, scores are coarse, and "no worse" is the
eligibility contract. When multiple siblings are eligible, full success wins, then secondary
score, then lower proposal index. Every proposed delta, whether selected, rejected, or invalid
before eval, is recorded in the archive with its verdict. The run as a whole is only as
reproducible as its providers because proposals and rollouts sample real models at temperature.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import DEFAULT_K, ClosedLoopReport, evaluate_closed_loop
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta, apply_delta
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import SandboxUsage
from wmh.harness.mutate import render_task_attempt_evidence
from wmh.harness.proposer import DeltaProposer, ProposalFailure
from wmh.harness.runtime import (
    HarnessSearchCancelled,
    RuntimeCancelled,
    TokenUsage,
    combine_usage,
)
from wmh.harness.scoring import (
    MAX_TASK_DESCRIPTION_CHARS,
    MAX_TASK_EVIDENCE_CHARS,
    HarnessScorer,
    HarnessScoreReport,
    ScoreCapabilities,
    ScoreRequest,
    ScoreRunHealth,
    ScoreRunHealthError,
    TaskScore,
    cluster_score_failures,
    render_score_evidence,
    suite_score,
    suite_secondary_score,
)
from wmh.providers.base import Provider

if TYPE_CHECKING:
    # Only the annotation: pi_e2b (the optional e2b extra's consumer) is imported lazily where
    # the pool is actually constructed.
    from wmh.harness.pi_e2b import E2BSandboxPool

# Non-regression tolerance: fmean over identical verdicts must compare equal, never fail a gate
# on float noise.
_TIE_EPS = 1e-9

ALL_PASS_MECHANISM = "none: all tasks pass"

FailureClusterKey = tuple[str, str, tuple[str, ...]]

# Reports (iteration, champion name, success rate, changed); iteration 0 is the seed.
CreateProgress = Callable[[int, str, float, bool], None]


class DeltaArchive(BaseModel):
    """The full search record: a root snapshot plus every audited delta, in proposal order.

    Accepted deltas form the lineage (`parent_doc_hash -> delta -> child_doc_hash`); any doc in it
    is reconstructable by folding them from the seed. Rejected and invalid deltas are kept too —
    with their verdicts — because "which kinds of edits fail on which failure classes" is as
    queryable a question as which succeed.
    """

    seed: HarnessDoc
    deltas: list[HarnessDelta] = Field(default_factory=list)

    def accepted(self) -> list[HarnessDelta]:
        return [d for d in self.deltas if d.verdict is not None and d.verdict.accepted]

    def reconstruct(self, doc_hash: str) -> HarnessDoc:
        """Fold accepted deltas from the seed until `doc_hash` is produced."""
        docs = {self.seed.doc_hash: self.seed}
        for delta in self.accepted():
            parent = docs.get(delta.parent_doc_hash)
            if parent is not None:
                child = apply_delta(parent, delta.model_copy(deep=True), parent.name)
                docs[child.doc_hash] = child
        if doc_hash not in docs:
            raise ValueError(f"doc {doc_hash[:12]} is not in this archive's accepted lineage")
        return docs[doc_hash]


class ProposalRecord(BaseModel):
    """One proposal in an iteration's sibling batch, in stable proposal order.

    A dead proposal (every outcome but ``scored``) ends before a full-split evaluation. Scored
    proposals distinguish gate eligibility from final selection: several siblings may satisfy
    the frozen non-regression gate, but only the best eligible sibling is selected. The
    ``champion_score`` is always the frozen pre-iteration champion score, which is the comparison
    baseline and the honest plotting level for a dead proposal.
    """

    iteration: int = Field(ge=1)
    proposal_index: int = Field(ge=1)
    outcome: Literal["scored", "screened", "invalid", "unusable", "proposer_error"]
    candidate: str | None = None
    candidate_doc_hash: str | None = None
    delta_id: str | None = None
    trigger: FailureSignature | None = None
    expected_effect: str | None = None
    ops: list[str] = Field(default_factory=list)  # "replace prompt:main" style summaries
    rationales: list[str] = Field(default_factory=list)
    reason: str | None = None
    score: float | None = None  # full-suite success rate; scored proposals only
    gate_eligible: bool | None = None  # frozen non-regression gate; scored proposals only
    selected: bool = False  # true only for the iteration winner
    screen_child: float | None = None  # trigger-cluster means; screened attempts only
    screen_parent: float | None = None
    screen_child_secondary: float | None = None  # denser secondary screen signal
    screen_parent_secondary: float | None = None
    champion_score: float

    @model_validator(mode="after")
    def _validate_state(self) -> ProposalRecord:
        """Reject impossible scored, dead, and selected record combinations."""
        if self.outcome == "scored":
            if self.score is None or self.gate_eligible is None:
                raise ValueError("scored proposals require score and gate_eligible")
            if self.candidate is None or self.candidate_doc_hash is None or self.delta_id is None:
                raise ValueError("scored proposals require candidate and delta identities")
        elif self.score is not None or self.gate_eligible is not None or self.selected:
            raise ValueError("dead proposals cannot carry score, gate eligibility, or selection")
        if self.selected and not self.gate_eligible:
            raise ValueError("selected proposals must be gate eligible")
        return self


class CreateResult(BaseModel):
    """What a create run produced: the champion, its score, and the full search record."""

    best: HarnessDoc
    best_score: float
    archive: DeltaArchive
    reports: dict[str, ClosedLoopReport] = Field(default_factory=dict)  # by doc_hash
    holdout_reports: dict[str, ClosedLoopReport] = Field(default_factory=dict)  # by doc_hash
    suite: list[str] = Field(default_factory=list)  # final regression suite (task ids)
    skipped: int = 0  # proposals unusable or invalid before evaluation
    proposal_records: list[ProposalRecord] = Field(default_factory=list)
    screened: int = 0  # deltas rejected at the cheap trigger-cluster screen (no full eval spent)
    confirmations: int = 0  # narrow vetoes retried at higher k (see `narrow_failing_tiers`)
    iterations: int = 0
    proposal_batch_size: int = 1
    # Spend meters over the WHOLE search (seed, screens, full splits, holdout, confirmations).
    # worker_usage: worker-LLM tokens from self-metering runtimes (the pi worker path; None on
    # provider-wrapped runtimes, which are metered upstream). sandbox_usage: E2B sandbox count +
    # lifetime seconds (None on the local backend).
    worker_usage: TokenUsage | None = None
    sandbox_usage: SandboxUsage | None = None


class SearchResult(BaseModel):
    """Benchmark-neutral champion and audit record returned by ``search_harness``."""

    best: HarnessDoc
    best_score: float
    archive: DeltaArchive
    reports: dict[str, HarnessScoreReport] = Field(default_factory=dict)
    holdout_reports: dict[str, HarnessScoreReport] = Field(default_factory=dict)
    suite: list[str] = Field(default_factory=list)
    skipped: int = 0
    proposal_records: list[ProposalRecord] = Field(default_factory=list)
    screened: int = 0
    confirmations: int = 0
    iterations: int = 0
    proposal_batch_size: int = 1


@dataclass
class _ScoredProposal:
    """One neutral scored sibling awaiting iteration-level winner selection."""

    proposal_index: int
    child: HarnessDoc
    delta: HarnessDelta
    report: HarnessScoreReport
    gate: GateRecord
    record: ProposalRecord


def select_failure_cluster(
    clusters: list[FailureSignature],
    expansion_counts: dict[FailureClusterKey, int],
    *,
    parent_doc_hash: str,
) -> FailureSignature:
    """Choose a high-impact cluster without getting trapped on one exhausted failure.

    A cluster's priority is ``task_count / (1 + prior_iterations_on_this_parent)``. Large mechanisms
    still receive proportionally more search budget, but equally sized singleton failures rotate
    after one batch instead of a deterministic ``clusters[0]`` absorbing the entire run. The
    stable mechanism/task ordering resolves exact ties without entropy.
    """
    if not clusters:
        raise ValueError("cannot select from an empty failure-cluster list")

    def _priority(cluster: FailureSignature) -> tuple[float, str, tuple[str, ...]]:
        key = _failure_cluster_key(parent_doc_hash, cluster)
        prior_iterations = expansion_counts.get(key, 0)
        return (
            -(len(cluster.task_ids) / (1 + prior_iterations)),
            cluster.mechanism,
            tuple(cluster.task_ids),
        )

    return min(clusters, key=_priority)


def _failure_cluster_key(parent_doc_hash: str, cluster: FailureSignature) -> FailureClusterKey:
    return parent_doc_hash, cluster.mechanism, tuple(cluster.task_ids)


def narrow_failing_tiers(
    verdict: GateRecord,
    *,
    k: int,
    n_suite: int,
    n_holdout: int,
    margin_attempts: int = 2,
) -> list[str] | None:
    """Which tiers vetoed this delta narrowly enough to deserve a re-measurement.

    Eligible only when the delta strictly won the full split: the question a confirmation
    answers is "was this win vetoed by measurement noise?", not "can a loser get lucky?".
    A tier's veto is narrow when its regression is at most `margin_attempts` single-attempt
    flips wide (one flip changes a tier mean by 1/(k*n)). Returns the narrowly-failing tier
    names, or None when the delta is ineligible (no win, a wide veto, or no veto at all).
    """
    if verdict.accepted or verdict.full_delta <= _TIE_EPS:
        return None
    # A confirmation may only revisit the explicitly returned binary vetoes. Do not let it erase
    # a separate tied-success dense veto on another tier.
    if abs(verdict.suite_delta) <= _TIE_EPS and verdict.suite_secondary_delta < -_TIE_EPS:
        return None
    if (
        verdict.holdout_delta is not None
        and abs(verdict.holdout_delta) <= _TIE_EPS
        and verdict.holdout_secondary_delta is not None
        and verdict.holdout_secondary_delta < -_TIE_EPS
    ):
        return None
    tiers: list[str] = []
    if verdict.suite_delta < -_TIE_EPS:
        if n_suite == 0 or verdict.suite_delta < -(margin_attempts / (k * n_suite)) - _TIE_EPS:
            return None
        tiers.append("suite")
    if verdict.holdout_delta is not None and verdict.holdout_delta < -_TIE_EPS:
        if (
            n_holdout == 0
            or verdict.holdout_delta < -(margin_attempts / (k * n_holdout)) - _TIE_EPS
        ):
            return None
        tiers.append("holdout")
    return tiers or None


def gate_score_delta(
    delta: HarnessDelta,
    *,
    child: HarnessScoreReport,
    champion: HarnessScoreReport,
    best_full: float,
    suite: list[str],
    child_holdout: HarnessScoreReport | None = None,
    champion_holdout: HarnessScoreReport | None = None,
) -> GateRecord:
    """Apply the search's lexicographic non-regression gate to neutral scores."""
    suite_delta = suite_score(child, suite) - suite_score(champion, suite)
    suite_secondary_delta = suite_secondary_score(child, suite) - suite_secondary_score(
        champion, suite
    )
    full_delta = child.score - best_full
    full_secondary_delta = child.secondary_score - champion.secondary_score
    holdout_delta = (
        child_holdout.score - champion_holdout.score
        if child_holdout is not None and champion_holdout is not None
        else None
    )
    holdout_secondary_delta = (
        child_holdout.secondary_score - champion_holdout.secondary_score
        if child_holdout is not None and champion_holdout is not None
        else None
    )
    failures: list[str] = []
    if suite_delta < -_TIE_EPS:
        failures.append(f"suite regressed by {-suite_delta:.3f}")
    elif abs(suite_delta) <= _TIE_EPS and suite_secondary_delta < -_TIE_EPS:
        failures.append(f"suite secondary score regressed by {-suite_secondary_delta:.3f}")
    if full_delta < -_TIE_EPS:
        failures.append(f"full split {child.score:.3f} below best {best_full:.3f}")
    elif abs(full_delta) <= _TIE_EPS and full_secondary_delta < -_TIE_EPS:
        failures.append(f"full-split secondary score regressed by {-full_secondary_delta:.3f}")
    if holdout_delta is not None and holdout_delta < -_TIE_EPS:
        failures.append(f"held-out regressed by {-holdout_delta:.3f}")
    elif (
        holdout_delta is not None
        and abs(holdout_delta) <= _TIE_EPS
        and holdout_secondary_delta is not None
        and holdout_secondary_delta < -_TIE_EPS
    ):
        failures.append(f"held-out secondary score regressed by {-holdout_secondary_delta:.3f}")
    flipped = sum(
        1
        for task_id in delta.trigger.task_ids
        if (task := child.per_task.get(task_id)) is not None and task.passed
    )
    effect = (
        f"trigger cluster: {flipped}/{len(delta.trigger.task_ids)} tasks now pass"
        if delta.trigger.task_ids
        else "no trigger cluster (all-pass parent)"
    )
    accepted = not failures
    reason = ("accepted; " if accepted else "rejected: " + "; ".join(failures) + "; ") + effect
    return GateRecord(
        suite_delta=suite_delta,
        suite_secondary_delta=suite_secondary_delta,
        full_delta=full_delta,
        full_secondary_delta=full_secondary_delta,
        holdout_delta=holdout_delta,
        holdout_secondary_delta=holdout_secondary_delta,
        accepted=accepted,
        reason=reason,
    )


def _snapshot_score_report(report: HarnessScoreReport) -> HarnessScoreReport:
    """Revalidate and detach a scorer-owned report before it enters search state."""
    snapshot = HarnessScoreReport.model_validate(report.model_dump(mode="json"))
    return snapshot.model_copy(
        update={"per_task": dict(sorted(snapshot.per_task.items()))},
        deep=True,
    )


def _score_report_fingerprint(report: HarnessScoreReport) -> str:
    """Canonical content identity used to detect evaluation-id collisions."""
    content = report.model_dump_json(exclude={"evaluation_id"})
    return hashlib.blake2b(content.encode("utf-8"), digest_size=32).hexdigest()


def search_harness(
    name: str,
    seed_doc: HarnessDoc,
    scorer: HarnessScorer,
    proposer: DeltaProposer,
    *,
    iterations: int = 5,
    proposal_batch_size: int = 1,
    screen_proposals: bool = True,
    holdout_scorer: HarnessScorer | None = None,
    confirm_narrow_vetoes: bool = True,
    on_progress: CreateProgress | None = None,
    on_note: Callable[[str], None] | None = None,
    on_proposal: Callable[[ProposalRecord], None] | None = None,
    on_accept: Callable[[HarnessDoc, HarnessDelta, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> SearchResult:
    """Search over harness deltas using normalized scores from an injected evaluator.

    The seed fixes the discovery task matrix and attempt count. Every candidate must return that
    same matrix before it can enter the non-regression gate. Optional screening and confirmation
    stages are admitted only when the scorer declares the required capabilities.

    Args:
        name: Name assigned to the final champion.
        seed_doc: Initial harness document.
        scorer: Discovery evaluator and candidate eligibility boundary.
        proposer: Existing delta proposer fed bounded scorer evidence.
        iterations: Number of proposal batches. Zero performs seed qualification only.
        proposal_batch_size: Sibling proposals evaluated against each frozen champion.
        screen_proposals: Whether to prefilter each child on its trigger task subset.
        holdout_scorer: Optional independent evaluator used as a third gate tier.
        confirm_narrow_vetoes: Whether capable scorers remeasure narrow gate vetoes.

    Returns:
        The champion, normalized reports, and complete delta audit record.

    Raises:
        ValueError: If scorer capabilities, task matrices, attempts, or identities drift.
        HarnessSearchCancelled: If cancellation is requested at a search boundary.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    if proposal_batch_size < 1:
        raise ValueError(f"proposal_batch_size must be positive, got {proposal_batch_size}")
    discovery_attempts = scorer.default_attempts
    if discovery_attempts < 1:
        raise ValueError("scorer.default_attempts must be positive")
    if screen_proposals and not scorer.capabilities.task_subsets:
        raise ValueError(
            "scorer cannot evaluate task subsets; pass screen_proposals=False to avoid "
            "unsupported paid screens"
        )
    if confirm_narrow_vetoes and not scorer.capabilities.attempt_overrides:
        raise ValueError("scorer cannot override attempt counts; pass confirm_narrow_vetoes=False")
    holdout_attempts = holdout_scorer.default_attempts if holdout_scorer is not None else None
    if holdout_scorer is not None:
        assert holdout_attempts is not None
        if holdout_attempts < 1:
            raise ValueError("holdout_scorer.default_attempts must be positive")
        if confirm_narrow_vetoes and not holdout_scorer.capabilities.attempt_overrides:
            raise ValueError(
                "holdout scorer cannot override attempt counts; pass confirm_narrow_vetoes=False"
            )
        if confirm_narrow_vetoes and holdout_attempts != discovery_attempts:
            raise ValueError(
                "confirmation requires discovery and holdout scorers to use the same "
                "default_attempts"
            )
    seed_error = scorer.validate_candidate(seed_doc)
    if seed_error is not None:
        raise ValueError(f"seed is not eligible for scoring: {seed_error}")
    if holdout_scorer is not None:
        holdout_seed_error = holdout_scorer.validate_candidate(seed_doc)
        if holdout_seed_error is not None:
            raise ValueError(f"holdout seed is not eligible for scoring: {holdout_seed_error}")

    evaluations_by_id: dict[str, tuple[str, str, str]] = {}

    def _check_cancelled() -> None:
        if should_cancel is not None and should_cancel():
            raise HarnessSearchCancelled("harness search cancelled")

    def _note(message: str) -> None:
        if on_note is not None:
            on_note(message)

    def _score(
        active_scorer: HarnessScorer,
        doc: HarnessDoc,
        *,
        purpose: Literal["seed", "screen", "full", "holdout", "confirmation"],
        task_ids: list[str] | None = None,
        attempts: int | None = None,
        expected_task_ids: frozenset[str] | None = None,
    ) -> HarnessScoreReport:
        _check_cancelled()
        request = ScoreRequest(
            purpose=purpose,
            task_ids=tuple(task_ids) if task_ids is not None else None,
            attempts=attempts,
        )
        report = _snapshot_score_report(active_scorer.score(doc, request=request))
        if report.run_health is not ScoreRunHealth.VALID:
            raise ScoreRunHealthError(report.evaluation_id, report.run_health)
        scorer_attempts = (
            discovery_attempts
            if active_scorer is scorer
            else holdout_attempts
            if active_scorer is holdout_scorer
            else None
        )
        if scorer_attempts is None:
            raise ValueError("search received a report from an unknown scorer")
        expected_attempts = attempts or scorer_attempts
        if report.attempts != expected_attempts:
            raise ValueError(
                f"scorer returned attempts={report.attempts}; expected attempts={expected_attempts}"
            )
        required_task_ids = frozenset(task_ids) if task_ids is not None else expected_task_ids
        if required_task_ids is not None and set(report.per_task) != required_task_ids:
            missing = sorted(required_task_ids - set(report.per_task))
            extra = sorted(set(report.per_task) - required_task_ids)
            raise ValueError(
                f"scorer returned the wrong task set; missing={missing}, extra={extra}"
            )
        evaluation_record = (
            doc.execution_hash,
            request.model_dump_json(),
            _score_report_fingerprint(report),
        )
        prior_record = evaluations_by_id.get(report.evaluation_id)
        if prior_record is not None and prior_record != evaluation_record:
            raise ValueError(
                f"evaluation_id {report.evaluation_id!r} identifies a different report"
            )
        evaluations_by_id[report.evaluation_id] = evaluation_record
        _check_cancelled()
        return report

    docs: dict[str, HarnessDoc] = {seed_doc.doc_hash: seed_doc}
    reports: dict[str, HarnessScoreReport] = {}
    holdout_reports: dict[str, HarnessScoreReport] = {}
    archive = DeltaArchive(seed=seed_doc)
    failure_cluster_expansions: dict[FailureClusterKey, int] = {}
    skipped = 0
    screened = 0
    confirmations = 0

    seed_report = _score(scorer, seed_doc, purpose="seed")
    if not seed_report.per_task:
        raise ValueError("seed score report contains no tasks")
    discovery_task_ids = frozenset(seed_report.per_task)
    reports[seed_doc.doc_hash] = seed_report
    holdout_task_ids: frozenset[str] | None = None
    if holdout_scorer is not None:
        seed_holdout = _score(holdout_scorer, seed_doc, purpose="holdout")
        if not seed_holdout.per_task:
            raise ValueError("holdout seed score report contains no tasks")
        holdout_task_ids = frozenset(seed_holdout.per_task)
        holdout_reports[seed_doc.doc_hash] = seed_holdout
    if on_progress is not None:
        on_progress(0, seed_doc.name, seed_report.score, True)

    champion_hash = seed_doc.doc_hash
    best_full = seed_report.score
    suite = sorted(task_id for task_id, task in seed_report.per_task.items() if task.passed)
    proposal_records: list[ProposalRecord] = []

    def _stage_dead(records: list[ProposalRecord], record: ProposalRecord) -> None:
        records.append(record)
        _note(
            _dead_proposal_note(
                record,
                iterations=iterations,
                batch_size=proposal_batch_size,
            )
        )

    for iteration_index in range(1, iterations + 1):
        _check_cancelled()
        scorer.before_proposal_batch()
        if holdout_scorer is not None and holdout_scorer is not scorer:
            holdout_scorer.before_proposal_batch()
        parent = docs[champion_hash]
        parent_report = reports[champion_hash]
        frozen_champion_hash = champion_hash
        frozen_champion_score = parent_report.score
        frozen_best_full = best_full
        frozen_suite = list(suite)
        frozen_champion_holdout = holdout_reports.get(frozen_champion_hash)
        clusters = cluster_score_failures(parent_report)
        batch_cluster_key: FailureClusterKey | None = None
        batch_cluster_expansion_recorded = False
        if clusters:
            trigger = select_failure_cluster(
                clusters,
                failure_cluster_expansions,
                parent_doc_hash=parent.doc_hash,
            )
            batch_cluster_key = _failure_cluster_key(parent.doc_hash, trigger)
        else:
            trigger = FailureSignature(mechanism=ALL_PASS_MECHANISM)
        evidence = render_score_evidence(trigger, parent_report)
        try:
            batch = proposer.propose_batch(
                parent,
                trigger,
                evidence,
                history=archive.deltas,
                count=proposal_batch_size,
                should_cancel=should_cancel,
            )
            if len(batch) != proposal_batch_size:
                raise ValueError(
                    f"proposer returned {len(batch)} proposals; expected {proposal_batch_size}"
                )
        except HarnessSearchCancelled:
            raise
        except Exception as error:  # noqa: BLE001
            batch = [ProposalFailure(reason=str(error))] * proposal_batch_size
        _check_cancelled()

        batch_records: list[ProposalRecord] = []
        batch_deltas: list[HarnessDelta] = []
        scored_proposals: list[_ScoredProposal] = []
        seen_delta_ids = {delta.delta_id for delta in archive.deltas}
        seen_child_hashes = {
            delta.child_doc_hash for delta in archive.deltas if delta.child_doc_hash is not None
        }

        for proposal_index, proposed in enumerate(batch, 1):
            _check_cancelled()
            label = _proposal_label(
                iteration_index,
                proposal_index,
                iterations=iterations,
                batch_size=proposal_batch_size,
            )
            if isinstance(proposed, ProposalFailure):
                skipped += 1
                _stage_dead(
                    batch_records,
                    ProposalRecord(
                        iteration=iteration_index,
                        proposal_index=proposal_index,
                        outcome="proposer_error",
                        trigger=trigger,
                        reason=proposed.reason,
                        champion_score=frozen_champion_score,
                    ),
                )
                continue
            if proposed is None:
                skipped += 1
                _stage_dead(
                    batch_records,
                    ProposalRecord(
                        iteration=iteration_index,
                        proposal_index=proposal_index,
                        outcome="unusable",
                        trigger=trigger,
                        reason="unparseable or truncated meta reply",
                        champion_score=frozen_champion_score,
                    ),
                )
                continue
            delta = proposed
            ops_summary = [f"{op.op} {op.surface_id}" for op in delta.ops]
            rationales = [op.rationale[:1_000] for op in delta.ops]
            expected_effect = delta.expected_effect[:1_000]
            if delta.delta_id in seen_delta_ids:
                delta.verdict = GateRecord(
                    accepted=False,
                    reason="invalid before eval: duplicate of an already-proposed delta",
                )
                batch_deltas.append(delta)
                skipped += 1
                _stage_dead(
                    batch_records,
                    ProposalRecord(
                        iteration=iteration_index,
                        proposal_index=proposal_index,
                        outcome="invalid",
                        delta_id=delta.delta_id,
                        trigger=delta.trigger,
                        expected_effect=expected_effect,
                        ops=ops_summary,
                        rationales=rationales,
                        reason="duplicate of an already-proposed delta",
                        champion_score=frozen_champion_score,
                    ),
                )
                continue
            seen_delta_ids.add(delta.delta_id)
            batch_deltas.append(delta)
            try:
                child = apply_delta(
                    parent,
                    delta,
                    f"{name}-i{iteration_index}-p{proposal_index}",
                )
            except ValueError as error:
                delta.verdict = GateRecord(accepted=False, reason=f"invalid before eval: {error}")
                skipped += 1
                _stage_dead(
                    batch_records,
                    ProposalRecord(
                        iteration=iteration_index,
                        proposal_index=proposal_index,
                        outcome="invalid",
                        delta_id=delta.delta_id,
                        trigger=delta.trigger,
                        expected_effect=expected_effect,
                        ops=ops_summary,
                        rationales=rationales,
                        reason=f"invalid before eval: {error}",
                        champion_score=frozen_champion_score,
                    ),
                )
                continue
            if child.doc_hash in seen_child_hashes:
                delta.verdict = GateRecord(
                    accepted=False,
                    reason="invalid before eval: duplicate of an already-proposed child",
                )
                skipped += 1
                _stage_dead(
                    batch_records,
                    ProposalRecord(
                        iteration=iteration_index,
                        proposal_index=proposal_index,
                        outcome="invalid",
                        candidate=child.name,
                        candidate_doc_hash=child.doc_hash,
                        delta_id=delta.delta_id,
                        trigger=delta.trigger,
                        expected_effect=expected_effect,
                        ops=ops_summary,
                        rationales=rationales,
                        reason="duplicate of an already-proposed child",
                        champion_score=frozen_champion_score,
                    ),
                )
                continue
            seen_child_hashes.add(child.doc_hash)
            validation_error = scorer.validate_candidate(child)
            if validation_error is None and holdout_scorer is not None:
                validation_error = holdout_scorer.validate_candidate(child)
            if validation_error is not None:
                reason = f"invalid before eval: {validation_error}"
                delta.verdict = GateRecord(accepted=False, reason=reason)
                skipped += 1
                _stage_dead(
                    batch_records,
                    ProposalRecord(
                        iteration=iteration_index,
                        proposal_index=proposal_index,
                        outcome="invalid",
                        candidate=child.name,
                        candidate_doc_hash=child.doc_hash,
                        delta_id=delta.delta_id,
                        trigger=delta.trigger,
                        expected_effect=expected_effect,
                        ops=ops_summary,
                        rationales=rationales,
                        reason=reason,
                        champion_score=frozen_champion_score,
                    ),
                )
                continue

            if batch_cluster_key is not None and not batch_cluster_expansion_recorded:
                failure_cluster_expansions[batch_cluster_key] = (
                    failure_cluster_expansions.get(batch_cluster_key, 0) + 1
                )
                batch_cluster_expansion_recorded = True

            screen_child_value: float | None = None
            screen_parent_value: float | None = None
            screen_child_secondary: float | None = None
            screen_parent_secondary: float | None = None
            screen_task_ids = sorted(trigger.task_ids)
            if screen_proposals and screen_task_ids:
                screen_report = _score(
                    scorer,
                    child,
                    purpose="screen",
                    task_ids=screen_task_ids,
                )
                parent_mean = suite_score(parent_report, screen_task_ids)
                child_mean = suite_score(screen_report, screen_task_ids)
                parent_secondary = suite_secondary_score(parent_report, screen_task_ids)
                child_secondary = suite_secondary_score(screen_report, screen_task_ids)
                screen_child_value = child_mean
                screen_parent_value = parent_mean
                screen_child_secondary = child_secondary
                screen_parent_secondary = parent_secondary
                feedback_error = _record_score_evaluation(
                    proposer,
                    delta,
                    stage="screen",
                    report=screen_report,
                    summary=(
                        f"trigger score {child_mean:.3f} vs parent {parent_mean:.3f}; "
                        f"secondary score {child_secondary:.3f} vs parent {parent_secondary:.3f}"
                    ),
                )
                if feedback_error is not None:
                    _note(
                        f"{label}: screen feedback could not be persisted "
                        f"({feedback_error}); continuing"
                    )
                success_regressed = child_mean < parent_mean - _TIE_EPS
                success_tied = abs(child_mean - parent_mean) <= _TIE_EPS
                secondary_did_not_improve = child_secondary <= parent_secondary + _TIE_EPS
                if success_regressed or (success_tied and secondary_did_not_improve):
                    delta.verdict = GateRecord(
                        accepted=False,
                        reason=(
                            f"screened out: trigger score {child_mean:.2f} vs parent "
                            f"{parent_mean:.2f}; secondary score {child_secondary:.2f} vs "
                            f"parent {parent_secondary:.2f} over {len(screen_task_ids)} "
                            f"task(s), attempts={discovery_attempts}; the delta did not "
                            "improve its own target"
                        ),
                    )
                    screened += 1
                    _stage_dead(
                        batch_records,
                        ProposalRecord(
                            iteration=iteration_index,
                            proposal_index=proposal_index,
                            outcome="screened",
                            candidate=child.name,
                            candidate_doc_hash=child.doc_hash,
                            delta_id=delta.delta_id,
                            trigger=delta.trigger,
                            expected_effect=expected_effect,
                            ops=ops_summary,
                            rationales=rationales,
                            reason=delta.verdict.reason,
                            screen_child=child_mean,
                            screen_parent=parent_mean,
                            screen_child_secondary=child_secondary,
                            screen_parent_secondary=parent_secondary,
                            champion_score=frozen_champion_score,
                        ),
                    )
                    continue

            child_report = _score(
                scorer,
                child,
                purpose="full",
                expected_task_ids=discovery_task_ids,
            )
            pre_verdict = gate_score_delta(
                delta,
                child=child_report,
                champion=parent_report,
                best_full=frozen_best_full,
                suite=frozen_suite,
            )
            could_accept = pre_verdict.accepted or (
                confirm_narrow_vetoes
                and narrow_failing_tiers(
                    pre_verdict,
                    k=discovery_attempts,
                    n_suite=len(frozen_suite),
                    n_holdout=0,
                )
                is not None
            )
            if holdout_scorer is not None and could_accept:
                child_holdout = _score(
                    holdout_scorer,
                    child,
                    purpose="holdout",
                    expected_task_ids=holdout_task_ids,
                )
                holdout_reports[child.doc_hash] = child_holdout
                if frozen_champion_holdout is None:
                    frozen_champion_holdout = _score(
                        holdout_scorer,
                        parent,
                        purpose="holdout",
                        expected_task_ids=holdout_task_ids,
                    )
                    holdout_reports[frozen_champion_hash] = frozen_champion_holdout
                verdict = gate_score_delta(
                    delta,
                    child=child_report,
                    champion=parent_report,
                    best_full=frozen_best_full,
                    suite=frozen_suite,
                    child_holdout=child_holdout,
                    champion_holdout=frozen_champion_holdout,
                )
            else:
                verdict = pre_verdict
            tiers = (
                narrow_failing_tiers(
                    verdict,
                    k=discovery_attempts,
                    n_suite=len(frozen_suite),
                    n_holdout=(
                        len(frozen_champion_holdout.per_task)
                        if frozen_champion_holdout is not None
                        else 0
                    ),
                )
                if confirm_narrow_vetoes
                else None
            )
            if tiers:
                confirmations += 1
                confirmed_ok = True
                notes: list[str] = []
                for tier in tiers:
                    active_scorer = scorer if tier == "suite" else holdout_scorer
                    assert active_scorer is not None
                    task_ids = frozen_suite if tier == "suite" else None
                    attempts = 2 * (
                        discovery_attempts if tier == "suite" else holdout_attempts or 0
                    )
                    child_re = _score(
                        active_scorer,
                        child,
                        purpose="confirmation",
                        task_ids=task_ids,
                        attempts=attempts,
                        expected_task_ids=(
                            discovery_task_ids if tier == "suite" else holdout_task_ids
                        ),
                    )
                    champion_re = _score(
                        active_scorer,
                        parent,
                        purpose="confirmation",
                        task_ids=task_ids,
                        attempts=attempts,
                        expected_task_ids=(
                            discovery_task_ids if tier == "suite" else holdout_task_ids
                        ),
                    )
                    score_delta = child_re.score - champion_re.score
                    secondary_delta = child_re.secondary_score - champion_re.secondary_score
                    notes.append(
                        f"{tier} re-measured at attempts={attempts}: score "
                        f"{score_delta:+.3f}, secondary score {secondary_delta:+.3f}"
                    )
                    if score_delta < -_TIE_EPS or (
                        abs(score_delta) <= _TIE_EPS and secondary_delta < -_TIE_EPS
                    ):
                        confirmed_ok = False
                outcome = "veto overturned" if confirmed_ok else "regression confirmed"
                verdict = GateRecord(
                    suite_delta=verdict.suite_delta,
                    suite_secondary_delta=verdict.suite_secondary_delta,
                    full_delta=verdict.full_delta,
                    full_secondary_delta=verdict.full_secondary_delta,
                    holdout_delta=verdict.holdout_delta,
                    holdout_secondary_delta=verdict.holdout_secondary_delta,
                    accepted=confirmed_ok,
                    reason=(
                        f"confirmation re-run ({outcome}): {'; '.join(notes)} | initially: "
                        f"{verdict.reason}"
                    ),
                )

            docs[child.doc_hash] = child
            reports[child.doc_hash] = child_report
            record = ProposalRecord(
                iteration=iteration_index,
                proposal_index=proposal_index,
                outcome="scored",
                candidate=child.name,
                candidate_doc_hash=child.doc_hash,
                delta_id=delta.delta_id,
                trigger=delta.trigger,
                expected_effect=expected_effect,
                ops=ops_summary,
                rationales=rationales,
                reason=verdict.reason,
                score=child_report.score,
                gate_eligible=verdict.accepted,
                screen_child=screen_child_value,
                screen_parent=screen_parent_value,
                screen_child_secondary=screen_child_secondary,
                screen_parent_secondary=screen_parent_secondary,
                champion_score=frozen_champion_score,
            )
            batch_records.append(record)
            scored_proposals.append(
                _ScoredProposal(
                    proposal_index=proposal_index,
                    child=child,
                    delta=delta,
                    report=child_report,
                    gate=verdict,
                    record=record,
                )
            )

        _check_cancelled()
        eligible = [candidate for candidate in scored_proposals if candidate.gate.accepted]
        winner = (
            max(
                eligible,
                key=lambda candidate: (
                    candidate.report.score,
                    candidate.report.secondary_score,
                    -candidate.proposal_index,
                ),
            )
            if eligible
            else None
        )
        for candidate in scored_proposals:
            gate_eligible = candidate.gate.accepted
            if winner is candidate:
                final_gate = candidate.gate
            elif gate_eligible:
                assert winner is not None
                final_gate = candidate.gate.model_copy(
                    update={
                        "accepted": False,
                        "reason": (
                            "gate eligible but not selected: "
                            f"proposal {winner.proposal_index} ranked higher by full score, "
                            "secondary score, then proposal order | " + candidate.gate.reason
                        ),
                    }
                )
            else:
                final_gate = candidate.gate
            candidate.delta.verdict = final_gate
            candidate.record.gate_eligible = gate_eligible
            candidate.record.selected = winner is candidate
            candidate.record.reason = final_gate.reason

        for candidate in scored_proposals:
            assert candidate.delta.verdict is not None
            feedback_error = _record_score_evaluation(
                proposer,
                candidate.delta,
                stage="full",
                report=candidate.report,
                summary=candidate.delta.verdict.reason,
            )
            if feedback_error is not None:
                _note(
                    f"iteration {iteration_index}/{iterations} proposal "
                    f"{candidate.proposal_index}/{proposal_batch_size}: full feedback could "
                    f"not be persisted ({feedback_error}); continuing"
                )

        _check_cancelled()
        archive.deltas.extend(batch_deltas)
        if winner is not None:
            champion_hash = winner.child.doc_hash
            best_full = max(best_full, winner.report.score)
            suite = sorted(
                set(suite)
                | {task_id for task_id, task in winner.report.per_task.items() if task.passed}
            )
            if on_accept is not None:
                on_accept(winner.child, winner.delta, winner.report.score)

        proposal_records.extend(batch_records)
        if on_proposal is not None:
            for record in batch_records:
                on_proposal(record)
        if on_progress is not None:
            champion = docs[champion_hash]
            on_progress(
                iteration_index,
                champion.name,
                reports[champion_hash].score,
                winner is not None,
            )

    _check_cancelled()
    best = docs[champion_hash].model_copy(update={"name": name, "version": 0})
    return SearchResult(
        best=best,
        best_score=reports[champion_hash].score,
        archive=archive,
        reports={doc_hash: report.model_copy(deep=True) for doc_hash, report in reports.items()},
        holdout_reports={
            doc_hash: report.model_copy(deep=True) for doc_hash, report in holdout_reports.items()
        },
        suite=suite,
        skipped=skipped,
        proposal_records=proposal_records,
        screened=screened,
        confirmations=confirmations,
        iterations=iterations,
        proposal_batch_size=proposal_batch_size,
    )


ClosedLoopScoreFn = Callable[[HarnessDoc, list[TaskSpec], int], ClosedLoopReport]


class _ClosedLoopHarnessScorer:
    """Adapt the existing world-model evaluation into normalized search scores."""

    capabilities = ScoreCapabilities(task_subsets=True, attempt_overrides=True)

    def __init__(
        self,
        tasks: list[TaskSpec],
        *,
        default_attempts: int,
        evaluate: ClosedLoopScoreFn,
        validate_candidate: Callable[[HarnessDoc], str | None],
        before_proposal_batch: Callable[[], None],
    ) -> None:
        self._tasks = list(tasks)
        self._by_id = {task.task_id: task for task in tasks}
        if len(self._by_id) != len(tasks):
            raise ValueError("closed-loop search task ids must be unique")
        self.default_attempts = default_attempts
        self._evaluate = evaluate
        self._validate = validate_candidate
        self._before_proposal = before_proposal_batch
        self.raw_reports: dict[str, ClosedLoopReport] = {}

    def validate_candidate(self, candidate: HarnessDoc) -> str | None:
        return self._validate(candidate)

    def before_proposal_batch(self) -> None:
        self._before_proposal()

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
        if request.task_ids is None:
            tasks = self._tasks
        else:
            unknown = sorted(set(request.task_ids) - set(self._by_id))
            if unknown:
                raise ValueError(f"score request contains unknown task ids: {unknown}")
            selected = set(request.task_ids)
            tasks = [task for task in self._tasks if task.task_id in selected]
        attempts = request.attempts or self.default_attempts
        raw = self._evaluate(candidate, tasks, attempts)
        if raw.k != attempts:
            raise ValueError(f"closed-loop evaluator returned k={raw.k} for attempts={attempts}")
        _validate_closed_loop_report(tasks, raw, attempts=attempts)
        if request.purpose in ("seed", "full", "holdout"):
            self.raw_reports[candidate.doc_hash] = raw
        return _closed_loop_score(candidate, tasks, raw, request)


def _validate_closed_loop_report(
    tasks: list[TaskSpec], report: ClosedLoopReport, *, attempts: int
) -> None:
    """Reject a raw scorecard unless its full task-attempt matrix is internally consistent."""
    expected_task_ids = {task.task_id for task in tasks}
    actual_task_ids = set(report.per_task)
    if actual_task_ids != expected_task_ids:
        missing = sorted(expected_task_ids - actual_task_ids)
        extra = sorted(actual_task_ids - expected_task_ids)
        raise ValueError(
            f"closed-loop evaluator returned wrong task set; missing={missing}, extra={extra}"
        )

    success_rates: list[float] = []
    secondary_scores: list[float] = []
    for task_id in sorted(expected_task_ids):
        outcome = report.per_task[task_id]
        if outcome.task_id != task_id:
            raise ValueError(
                f"closed-loop report key {task_id!r} does not match task_id {outcome.task_id!r}"
            )
        if outcome.passes != attempts:
            raise ValueError(
                f"closed-loop task {task_id!r} returned passes={outcome.passes}; "
                f"expected {attempts}"
            )
        if len(outcome.verdicts) != attempts:
            raise ValueError(
                f"closed-loop task {task_id!r} returned {len(outcome.verdicts)} verdicts; "
                f"expected {attempts}"
            )
        if len(outcome.attempts) != attempts:
            raise ValueError(
                f"closed-loop task {task_id!r} returned {len(outcome.attempts)} evidence "
                f"records; expected {attempts}"
            )

        success_rate = (
            sum(1.0 if verdict.passed else 0.0 for verdict in outcome.verdicts) / attempts
        )
        secondary_score = sum(verdict.fraction for verdict in outcome.verdicts) / attempts
        if not isclose(outcome.success_rate, success_rate, rel_tol=0.0, abs_tol=_TIE_EPS):
            raise ValueError(
                f"closed-loop task {task_id!r} success_rate={outcome.success_rate!r} is "
                f"inconsistent with verdicts={success_rate!r}"
            )
        if not isclose(outcome.mean_fraction, secondary_score, rel_tol=0.0, abs_tol=_TIE_EPS):
            raise ValueError(
                f"closed-loop task {task_id!r} mean_fraction={outcome.mean_fraction!r} is "
                f"inconsistent with verdicts={secondary_score!r}"
            )
        success_rates.append(success_rate)
        secondary_scores.append(secondary_score)

    aggregate_score = sum(success_rates) / len(success_rates) if success_rates else 0.0
    aggregate_secondary = sum(secondary_scores) / len(secondary_scores) if secondary_scores else 0.0
    if not isclose(report.success_rate, aggregate_score, rel_tol=0.0, abs_tol=_TIE_EPS):
        raise ValueError(
            f"closed-loop report success_rate={report.success_rate!r} is inconsistent with "
            f"per-task mean={aggregate_score!r}"
        )
    if not isclose(report.mean_fraction, aggregate_secondary, rel_tol=0.0, abs_tol=_TIE_EPS):
        raise ValueError(
            f"closed-loop report mean_fraction={report.mean_fraction!r} is inconsistent with "
            f"per-task mean={aggregate_secondary!r}"
        )


def _closed_loop_score(
    candidate: HarnessDoc,
    tasks: list[TaskSpec],
    report: ClosedLoopReport,
    request: ScoreRequest,
) -> HarnessScoreReport:
    """Project a gold-judged report onto the benchmark-neutral search contract."""
    task_scores: dict[str, TaskScore] = {}
    for task in tasks:
        outcome = report.per_task.get(task.task_id)
        if outcome is None:
            raise ValueError(f"closed-loop report is missing task {task.task_id!r}")
        mechanisms: list[str] = []
        seen: set[str] = set()
        for verdict in outcome.verdicts:
            for assertion in verdict.assertions:
                if not assertion.passed and assertion.assertion not in seen:
                    seen.add(assertion.assertion)
                    mechanisms.append(assertion.assertion)
        description = _bounded_score_text(
            task.instruction,
            limit=MAX_TASK_DESCRIPTION_CHARS,
            label="description",
        )
        evidence = _bounded_score_text(
            render_task_attempt_evidence(outcome),
            limit=MAX_TASK_EVIDENCE_CHARS,
            label="evidence",
        )
        task_scores[task.task_id] = TaskScore(
            task_id=task.task_id,
            score=outcome.success_rate,
            secondary_score=outcome.mean_fraction,
            passed=outcome.success_rate >= 1.0 - _TIE_EPS,
            description=description,
            mechanisms=tuple(mechanisms),
            evidence=evidence,
        )
    identity_input = "\x00".join(
        (
            candidate.execution_hash,
            request.purpose,
            str(report.k),
            *(task.task_id for task in tasks),
            report.model_dump_json(),
        )
    )
    digest = hashlib.blake2b(identity_input.encode("utf-8"), digest_size=16).hexdigest()
    return HarnessScoreReport(
        evaluation_id=f"closed-loop:{digest}",
        label=report.label,
        score=report.success_rate,
        secondary_score=report.mean_fraction,
        attempts=report.k,
        run_health=ScoreRunHealth.VALID,
        per_task=task_scores,
    )


def _bounded_score_text(value: str, *, limit: int, label: str) -> str:
    """Bound one proposer-facing field while preserving an explicit truncation marker."""
    if len(value) <= limit:
        return value
    marker = f"\n...[{label} truncated; original_chars={len(value)}]"
    return value[: limit - len(marker)] + marker


def create_harness(
    name: str,
    seed_doc: HarnessDoc,
    tasks: list[TaskSpec],
    world_model: WorldModel,
    agent_provider: Provider,
    proposer: DeltaProposer,
    judge: GoldJudge,
    *,
    iterations: int = 5,
    proposal_batch_size: int = 1,
    k: int = DEFAULT_K,
    holdout: list[TaskSpec] | None = None,
    confirm_narrow_vetoes: bool = True,
    harness_backend: Literal["local", "e2b"] = "local",
    eval_concurrency: int | None = None,
    e2b_template: str | None = None,
    e2b_metadata: dict[str, str] | None = None,
    on_progress: CreateProgress | None = None,
    on_note: Callable[[str], None] | None = None,
    on_proposal: Callable[[ProposalRecord], None] | None = None,
    on_accept: Callable[[HarnessDoc, HarnessDelta, float], None] | None = None,
    on_sandbox_usage: Callable[[SandboxUsage], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> CreateResult:
    """Run the existing world-model search through the benchmark-neutral scorer seam."""
    if harness_backend not in ("local", "e2b"):
        raise ValueError(f"unknown harness_backend {harness_backend!r}; choose local or e2b")
    if harness_backend == "e2b" and seed_doc.runtime_kind() != "pi-node":
        raise ValueError(
            "harness_backend='e2b' runs the pi-node harness process in sandboxes; seed "
            f"runtime kind is {seed_doc.runtime_kind()!r}, which already runs in-process; "
            "use harness_backend='local'"
        )

    sandbox_pool: E2BSandboxPool | None = None
    if harness_backend == "e2b":
        from wmh.harness.pi_e2b import E2BSandboxPool as _Pool

        sandbox_pool = _Pool(template=e2b_template, metadata=e2b_metadata)

    worker_usages: list[TokenUsage | None] = []
    cancelled: HarnessSearchCancelled | None = None
    result: CreateResult | None = None

    def _check_cancelled() -> None:
        if should_cancel is not None and should_cancel():
            raise HarnessSearchCancelled("harness search cancelled")

    def _validate_candidate(candidate: HarnessDoc) -> str | None:
        if harness_backend == "e2b" and candidate.runtime_kind() != "pi-node":
            return (
                f"runtime kind {candidate.runtime_kind()!r} cannot run on "
                "harness_backend='e2b' (pi-node only)"
            )
        return None

    def _before_proposal_batch() -> None:
        if sandbox_pool is not None:
            sandbox_pool.retire_idle()

    def _evaluate(
        candidate: HarnessDoc,
        split: list[TaskSpec],
        attempts: int,
    ) -> ClosedLoopReport:
        _check_cancelled()
        if harness_backend == "local":
            concurrency = eval_concurrency if eval_concurrency is not None else 1
            if concurrency != 1 and candidate.runtime_kind() == "pi-node":
                raise ValueError(
                    "pi-node harnesses run one episode at a time under harness_backend='local' "
                    "(single runner port/channel); use eval_concurrency=1 or "
                    "harness_backend='e2b'"
                )
            runtime = candidate.runtime(agent_provider)
        else:
            concurrency = eval_concurrency if eval_concurrency is not None else 0
            runtime = candidate.runtime(
                agent_provider,
                backend="e2b",
                e2b_pool=sandbox_pool,
                should_cancel=should_cancel,
            )
        try:
            report = evaluate_closed_loop(
                split,
                world_model,
                agent_provider,
                judge,
                label=candidate.name,
                k=attempts,
                concurrency=concurrency,
                runtime=runtime,
                should_cancel=should_cancel,
            )
        except RuntimeCancelled as error:
            raise HarnessSearchCancelled(
                "harness search cancelled", worker_usage=error.worker_usage
            ) from error
        worker_usages.append(report.worker_usage)
        _check_cancelled()
        return report

    try:
        scorer = _ClosedLoopHarnessScorer(
            tasks,
            default_attempts=k,
            evaluate=_evaluate,
            validate_candidate=_validate_candidate,
            before_proposal_batch=_before_proposal_batch,
        )
        holdout_scorer = (
            _ClosedLoopHarnessScorer(
                holdout,
                default_attempts=k,
                evaluate=_evaluate,
                validate_candidate=_validate_candidate,
                before_proposal_batch=lambda: None,
            )
            if holdout
            else None
        )
        search_result = search_harness(
            name,
            seed_doc,
            scorer,
            proposer,
            iterations=iterations,
            proposal_batch_size=proposal_batch_size,
            screen_proposals=True,
            holdout_scorer=holdout_scorer,
            confirm_narrow_vetoes=confirm_narrow_vetoes,
            on_progress=on_progress,
            on_note=on_note,
            on_proposal=on_proposal,
            on_accept=on_accept,
            should_cancel=should_cancel,
        )
        result = CreateResult(
            best=search_result.best,
            best_score=search_result.best_score,
            archive=search_result.archive,
            reports=scorer.raw_reports,
            holdout_reports=(holdout_scorer.raw_reports if holdout_scorer is not None else {}),
            suite=search_result.suite,
            skipped=search_result.skipped,
            proposal_records=search_result.proposal_records,
            screened=search_result.screened,
            confirmations=search_result.confirmations,
            iterations=search_result.iterations,
            proposal_batch_size=search_result.proposal_batch_size,
            worker_usage=combine_usage(worker_usages),
        )
        return result
    except HarnessSearchCancelled as error:
        error.worker_usage = combine_usage([*worker_usages, error.worker_usage])
        cancelled = error
        raise
    finally:
        if sandbox_pool is not None:
            sandbox_pool.close()
            usage = sandbox_pool.usage()
            if result is not None:
                result.sandbox_usage = usage
            if cancelled is not None:
                cancelled.sandbox_usage = usage
            if on_sandbox_usage is not None:
                on_sandbox_usage(usage)


def _record_score_evaluation(
    proposer: DeltaProposer,
    delta: HarnessDelta,
    *,
    stage: str,
    report: HarnessScoreReport,
    summary: str,
) -> str | None:
    """Persist neutral evaluation evidence when the proposer supports feedback."""
    recorder = getattr(proposer, "record_evaluation", None)
    if not callable(recorder):
        return None
    content = (
        f"# Candidate evaluation: {stage}\n\n"
        f"Delta: {delta.delta_id}\n\n"
        f"Expected effect: {delta.expected_effect}\n\n"
        f"Outcome: {summary}\n\n"
        f"{render_score_evidence(delta.trigger, report)}"
    )
    try:
        recorder(delta, stage=stage, content=content)
    except HarnessSearchCancelled:
        raise
    except Exception as error:  # noqa: BLE001
        return str(error)
    return None


def _proposal_label(
    iteration_index: int, proposal_index: int, *, iterations: int, batch_size: int
) -> str:
    """Human-readable proposal identity that stays concise for singleton batches."""
    if batch_size == 1:
        return f"iteration {iteration_index}/{iterations}"
    return f"iteration {iteration_index}/{iterations} proposal {proposal_index}/{batch_size}"


def _dead_proposal_note(record: ProposalRecord, *, iterations: int, batch_size: int) -> str:
    """Render one dead proposal for the lightweight narration callback."""
    label = _proposal_label(
        record.iteration,
        record.proposal_index,
        iterations=iterations,
        batch_size=batch_size,
    )
    reason = record.reason or "no reason reported"
    if record.outcome == "proposer_error":
        return f"{label}: proposer call failed ({reason}); skipped"
    if record.outcome == "unusable":
        return f"{label}: proposal unusable ({reason}); skipped"
    if record.outcome == "invalid":
        return f"{label}: {reason}; skipped"
    return f"{label}: {reason}"
