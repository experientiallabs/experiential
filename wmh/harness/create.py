"""`wmh harness create`: budgeted search over harness deltas, gated by non-regression.

The loop: select a parent from the accepted pool (stepping-stone weighting — good scores favored,
already-expanded parents discounted), cluster the parent's failures into mechanisms, ask the
proposer for a sibling batch of `HarnessDelta` objects against a size-weighted, expansion-
discounted cluster, apply each atomically, score each child closed-loop against the world model,
and gate acceptance:

- **Tier 1 — regression suite**: the child's score on the suite (tasks the search has already
  mastered) must not drop below the champion's. Newly-passing tasks promote into the suite on
  accept, so wins are locked in and later deltas cannot quietly trade them away.
- **Tier 2 — full split**: the child's overall success rate must be at least the best seen.
- **Tier 3 — held-out (optional)**: with a holdout task file, the child must also be no worse than
  the champion on tasks the proposer never saw evidence from.

Ties pass every tier: with k passes per task, scores are coarse, and "no worse" is the contract.
Every proposed delta — accepted, rejected, or invalid-before-eval — is recorded in the archive
with its verdict, so the archive is a queryable lineage of audited updates, not a pile of
snapshots. Parent SELECTION is deterministic — a blake2b hash of the iteration index replaces RNG
(per the repo's no-entropy rule) — but the run as a whole is only as reproducible as its
providers: proposals and rollouts sample real models at temperature.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import DEFAULT_K, ClosedLoopReport, evaluate_closed_loop
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta, apply_delta
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import SandboxUsage
from wmh.harness.mutate import render_evidence
from wmh.harness.proposer import DeltaProposer, ProposalFailure
from wmh.harness.runtime import (
    HarnessSearchCancelled,
    RuntimeCancelled,
    TokenUsage,
    combine_usage,
)
from wmh.providers.base import Provider

if TYPE_CHECKING:
    # Only the annotation: pi_e2b (the optional e2b extra's consumer) is imported lazily where
    # the pool is actually constructed.
    from wmh.harness.pi_e2b import E2BSandboxPool

# Selection floor: a zero-scoring variant keeps a small chance of being expanded, so early
# pools with no successes still make progress instead of dividing by zero interest.
_SELECTION_FLOOR = 0.05

# Non-regression tolerance: fmean over identical verdicts must compare equal, never fail a gate
# on float noise.
_TIE_EPS = 1e-9

ALL_PASS_MECHANISM = "none: all tasks pass"

FailureClusterKey = tuple[str, str, tuple[str, ...]]

# Reports progress as (iteration, variant name, success_rate, accepted); iteration 0 is the seed.
CreateProgress = Callable[[int, str, float, bool], None]


class PoolEntry(BaseModel):
    """One accepted variant, as the parent-selection pool sees it."""

    doc_hash: str
    name: str
    success_rate: float


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


class IterationRecord(BaseModel):
    """One proposal attempt, scored or dead, in order: the complete search history.

    Every search round proposes a sibling batch from one selected parent. A dead attempt
    (every outcome but "scored") ends early: the proposal died before a full-split eval,
    the search moves straight to the next sibling, and the record here is what remains of it.
    Callers render these as records and plot points, so a run full of dead proposals shows its
    work instead of looking like it never iterated. `champion_score` is the champion's
    full-suite rate when the iteration resolved: the honest y-level for plotting a dead
    iteration (the line it failed to move).
    """

    iteration: int
    round: int
    proposal_index: int
    outcome: Literal["scored", "screened", "invalid", "unusable", "proposer_error"]
    candidate: str | None = None
    candidate_doc_hash: str | None = None
    delta_id: str | None = None
    trigger: FailureSignature | None = None
    expected_effect: str | None = None
    ops: list[str] = Field(default_factory=list)  # "replace prompt:main" style summaries
    rationales: list[str] = Field(default_factory=list)
    reason: str | None = None
    score: float | None = None  # full-suite success rate; scored attempts only
    accepted: bool | None = None  # gate verdict; scored attempts only
    screen_child: float | None = None  # trigger-cluster means; screened attempts only
    screen_parent: float | None = None
    screen_child_fraction: float | None = None  # denser assertion-level screen signal
    screen_parent_fraction: float | None = None
    champion_score: float | None = None


class CreateResult(BaseModel):
    """What a create run produced: the champion, its score, and the full search record."""

    best: HarnessDoc
    best_score: float
    archive: DeltaArchive
    reports: dict[str, ClosedLoopReport] = Field(default_factory=dict)  # by doc_hash
    holdout_reports: dict[str, ClosedLoopReport] = Field(default_factory=dict)  # by doc_hash
    suite: list[str] = Field(default_factory=list)  # final regression suite (task ids)
    skipped: int = 0  # iterations whose proposal was unusable or invalid (they end early)
    iteration_records: list[IterationRecord] = Field(default_factory=list)  # every iteration
    screened: int = 0  # deltas rejected at the cheap trigger-cluster screen (no full eval spent)
    confirmations: int = 0  # narrow vetoes retried at higher k (see `narrow_failing_tiers`)
    rounds: int = 0
    proposal_batch_size: int = 1
    # Spend meters over the WHOLE search (seed, screens, full splits, holdout, confirmations).
    # worker_usage: worker-LLM tokens from self-metering runtimes (the pi worker path; None on
    # provider-wrapped runtimes, which are metered upstream). sandbox_usage: E2B sandbox count +
    # lifetime seconds (None on the local backend).
    worker_usage: TokenUsage | None = None
    sandbox_usage: SandboxUsage | None = None


def cluster_failures(report: ClosedLoopReport, tasks: list[TaskSpec]) -> list[FailureSignature]:
    """Group failing tasks into mechanisms, deterministically — no LLM, no entropy.

    Two failing tasks share a mechanism when they share an unmet gold assertion (connected
    components over the task/assertion graph). The cluster's `mechanism` label is its most common
    unmet assertion (ties broken lexicographically); clusters are ordered largest-first so the
    size-weighted, expansion-discounted selector has a stable base order. A failing task whose
    verdicts carry no per-assertion detail (an unparseable judge reply) forms its own cluster.
    """
    unmet_by_task: dict[str, list[str]] = {}
    for task in tasks:
        outcome = report.per_task.get(task.task_id)
        if outcome is None or outcome.success_rate >= 1.0:
            continue
        seen: set[str] = set()
        unmet: list[str] = []
        for verdict in outcome.verdicts:
            for result in verdict.assertions:
                if not result.passed and result.assertion not in seen:
                    seen.add(result.assertion)
                    unmet.append(result.assertion)
        unmet_by_task[task.task_id] = unmet

    clusters: list[FailureSignature] = []
    assigned: set[str] = set()
    for task_id in sorted(unmet_by_task):
        if task_id in assigned:
            continue
        # Flood-fill the component: tasks connected through shared unmet assertions.
        member_ids = {task_id}
        assertions = set(unmet_by_task[task_id])
        grew = True
        while grew:
            grew = False
            for other, other_unmet in unmet_by_task.items():
                if other in member_ids or not assertions.intersection(other_unmet):
                    continue
                member_ids.add(other)
                assertions.update(other_unmet)
                grew = True
        assigned.update(member_ids)
        counts: dict[str, int] = {}
        for member in member_ids:
            for assertion in unmet_by_task[member]:
                counts[assertion] = counts.get(assertion, 0) + 1
        # Most common unmet assertion labels the cluster; max() keeps the first (lexicographically
        # smallest) among ties because the candidates are pre-sorted.
        mechanism = (
            max(sorted(counts), key=lambda a: counts[a])
            if counts
            else "run failed without per-assertion verdicts"
        )
        clusters.append(
            FailureSignature(
                mechanism=mechanism,
                task_ids=sorted(member_ids),
                unmet_assertions=sorted(assertions),
            )
        )
    clusters.sort(key=lambda c: (-len(c.task_ids), c.mechanism))
    return clusters


def select_failure_cluster(
    clusters: list[FailureSignature],
    expansion_counts: dict[FailureClusterKey, int],
    *,
    parent_doc_hash: str,
) -> FailureSignature:
    """Choose a high-impact cluster without getting trapped on one exhausted failure.

    A cluster's priority is ``task_count / (1 + prior_rounds_on_this_parent)``. Large mechanisms
    still receive proportionally more search budget, but equally sized singleton failures rotate
    after one batch instead of a deterministic ``clusters[0]`` absorbing the entire run. The
    stable mechanism/task ordering resolves exact ties without entropy.
    """
    if not clusters:
        raise ValueError("cannot select from an empty failure-cluster list")

    def _priority(cluster: FailureSignature) -> tuple[float, str, tuple[str, ...]]:
        key = _failure_cluster_key(parent_doc_hash, cluster)
        rounds = expansion_counts.get(key, 0)
        return (
            -(len(cluster.task_ids) / (1 + rounds)),
            cluster.mechanism,
            tuple(cluster.task_ids),
        )

    return min(clusters, key=_priority)


def _failure_cluster_key(parent_doc_hash: str, cluster: FailureSignature) -> FailureClusterKey:
    return parent_doc_hash, cluster.mechanism, tuple(cluster.task_ids)


def select_parent(
    entries: list[PoolEntry], children_counts: dict[str, int], seed: int
) -> PoolEntry:
    """Pick the next parent by stepping-stone weighting, deterministically.

    Weight = (success_rate + floor) / (1 + children_count): better variants are favored, but each
    expansion discounts a parent so the search spreads over the pool instead of hill-climbing one
    lineage. The "random" point is blake2b(seed) mapped to [0, 1) over the cumulative weights —
    reproducible, no RNG.
    """
    if not entries:
        raise ValueError("cannot select a parent from an empty pool")
    weights = [
        (entry.success_rate + _SELECTION_FLOOR) / (1 + children_counts.get(entry.doc_hash, 0))
        for entry in entries
    ]
    point = _fraction(str(seed)) * sum(weights)
    cumulative = 0.0
    for entry, weight in zip(entries, weights, strict=True):
        cumulative += weight
        if point < cumulative:
            return entry
    return entries[-1]  # floating-point edge: the point landed exactly on the total


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
    if abs(verdict.suite_delta) <= _TIE_EPS and verdict.suite_fraction_delta < -_TIE_EPS:
        return None
    if (
        verdict.holdout_delta is not None
        and abs(verdict.holdout_delta) <= _TIE_EPS
        and verdict.holdout_fraction_delta is not None
        and verdict.holdout_fraction_delta < -_TIE_EPS
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


def gate_delta(
    delta: HarnessDelta,
    *,
    child: ClosedLoopReport,
    champion: ClosedLoopReport,
    best_full: float,
    suite: list[str],
    child_holdout: ClosedLoopReport | None = None,
    champion_holdout: ClosedLoopReport | None = None,
) -> GateRecord:
    """Lexicographic success/partial-credit gate plus optional held-out acceptance.

    End-to-end task success remains the primary objective. When a tier's binary score ties, its
    assertion-level fraction must not regress. This admits useful partial-progress stepping
    stones without promoting a target-local improvement that silently damages more work across
    the full split.
    """
    suite_delta = _suite_rate(child, suite) - _suite_rate(champion, suite)
    suite_fraction_delta = _suite_fraction(child, suite) - _suite_fraction(champion, suite)
    full_delta = child.success_rate - best_full
    full_fraction_delta = child.mean_fraction - champion.mean_fraction
    holdout_delta = (
        child_holdout.success_rate - champion_holdout.success_rate
        if child_holdout is not None and champion_holdout is not None
        else None
    )
    holdout_fraction_delta = (
        child_holdout.mean_fraction - champion_holdout.mean_fraction
        if child_holdout is not None and champion_holdout is not None
        else None
    )
    failures: list[str] = []
    if suite_delta < -_TIE_EPS:
        failures.append(f"suite regressed by {-suite_delta:.3f}")
    elif abs(suite_delta) <= _TIE_EPS and suite_fraction_delta < -_TIE_EPS:
        failures.append(f"suite assertion fraction regressed by {-suite_fraction_delta:.3f}")
    if full_delta < -_TIE_EPS:
        failures.append(f"full split {child.success_rate:.3f} below best {best_full:.3f}")
    elif abs(full_delta) <= _TIE_EPS and full_fraction_delta < -_TIE_EPS:
        failures.append(f"full-split assertion fraction regressed by {-full_fraction_delta:.3f}")
    if holdout_delta is not None and holdout_delta < -_TIE_EPS:
        failures.append(f"held-out regressed by {-holdout_delta:.3f}")
    elif (
        holdout_delta is not None
        and abs(holdout_delta) <= _TIE_EPS
        and holdout_fraction_delta is not None
        and holdout_fraction_delta < -_TIE_EPS
    ):
        failures.append(f"held-out assertion fraction regressed by {-holdout_fraction_delta:.3f}")
    flipped = sum(
        1
        for task_id in delta.trigger.task_ids
        if (outcome := child.per_task.get(task_id)) is not None and outcome.success_rate >= 1.0
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
        suite_fraction_delta=suite_fraction_delta,
        full_delta=full_delta,
        full_fraction_delta=full_fraction_delta,
        holdout_delta=holdout_delta,
        holdout_fraction_delta=holdout_fraction_delta,
        accepted=accepted,
        reason=reason,
    )


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
    on_iteration: Callable[[IterationRecord], None] | None = None,
    on_accept: Callable[[HarnessDoc, HarnessDelta, float], None] | None = None,
    on_sandbox_usage: Callable[[SandboxUsage], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> CreateResult:
    """Search for a better harness under a fixed eval budget; the champion is renamed to `name`.

    Scores the seed first, then runs `iterations` proposal rounds. Each round asks the proposer
    for `proposal_batch_size` sibling deltas against one selected parent before any sibling is
    evaluated. A dead proposal (unusable/invalid proposal, or one
    screened out on its own trigger cluster) ends early and cheaply: it is counted
    (`skipped`/`screened`), recorded in `iteration_records`, narrated via `on_note`, and
    the search moves straight to the next iteration. Never fatal. `on_accept` fires the
    moment a delta is accepted, with the new champion doc, its delta (verdict attached),
    and its full-suite score, so callers can persist champions in real time instead of
    waiting for the search to finish.

    ``on_sandbox_usage`` fires after the shared E2B pool is successfully closed on every exit
    path, including failures and cancellation, so callers can persist already-incurred evaluator
    spend without requiring a partial result. An unproven close raises instead of publishing a
    falsely final meter.
    ``should_cancel`` is checked before and after every score wave, before each proposal slot,
    and after each batched proposer call. E2B runtimes also poll it while waiting for runner
    frames, so cancellation aborts the active wave without judging partial cells. A provider/tool
    call already in progress remains bounded by that call's own timeout. Cancellation raises
    :class:`HarnessSearchCancelled` while the normal ``finally`` path retires sandbox resources.

    Every rollout scores against the world-model simulation — the environment is always sim.
    `harness_backend` picks where the harness PROCESS executes: `local` (the default) runs it
    in/from this process exactly as before; `e2b` runs the real pi agent inside E2B sandboxes
    (pi-node seeds only — that harness's context management is the thing under search), with its
    tool calls still answered by the world model host-side. One `E2BSandboxPool` is shared across
    score waves within a proposal batch, then its idle runners are retired before the next batch's
    proposer call. This amortizes bootstrap work across sibling proposals without carrying E2B
    command streams through the potentially long proposal gap between rounds. `eval_concurrency`
    is how many (task, attempt) cells run at once; `None` means the backend default — 1
    (sequential) for local, 0 (every cell at once, one pooled sandbox each) for e2b.
    `e2b_template` names a prebaked sandbox template (node 22 + the pi runner deps) so e2b
    rollouts skip bootstrap installs. `e2b_metadata` tags every sandbox created by the shared
    evaluation pool, including fresh replacements for retired runners.

    Verification is staged by cost: a child is first SCREENED on its own trigger cluster (the
    2-3 failing tasks its delta claims to fix, k passes). The screen is lexicographic: full-task
    success first, then assertion-level partial credit, so a real partial fix is not flattened
    into a binary tie. If neither signal improves, the delta is rejected and archived for a
    fraction of a full eval's cost, and no `on_progress` event fires. Repeated batches discount
    their cluster on that parent, preventing one environment-limited singleton from absorbing the
    whole run. Held-out evals run only for children that pass tiers 1-2, so a bad delta costs at
    most one full-split eval. Every judged delta (screened, rejected, or accepted) is fed back to
    the proposer as history, so it iterates instead of re-proposing rejected ideas.

    Symmetrically, a REJECTION can be noise: with k passes over small tiers, one unlucky attempt
    can veto a genuine win. When `confirm_narrow_vetoes` is set, a delta that strictly won the
    full split but failed suite/holdout within `narrow_failing_tiers`' margin gets that tier
    re-measured — child AND champion, at 2k — and the re-measurement decides. The verdict records
    the retrial either way, so the archive shows which accepts needed confirmation.
    """
    if harness_backend not in ("local", "e2b"):
        raise ValueError(f"unknown harness_backend {harness_backend!r}; choose local or e2b")
    if proposal_batch_size < 1:
        raise ValueError(f"proposal_batch_size must be positive, got {proposal_batch_size}")
    if harness_backend == "e2b" and seed_doc.runtime_kind() != "pi-node":
        raise ValueError(
            "harness_backend='e2b' runs the pi-node harness process in sandboxes; seed "
            f"runtime kind is {seed_doc.runtime_kind()!r}, which already runs in-process — "
            "use harness_backend='local'"
        )
    sandbox_pool: E2BSandboxPool | None = None
    if harness_backend == "e2b":
        # Lazy: the e2b backend is an optional extra; local searches must import none of it.
        from wmh.harness.pi_e2b import E2BSandboxPool as _Pool

        sandbox_pool = _Pool(template=e2b_template, metadata=e2b_metadata)

    cancelled: HarnessSearchCancelled | None = None
    result: CreateResult | None = None
    try:

        def _check_cancelled() -> None:
            if should_cancel is not None and should_cancel():
                raise HarnessSearchCancelled("harness search cancelled")

        def _note(message: str) -> None:
            # Narration for iterations that produce NO on_progress event (unusable/invalid/screened
            # proposals): without it a run whose proposals all fail looks like it never iterated.
            if on_note is not None:
                on_note(message)

        docs: dict[str, HarnessDoc] = {seed_doc.doc_hash: seed_doc}
        worker_usages: list[TokenUsage | None] = []
        reports: dict[str, ClosedLoopReport] = {}
        holdout_reports: dict[str, ClosedLoopReport] = {}
        archive = DeltaArchive(seed=seed_doc)
        children_counts: dict[str, int] = {}
        failure_cluster_expansions: dict[FailureClusterKey, int] = {}
        skipped = 0
        screened = 0
        confirmations = 0

        def _score(
            doc: HarnessDoc, split: list[TaskSpec], *, k_override: int | None = None
        ) -> ClosedLoopReport:
            _check_cancelled()
            k_eff = k if k_override is None else k_override
            if harness_backend == "local":
                concurrency = eval_concurrency if eval_concurrency is not None else 1
                if concurrency != 1 and doc.runtime_kind() == "pi-node":
                    # Local pi runtimes are single-episode resources (one runner port/workdir, or
                    # one RunnerLink channel): parallel cells would collide. Checked per-doc because
                    # a delta can flip param:runtime-kind mid-search.
                    raise ValueError(
                        "pi-node harnesses run one episode at a time under harness_backend='local' "
                        "(single runner port/channel); use eval_concurrency=1 or "
                        "harness_backend='e2b'"
                    )
                runtime = doc.runtime(agent_provider)
            else:
                # The pi process runs in pooled sandboxes; every cell at once by default. Tool calls
                # still route to the world model — the environment is sim regardless of backend.
                concurrency = eval_concurrency if eval_concurrency is not None else 0
                runtime = doc.runtime(
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
                    label=doc.name,
                    k=k_eff,
                    concurrency=concurrency,
                    runtime=runtime,
                    should_cancel=should_cancel,
                )
            except RuntimeCancelled as exc:
                # Cancelled cells are not scoreable outcomes: do not judge them. Converting at the
                # search boundary preserves the public cancellation contract while the surrounding
                # finally closes the shared pool and retires every active evaluator sandbox.
                raise HarnessSearchCancelled(
                    "harness search cancelled", worker_usage=exc.worker_usage
                ) from exc
            # Tally the pi worker's self-metered tokens across every score wave (seed, screens,
            # full splits, holdout, confirmations): its LLM calls bypass the Provider, so this is
            # the only record. None on backends whose runtimes don't self-meter (local).
            worker_usages.append(report.worker_usage)
            # Append first so a cancellation that lands after evaluation but before
            # the report is consumed still carries the completed wave's spend.
            _check_cancelled()
            return report

        seed_report = _score(seed_doc, tasks)
        reports[seed_doc.doc_hash] = seed_report
        if holdout:
            holdout_reports[seed_doc.doc_hash] = _score(seed_doc, holdout)
        if on_progress is not None:
            on_progress(0, seed_doc.name, seed_report.success_rate, True)

        pool = [
            PoolEntry(
                doc_hash=seed_doc.doc_hash,
                name=seed_doc.name,
                success_rate=seed_report.success_rate,
            )
        ]
        champion_hash = seed_doc.doc_hash
        best_full = seed_report.success_rate
        # The regression suite: tasks the champion lineage has fully passed. Wins promote in
        # on accept.
        suite = sorted(
            task.task_id
            for task in tasks
            if seed_report.per_task[task.task_id].success_rate >= 1.0 - _TIE_EPS
        )

        iteration_records: list[IterationRecord] = []

        def _dead(record: IterationRecord, note: str) -> None:
            # A dead iteration is a first-class search event: recorded, streamed, narrated.
            # It ends early (no full eval was spent) and the search moves straight on.
            iteration_records.append(record)
            if on_iteration is not None:
                on_iteration(record)
            _note(note)

        proposal_queue: list[
            tuple[
                HarnessDoc,
                ClosedLoopReport,
                FailureSignature,
                HarnessDelta | ProposalFailure | None,
            ]
        ] = []
        batch_cluster_key: FailureClusterKey | None = None
        batch_cluster_expansion_recorded = False
        total_attempts = iterations * proposal_batch_size
        for i in range(1, total_attempts + 1):
            _check_cancelled()
            round_index = ((i - 1) // proposal_batch_size) + 1
            proposal_index = ((i - 1) % proposal_batch_size) + 1
            champion_score = reports[champion_hash].success_rate
            if proposal_index == 1:
                # The previous round's eval runners would otherwise sit idle through this
                # potentially long proposer call. Retire only idle runners now; subsequent score
                # waves for this batch still share the newly warmed pool.
                if sandbox_pool is not None:
                    sandbox_pool.retire_idle()
                parent_entry = select_parent(pool, children_counts, seed=round_index)
                parent = docs[parent_entry.doc_hash]
                parent_report = reports[parent_entry.doc_hash]
                clusters = cluster_failures(parent_report, tasks)
                batch_cluster_key = None
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
                evidence = render_evidence(trigger, parent_report, tasks)
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
                            f"proposer returned {len(batch)} proposals; "
                            f"expected {proposal_batch_size}"
                        )
                except HarnessSearchCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - provider/agent/transport failure
                    batch = [ProposalFailure(reason=str(exc))] * proposal_batch_size
                _check_cancelled()
                proposal_queue.extend((parent, parent_report, trigger, delta) for delta in batch)

            parent, parent_report, trigger, delta = proposal_queue.pop(0)
            label = _proposal_label(
                round_index,
                proposal_index,
                rounds=iterations,
                batch_size=proposal_batch_size,
            )
            if isinstance(delta, ProposalFailure):
                skipped += 1
                _dead(
                    IterationRecord(
                        iteration=i,
                        round=round_index,
                        proposal_index=proposal_index,
                        outcome="proposer_error",
                        trigger=trigger,
                        reason=delta.reason,
                        champion_score=champion_score,
                    ),
                    f"{label}: proposer call failed ({delta.reason}); skipped",
                )
                continue
            if delta is None:
                skipped += 1
                _dead(
                    IterationRecord(
                        iteration=i,
                        round=round_index,
                        proposal_index=proposal_index,
                        outcome="unusable",
                        trigger=trigger,
                        reason="unparseable or truncated meta reply",
                        champion_score=champion_score,
                    ),
                    f"{label}: proposal unusable (unparseable or truncated meta reply); skipped",
                )
                continue
            ops_summary = [f"{op.op} {op.surface_id}" for op in delta.ops]
            rationales = [op.rationale[:1_000] for op in delta.ops]
            expected_effect = delta.expected_effect[:1_000]
            if any(d.delta_id == delta.delta_id for d in archive.deltas):
                # The proposer re-proposed a delta this run already judged. Re-evaluating it
                # would spend a screen (or worse) to learn a known verdict; skip without spend.
                # The duplicate is NOT re-archived; the original carries the verdict.
                skipped += 1
                _dead(
                    IterationRecord(
                        iteration=i,
                        round=round_index,
                        proposal_index=proposal_index,
                        outcome="invalid",
                        delta_id=delta.delta_id,
                        trigger=delta.trigger,
                        expected_effect=expected_effect,
                        ops=ops_summary,
                        rationales=rationales,
                        reason="duplicate of an already-judged delta",
                        champion_score=champion_score,
                    ),
                    f"{label}: proposal duplicates an already-judged delta; skipped",
                )
                continue
            try:
                child_name = (
                    f"{name}-g{round_index}"
                    if proposal_batch_size == 1
                    else f"{name}-g{round_index}-p{proposal_index}"
                )
                child = apply_delta(parent, delta, child_name)
            except ValueError as exc:
                delta.verdict = GateRecord(accepted=False, reason=f"invalid before eval: {exc}")
                archive.deltas.append(delta)
                skipped += 1
                _dead(
                    IterationRecord(
                        iteration=i,
                        round=round_index,
                        proposal_index=proposal_index,
                        outcome="invalid",
                        delta_id=delta.delta_id,
                        trigger=delta.trigger,
                        expected_effect=expected_effect,
                        ops=ops_summary,
                        rationales=rationales,
                        reason=f"invalid before eval: {exc}",
                        champion_score=champion_score,
                    ),
                    f"{label}: delta invalid before eval ({exc}); skipped",
                )
                continue
            if harness_backend == "e2b" and child.runtime_kind() != "pi-node":
                # A delta that abandons the pi-node runtime cannot execute on this backend:
                # `doc.runtime(backend="e2b")` would raise mid-score and abort the whole search.
                # Reject-and-archive it like any other invalid-before-eval proposal.
                delta.verdict = GateRecord(
                    accepted=False,
                    reason=(
                        f"invalid before eval: runtime kind {child.runtime_kind()!r} cannot "
                        "run on harness_backend='e2b' (pi-node only)"
                    ),
                )
                archive.deltas.append(delta)
                skipped += 1
                _dead(
                    IterationRecord(
                        iteration=i,
                        round=round_index,
                        proposal_index=proposal_index,
                        outcome="invalid",
                        candidate=child.name,
                        candidate_doc_hash=child.doc_hash,
                        delta_id=delta.delta_id,
                        trigger=delta.trigger,
                        expected_effect=expected_effect,
                        ops=ops_summary,
                        rationales=rationales,
                        reason=str(delta.verdict.reason),
                        champion_score=champion_score,
                    ),
                    f"{label}: delta abandoned the pi-node runtime "
                    "(e2b runs pi-node only); skipped",
                )
                continue

            # Discount the selected cluster only when this batch produces its first child that can
            # actually enter evaluation. Parsed-but-duplicate, inapplicable, or backend-invalid
            # deltas teach the search nothing about that failure mechanism. Sibling proposals share
            # one batch expansion, so later evaluable children must not discount it again.
            if batch_cluster_key is not None and not batch_cluster_expansion_recorded:
                failure_cluster_expansions[batch_cluster_key] = (
                    failure_cluster_expansions.get(batch_cluster_key, 0) + 1
                )
                batch_cluster_expansion_recorded = True

            # Only an evaluable child counts as a real expansion. Provider faults,
            # unusable replies, duplicates, and invalid deltas leave the parent fresh.
            children_counts[parent.doc_hash] = children_counts.get(parent.doc_hash, 0) + 1

            # Cheap screen: before a full-split eval, the delta must improve the very cluster it
            # was proposed to fix. Compare task success first, then assertion-level partial credit:
            # a 0%-to-75% assertion lift must not be flattened into the same "0 vs 0" as no effect.
            screen_child_value: float | None = None
            screen_parent_value: float | None = None
            screen_child_fraction_value: float | None = None
            screen_parent_fraction_value: float | None = None
            screen_tasks = [t for t in tasks if t.task_id in set(trigger.task_ids)]
            if screen_tasks:
                screen_report = _score(child, screen_tasks)
                parent_mean = _suite_rate(parent_report, sorted(trigger.task_ids))
                child_mean = _suite_rate(screen_report, sorted(trigger.task_ids))
                parent_fraction = _suite_fraction(parent_report, sorted(trigger.task_ids))
                child_fraction = _suite_fraction(screen_report, sorted(trigger.task_ids))
                screen_child_value = child_mean
                screen_parent_value = parent_mean
                screen_child_fraction_value = child_fraction
                screen_parent_fraction_value = parent_fraction
                success_regressed = child_mean < parent_mean - _TIE_EPS
                success_tied = abs(child_mean - parent_mean) <= _TIE_EPS
                fraction_did_not_improve = child_fraction <= parent_fraction + _TIE_EPS
                feedback_error = _record_proposer_evaluation(
                    proposer,
                    delta,
                    stage="screen",
                    report=screen_report,
                    tasks=screen_tasks,
                    summary=(
                        f"trigger success {child_mean:.3f} vs parent {parent_mean:.3f}; "
                        f"assertion fraction {child_fraction:.3f} vs parent "
                        f"{parent_fraction:.3f}"
                    ),
                )
                if feedback_error is not None:
                    _note(
                        f"{label}: screen feedback could not be persisted "
                        f"({feedback_error}); continuing"
                    )
                if success_regressed or (success_tied and fraction_did_not_improve):
                    delta.verdict = GateRecord(
                        accepted=False,
                        reason=(
                            f"screened out: trigger success {child_mean:.2f} vs parent "
                            f"{parent_mean:.2f}; assertion fraction {child_fraction:.2f} vs "
                            f"parent {parent_fraction:.2f} over {len(screen_tasks)} task(s), "
                            f"k={k}; "
                            "the delta did not improve its own target"
                        ),
                    )
                    archive.deltas.append(delta)
                    screened += 1
                    _dead(
                        IterationRecord(
                            iteration=i,
                            round=round_index,
                            proposal_index=proposal_index,
                            outcome="screened",
                            candidate=child.name,
                            candidate_doc_hash=child.doc_hash,
                            delta_id=delta.delta_id,
                            trigger=delta.trigger,
                            expected_effect=expected_effect,
                            ops=ops_summary,
                            rationales=rationales,
                            reason=str(delta.verdict.reason),
                            screen_child=child_mean,
                            screen_parent=parent_mean,
                            screen_child_fraction=child_fraction,
                            screen_parent_fraction=parent_fraction,
                            champion_score=champion_score,
                        ),
                        f"{label}: screened out: trigger success "
                        f"{child_mean:.2f} vs parent {parent_mean:.2f}; assertion fraction "
                        f"{child_fraction:.2f} vs parent {parent_fraction:.2f}",
                    )
                    continue

            child_report = _score(child, tasks)
            pre_verdict = gate_delta(
                delta,
                child=child_report,
                champion=reports[champion_hash],
                best_full=best_full,
                suite=suite,
            )
            # The held-out tier is measured for every candidate that could still be accepted —
            # including candidates whose suite/full veto is narrow enough for a confirmation
            # re-run. A confirmation may overturn a veto; it must never bypass the held-out tier.
            could_accept = pre_verdict.accepted or (
                confirm_narrow_vetoes
                and narrow_failing_tiers(pre_verdict, k=k, n_suite=len(suite), n_holdout=0)
                is not None
            )
            if holdout and could_accept:
                child_holdout = _score(child, holdout)
                holdout_reports[child.doc_hash] = child_holdout
                if champion_hash not in holdout_reports:
                    holdout_reports[champion_hash] = _score(docs[champion_hash], holdout)
                verdict = gate_delta(
                    delta,
                    child=child_report,
                    champion=reports[champion_hash],
                    best_full=best_full,
                    suite=suite,
                    child_holdout=child_holdout,
                    champion_holdout=holdout_reports[champion_hash],
                )
            else:
                verdict = pre_verdict
            tiers = (
                narrow_failing_tiers(verdict, k=k, n_suite=len(suite), n_holdout=len(holdout or []))
                if confirm_narrow_vetoes
                else None
            )
            if tiers:
                confirmations += 1
                confirmed_ok = True
                notes: list[str] = []
                for tier in tiers:
                    tier_tasks = (
                        [t for t in tasks if t.task_id in set(suite)]
                        if tier == "suite"
                        else list(holdout or [])
                    )
                    child_re = _score(child, tier_tasks, k_override=2 * k)
                    champ_re = _score(docs[champion_hash], tier_tasks, k_override=2 * k)
                    re_delta = child_re.success_rate - champ_re.success_rate
                    re_fraction_delta = child_re.mean_fraction - champ_re.mean_fraction
                    notes.append(
                        f"{tier} re-measured at k={2 * k}: success {re_delta:+.3f}, "
                        f"assertion fraction {re_fraction_delta:+.3f}"
                    )
                    if re_delta < -_TIE_EPS or (
                        abs(re_delta) <= _TIE_EPS and re_fraction_delta < -_TIE_EPS
                    ):
                        confirmed_ok = False
                outcome = "veto overturned" if confirmed_ok else "regression confirmed"
                verdict = GateRecord(
                    suite_delta=verdict.suite_delta,
                    suite_fraction_delta=verdict.suite_fraction_delta,
                    full_delta=verdict.full_delta,
                    full_fraction_delta=verdict.full_fraction_delta,
                    holdout_delta=verdict.holdout_delta,
                    holdout_fraction_delta=verdict.holdout_fraction_delta,
                    accepted=confirmed_ok,
                    reason=f"confirmation re-run ({outcome}): {'; '.join(notes)} | initially: "
                    + verdict.reason,
                )
            if verdict.accepted:
                # Scoring checks cancellation on return, but gating and confirmation bookkeeping
                # happen afterward. Check again at the acceptance commit boundary so a cancelled
                # child never enters lineage or reaches the persistence callback.
                _check_cancelled()
            feedback_error = _record_proposer_evaluation(
                proposer,
                delta,
                stage="full",
                report=child_report,
                tasks=tasks,
                summary=verdict.reason,
            )
            if feedback_error is not None:
                _note(
                    f"{label}: full feedback could not be persisted ({feedback_error}); continuing"
                )
            delta.verdict = verdict
            archive.deltas.append(delta)
            docs[child.doc_hash] = child
            reports[child.doc_hash] = child_report
            if verdict.accepted:
                pool.append(
                    PoolEntry(
                        doc_hash=child.doc_hash,
                        name=child.name,
                        success_rate=child_report.success_rate,
                    )
                )
                champion_hash = child.doc_hash
                best_full = max(best_full, child_report.success_rate)
                promoted = {
                    task_id
                    for task_id, outcome in child_report.per_task.items()
                    if outcome.success_rate >= 1.0 - _TIE_EPS
                }
                suite = sorted(set(suite) | promoted)
                if on_accept is not None:
                    on_accept(child, delta, child_report.success_rate)
            record = IterationRecord(
                iteration=i,
                round=round_index,
                proposal_index=proposal_index,
                outcome="scored",
                candidate=child.name,
                candidate_doc_hash=child.doc_hash,
                delta_id=delta.delta_id,
                trigger=delta.trigger,
                expected_effect=expected_effect,
                ops=ops_summary,
                rationales=rationales,
                reason=str(verdict.reason),
                score=child_report.success_rate,
                accepted=verdict.accepted,
                screen_child=screen_child_value,
                screen_parent=screen_parent_value,
                screen_child_fraction=screen_child_fraction_value,
                screen_parent_fraction=screen_parent_fraction_value,
                champion_score=reports[champion_hash].success_rate,
            )
            iteration_records.append(record)
            if on_iteration is not None:
                on_iteration(record)
            if on_progress is not None:
                on_progress(i, child.name, child_report.success_rate, verdict.accepted)

        _check_cancelled()
        best = docs[champion_hash].model_copy(update={"name": name, "version": 0})
        result = CreateResult(
            best=best,
            best_score=reports[champion_hash].success_rate,
            archive=archive,
            reports=reports,
            holdout_reports=holdout_reports,
            suite=suite,
            skipped=skipped,
            iteration_records=iteration_records,
            screened=screened,
            confirmations=confirmations,
            rounds=iterations,
            proposal_batch_size=proposal_batch_size,
            worker_usage=combine_usage(worker_usages),
        )
        return result
    except HarnessSearchCancelled as error:
        # A cancellation inside evaluation carries that wave's completed and
        # partial cells. Waves that returned normally were appended above. The
        # search exception is the authoritative aggregate for callers because
        # cancellation intentionally has no partial CreateResult.
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


def _suite_rate(report: ClosedLoopReport, suite: list[str]) -> float:
    """Mean per-task success over the regression suite; an empty suite constrains nothing."""
    if not suite:
        return 1.0
    rates = [
        outcome.success_rate
        for task_id in suite
        if (outcome := report.per_task.get(task_id)) is not None
    ]
    # Dividing by len(suite), not len(rates): a suite task missing from the report counts as 0
    # (fail-closed), though suite tasks are always a subset of the scored split in practice.
    return sum(rates) / len(suite)


def _suite_fraction(report: ClosedLoopReport, suite: list[str]) -> float:
    """Mean assertion completion over a task subset; missing tasks fail closed."""
    if not suite:
        return 1.0
    fractions = [
        outcome.mean_fraction
        for task_id in suite
        if (outcome := report.per_task.get(task_id)) is not None
    ]
    return sum(fractions) / len(suite)


def _record_proposer_evaluation(
    proposer: DeltaProposer,
    delta: HarnessDelta,
    *,
    stage: str,
    report: ClosedLoopReport,
    tasks: list[TaskSpec],
    summary: str,
) -> str | None:
    """Best-effort durable trace feedback; evaluation correctness never depends on telemetry."""
    recorder = getattr(proposer, "record_evaluation", None)
    if not callable(recorder):
        return None
    content = (
        f"# Candidate evaluation: {stage}\n\n"
        f"Delta: {delta.delta_id}\n\n"
        f"Expected effect: {delta.expected_effect}\n\n"
        f"Outcome: {summary}\n\n"
        f"{render_evidence(delta.trigger, report, tasks)}"
    )
    try:
        recorder(delta, stage=stage, content=content)
    except HarnessSearchCancelled:
        raise
    except Exception as error:  # noqa: BLE001 - optional E2B feedback must not abort scored work
        return str(error)
    return None


def _proposal_label(round_index: int, proposal_index: int, *, rounds: int, batch_size: int) -> str:
    """Human-readable attempt identity that stays concise for singleton batches."""
    if batch_size == 1:
        return f"iteration {round_index}/{rounds}"
    return f"round {round_index}/{rounds} proposal {proposal_index}/{batch_size}"


def _fraction(text: str) -> float:
    """Stable hash of `text` mapped to [0, 1) — the repo's RNG-free randomness convention."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64
