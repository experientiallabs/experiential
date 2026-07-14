"""`wmh harness create`: budgeted search over harness deltas, gated by non-regression.

The loop: select a parent from the accepted pool (stepping-stone weighting — good scores favored,
already-expanded parents discounted), cluster the parent's failures into mechanisms, ask the
proposer for one `HarnessDelta` against the largest cluster, apply it atomically, score the child
closed-loop against the world model, and gate acceptance:

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
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import DEFAULT_K, ClosedLoopReport, evaluate_closed_loop
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta, apply_delta
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import SandboxUsage
from wmh.harness.mutate import propose_delta, render_evidence
from wmh.harness.runtime import TokenUsage, combine_usage
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
    """One search iteration, scored or dead, in order: the complete search history.

    Every iteration proposes exactly one delta. A dead iteration (every outcome but
    "scored") ends early: the proposal died before a full-split eval, the search moves
    straight to the next iteration, and the record here is what remains of it. Callers
    render these as records and plot points, so a run full of dead proposals shows its
    work instead of looking like it never iterated. `champion_score` is the champion's
    full-suite rate when the iteration resolved: the honest y-level for plotting a dead
    iteration (the line it failed to move).
    """

    iteration: int
    outcome: Literal["scored", "screened", "invalid", "unusable", "proposer_error"]
    candidate: str | None = None
    delta_id: str | None = None
    ops: list[str] = Field(default_factory=list)  # "replace prompt:main" style summaries
    reason: str | None = None
    score: float | None = None  # full-suite success rate; scored attempts only
    accepted: bool | None = None  # gate verdict; scored attempts only
    screen_child: float | None = None  # trigger-cluster means; screened attempts only
    screen_parent: float | None = None
    champion_score: float | None = None


@dataclass(frozen=True, slots=True)
class _DeadAttempt:
    """A generation member that died before its trigger-cluster screen."""

    record: IterationRecord
    note: str


@dataclass(frozen=True, slots=True)
class _GenerationCandidate:
    """One applied proposal waiting for its real trigger-cluster screen."""

    iteration: int
    parent: HarnessDoc
    parent_report: ClosedLoopReport
    trigger: FailureSignature
    delta: HarnessDelta
    child: HarnessDoc
    ops_summary: list[str]
    screen_tasks: list[TaskSpec]


@dataclass(frozen=True, slots=True)
class _ProposalContext:
    """Immutable inputs for one independently generated breadth proposal."""

    iteration: int
    parent: HarnessDoc
    parent_report: ClosedLoopReport
    trigger: FailureSignature
    evidence: str


@dataclass(frozen=True, slots=True)
class _ProposalResult:
    """One proposer outcome, including a non-fatal provider failure when present."""

    context: _ProposalContext
    delta: HarnessDelta | None
    error: str | None = None


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
    caller's "attack the biggest cluster" policy is a stable choice. A failing task whose verdicts
    carry no per-assertion detail (an unparseable judge reply) forms its own cluster.
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
    """Two-tier (plus optional held-out) acceptance; ties pass. Also audits `expected_effect`."""
    suite_delta = _suite_rate(child, suite) - _suite_rate(champion, suite)
    full_delta = child.success_rate - best_full
    holdout_delta = (
        child_holdout.success_rate - champion_holdout.success_rate
        if child_holdout is not None and champion_holdout is not None
        else None
    )
    failures: list[str] = []
    if suite_delta < -_TIE_EPS:
        failures.append(f"suite regressed by {-suite_delta:.3f}")
    if full_delta < -_TIE_EPS:
        failures.append(f"full split {child.success_rate:.3f} below best {best_full:.3f}")
    if holdout_delta is not None and holdout_delta < -_TIE_EPS:
        failures.append(f"held-out regressed by {-holdout_delta:.3f}")
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
        full_delta=full_delta,
        holdout_delta=holdout_delta,
        accepted=accepted,
        reason=reason,
    )


def create_harness(
    name: str,
    seed_doc: HarnessDoc,
    tasks: list[TaskSpec],
    world_model: WorldModel,
    agent_provider: Provider,
    meta_provider: Provider,
    judge: GoldJudge,
    *,
    iterations: int = 5,
    k: int = DEFAULT_K,
    holdout: list[TaskSpec] | None = None,
    confirm_narrow_vetoes: bool = True,
    harness_backend: Literal["local", "e2b"] = "local",
    eval_concurrency: int | None = None,
    iteration_concurrency: int = 1,
    e2b_template: str | None = None,
    on_progress: CreateProgress | None = None,
    on_note: Callable[[str], None] | None = None,
    on_iteration: Callable[[IterationRecord], None] | None = None,
    on_accept: Callable[[HarnessDoc, HarnessDelta, float], None] | None = None,
) -> CreateResult:
    """Search for a better harness under a fixed eval budget; the champion is renamed to `name`.

    Scores the seed first, then runs `iterations` propose-apply-SCREEN-score-gate
    iterations, one proposal each. ``iteration_concurrency`` groups proposals into bounded
    breadth generations: each proposal still gets the full parent, evidence, provider budget,
    and verdict-bearing history from earlier generations, while the independent proposer calls
    and trigger-cluster screens within the generation run concurrently. A dead iteration
    (unusable/invalid proposal, or one screened out on its own trigger cluster) ends early and
    cheaply: it is counted
    (`skipped`/`screened`), recorded in `iteration_records`, narrated via `on_note`, and
    the search moves straight to the next iteration. Never fatal. `on_accept` fires the
    moment a delta is accepted, with the new champion doc, its delta (verdict attached),
    and its full-suite score, so callers can persist champions in real time instead of
    waiting for the search to finish.

    Every rollout scores against the world-model simulation — the environment is always sim.
    `harness_backend` picks where the harness PROCESS executes: `local` (the default) runs it
    in/from this process exactly as before; `e2b` runs the real pi agent inside E2B sandboxes
    (pi-node seeds only — that harness's context management is the thing under search), with its
    tool calls still answered by the world model host-side. One `E2BSandboxPool` is shared across
    every `_score` wave so sandbox bootstraps amortize over the whole search. `eval_concurrency`
    is how many (task, attempt) cells run at once; `None` means the backend default — 1
    (sequential) for local, 0 (every cell at once, one pooled sandbox each) for e2b.
    `e2b_template` names a prebaked sandbox template (node 22 + the pi runner deps) so e2b
    rollouts skip bootstrap installs. Iteration concurrency greater than one requires the e2b
    backend: separate documents then share the thread-safe sandbox pool while retaining one
    sandbox per real (task, attempt) cell.

    Verification is staged by cost: a child is first SCREENED on its own trigger cluster (the
    2-3 failing tasks its delta claims to fix, k passes) — if the cluster did not improve over
    the parent, the delta is rejected and archived for a fraction of a full eval's cost, and no
    `on_progress` event fires. Held-out evals run only for children that pass tiers 1-2, so a
    bad delta costs at most one full-split eval. Every judged delta (screened, rejected, or
    accepted) is fed back to later generations as verdict-bearing history. Members of one breadth
    generation deliberately share the same immutable history snapshot so provider latency can
    overlap without fabricating verdicts that do not exist yet.

    Symmetrically, a REJECTION can be noise: with k passes over small tiers, one unlucky attempt
    can veto a genuine win. When `confirm_narrow_vetoes` is set, a delta that strictly won the
    full split but failed suite/holdout within `narrow_failing_tiers`' margin gets that tier
    re-measured — child AND champion, at 2k — and the re-measurement decides. The verdict records
    the retrial either way, so the archive shows which accepts needed confirmation.
    """
    if harness_backend not in ("local", "e2b"):
        raise ValueError(f"unknown harness_backend {harness_backend!r}; choose local or e2b")
    if iteration_concurrency < 1:
        raise ValueError("iteration_concurrency must be >= 1")
    if iteration_concurrency > 1 and harness_backend != "e2b":
        raise ValueError("iteration_concurrency > 1 requires harness_backend='e2b'")
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

        sandbox_pool = _Pool(template=e2b_template)

    try:

        def _note(message: str) -> None:
            # Narration for iterations that produce NO on_progress event (unusable/invalid/screened
            # proposals): without it a run whose proposals all fail looks like it never iterated.
            if on_note is not None:
                on_note(message)

        docs: dict[str, HarnessDoc] = {seed_doc.doc_hash: seed_doc}
        worker_usages: list[TokenUsage | None] = []
        worker_usage_lock = threading.Lock()
        reports: dict[str, ClosedLoopReport] = {}
        holdout_reports: dict[str, ClosedLoopReport] = {}
        archive = DeltaArchive(seed=seed_doc)
        children_counts: dict[str, int] = {}
        skipped = 0
        screened = 0
        confirmations = 0

        def _score(
            doc: HarnessDoc, split: list[TaskSpec], *, k_override: int | None = None
        ) -> ClosedLoopReport:
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
                runtime = doc.runtime(agent_provider, backend="e2b", e2b_pool=sandbox_pool)
            report = evaluate_closed_loop(
                split,
                world_model,
                agent_provider,
                judge,
                label=doc.name,
                k=k_eff,
                concurrency=concurrency,
                runtime=runtime,
            )
            # Tally the pi worker's self-metered tokens across every score wave (seed, screens,
            # full splits, holdout, confirmations): its LLM calls bypass the Provider, so this is
            # the only record. None on backends whose runtimes don't self-meter (local).
            with worker_usage_lock:
                worker_usages.append(report.worker_usage)
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

        def _screen_generation(
            candidates: list[_GenerationCandidate],
        ) -> dict[int, ClosedLoopReport]:
            """Run each generation member's full k-pass trigger screen concurrently."""
            screenable = [candidate for candidate in candidates if candidate.screen_tasks]
            if not screenable:
                return {}
            if len(screenable) == 1:
                candidate = screenable[0]
                return {candidate.iteration: _score(candidate.child, candidate.screen_tasks)}
            screen_reports: dict[int, ClosedLoopReport] = {}
            executor = ThreadPoolExecutor(max_workers=min(iteration_concurrency, len(screenable)))
            futures: dict[Future[ClosedLoopReport], _GenerationCandidate] = {
                executor.submit(_score, candidate.child, candidate.screen_tasks): candidate
                for candidate in screenable
            }
            try:
                for future in as_completed(futures):
                    candidate = futures[future]
                    screen_reports[candidate.iteration] = future.result()
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
            return screen_reports

        def _propose_generation(contexts: list[_ProposalContext]) -> dict[int, _ProposalResult]:
            """Run a generation's full-budget proposer calls against one history snapshot."""
            history = list(archive.deltas)

            def _propose(context: _ProposalContext) -> HarnessDelta | None:
                return propose_delta(
                    context.parent,
                    context.trigger,
                    context.evidence,
                    meta_provider,
                    history=history,
                )

            if len(contexts) == 1:
                context = contexts[0]
                try:
                    delta = _propose(context)
                except Exception as exc:  # noqa: BLE001 - provider/transport failure, by design
                    return {
                        context.iteration: _ProposalResult(
                            context=context,
                            delta=None,
                            error=str(exc),
                        )
                    }
                return {context.iteration: _ProposalResult(context=context, delta=delta)}

            results: dict[int, _ProposalResult] = {}
            executor = ThreadPoolExecutor(max_workers=min(iteration_concurrency, len(contexts)))
            futures: dict[Future[HarnessDelta | None], _ProposalContext] = {
                executor.submit(_propose, context): context for context in contexts
            }
            try:
                for future in as_completed(futures):
                    context = futures[future]
                    try:
                        delta = future.result()
                    except Exception as exc:  # noqa: BLE001 - one failed proposal is one dead turn
                        results[context.iteration] = _ProposalResult(
                            context=context,
                            delta=None,
                            error=str(exc),
                        )
                    else:
                        results[context.iteration] = _ProposalResult(
                            context=context,
                            delta=delta,
                        )
            finally:
                executor.shutdown(wait=True)
            return results

        for generation_start in range(1, iterations + 1, iteration_concurrency):
            generation_stop = min(iterations + 1, generation_start + iteration_concurrency)
            proposal_contexts: list[_ProposalContext] = []
            generation_size = generation_stop - generation_start
            for i in range(generation_start, generation_stop):
                parent_entry = select_parent(pool, children_counts, seed=i)
                parent = docs[parent_entry.doc_hash]
                parent_report = reports[parent_entry.doc_hash]
                # Count the expansion attempt up front so a parent whose proposals keep failing is
                # progressively discounted instead of re-selected every iteration.
                children_counts[parent.doc_hash] = children_counts.get(parent.doc_hash, 0) + 1

                clusters = cluster_failures(parent_report, tasks)
                trigger = (
                    clusters[0] if clusters else FailureSignature(mechanism=ALL_PASS_MECHANISM)
                )
                evidence = render_evidence(trigger, parent_report, tasks)
                if generation_size > 1:
                    slot = i - generation_start + 1
                    evidence += (
                        f"\n\nBreadth candidate {slot}/{generation_size}: independently explore "
                        "a distinct high-value approach. Other candidates in this generation "
                        "receive the same evidence and are evaluated independently."
                    )
                proposal_contexts.append(
                    _ProposalContext(
                        iteration=i,
                        parent=parent,
                        parent_report=parent_report,
                        trigger=trigger,
                        evidence=evidence,
                    )
                )

            proposal_results = _propose_generation(proposal_contexts)
            attempts: list[_DeadAttempt | _GenerationCandidate] = []
            for i in range(generation_start, generation_stop):
                proposal = proposal_results[i]
                parent = proposal.context.parent
                parent_report = proposal.context.parent_report
                trigger = proposal.context.trigger
                if proposal.error is not None:
                    skipped += 1
                    attempts.append(
                        _DeadAttempt(
                            record=IterationRecord(
                                iteration=i,
                                outcome="proposer_error",
                                reason=proposal.error,
                            ),
                            note=(
                                f"iteration {i}/{iterations}: proposer call failed "
                                f"({proposal.error}); skipped"
                            ),
                        )
                    )
                    continue
                delta = proposal.delta
                if delta is None:
                    skipped += 1
                    attempts.append(
                        _DeadAttempt(
                            record=IterationRecord(
                                iteration=i,
                                outcome="unusable",
                                reason="unparseable or truncated meta reply",
                            ),
                            note=(
                                f"iteration {i}/{iterations}: proposal unusable (unparseable or "
                                "truncated meta reply); skipped"
                            ),
                        )
                    )
                    continue
                ops_summary = [f"{op.op} {op.surface_id}" for op in delta.ops]
                if any(existing.delta_id == delta.delta_id for existing in archive.deltas):
                    # Pending generation members count too: evaluating a duplicate would spend a
                    # real screen to learn the same verdict as its identical sibling.
                    skipped += 1
                    attempts.append(
                        _DeadAttempt(
                            record=IterationRecord(
                                iteration=i,
                                outcome="invalid",
                                delta_id=delta.delta_id,
                                ops=ops_summary,
                                reason="duplicate of an already-proposed delta",
                            ),
                            note=(
                                f"iteration {i}/{iterations}: proposal duplicates an "
                                "already-proposed delta; skipped"
                            ),
                        )
                    )
                    continue
                try:
                    child = apply_delta(parent, delta, f"{name}-g{i}")
                except ValueError as exc:
                    delta.verdict = GateRecord(accepted=False, reason=f"invalid before eval: {exc}")
                    archive.deltas.append(delta)
                    skipped += 1
                    attempts.append(
                        _DeadAttempt(
                            record=IterationRecord(
                                iteration=i,
                                outcome="invalid",
                                delta_id=delta.delta_id,
                                ops=ops_summary,
                                reason=f"invalid before eval: {exc}",
                            ),
                            note=(
                                f"iteration {i}/{iterations}: delta invalid before eval ({exc}); "
                                "skipped"
                            ),
                        )
                    )
                    continue
                if harness_backend == "e2b" and child.runtime_kind() != "pi-node":
                    delta.verdict = GateRecord(
                        accepted=False,
                        reason=(
                            f"invalid before eval: runtime kind {child.runtime_kind()!r} cannot "
                            "run on harness_backend='e2b' (pi-node only)"
                        ),
                    )
                    archive.deltas.append(delta)
                    skipped += 1
                    attempts.append(
                        _DeadAttempt(
                            record=IterationRecord(
                                iteration=i,
                                outcome="invalid",
                                candidate=child.name,
                                delta_id=delta.delta_id,
                                ops=ops_summary,
                                reason=str(delta.verdict.reason),
                            ),
                            note=(
                                f"iteration {i}/{iterations}: delta abandoned the pi-node "
                                "runtime (e2b runs pi-node only); skipped"
                            ),
                        )
                    )
                    continue

                # Archive in deterministic iteration order. Its verdict is attached after all
                # independent screens complete, then becomes history for the next generation.
                archive.deltas.append(delta)
                attempts.append(
                    _GenerationCandidate(
                        iteration=i,
                        parent=parent,
                        parent_report=parent_report,
                        trigger=trigger,
                        delta=delta,
                        child=child,
                        ops_summary=ops_summary,
                        screen_tasks=[
                            task for task in tasks if task.task_id in set(trigger.task_ids)
                        ],
                    )
                )

            candidates = [
                attempt for attempt in attempts if isinstance(attempt, _GenerationCandidate)
            ]
            generation_screens = _screen_generation(candidates)

            # Resolve in iteration order. Screens overlapped, but gates and champion promotion
            # stay serial and deterministic, so every later verdict compares with the same
            # champion/suite state the ordinary search would expose at that boundary.
            for attempt in attempts:
                if isinstance(attempt, _DeadAttempt):
                    record = attempt.record.model_copy(
                        update={"champion_score": reports[champion_hash].success_rate}
                    )
                    _dead(record, attempt.note)
                    continue

                i = attempt.iteration
                parent = attempt.parent
                parent_report = attempt.parent_report
                trigger = attempt.trigger
                delta = attempt.delta
                child = attempt.child
                ops_summary = attempt.ops_summary
                champion_score = reports[champion_hash].success_rate
                screen_report = generation_screens.get(i)
                if screen_report is not None:
                    parent_mean = _suite_rate(parent_report, sorted(trigger.task_ids))
                    child_mean = _suite_rate(screen_report, sorted(trigger.task_ids))
                    if child_mean <= parent_mean + _TIE_EPS:
                        delta.verdict = GateRecord(
                            accepted=False,
                            reason=(
                                f"screened out: trigger cluster {child_mean:.2f} vs parent "
                                f"{parent_mean:.2f} over {len(attempt.screen_tasks)} task(s), "
                                f"k={k}; the delta did not improve its own target"
                            ),
                        )
                        screened += 1
                        _dead(
                            IterationRecord(
                                iteration=i,
                                outcome="screened",
                                candidate=child.name,
                                delta_id=delta.delta_id,
                                ops=ops_summary,
                                reason=str(delta.verdict.reason),
                                screen_child=child_mean,
                                screen_parent=parent_mean,
                                champion_score=champion_score,
                            ),
                            f"iteration {i}/{iterations}: screened out: trigger cluster "
                            f"{child_mean:.2f} vs parent {parent_mean:.2f}",
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
                # The held-out tier is measured for every candidate that could still be accepted
                # — including candidates whose suite/full veto is narrow enough for a
                # confirmation re-run. A confirmation may overturn a veto; it must never bypass
                # the held-out tier.
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
                    narrow_failing_tiers(
                        verdict,
                        k=k,
                        n_suite=len(suite),
                        n_holdout=len(holdout or []),
                    )
                    if confirm_narrow_vetoes
                    else None
                )
                if tiers:
                    confirmations += 1
                    confirmed_ok = True
                    notes: list[str] = []
                    for tier in tiers:
                        tier_tasks = (
                            [task for task in tasks if task.task_id in set(suite)]
                            if tier == "suite"
                            else list(holdout or [])
                        )
                        child_re = _score(child, tier_tasks, k_override=2 * k)
                        champ_re = _score(docs[champion_hash], tier_tasks, k_override=2 * k)
                        re_delta = child_re.success_rate - champ_re.success_rate
                        notes.append(f"{tier} re-measured at k={2 * k}: {re_delta:+.3f}")
                        if re_delta < -_TIE_EPS:
                            confirmed_ok = False
                    outcome = "veto overturned" if confirmed_ok else "regression confirmed"
                    verdict = GateRecord(
                        suite_delta=verdict.suite_delta,
                        full_delta=verdict.full_delta,
                        holdout_delta=verdict.holdout_delta,
                        accepted=confirmed_ok,
                        reason=(
                            f"confirmation re-run ({outcome}): {'; '.join(notes)} | initially: "
                            + verdict.reason
                        ),
                    )
                delta.verdict = verdict
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
                    outcome="scored",
                    candidate=child.name,
                    delta_id=delta.delta_id,
                    ops=ops_summary,
                    reason=str(verdict.reason),
                    score=child_report.success_rate,
                    accepted=verdict.accepted,
                    champion_score=reports[champion_hash].success_rate,
                )
                iteration_records.append(record)
                if on_iteration is not None:
                    on_iteration(record)
                if on_progress is not None:
                    on_progress(i, child.name, child_report.success_rate, verdict.accepted)

        best = docs[champion_hash].model_copy(update={"name": name, "version": 0})
        sandbox_usage = None
        if sandbox_pool is not None:
            sandbox_pool.close()  # idempotent; finalize lifetimes so the meter is complete
            sandbox_usage = sandbox_pool.usage()
        return CreateResult(
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
            worker_usage=combine_usage(worker_usages),
            sandbox_usage=sandbox_usage,
        )
    finally:
        if sandbox_pool is not None:
            sandbox_pool.close()


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


def _fraction(text: str) -> float:
    """Stable hash of `text` mapped to [0, 1) — the repo's RNG-free randomness convention."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64
