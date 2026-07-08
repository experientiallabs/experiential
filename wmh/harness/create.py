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
from collections.abc import Callable

from pydantic import BaseModel, Field

from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import DEFAULT_K, ClosedLoopReport, evaluate_closed_loop
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta, apply_delta
from wmh.harness.doc import HarnessDoc
from wmh.harness.mutate import propose_delta, render_evidence
from wmh.providers.base import Provider

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


class CreateResult(BaseModel):
    """What a create run produced: the champion, its score, and the full search record."""

    best: HarnessDoc
    best_score: float
    archive: DeltaArchive
    reports: dict[str, ClosedLoopReport] = Field(default_factory=dict)  # by doc_hash
    holdout_reports: dict[str, ClosedLoopReport] = Field(default_factory=dict)  # by doc_hash
    suite: list[str] = Field(default_factory=list)  # final regression suite (task ids)
    skipped: int = 0  # iterations lost to unusable or invalid proposals
    screened: int = 0  # deltas rejected at the cheap trigger-cluster screen (no full eval spent)
    confirmations: int = 0  # narrow vetoes retried at higher k (see `narrow_failing_tiers`)


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
    on_progress: CreateProgress | None = None,
) -> CreateResult:
    """Search for a better harness under a fixed eval budget; the champion is renamed to `name`.

    Scores the seed first, then runs `iterations` propose-apply-SCREEN-score-gate steps. An
    iteration whose proposal is unusable or fails atomic application is skipped (counted in
    `skipped`; an invalid delta is still archived with its rejection verdict), never fatal.

    Verification is staged by cost: a child is first SCREENED on its own trigger cluster (the
    2-3 failing tasks its delta claims to fix, k passes) — if the cluster did not improve over
    the parent, the delta is rejected and archived for a fraction of a full eval's cost, and no
    `on_progress` event fires. Held-out evals run only for children that pass tiers 1-2, so a
    bad delta costs at most one full-split eval. Every judged delta (screened, rejected, or
    accepted) is fed back to the proposer as history, so it iterates instead of re-proposing
    rejected ideas.

    Symmetrically, a REJECTION can be noise: with k passes over small tiers, one unlucky attempt
    can veto a genuine win. When `confirm_narrow_vetoes` is set, a delta that strictly won the
    full split but failed suite/holdout within `narrow_failing_tiers`' margin gets that tier
    re-measured — child AND champion, at 2k — and the re-measurement decides. The verdict records
    the retrial either way, so the archive shows which accepts needed confirmation.
    """
    docs: dict[str, HarnessDoc] = {seed_doc.doc_hash: seed_doc}
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
        return evaluate_closed_loop(
            split,
            world_model,
            agent_provider,
            judge,
            label=doc.name,
            k=k if k_override is None else k_override,
            runtime=doc.runtime(agent_provider),
        )

    seed_report = _score(seed_doc, tasks)
    reports[seed_doc.doc_hash] = seed_report
    if holdout:
        holdout_reports[seed_doc.doc_hash] = _score(seed_doc, holdout)
    if on_progress is not None:
        on_progress(0, seed_doc.name, seed_report.success_rate, True)

    pool = [
        PoolEntry(
            doc_hash=seed_doc.doc_hash, name=seed_doc.name, success_rate=seed_report.success_rate
        )
    ]
    champion_hash = seed_doc.doc_hash
    best_full = seed_report.success_rate
    # The regression suite: tasks the champion lineage has fully passed. Wins promote in on accept.
    suite = sorted(
        task.task_id
        for task in tasks
        if seed_report.per_task[task.task_id].success_rate >= 1.0 - _TIE_EPS
    )

    for i in range(1, iterations + 1):
        parent_entry = select_parent(pool, children_counts, seed=i)
        parent = docs[parent_entry.doc_hash]
        parent_report = reports[parent_entry.doc_hash]
        # Count the expansion attempt up front so a parent whose proposals keep failing is
        # progressively discounted instead of re-selected every iteration.
        children_counts[parent.doc_hash] = children_counts.get(parent.doc_hash, 0) + 1

        clusters = cluster_failures(parent_report, tasks)
        trigger = clusters[0] if clusters else FailureSignature(mechanism=ALL_PASS_MECHANISM)
        evidence = render_evidence(trigger, parent_report, tasks)
        delta = propose_delta(parent, trigger, evidence, meta_provider, history=archive.deltas)
        if delta is None:
            skipped += 1
            continue
        try:
            child = apply_delta(parent, delta, f"{name}-g{i}")
        except ValueError as exc:
            delta.verdict = GateRecord(accepted=False, reason=f"invalid before eval: {exc}")
            archive.deltas.append(delta)
            skipped += 1
            continue

        # Cheap screen: before a full-split eval, the delta must improve the very cluster it
        # was proposed to fix. A delta that cannot beat its parent on its own target is noise.
        screen_tasks = [t for t in tasks if t.task_id in set(trigger.task_ids)]
        if screen_tasks:
            screen_report = _score(child, screen_tasks)
            parent_mean = _suite_rate(parent_report, sorted(trigger.task_ids))
            child_mean = _suite_rate(screen_report, sorted(trigger.task_ids))
            if child_mean <= parent_mean + _TIE_EPS:
                delta.verdict = GateRecord(
                    accepted=False,
                    reason=(
                        f"screened out: trigger cluster {child_mean:.2f} vs parent "
                        f"{parent_mean:.2f} over {len(screen_tasks)} task(s), k={k} — "
                        "the delta did not improve its own target"
                    ),
                )
                archive.deltas.append(delta)
                screened += 1
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
            and narrow_failing_tiers(pre_verdict, k=k, n_suite=len(suite), n_holdout=0) is not None
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
                notes.append(f"{tier} re-measured at k={2 * k}: {re_delta:+.3f}")
                if re_delta < -_TIE_EPS:
                    confirmed_ok = False
            outcome = "veto overturned" if confirmed_ok else "regression confirmed"
            verdict = GateRecord(
                suite_delta=verdict.suite_delta,
                full_delta=verdict.full_delta,
                holdout_delta=verdict.holdout_delta,
                accepted=confirmed_ok,
                reason=f"confirmation re-run ({outcome}): {'; '.join(notes)} | initially: "
                + verdict.reason,
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
        if on_progress is not None:
            on_progress(i, child.name, child_report.success_rate, verdict.accepted)

    best = docs[champion_hash].model_copy(update={"name": name, "version": 0})
    return CreateResult(
        best=best,
        best_score=reports[champion_hash].success_rate,
        archive=archive,
        reports=reports,
        holdout_reports=holdout_reports,
        suite=suite,
        skipped=skipped,
        screened=screened,
        confirmations=confirmations,
    )


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
