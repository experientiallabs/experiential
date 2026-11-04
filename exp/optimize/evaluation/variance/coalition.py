"""The behavioral coalition authors read: every mining attempt, tier, selector, critic and contrast.

A rubric must rest on the combined behavior of many actors, not on one failed trajectory or one
oracle solution. :class:`CoalitionEvidence` gathers, for one decision family:

- **attempts**: every one of the G grouped attempts of every tier, with its operational outcome
  and its judged label. A label made by the family's current judge is ``judged``; one made by an
  earlier discovery judge is ``historical`` and never stands in for the current judge's rate;
- **behavior patterns**: distinct mistake patterns, recoveries and valid alternative solutions,
  each naming the attempts that show it. They are analysis (``hypothesis``) until a source
  establishes them;
- **tiers**: per-tier behavior. Each council actor is its own tier, so tiers are never pooled;
- **selector choices**: every point selector's chosen steps and rationale;
- **critic comparisons**: every critic estimate beside the judged rate it predicted;
- **contrast outcomes**: judged pass, fail and unscored counts per option and arm;
- **step labels**: optional TraceML step labels read from a ``traceml-labels`` artifact.

Every row keeps its source locator and evidence status. Tier labels and model names identify
who behaved how; they are never a criterion, which :meth:`CoalitionEvidence.actor_names` lets
the authors' host checks enforce. :meth:`CoalitionEvidence.check_exposure` refuses any attempt or
step label from a final-evaluation trace or lineage.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.behavior_differences import BehaviorDifferenceEvidence
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.judged import (
    JudgedCell,
    JudgedStudy,
    judged_episode,
    slot_coverage,
)
from exp.simulation.continuations import load_continuation_checkpoint

STEP_LABELS_TYPE = "traceml-labels"
"""Artifact type of an optional TraceML step-label reference.

Its ``labels.json`` holds ``traces``, each with a ``trace_id``, step labels (``index``,
``dominant``, ``uncertain``) and ``pivots`` and ``returns`` (``kind``, ``step``, ``triggers``).
The coalition reads that JSON shape directly rather than depending on the labeller's module.
"""

CoalitionStatus = Literal["observed", "judged", "historical", "hypothesis"]
"""How far a coalition row is established: a harness record, a label of the current judge, a
label of an earlier judge, or model analysis no source has tested."""

AttemptLabel = Literal["pass", "fail", "unassessable", "unjudged"]
"""An attempt's judged verdict, or ``unjudged`` when no judging was supplied."""

_ID = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"


class CoalitionAttempt(ContractModel):
    """One grouped actor attempt on a mining problem.

    Attributes:
        attempt_id: Stable identity, cited as ``attempt:<attempt_id>``.
        tier: The tier the attempt was sampled under.
        lineage: Leakage lineage of the attempted task; checked against the exposure plan.
        trace_id: Source trace identity, when the attempt replays one.
        outcome: Operational outcome of the run, never a success label.
        label: Judged verdict, or ``unjudged``.
        label_status: ``judged`` for the current judge, ``historical`` for another judge,
            ``observed`` when unjudged.
        judge: Judge definition that made the label.
        assessment: Saved assessment holding the label.
        steps: Saved checkpoints of the attempt.
        source: Locator of the saved attempt.
    """

    attempt_id: str = Field(pattern=_ID)
    tier: str = Field(min_length=1)
    lineage: str = Field(min_length=1)
    trace_id: str | None = None
    outcome: Literal["completed", "failed", "outcome_unknown"]
    label: AttemptLabel
    label_status: CoalitionStatus
    judge: Sha256 | None = None
    assessment: ArtifactInput | None = None
    steps: int = Field(ge=0)
    source: str = Field(min_length=1)


class BehaviorPattern(ContractModel):
    """One distinct behavior seen across attempts.

    Attributes:
        pattern_id: Stable identity, cited as ``pattern:<pattern_id>``.
        kind: A mistake, a recovery after a mistake, or a valid alternative solution.
        description: The behavior, without actor or model names.
        attempts: Attempts that show it.
        status: Evidence status; analysis is a ``hypothesis``.
        supported_by: Locators that show it.
    """

    pattern_id: str = Field(pattern=_ID)
    kind: Literal["mistake", "recovery", "valid_alternative"]
    description: str = Field(min_length=1, max_length=1500)
    attempts: tuple[str, ...] = Field(min_length=1)
    status: CoalitionStatus = "hypothesis"
    supported_by: tuple[str, ...] = ()


class TierBehavior(ContractModel):
    """How one tier's attempts were labelled.

    Attributes:
        tier: Tier label, cited as ``tier:<tier>``.
        model: Actor model of the tier.
        attempts: Attempts sampled.
        passed: Labelled pass.
        failed: Labelled fail.
        unscored: Unassessable or unjudged; never counted as either.
        patterns: Patterns any of its attempts show.
    """

    tier: str = Field(min_length=1)
    model: str = Field(min_length=1)
    attempts: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    unscored: int = Field(ge=0)
    patterns: tuple[str, ...] = ()


class SelectorChoice(ContractModel):
    """One point a selector chose, with its rationale.

    Attributes:
        tier: Tier whose attempt the selector read.
        draw: Selector draw number.
        step: The chosen step.
        decision: The decision the selector saw there.
        alternatives: Alternatives it proposed.
        rationale: Why it chose the step.
        status: Always ``hypothesis``.
    """

    tier: str
    draw: int = Field(ge=0)
    step: int = Field(ge=0)
    decision: str
    alternatives: tuple[str, ...] = ()
    rationale: str
    status: Literal["hypothesis"] = "hypothesis"


class CriticComparison(ContractModel):
    """One critic estimate beside the judged outcomes it predicted.

    Attributes:
        tier: Tier of the continuations.
        draw: Critic draw number.
        step: Estimated step.
        estimate: Predicted success probability.
        successes: Judged passes from the step.
        failures: Judged failures from the step.
        unknown: Unscored continuations from the step.
        judge: Judge definition that labelled the continuations.
        status: Always ``hypothesis`` for the estimate itself.
    """

    tier: str
    draw: int = Field(ge=0)
    step: int = Field(ge=0)
    estimate: float = Field(ge=0.0, le=1.0)
    successes: int = Field(ge=0)
    failures: int = Field(ge=0)
    unknown: int = Field(ge=0)
    judge: Sha256
    status: Literal["hypothesis"] = "hypothesis"

    @property
    def judged_rate(self) -> float | None:
        """Judged success rate over scored continuations, None when none was scored."""
        scored = self.successes + self.failures
        return self.successes / scored if scored else None


class ContrastOutcome(ContractModel):
    """Judged outcomes of one competing option at the same checkpoint.

    Attributes:
        contrast_id: Saved action-contrast result.
        arm: Which arm imposed the option.
        option: The imposed option.
        passed: Judged pass.
        failed: Judged fail.
        unscored: Planned cells without a pass or fail label.
        judge: Judge definition that labelled the cells.
        status: Always ``judged``.
    """

    contrast_id: str = Field(min_length=1)
    arm: Literal["baseline", "alternative"]
    option: str = Field(min_length=1)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    unscored: int = Field(ge=0)
    judge: Sha256
    status: Literal["judged"] = "judged"


class StepLabel(ContractModel):
    """One TraceML step label, analysis rather than authority.

    Attributes:
        attempt_id: Labelled attempt, when the label names one.
        trace_id: Labelled source trace, when the label names one.
        step: Labelled step.
        label: The step label, such as a pivot or a return.
        detail: Optional explanation.
        status: Always ``hypothesis``.
    """

    attempt_id: str | None = None
    trace_id: str | None = None
    step: int = Field(ge=0)
    label: str = Field(min_length=1, max_length=200)
    detail: str | None = Field(default=None, max_length=1500)
    status: Literal["hypothesis"] = "hypothesis"


class CoalitionEvidence(ContractModel):
    """The whole behavioral coalition of one decision family.

    Attributes:
        attempts: Every grouped attempt of every tier.
        patterns: Distinct mistakes, recoveries and valid alternatives.
        tiers: Per-tier behavior.
        selections: Every selector choice.
        critics: Every critic estimate with its judged outcome.
        contrasts: Judged outcomes per option and arm.
        step_labels: TraceML step labels, when supplied.
        references: Saved artifacts the coalition was read from.
        unavailable: Material that was not supplied, named rather than presumed.
    """

    attempts: tuple[CoalitionAttempt, ...] = ()
    patterns: tuple[BehaviorPattern, ...] = ()
    tiers: tuple[TierBehavior, ...] = ()
    selections: tuple[SelectorChoice, ...] = ()
    critics: tuple[CriticComparison, ...] = ()
    contrasts: tuple[ContrastOutcome, ...] = ()
    step_labels: tuple[StepLabel, ...] = ()
    behavior_differences: tuple[BehaviorDifferenceEvidence, ...] = ()
    references: tuple[ArtifactInput, ...] = ()
    unavailable: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Require distinct identities and rows that name known attempts and tiers."""
        attempts = [row.attempt_id for row in self.attempts]
        tiers = [row.tier for row in self.tiers]
        for name, identities in (
            ("attempt", attempts),
            ("pattern", [row.pattern_id for row in self.patterns]),
            ("tier", tiers),
        ):
            if len(set(identities)) != len(identities):
                raise ValueError(f"coalition {name} identities must be distinct")
        known = set(attempts)
        named = {a for row in self.patterns for a in row.attempts}
        named |= {row.attempt_id for row in self.step_labels if row.attempt_id is not None}
        named |= {
            a
            for group in self.behavior_differences
            for pair in group.pairs
            for a in (pair.good_attempt, pair.poor_attempt)
        }
        if named - known:
            raise ValueError(f"coalition rows name unknown attempts {sorted(named - known)}")
        sampled = {row.tier for row in (*self.attempts, *self.selections, *self.critics)}
        if self.tiers and sampled - set(tiers):
            raise ValueError(f"coalition rows name undeclared tiers {sorted(sampled - set(tiers))}")
        if any(row.attempt_id is None and row.trace_id is None for row in self.step_labels):
            raise ValueError("a step label names an attempt or a trace")
        return self

    def locators(self) -> frozenset[str]:
        """Every locator an author may cite for coalition rows."""
        return frozenset(
            {
                *(f"attempt:{row.attempt_id}" for row in self.attempts),
                *(f"pattern:{row.pattern_id}" for row in self.patterns),
                *(f"tier:{row.tier}" for row in self.tiers),
                *(f"selection:{row.tier}.{row.draw}.{row.step}" for row in self.selections),
                *(f"critic:{row.tier}.{row.draw}.{row.step}" for row in self.critics),
                *(f"contrast:{row.contrast_id}.{row.arm}" for row in self.contrasts),
                *(f"steplabel:{index}" for index in range(len(self.step_labels))),
                *(
                    f"behavior-diff:{row.source.artifact_id}/{row.group_id}"
                    for row in self.behavior_differences
                ),
            }
        )

    def actor_names(self) -> frozenset[str]:
        """Tier labels and model names, which no authored criterion may mention."""
        names = {row.tier for row in self.tiers} | {row.model for row in self.tiers}
        names |= {row.model.rsplit("/", 1)[-1] for row in self.tiers}
        names |= {row.tier for row in self.attempts}
        return frozenset(name for name in names if len(name) >= 3 and name != "default")

    def mentions(self, texts: Sequence[str]) -> tuple[str, ...]:
        """Return the tier labels and model names any text mentions, sorted.

        Args:
            texts: Authored criterion text.

        Returns:
            Every mentioned name, matched case-insensitively at word boundaries.
        """
        joined = "\n".join(texts).lower()
        return tuple(
            sorted(
                name
                for name in self.actor_names()
                if re.search(rf"(?<![\w.-]){re.escape(name.lower())}(?![\w-])", joined)
            )
        )

    def check_exposure(self, exposure: ExposurePlan) -> None:
        """Refuse attempts and step labels from final-evaluation traces or lineages.

        Every attempt's task lineage must be in the plan. A trace the plan does not list (an actor
        run rather than a source trace) is not a held-out source; a listed one is checked.

        Args:
            exposure: The frozen data roles.

        Raises:
            ValueError: A row's trace or lineage is reserved for final evaluation, or a lineage is
                unknown to the plan.
        """
        held: list[str] = []
        for row in self.attempts:
            role = exposure.roles.get(row.lineage)
            if role is None:
                raise ValueError(f"attempt {row.attempt_id} has lineage outside the exposure plan")
            if role == "final_evaluation":
                held.append(row.attempt_id)
        traces = [row.trace_id for row in (*self.attempts, *self.step_labels) if row.trace_id]
        held += [
            trace
            for trace in traces
            if exposure.roles.get(exposure.traces.get(trace, "")) == "final_evaluation"
        ]
        if held:
            raise ValueError(
                f"{len(held)} coalition rows come from final evaluation, for example "
                f"{sorted(held)[0]}; authors may read only world-learning and mining behavior"
            )


def mining_coalition(
    project: ProjectStore,
    reports: Sequence[ActorCriticReport],
    *,
    judged_groups: Mapping[str, JudgedStudy] | None = None,
    judge: str | None = None,
    patterns: Sequence[BehaviorPattern] = (),
    contrasts: Sequence[ContrastOutcome] = (),
    step_labels: ArtifactInput | None = None,
    behavior_differences: Sequence[BehaviorDifferenceEvidence] = (),
) -> CoalitionEvidence:
    """Gather the coalition from one error-mining report per tier.

    Args:
        project: Project holding the reports' saved groups and judgings.
        reports: One report per tier; each council actor is its own tier.
        judged_groups: Judging of each tier's group of attempts, keyed by tier.
        judge: The family's current judge definition; labels of any other are ``historical``.
        patterns: Mistake, recovery and alternative patterns found across the attempts.
        contrasts: Judged outcomes per option, from :func:`contrast_outcomes`.
        step_labels: Optional ``traceml-labels`` artifact.

    Returns:
        The coalition, naming every unsupplied material in ``unavailable``.

    Raises:
        ValueError: Two reports share a tier, or a judging covers another group.
    """
    store = project.artifacts
    judged_groups = judged_groups or {}
    attempts: list[CoalitionAttempt] = []
    tiers: list[tuple[str, str]] = []
    selections: list[SelectorChoice] = []
    critics: list[CriticComparison] = []
    references: set[ArtifactInput] = set()
    unavailable: list[str] = []
    for report in reports:
        snapshot = report.spec.initial.study.actor_model
        tier = report.spec.actor_tier if report.group_tier is None else report.group_tier.label
        if report.group_tier is not None:
            snapshot = report.group_tier.actor_model
        if tier in {name for name, _ in tiers}:
            raise ValueError(f"two reports share tier {tier}; each actor is its own tier")
        tiers.append((tier, f"{snapshot.provider}/{snapshot.model_id}"))
        study = load_study(store, report.group)
        start = load_continuation_checkpoint(store, report.spec.initial.study.checkpoint)
        judged = judged_groups.get(tier)
        if judged is not None and judged.study != report.group:
            raise ValueError(f"the judging supplied for tier {tier} covers another group")
        if judged is None:
            unavailable.append(f"tier {tier}: its group of attempts was not judged")
        slots = {} if judged is None else {slot.position: slot for slot in judged.branches}
        for branch in study.branches:
            slot = slots.get(branch.index)
            assessment = (
                None
                if slot is None or judged is None
                else (judged_episode(project, slot, judged.judge))
            )
            label: AttemptLabel = "unjudged"
            if judged is not None:
                label = "unassessable"
                if assessment is not None and assessment.status == "judged":
                    label = assessment.verdict
            attempts.append(
                CoalitionAttempt(
                    attempt_id=f"{tier}.g{branch.index}",
                    tier=tier,
                    lineage=start.task.lineage_group_id,
                    outcome=branch.status,
                    label=label,
                    label_status="observed"
                    if judged is None
                    else "judged"
                    if judged.judge == judge
                    else "historical",
                    judge=None if judged is None else judged.judge,
                    assessment=None
                    if slot is None or slot.episode is None
                    else (slot.episode.assessment),
                    steps=len(branch.checkpoints),
                    source=f"{report.group.artifact_id}#branch-{branch.index}",
                )
            )
        references.add(report.group)
        references |= {row.judged for row in report.points if row.judged is not None}
        selections += [
            SelectorChoice(
                tier=tier,
                draw=draw.draw,
                step=point.step,
                decision=point.decision,
                alternatives=point.alternatives,
                rationale=point.rationale,
            )
            for draw in report.judges
            if draw.selection is not None
            for point in draw.selection.points
        ]
        outcomes = {row.outcome.step: row.outcome for row in report.points}
        critics += [
            CriticComparison(
                tier=tier,
                draw=draw.draw,
                step=estimate.step,
                estimate=estimate.probability,
                successes=outcomes[estimate.step].successes,
                failures=outcomes[estimate.step].failures,
                unknown=outcomes[estimate.step].unknown,
                judge=report.spec.episode_judge.identity,
            )
            for draw in report.critics
            if draw.assessment is not None
            for estimate in draw.assessment.estimates
            if estimate.step in outcomes
        ]
    labels = () if step_labels is None else load_step_labels(project, step_labels)
    if step_labels is None:
        unavailable.append("no TraceML step labels were supplied")
    else:
        references.add(step_labels)
    if not contrasts:
        unavailable.append("no judged action-contrast outcomes were supplied")
    if not patterns:
        unavailable.append("no mistake, recovery or alternative patterns were supplied")
    references.update(group.source for group in behavior_differences)
    return CoalitionEvidence(
        attempts=tuple(attempts),
        behavior_differences=tuple(behavior_differences),
        patterns=tuple(patterns),
        tiers=tuple(_tier(name, model, attempts, patterns) for name, model in tiers),
        selections=tuple(selections),
        critics=tuple(critics),
        contrasts=tuple(contrasts),
        step_labels=labels,
        references=tuple(sorted(references, key=lambda ref: ref.artifact_id)),
        unavailable=tuple(unavailable),
    )


def contrast_outcomes(
    project: ProjectStore,
    cells: Sequence[JudgedCell],
    *,
    judge: str,
    options: Mapping[str, str],
) -> tuple[ContrastOutcome, ...]:
    """Count judged outcomes per contrast and arm, every unscored cell kept.

    Args:
        project: Project holding the assessments.
        cells: Judged cells of one or more contrasts.
        judge: The judge that labelled them.
        options: The option each arm imposed, keyed by ``baseline`` and ``alternative``.

    Returns:
        One outcome per (contrast, arm), in first-seen order.
    """
    groups: dict[tuple[str, Literal["baseline", "alternative"]], list[JudgedCell]] = {}
    for cell in cells:
        groups.setdefault((cell.contrast.artifact_id, cell.arm), []).append(cell)
    rows: list[ContrastOutcome] = []
    for (contrast, arm), members in groups.items():
        coverage = slot_coverage(project, [cell.slot for cell in members], judge)
        rows.append(
            ContrastOutcome(
                contrast_id=contrast,
                arm=arm,
                option=options[arm],
                passed=coverage.passed,
                failed=coverage.failed,
                unscored=coverage.unscored,
                judge=judge,
            )
        )
    return tuple(rows)


def load_step_labels(project: ProjectStore, reference: ArtifactInput) -> tuple[StepLabel, ...]:
    """Read TraceML step labels, pivots and returns from one pinned ``traceml-labels`` artifact.

    Args:
        project: Project holding the artifact.
        reference: The pinned artifact.

    Returns:
        One label per labelled step (its dominant action) and per pivot or return, each a
        ``hypothesis``; an uncertain step says so in its detail.

    Raises:
        ValueError: The reference is not a pinned ``traceml-labels`` artifact or its traces are
            malformed.
    """
    stored = project.artifacts.read(reference.artifact_id)
    manifest = stored.manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != STEP_LABELS_TYPE:
        raise ValueError(f"artifact {reference.artifact_id} is not a pinned {STEP_LABELS_TYPE}")
    content = json.loads(stored.payloads["labels.json"])
    traces = content.get("traces") if isinstance(content, dict) else None
    if not isinstance(traces, list):
        raise ValueError(f"{STEP_LABELS_TYPE} holds a list of labelled traces")
    labels: list[StepLabel] = []
    for trace in traces:
        if not isinstance(trace, dict) or not isinstance(trace.get("trace_id"), str):
            raise ValueError(f"every {STEP_LABELS_TYPE} trace is an object with a trace_id")
        trace_id = str(trace["trace_id"])
        for row in _rows(trace, "steps"):
            uncertain = " (uncertain)" if row.get("uncertain") else ""
            labels.append(
                StepLabel(
                    trace_id=trace_id,
                    step=_int(row, "index"),
                    label=str(row.get("dominant")),
                    detail=f"dominant action{uncertain}",
                )
            )
        for row in (*_rows(trace, "pivots"), *_rows(trace, "returns")):
            triggers = row.get("triggers")
            labels.append(
                StepLabel(
                    trace_id=trace_id,
                    step=_int(row, "step"),
                    label=str(row.get("kind")),
                    detail=f"triggers {triggers}"
                    + (" (uncertain)" if row.get("uncertain") else ""),
                )
            )
    return tuple(labels)


def _rows(trace: dict[str, object], key: str) -> list[dict[str, object]]:
    """Return one list of objects of a labelled trace, refusing anything else."""
    rows = trace.get(key, [])
    if not isinstance(rows, list):
        raise ValueError(f"{STEP_LABELS_TYPE} {key} must be a list of objects")
    objects: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{STEP_LABELS_TYPE} {key} must be a list of objects")
        objects.append({str(name): value for name, value in row.items()})
    return objects


def _int(row: dict[str, object], key: str) -> int:
    """Return one non-negative integer field of a label row."""
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{STEP_LABELS_TYPE} rows need a non-negative integer {key}")
    return value


def _tier(
    name: str, model: str, attempts: Sequence[CoalitionAttempt], patterns: Sequence[BehaviorPattern]
) -> TierBehavior:
    """Summarize one tier's labels and the patterns its attempts show."""
    own = [row for row in attempts if row.tier == name]
    ids = {row.attempt_id for row in own}
    return TierBehavior(
        tier=name,
        model=model,
        attempts=len(own),
        passed=sum(row.label == "pass" for row in own),
        failed=sum(row.label == "fail" for row in own),
        unscored=sum(row.label in ("unassessable", "unjudged") for row in own),
        patterns=tuple(row.pattern_id for row in patterns if ids & set(row.attempts)),
    )
