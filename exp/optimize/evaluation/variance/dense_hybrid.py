"""Hybrid dense reward: decision-model tags score every step, the LLM judge only flagged ones.

The decision model (Clef, through the unchanged tagging seam) answers the behavior bank about
every actor step of an episode (:func:`tag_episode_steps`). Each frozen dense tag lists the bank
questions that propose it in ``sources``. :func:`route_steps` reads those probabilities per step:

- **fired**: a source question is above the threshold outside the ambiguity band;
- **uncertain**: a source question is inside the band, the step has no row (tagging refused),
  or the tag has no source question at all, so the decision model cannot speak for it;
- **clear**: every source question of the tag is below the band on every step.

A tag is routed to the agentic dense judge (:func:`~.dense_judge.judge_dense`, every prompt
variant, strict majority) when any step fired or is uncertain for it, or when a linter or other
caller candidate proposes it; the judge then sees only routed candidates, with the flagged steps
as hints. A clear tag is decided cheaply as ``absent``. That is the only decision tags make
alone: a fired tag is a hypothesis, never a finding, so observing ``modified_evaluation`` (a test
was changed) never becomes an established weakening without the judge's cited obligation,
change, result and weakening, and a legitimate test correction or an alternative valid repair
stays acceptable.

Findings from both halves are merged and aggregated by the unchanged
:func:`~.dense_reward.aggregate`, so per-finding rewards still sum to the bounded episode total,
uncertain findings stay unscored, and a judge that returns nothing makes the episode
unavailable rather than clean. :class:`HybridRouting` records the routing, the judge draws
actually spent against the full judge's ceiling, and, when a full-judge reference is available,
the recall of established findings (:func:`routing_recall`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.behavior_pairs import behavior_trajectory
from exp.optimize.evaluation.variance.dense_candidates import DenseCandidate
from exp.optimize.evaluation.variance.dense_judge import DenseRewardJudge, judge_dense
from exp.optimize.evaluation.variance.dense_prompt import dense_model_calls
from exp.optimize.evaluation.variance.dense_reward import (
    DenseFinding,
    DenseRewardDefinition,
    DenseRewardExport,
    EpisodeDense,
    OutcomeReference,
    aggregate,
)
from exp.optimize.evaluation.variance.dense_spans import ActorTrajectory, SampledTokens
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence
from exp.optimize.evaluation.variance.reward_states import (
    FINDING_MACHINE,
    FindingState,
    Lifecycle,
    Refusal,
    Transition,
)
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tag_selection import DEFAULT_AMBIGUITY_BAND, DEFAULT_FIRE_THRESHOLD
from exp.simulation.mining.tagging import TagMatrix, TraceTagger, TraceTaggingError
from exp.simulation.mining.traceml import StepViewSpec, TraceTrajectory, step_units

DENSE_HYBRID_PROTOCOL = "dense-hybrid-v1"
"""Version of routing, cheap decisions and merging."""

DENSE_HYBRID_TYPE = "dense-hybrid-assessment"
"""Artifact type of a saved hybrid judgment (``dense.json`` and ``routing.json``)."""

StepStatus = Literal["fired", "uncertain", "clear"]
"""What the decision model said about one tag on one step."""


class RoutingSpec(ContractModel):
    """Frozen routing thresholds.

    Attributes:
        threshold: Probability strictly above which a source question fires.
        ambiguity_band: Closed band inside which an answer is uncertain.
        view: Step rendering the tags were bought over.
    """

    threshold: float = Field(default=DEFAULT_FIRE_THRESHOLD, gt=0.0, lt=1.0)
    ambiguity_band: tuple[float, float] = DEFAULT_AMBIGUITY_BAND
    view: StepViewSpec = StepViewSpec()


class StepTag(ContractModel):
    """One tag on one step.

    Attributes:
        locator: The step's actor event id.
        tag_id: The dense tag.
        status: Fired, uncertain or clear.
        probability: Largest source probability, or None without a row or source.
        question: The source question that set the probability.
    """

    locator: str
    tag_id: str
    status: StepStatus
    probability: float | None = None
    question: str | None = None


class TagRoute(ContractModel):
    """Where one dense tag went.

    Attributes:
        tag_id: The dense tag.
        route: ``judge`` or ``cleared`` (decided absent by the decision tags).
        reasons: Why it was routed.
        flagged_steps: Steps that fired or were uncertain.
    """

    tag_id: str
    route: Literal["judge", "cleared"]
    reasons: tuple[str, ...] = ()
    flagged_steps: tuple[str, ...] = ()


class HybridRouting(ContractModel):
    """Routing, cost and recall of one hybrid judgment.

    Attributes:
        protocol: Hybrid protocol.
        steps: Actor steps the decision model tagged.
        untagged_steps: Steps without a row (treated as uncertain).
        step_tags: Every (step, tag) status.
        routes: One route per frozen tag.
        judge_candidates: Candidates sent to the judge.
        cleared: Tags decided absent without the judge.
        judge_draws: Judge draws actually made (0 when nothing was routed).
        judge_draw_ceiling: Draws the full judge may make on this episode.
        recall: Share of a full judge's established findings whose tag was routed, when known.
        missed: Established tags the routing cleared, when known.
    """

    protocol: str = DENSE_HYBRID_PROTOCOL
    steps: int = Field(ge=0)
    untagged_steps: tuple[str, ...] = ()
    step_tags: tuple[StepTag, ...] = ()
    routes: tuple[TagRoute, ...]
    judge_candidates: int = Field(ge=0)
    cleared: tuple[str, ...] = ()
    judge_draws: int = Field(default=0, ge=0)
    judge_draw_ceiling: int = Field(ge=0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    missed: tuple[str, ...] = ()
    transitions: tuple[Transition, ...] = ()
    refusals: tuple[Refusal, ...] = ()


class HybridDense(ContractModel):
    """A saved hybrid judgment.

    Attributes:
        export: The merged, aggregated dense export.
        routing: Routing, cost and recall.
        judged: The dense judge's saved assessment, when the judge ran.
    """

    export: DenseRewardExport
    routing: HybridRouting
    judged: ArtifactInput | None = None


def episode_steps(evidence: EpisodeEvidence) -> TraceTrajectory:
    """The episode's actor steps (messages and tool calls) with event-id locators."""
    return behavior_trajectory(evidence, tier="dense", attempt=0, task=evidence.episode_id)


def tag_episode_steps(
    evidence: EpisodeEvidence,
    tagger: TraceTagger,
    bank: QuestionBank,
    spec: RoutingSpec | None = None,
) -> TagMatrix | None:
    """Buy the decision model's answers for every actor step, or None when tagging is refused."""
    units = step_units(episode_steps(evidence), (spec or RoutingSpec()).view)
    if not units:
        return None
    try:
        return tagger.tag(units, bank)
    except TraceTaggingError:
        return None


def route_steps(
    evidence: EpisodeEvidence,
    matrix: TagMatrix | None,
    definition: DenseRewardDefinition,
    spec: RoutingSpec | None = None,
    *,
    proposed: Sequence[DenseCandidate] = (),
) -> tuple[tuple[StepTag, ...], tuple[TagRoute, ...], tuple[str, ...]]:
    """Route every frozen tag from the decision model's step answers.

    Args:
        evidence: The episode.
        matrix: Step-view answers, or None when tagging failed (every step uncertain).
        definition: Frozen tags and their source questions.
        spec: Thresholds.
        proposed: Other candidates (linters); a proposed tag is routed to the judge.

    Returns:
        Every (step, tag) status, one route per tag, and the untagged steps.
    """
    spec = spec or RoutingSpec()
    low, high = spec.ambiguity_band
    trajectory = episode_steps(evidence)
    units = step_units(trajectory, spec.view)
    columns = {} if matrix is None else {q: i for i, q in enumerate(matrix.question_ids)}
    rows: dict[str, tuple[float, ...] | None] = {}
    for step, unit in zip(trajectory.steps, units, strict=True):
        try:
            rows[step.locator] = None if matrix is None else matrix.values_for(unit)
        except (TraceTaggingError, ValueError):
            rows[step.locator] = None
    untagged = tuple(locator for locator, row in rows.items() if row is None)
    by_linter = {row.tag_id for row in proposed}
    step_tags: list[StepTag] = []
    routes: list[TagRoute] = []
    for tag in definition.tags:
        sources = [q for q in tag.sources if q in columns]
        flagged: list[str] = []
        reasons: list[str] = []
        for locator, row in rows.items():
            probability: float | None = None
            question: str | None = None
            if row is None or not sources:
                status: StepStatus = "uncertain"
            else:
                value, question = max((row[columns[q]], q) for q in sources)
                probability = value
                if value > spec.threshold and not low <= value <= high:
                    status = "fired"
                elif low <= value <= high:
                    status = "uncertain"
                else:
                    status = "clear"
            step_tags.append(
                StepTag(
                    locator=locator,
                    tag_id=tag.tag_id,
                    status=status,
                    probability=probability,
                    question=question,
                )
            )
            if status != "clear":
                flagged.append(locator)
        if not sources:
            reasons.append("no decision-model question covers this tag")
        elif untagged:
            reasons.append(f"{len(untagged)} steps have no decision-model answer")
        if sources and any(s.tag_id == tag.tag_id and s.status == "fired" for s in step_tags):
            reasons.append("a source question fired")
        if sources and any(
            s.tag_id == tag.tag_id and s.status == "uncertain" and s.probability is not None
            for s in step_tags
        ):
            reasons.append("a source question is inside the ambiguity band")
        if tag.tag_id in by_linter:
            reasons.append("a linter or caller candidate proposes it")
        route: Literal["judge", "cleared"] = (
            "judge" if flagged or tag.tag_id in by_linter or not trajectory.steps else "cleared"
        )
        if not trajectory.steps:
            reasons.append("the episode has no actor step to tag")
        routes.append(
            TagRoute(
                tag_id=tag.tag_id,
                route=route,
                reasons=tuple(reasons) or ("every step is clear",),
                flagged_steps=tuple(flagged),
            )
        )
    return tuple(step_tags), tuple(routes), untagged


def routed_candidates(
    routes: Sequence[TagRoute],
    step_tags: Sequence[StepTag],
    *,
    proposed: Sequence[DenseCandidate] = (),
) -> tuple[DenseCandidate, ...]:
    """One candidate per routed tag (``tag:<id>``), hinting the flagged steps, plus proposals.

    A routed tag's candidate keeps the full judge's identity, so both judges answer the same
    candidate and their findings compare directly.
    """
    found: list[DenseCandidate] = []
    for route in routes:
        if route.route != "judge":
            continue
        fired = [
            s
            for s in step_tags
            if s.tag_id == route.tag_id and s.status == "fired" and s.question is not None
        ]
        found.append(
            DenseCandidate(
                candidate_id=f"tag:{route.tag_id}",
                tag_id=route.tag_id,
                origin="error_tag" if route.flagged_steps else "definition",
                proposition=fired[0].question if fired else None,
                tagger_status=("fired" if fired else "uncertain" if route.flagged_steps else None),
                hint_refs=route.flagged_steps[:32],
                note="decision-model routing: " + "; ".join(route.reasons),
            )
        )
    seen = {row.candidate_id for row in found}
    found.extend(row for row in proposed if row.candidate_id not in seen)
    return tuple(found)


def _cleared(definition: DenseRewardDefinition, routes: Sequence[TagRoute]) -> list[DenseFinding]:
    """Absent findings for every cleared tag, decided by the decision model alone."""
    return [
        DenseFinding(
            finding_id=f"tag:{route.tag_id}",
            tag_id=route.tag_id,
            candidate_ids=(f"tag:{route.tag_id}",),
            decision="absent",
            rationale="decision-model tags: every source question is below the ambiguity band "
            "on every actor step",
            axis=definition.tag(route.tag_id).axis,
            polarity=definition.tag(route.tag_id).polarity,
            severity=definition.tag(route.tag_id).severity,
        )
        for route in routes
        if route.route == "cleared"
    ]


def _empty() -> EpisodeDense:
    """Nothing scored."""
    return EpisodeDense(
        total=0.0, credit=0.0, penalty=0.0, decisive=False, scored=0, unscored=0, diagnostics=0
    )


def judge_hybrid(
    project: ProjectStore,
    judge: DenseRewardJudge,
    evidence: EpisodeEvidence,
    trajectory: ActorTrajectory,
    matrix: TagMatrix | None,
    *,
    code_revision: str,
    spec: RoutingSpec | None = None,
    proposed: Sequence[DenseCandidate] = (),
    outcome: OutcomeReference | None = None,
    sampled: SampledTokens | None = None,
) -> tuple[ArtifactInput, HybridDense]:
    """Score every step with decision tags, judge only routed tags, merge and aggregate.

    Args:
        project: Project holding the judge's draws and the saved hybrid judgment.
        judge: Frozen dense definition and admitted settings.
        evidence: The complete episode evidence.
        trajectory: The actor trajectory spans index.
        matrix: The decision model's step answers (:func:`tag_episode_steps`), or None.
        code_revision: Producing source identity.
        spec: Routing thresholds.
        proposed: Linter or other candidates; their tags are always judged.
        outcome: The separate outcome reward, attached only.
        sampled: The actor's sampled tokens, when bound.

    Returns:
        The saved reference and the hybrid judgment; replayed without calls when saved.
    """
    spec = spec or RoutingSpec()
    definition = judge.definition
    step_tags, routes, untagged = route_steps(evidence, matrix, definition, spec, proposed=proposed)
    candidates = routed_candidates(routes, step_tags, proposed=proposed)
    identity = stable_id(
        DENSE_HYBRID_TYPE,
        {
            "protocol": DENSE_HYBRID_PROTOCOL,
            "definition": definition.identity,
            "evidence": evidence.identity,
            "matrix": None if matrix is None else sha256_json(matrix),
            "spec": sha256_json(spec),
            "candidates": sha256_json([row.model_dump(mode="json") for row in candidates]),
            "trajectory": sha256_json(trajectory),
            "sampled": None if sampled is None else sha256_json(sampled),
        },
    )
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        if store.exists(identity):
            reference = artifact_input(store.read(identity).manifest)
            saved = HybridDense.model_validate_json(store.read_bytes(identity, "hybrid.json"))
            if saved.export.evidence_sha256 != evidence.identity:
                raise ValueError(f"saved hybrid judgment {identity} judges something else")
            return reference, saved.model_copy(
                update={"export": saved.export.model_copy(update={"outcome": outcome})}
            )
        judged_ref: ArtifactInput | None = None
        judged: DenseRewardExport | None = None
        if candidates:
            judged_ref, judged = judge_dense(
                project,
                judge,
                evidence,
                candidates,
                trajectory,
                code_revision=code_revision,
                sampled=sampled,
            )
        draws = 0 if judged is None else sum(len(v.responses) for v in judged.variants)
        routing = HybridRouting(
            steps=len(episode_steps(evidence).steps),
            untagged_steps=untagged,
            step_tags=step_tags,
            routes=routes,
            judge_candidates=len(candidates),
            cleared=tuple(r.tag_id for r in routes if r.route == "cleared"),
            judge_draws=draws,
            judge_draw_ceiling=dense_model_calls(definition.profile),
        )
        available = judged is None or judged.status == "judged"
        findings: tuple[DenseFinding, ...] = ()
        if available:
            merged = [*(() if judged is None else judged.findings), *_cleared(definition, routes)]
            order = {event.event_id: event.sequence for event in evidence.events}
            big = len(order) + 1
            merged.sort(
                key=lambda row: min((order.get(r, big) for r in row.observed_refs), default=big)
            )
            findings, episode = aggregate(merged, definition)
        else:
            episode = _empty()
        states = finding_lifecycle(
            tagged=matrix is not None, routes=routes, candidates=candidates, findings=findings
        )
        routing = routing.model_copy(
            update={"transitions": tuple(states.transitions), "refusals": tuple(states.refusals)}
        )
        export = DenseRewardExport(
            episode_id=evidence.episode_id,
            evidence_sha256=evidence.identity,
            definition_sha256=definition.identity,
            status="judged" if available else "unavailable",
            findings=findings,
            episode=episode,
            trajectory=trajectory.kind,
            sampled_tokens=None if sampled is None else sampled.binding,
            granularity="action" if judged is None else judged.granularity,
            variants=() if judged is None else judged.variants,
            failure=None if available else (judged.failure if judged else None),
        )
        result = HybridDense(export=export, routing=routing, judged=judged_ref)
        manifest = store.write_json(
            artifact_id=identity,
            artifact_type=DENSE_HYBRID_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=() if judged_ref is None else (judged_ref,),
            ),
            files={"hybrid.json": result},
        )
        return artifact_input(manifest), result.model_copy(
            update={"export": export.model_copy(update={"outcome": outcome})}
        )


def finding_lifecycle(
    *,
    tagged: bool,
    routes: Sequence[TagRoute],
    candidates: Sequence[DenseCandidate],
    findings: Sequence[DenseFinding],
) -> Lifecycle[FindingState]:
    """Replay one hybrid judgment through :data:`~.reward_states.FINDING_MACHINE`.

    Every frozen tag and every proposed candidate starts as a candidate, is tagged (when the
    decision model answered), then cleared or routed. Each finding's decision is the next state,
    and aggregation moves it to scored or unscored. A routed candidate with no finding (the
    judge was unavailable) becomes uncertain and then unscored, and a cleared one absent and
    then unscored, since an unavailable episode scores nothing. Any move the table forbids is
    refused and recorded, never applied.
    """
    states = FINDING_MACHINE.lifecycle()
    routed = {row.candidate_id for row in candidates}
    cleared = {f"tag:{row.tag_id}" for row in routes if row.route == "cleared"}
    subjects = [f"tag:{row.tag_id}" for row in routes]
    subjects += [c for c in sorted(routed) if c not in subjects]
    decided = {
        candidate: row for row in findings for candidate in (row.finding_id, *row.candidate_ids)
    }
    for subject in subjects:
        states.start(subject)
        if tagged:
            states.move(subject, FindingState.TAGGED, "decision-model step tags answered")
        if subject in cleared:
            states.move(subject, FindingState.CLEARED, "every step clear")
        else:
            states.move(subject, FindingState.ROUTED, "flagged, uncertain or proposed")
        row = decided.get(subject)
        if row is None and subject in cleared:
            states.move(subject, FindingState.ABSENT, "every step clear")
            states.move(subject, FindingState.UNSCORED, "the episode is unavailable")
            continue
        if row is None:
            states.move(subject, FindingState.UNCERTAIN, "no judged finding")
            states.move(subject, FindingState.UNSCORED, "the judge returned nothing")
            continue
        states.move(subject, FindingState(row.decision), row.rationale[:200])
        if row.scored:
            states.move(subject, FindingState.SCORED, f"reward {row.reward}")
        else:
            states.move(subject, FindingState.UNSCORED, row.unscored_reason or "unscored")
    return states


def routing_recall(routing: HybridRouting, full: DenseRewardExport) -> HybridRouting:
    """Record how many of a full judge's established findings the routing sent to the judge.

    Args:
        routing: The hybrid routing of the same episode.
        full: A full dense judgment (every tag a candidate), for example an audit sample.

    Returns:
        The routing with ``recall`` (None when the full judge established nothing) and
        ``missed``, the established tags the decision tags cleared.
    """
    established = sorted({f.tag_id for f in full.findings if f.decision == "established"})
    if not established:
        return routing.model_copy(update={"recall": None, "missed": ()})
    cleared = set(routing.cleared)
    missed = tuple(tag for tag in established if tag in cleared)
    return routing.model_copy(
        update={"recall": 1.0 - len(missed) / len(established), "missed": missed}
    )
