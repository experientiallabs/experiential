"""Automatic rubric revision: a bounded loop driven by the problems judging reports.

Judging can show that a rubric, not an episode, is at fault. :func:`judged_problems` collects
those problems from saved judgments under one frozen judge:

- ``judge_reported``: the judge named a rubric problem (an ambiguous or insufficient criterion);
- ``calibration_miss``: an independently labelled calibration episode got another verdict;
- ``coverage_gap``: complete evidence was still unassessable because a criterion's evidence
  anchors were insufficient, a gap attributable to the rubric rather than the record;
- ``unjudged``: a target could not be judged at all, which blocks acceptance but is never sent
  to the reviser as a rubric fault;
- ``compatibility``: a world/rubric compatibility disagreement.

:func:`revise_until_stable` then runs at most ``max_rounds`` rounds. Each round sends the cited
problems (and every earlier rejected round) to one forced ``submit_rubric_revision`` draw, checks
the revised rubric with the rubric author's host checks, saves it as a new ``episode-rubric`` and
binds a new :class:`~exp.optimize.evaluation.variance.episodes.JudgeDefinition`: a new
experiment identity, so no old label is reused. It re-judges the calibration episodes and the
affected saved evidence under the new definition and accepts the revision only when no problem
remains, neither an old one nor a new one. Every round is recorded. When the round limit passes
without an accepted revision the family stays ``diagnostic`` under its original judge.

Labels made under any earlier definition stay valid only for that definition:
:func:`require_current_labels` refuses them under the revised judge.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelMessage, ModelRequest, ToolChoice
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.authoring_packet import (
    AuthoringPacket,
    bounded_reason,
    buy_author_draw,
    forced_tool_arguments,
    load_packet,
)
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence, JudgeDefinition, Verdict
from exp.optimize.evaluation.variance.judge import judge_episode
from exp.optimize.evaluation.variance.judged import (
    RUBRIC_PROBLEM,
    EpisodeJudge,
    JudgedSlot,
    judged_episode,
)
from exp.optimize.evaluation.variance.rubric_author import (
    RUBRIC_AUTHOR_PROMPT,
    Rubric,
    RubricAnswer,
    check_rubric_answer,
    judge_definition,
    load_rubric,
    save_rubric_content,
)
from exp.optimize.evaluation.variance.world_freeze import CompatibilityReview

RUBRIC_REVISION_PROTOCOL = "rubric-revision-v1"
"""Version of the revision prompt, answer schema, problem collection and acceptance rule."""

REVISION_TOOL = "submit_rubric_revision"
"""Name of the one tool a reviser must call."""

REVISION_TYPE = "rubric-revision"
"""Artifact type of a saved :class:`RubricRevision`."""

REVISION_PROMPT = (
    RUBRIC_AUTHOR_PROMPT
    + """
REVISION
You are revising a rubric that a frozen LLM judge already applied. The user message adds
current_rubric, problems and earlier_rounds to the packet. Each problem is evidence that the
rubric, not an episode, is at fault: the judge called a criterion ambiguous or insufficient, an
independently labelled calibration episode got another verdict, complete evidence was still
unassessable, or the rubric needs evidence the world does not record. Revise only what the cited
problems require, keep every requirement the sources support, and never weaken a requirement to
make a defective world or a failing calibration case pass. Earlier rejected rounds show changes
that did not resolve the problems or introduced new ones. For every problem_id say in resolutions
what you changed. If a problem cannot be resolved from the packet's sources, return status
needs_research. Call submit_rubric_revision exactly once.
"""
)

ProblemKind = Literal[
    "judge_reported", "calibration_miss", "coverage_gap", "unjudged", "compatibility"
]
"""Where a rubric problem was found."""


class RubricProblem(ContractModel):
    """One problem judging attributes to the rubric.

    Attributes:
        problem_id: ``<kind>:<subject>``; the same problem in a later round has the same id.
        kind: Where it was found.
        subject: The episode, calibration case or compatibility subject concerned.
        detail: What was found.
        sources: Saved assessments or packet locators that show it.
    """

    problem_id: str = Field(min_length=1)
    kind: ProblemKind
    subject: str = Field(min_length=1)
    detail: str = Field(min_length=1, max_length=2000)
    sources: tuple[str, ...] = ()


class CalibrationEpisode(ContractModel):
    """One independently labelled calibration case, instantiated as complete trusted evidence.

    Attributes:
        case_id: The rubric's calibration case.
        expected: The independently reviewed verdict.
        evidence: The episode evidence, bound to the judge's task and world.
    """

    case_id: str = Field(min_length=1)
    expected: Verdict
    evidence: EpisodeEvidence


class RevisionTargets(ContractModel):
    """What every revision is re-judged on.

    Attributes:
        calibration: Labelled calibration episodes.
        affected: Saved episode evidence whose labels the rubric problems touch.
    """

    calibration: tuple[CalibrationEpisode, ...] = ()
    affected: tuple[EpisodeEvidence, ...] = ()

    @property
    def identity(self) -> Sha256:
        """Digest of every target."""
        return sha256_json(self)


class Resolution(ContractModel):
    """How a revision addresses one cited problem.

    Attributes:
        problem_id: The cited problem.
        change: What was changed.
    """

    problem_id: str = Field(min_length=1)
    change: str = Field(min_length=1, max_length=1500)


class RevisionAnswer(RubricAnswer):
    """What a reviser answers: a rubric answer plus how each cited problem is resolved.

    Attributes:
        resolutions: One per cited problem.
    """

    resolutions: tuple[Resolution, ...] = ()


class RevisionRound(ContractModel):
    """One recorded revision round.

    Attributes:
        round: Round number, from 1.
        status: ``accepted``; ``rejected`` when problems remain or new ones appear; ``refused``
            when the revised rubric failed a host check; ``needs_research``; ``unavailable``.
        response: Saved reviser response, when one exists.
        rubric: The saved revised rubric, when one was checked.
        judge: Identity of the revised judge definition, when one was bound.
        cited: Problems the round was asked to resolve.
        remaining: Cited problems still present after re-judging.
        new: Problems the revision introduced.
        assessments: Saved assessments of every re-judged target.
        failure: Why the round produced no judged revision, when it did not.
    """

    round: int = Field(ge=1)
    status: Literal["accepted", "rejected", "refused", "needs_research", "unavailable"]
    response: ArtifactInput | None = None
    rubric: ArtifactInput | None = None
    judge: Sha256 | None = None
    cited: tuple[RubricProblem, ...]
    remaining: tuple[RubricProblem, ...] = ()
    new: tuple[RubricProblem, ...] = ()
    assessments: tuple[ArtifactInput, ...] = ()
    failure: str | None = None


class RubricRevision(ContractModel):
    """The whole bounded revision of one judge's rubric.

    Attributes:
        protocol: Revision protocol.
        packet_sha256: The packet the rubric was authored from.
        max_rounds: The configured round limit.
        initial: The judge definition whose judgments reported the problems.
        problems: Problems found under the initial judge.
        initial_assessments: Saved assessments of every target under the initial judge.
        rounds: Every round, in order.
        status: ``stable`` when no problem was found, ``revised`` when a round was accepted,
            ``needs_research`` when the reviser asked for evidence, otherwise ``diagnostic``.
        final: The judge definition to use: the accepted revision's, or the initial one.
    """

    protocol: str = RUBRIC_REVISION_PROTOCOL
    packet_sha256: Sha256
    max_rounds: int = Field(ge=1)
    initial: JudgeDefinition
    problems: tuple[RubricProblem, ...]
    initial_assessments: tuple[ArtifactInput, ...]
    rounds: tuple[RevisionRound, ...]
    status: Literal["stable", "revised", "needs_research", "diagnostic"]
    final: JudgeDefinition

    @property
    def trainable(self) -> bool:
        """Whether the final judge has no recorded rubric problem."""
        return self.status in ("stable", "revised")


def judged_problems(
    project: ProjectStore,
    judge: EpisodeJudge,
    targets: RevisionTargets,
    *,
    code_revision: str,
    compatibility: CompatibilityReview | None = None,
) -> tuple[tuple[RubricProblem, ...], tuple[ArtifactInput, ...]]:
    """Judge (or replay) every target under one judge and collect the rubric's problems.

    Args:
        project: Project holding the rubric, evidence and assessments.
        judge: The frozen judge and its admitted settings.
        targets: Calibration episodes and affected evidence.
        code_revision: Producing source identity.
        compatibility: The world/rubric compatibility review of this judge's rubric, if any.

    Returns:
        The problems, sorted by identity, and every saved assessment.
    """
    found: dict[str, RubricProblem] = {}
    references: list[ArtifactInput] = []

    def add(kind: ProblemKind, subject: str, detail: str, sources: tuple[str, ...]) -> None:
        key = f"{kind}:{subject}"
        if key not in found:
            found[key] = RubricProblem(
                problem_id=key, kind=kind, subject=subject, detail=detail[:2000], sources=sources
            )

    episodes: list[tuple[EpisodeEvidence, CalibrationEpisode | None]]
    episodes = [(case.evidence, case) for case in targets.calibration]
    episodes += [(evidence, None) for evidence in targets.affected]
    for evidence, case in episodes:
        reference, assessment = judge_episode(
            project,
            judge.definition,
            evidence,
            settings=judge.settings,
            code_revision=code_revision,
        )
        references.append(reference)
        sources = (reference.artifact_id,)
        subject = evidence.episode_id if case is None else case.case_id
        if assessment.status != "judged":
            add("unjudged", subject, assessment.failure or assessment.status, sources)
            continue
        if case is not None and assessment.verdict != case.expected:
            add(
                "calibration_miss",
                case.case_id,
                f"expected {case.expected}, judged {assessment.verdict}: "
                + "; ".join(f"{f.criterion_id} {f.status}" for f in assessment.findings),
                sources,
            )
        for item in assessment.limitations:
            if item.startswith(RUBRIC_PROBLEM):
                add("judge_reported", evidence.episode_id, item[len(RUBRIC_PROBLEM) :], sources)
        insufficient = [
            f.criterion_id for f in assessment.findings if f.status == "insufficient_evidence"
        ]
        if assessment.verdict == "unassessable" and evidence.complete and insufficient:
            add(
                "coverage_gap",
                evidence.episode_id,
                f"complete evidence left criteria {insufficient} insufficient",
                sources,
            )
    for row in () if compatibility is None else compatibility.disagreements:
        add(
            "compatibility", f"{row.kind}:{row.subject}", f"{row.world} / {row.rubric}", row.sources
        )
    return tuple(found[key] for key in sorted(found)), tuple(references)


def revise_until_stable(
    project: ProjectStore,
    settings: AuthorSettings,
    judge: EpisodeJudge,
    packet_reference: ArtifactInput,
    targets: RevisionTargets,
    *,
    identity: str,
    code_revision: str,
    max_rounds: int = 3,
    compatibility: Callable[[Rubric], CompatibilityReview] | None = None,
) -> RubricRevision:
    """Revise a judge's rubric until judging reports no problem, or the round limit passes.

    Args:
        project: Project holding the packet, rubric, evidence and every draw.
        settings: Frozen reviser model and allowance.
        judge: The frozen judge whose judgments report the problems.
        packet_reference: The saved authoring packet the rubric was authored from.
        targets: Calibration episodes and affected evidence, re-judged every round.
        identity: Stable identity of the revision.
        code_revision: Producing source identity.
        max_rounds: Most revision rounds before the family stays diagnostic.
        compatibility: Reviews a rubric against the frozen world, when one exists.

    Returns:
        The revision record, also saved as a ``rubric-revision`` artifact and replayed from it.

    Raises:
        ValueError: The judge's rubric was not authored from this packet, or a saved revision
            under this identity covers another judge or targets.
    """
    if max_rounds < 1:
        raise ValueError("a rubric revision needs at least one round")
    packet = load_packet(project, packet_reference)
    current = load_rubric(project, judge.definition.rubric)
    if current.packet_sha256 != packet.identity:
        raise ValueError("the judge's rubric was not authored from this packet")
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            saved = RubricRevision.model_validate_json(store.read_bytes(result_id, "revision.json"))
            if saved.initial != judge.definition or saved.max_rounds != max_rounds:
                raise ValueError(f"saved revision {result_id} covers another judge or limit")
            return saved
        review = None if compatibility is None else compatibility(current)
        problems, initial = judged_problems(
            project, judge, targets, code_revision=code_revision, compatibility=review
        )
        rounds: list[RevisionRound] = []
        final = judge.definition
        status: Literal["stable", "revised", "needs_research", "diagnostic"] = (
            "diagnostic" if problems else "stable"
        )
        for number in range(1, max_rounds + 1 if problems else 1):
            row = _round(
                project, settings, judge, packet, packet_reference, current, problems, rounds,
                targets, number, f"{identity}-round-{number}", code_revision, compatibility,
            )  # fmt: skip
            rounds.append(row)
            if row.status == "needs_research":
                status = "needs_research"
                break
            if row.status == "accepted" and row.rubric is not None:
                final = _definition(project, judge.definition, row.rubric)
                status = "revised"
                break
        revision = RubricRevision(
            packet_sha256=packet.identity,
            max_rounds=max_rounds,
            initial=judge.definition,
            problems=problems,
            initial_assessments=initial,
            rounds=tuple(rounds),
            status=status,
            final=final,
        )
        inputs = {packet_reference, judge.definition.rubric, *initial}
        for row in rounds:
            inputs |= {*row.assessments, *(r for r in (row.response, row.rubric) if r)}
        store.write_json(
            artifact_id=result_id,
            artifact_type=REVISION_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={"revision.json": revision},
        )
        return revision


def require_current_labels(
    project: ProjectStore, slots: Sequence[JudgedSlot], definition: JudgeDefinition
) -> None:
    """Refuse any label made under another judge definition than the current one.

    Args:
        project: Project holding the assessments.
        slots: Judged slots about to be used.
        definition: The current judge definition, for example a revision's ``final``.

    Raises:
        ValueError: A slot was judged under another definition; re-judge it instead.
    """
    for slot in slots:
        judged_episode(project, slot, definition.identity)


def _round(
    project: ProjectStore,
    settings: AuthorSettings,
    judge: EpisodeJudge,
    packet: AuthoringPacket,
    packet_reference: ArtifactInput,
    current: Rubric,
    problems: tuple[RubricProblem, ...],
    earlier: Sequence[RevisionRound],
    targets: RevisionTargets,
    number: int,
    identity: str,
    code_revision: str,
    compatibility: Callable[[Rubric], CompatibilityReview] | None,
) -> RevisionRound:
    """Run one revision round: draw, check, save, bind a new judge, re-judge and compare."""
    cited = tuple(problem for problem in problems if problem.kind != "unjudged")
    request = _request(packet, current, cited, earlier, settings.output_tokens)
    bought = buy_author_draw(
        project,
        settings,
        request,
        identity=identity,
        sources=(packet_reference, judge.definition.rubric),
        code_revision=code_revision,
    )
    if isinstance(bought, str):
        return RevisionRound(round=number, status="unavailable", cited=problems, failure=bought)
    response, raw = bought
    try:
        answer = RevisionAnswer.model_validate(forced_tool_arguments(raw, REVISION_TOOL))
        check_rubric_answer(answer, packet)
        missing = sorted({p.problem_id for p in cited} - {r.problem_id for r in answer.resolutions})
        if answer.status == "ready" and missing:
            raise ValueError(f"the revision resolves no change for problems {missing}")
        if answer.rubric is not None and answer.rubric == current.body:
            raise ValueError("the revised rubric is identical to the current rubric")
    except ValueError as error:
        return RevisionRound(
            round=number,
            status="refused",
            response=response,
            cited=problems,
            failure=bounded_reason(error),
        )
    if answer.rubric is None:
        return RevisionRound(
            round=number,
            status="needs_research",
            response=response,
            cited=problems,
            failure="; ".join(item.question for item in answer.research)[:2000],
        )
    revised = Rubric(
        family_id=current.family_id,
        packet_sha256=current.packet_sha256,
        body=answer.rubric,
        variants=current.variants,
        parent_sha256=current.identity,
        revision=current.revision + 1,
    )
    reference = save_rubric_content(project, revised, source=response, code_revision=code_revision)
    definition = _definition(project, judge.definition, reference)
    after, assessments = judged_problems(
        project,
        EpisodeJudge(definition=definition, settings=judge.settings),
        targets,
        code_revision=code_revision,
        compatibility=None if compatibility is None else compatibility(revised),
    )
    before = {problem.problem_id for problem in problems}
    remaining = tuple(problem for problem in after if problem.problem_id in before)
    new = tuple(problem for problem in after if problem.problem_id not in before)
    return RevisionRound(
        round=number,
        status="rejected" if after else "accepted",
        response=response,
        rubric=reference,
        judge=definition.identity,
        cited=problems,
        remaining=remaining,
        new=new,
        assessments=assessments,
    )


def _definition(
    project: ProjectStore, initial: JudgeDefinition, rubric: ArtifactInput
) -> JudgeDefinition:
    """Bind the initial judge's profile, task and world to a revised rubric.

    Raises:
        ValueError: The revised definition has the initial one's identity.
    """
    definition = judge_definition(
        project,
        initial.profile,
        rubric,
        task_sha256=initial.task_sha256,
        world_sha256=initial.world_sha256,
    )
    if definition.identity == initial.identity:
        raise ValueError("a revised rubric must start a new judge definition")
    return definition


def _request(
    packet: AuthoringPacket,
    current: Rubric,
    cited: tuple[RubricProblem, ...],
    earlier: Sequence[RevisionRound],
    output_tokens: int,
) -> ModelRequest:
    """Render one revision draw: the packet, the current rubric, the problems and past rounds."""
    payload: JsonObject = {
        "authoring_packet": packet.model_dump(mode="json"),
        "citable_locators": sorted(packet.locators()),
        "current_rubric": current.body.model_dump(mode="json"),
        "problems": [problem.model_dump(mode="json") for problem in cited],
        "earlier_rounds": [
            row.model_dump(mode="json", include={"round", "status", "remaining", "new", "failure"})
            for row in earlier
        ],
    }
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content=REVISION_PROMPT),
            ModelMessage(role="user", content=json.dumps(payload, sort_keys=True)),
        ),
        tools=(
            ToolSchema(
                name=REVISION_TOOL,
                description="Typed answer",
                input_schema=RevisionAnswer.model_json_schema(),
            ),
        ),
        tool_choice=ToolChoice(name=REVISION_TOOL),
        maximum_output_tokens=output_tokens,
    )


def revision_protocol_sha256() -> Sha256:
    """Return the identity of the revision prompt and answer schema."""
    return sha256_json(
        {
            "protocol": RUBRIC_REVISION_PROTOCOL,
            "prompt": REVISION_PROMPT,
            "schema": RevisionAnswer.model_json_schema(),
        }
    )


def revision_identity(judge: EpisodeJudge, targets: RevisionTargets, max_rounds: int) -> str:
    """Return a stable identity for revising one judge over one set of targets.

    Args:
        judge: The judge whose problems start the revision.
        targets: The re-judged targets.
        max_rounds: The round limit.

    Returns:
        A content-derived identity.
    """
    return stable_id(
        REVISION_TYPE,
        {
            "protocol": RUBRIC_REVISION_PROTOCOL,
            "judge": judge.identity,
            "targets": targets.identity,
            "max_rounds": max_rounds,
        },
    )


def load_revision(project: ProjectStore, reference: ArtifactInput) -> RubricRevision:
    """Read one exact saved revision record.

    Raises:
        ValueError: The reference is not the saved revision it pins.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        REVISION_TYPE
    ):
        raise ValueError(f"artifact {reference.artifact_id} is not a pinned {REVISION_TYPE}")
    return RubricRevision.model_validate_json(stored.payloads["revision.json"])
