"""Astra's review of one checkpoint: a bounded summary plus sampled evidence, cited answers only.

Astra is the pipeline's open-weight tag author (a ``glm-*-flash`` or ``deepseek-*`` alias). At a
checkpoint it reads the statistics code computed (flagged subjects first, at most
``max_subjects``; the halts; at most ``max_documents`` sampled evidence documents), never the raw
corpus, so one review is one bounded role draw per stance through the shared buy-or-replay role
runner (:func:`~.family_stages.run_role`).

The answer is forced (:class:`ReviewAnswer`). The host checks structure and evidence only, never
meaning: every citation must name a subject of this checkpoint and one of its statistics or flags;
a proposal's kind must be one this checkpoint makes and its targets subjects of the kinds that
kind may name; a retirement or bar revision must target a flagged subject; a question merge must
name a flagged near-duplicate pair and the question to keep; a family split must target a family
flagged ``mixed`` and name separating tags the statistics list for it; a family merge must name a
flagged indistinct pair; a drift flag must target a subject the statistics flagged; a halt
finding must cite a flag. A refused draft goes back through the runner's repair turn; a stance
whose answer never passes is recorded as a failure, and the checkpoint keeps its statistics.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue
from exp.common.project import ProjectStore
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.evidence_tools import EvidenceToolProfile
from exp.optimize.evaluation.variance.family_stages import run_role
from exp.optimize.evaluation.variance.rubric_evolution import AUTHOR_ALIASES
from exp.optimize.evaluation.variance.tag_monitor_contracts import (
    CHECKPOINT_NUMBERS,
    PROPOSAL_KINDS,
    TARGET_KINDS,
    CheckpointStats,
    Citation,
    Finding,
    MonitorSpec,
    ProposalKind,
    ReviewRecord,
    SubjectStats,
)

REVIEW_TOOL = "submit_tag_monitor_review"
"""Astra's forced answer tool."""

REVIEW_STANCES = {
    "statistics_first": (
        "Start from the flagged statistics. For each, decide from the numbers and the sampled "
        "evidence whether it is a real problem for the tags (a question that cannot split the "
        "data, two questions asking the same thing, a family that mixes environments, a meta-rule "
        "that cannot discriminate, a tag the decision model and the judge read differently, a tag "
        "drifting with reward). Say nothing about a flag you cannot support."
    ),
    "consequence_first": (
        "Start from what the next phase will consume: environments, rubrics or training. Raise "
        "the findings whose tags would corrupt that phase if left as they are, and propose the "
        "smallest change the owning path could make."
    ),
}
"""Review stances by name."""

INSTRUCTION = """You are Astra, the tag author of a decision-variance pipeline, monitoring the tags
you authored at one checkpoint. The documents hold statistics computed by code from answers the
decision model (Clef) already gave, and sampled evidence rows. You judge what the statistics mean.

Answer with findings and proposals. Every finding and proposal cites at least one statistic as
{subject, stat}: stat is a value name of that subject, or flag:<name> for one of its flags. Cite
only subjects and statistics shown. Proposals are never applied by you; each goes to the path
that owns the change. Allowed proposal kinds at this checkpoint are listed in the summary.

- retire_question / rewrite_question: target flagged questions (rewrite gives the new text).
- merge_questions: target two near-duplicate questions; set keep to the one to keep.
- split_family: target a family flagged mixed; refs lists separating tags (tag:...) from its refs.
- merge_families: target the two families of an indistinct family_pair.
- revise_bar_rule: target a flagged bar_rule.
- flag_drift: target a drifting or disagreeing tag.
Use severity halt only for a finding that must block the next phase, citing a flag."""


class AstraSettings(ContractModel):
    """Astra's frozen review settings.

    Attributes:
        author: The open-weight author alias and its allowance.
        profile: How each draw reads its evidence; one forced response by default.
        stances: Review stances, each one draw; their accepted answers are pooled.
    """

    author: AuthorSettings
    profile: EvidenceToolProfile = EvidenceToolProfile()
    stances: tuple[str, ...] = Field(default=("statistics_first",), min_length=1)

    @model_validator(mode="after")
    def _open_weight(self) -> AstraSettings:
        """Require an open-weight author alias and known stances."""
        if AUTHOR_ALIASES.fullmatch(self.author.model_alias) is None:
            raise ValueError(f"Astra must be an open-weight author, not {self.author.model_alias}")
        if set(self.stances) - set(REVIEW_STANCES):
            raise ValueError(f"review stances must be members of {sorted(REVIEW_STANCES)}")
        return self


class TagMonitorSettings(ContractModel):
    """The settings of one monitor checkpoint (a front-door slot or the epoch hook).

    Attributes:
        astra: Astra's review settings, or None to record statistics only.
        spec: Thresholds and bounds.
    """

    astra: AstraSettings | None = None
    spec: MonitorSpec = MonitorSpec()


class FindingAnswer(ContractModel):
    """One finding as Astra writes it.

    Attributes:
        subject: What it is about.
        statement: The conclusion.
        severity: ``info``, ``warn`` or ``halt``.
        cites: Statistics it rests on.
    """

    subject: str = Field(min_length=1, max_length=512)
    statement: str = Field(min_length=1, max_length=1500)
    severity: str = Field(default="warn", pattern="^(info|warn|halt)$")
    cites: list[Citation] = Field(min_length=1, max_length=32)


class ProposalAnswer(ContractModel):
    """One proposal as Astra writes it.

    Attributes:
        kind: The proposal kind.
        targets: Subjects it names.
        keep: For a question merge, the question to keep.
        rewrite: For a rewrite, the proposed question text.
        refs: Supporting subjects named in the statistics (a split's separating tags).
        rationale: Why.
        cites: Statistics it rests on.
    """

    kind: ProposalKind
    targets: list[str] = Field(min_length=1, max_length=16)
    keep: str | None = None
    rewrite: str | None = Field(default=None, max_length=1200)
    refs: list[str] = Field(default_factory=list, max_length=16)
    rationale: str = Field(min_length=1, max_length=2000)
    cites: list[Citation] = Field(min_length=1, max_length=32)


class ReviewAnswer(ContractModel):
    """Astra's forced answer.

    Attributes:
        findings: Cited findings.
        proposals: Cited, typed proposals.
    """

    findings: list[FindingAnswer] = Field(default_factory=list, max_length=64)
    proposals: list[ProposalAnswer] = Field(default_factory=list, max_length=64)


def review_costs(project: ProjectStore, settings: AstraSettings) -> tuple[CostComponent, ...]:
    """Price one checkpoint review: every stance's draw at its profile's model-call budget."""
    draws = len(settings.stances) * settings.profile.budget.model_calls
    return tuple(row.scaled(draws) for row in evaluation_costs(project, settings.author))


def review_documents(stats: CheckpointStats, spec: MonitorSpec) -> dict[str, JsonValue]:
    """Return the bounded packet: summary, subjects (flagged first), halts, sampled evidence."""
    flagged = [row for row in stats.subjects if row.flags]
    quiet = [row for row in stats.subjects if not row.flags and row.kind in ("bank", "family")]
    shown = (flagged + quiet)[: spec.max_subjects]
    counts: dict[str, int] = {}
    for row in stats.subjects:
        for flag in row.flags:
            counts[flag] = counts.get(flag, 0) + 1
    summary: JsonObject = {
        "checkpoint": stats.checkpoint,
        "number": CHECKPOINT_NUMBERS[stats.checkpoint],
        "scope": stats.scope,
        "subjects": len(stats.subjects),
        "flagged": len(flagged),
        "shown": len(shown),
        "flag_counts": dict(sorted(counts.items())),
        "allowed_proposals": list(PROPOSAL_KINDS[stats.checkpoint]),
    }
    documents: dict[str, JsonValue] = {
        "summary": summary,
        "subjects": [row.model_dump(mode="json") for row in shown],
        "halts": [row.model_dump(mode="json") for row in stats.halts],
    }
    for key in sorted(stats.evidence)[: spec.max_documents]:
        documents[key] = stats.evidence[key]
    return documents


def check_review(answer: JsonObject, stats: CheckpointStats) -> str | None:
    """Return why Astra's answer is refused, or None when every citation and type holds."""
    try:
        parsed = ReviewAnswer.model_validate(answer)
    except ValueError as error:
        return f"the answer does not match the review schema: {error}"
    subjects = {row.subject: row for row in stats.subjects}
    problems: list[str] = []
    for finding in parsed.findings:
        problems.extend(_citations(finding.cites, subjects))
        if finding.subject not in subjects:
            problems.append(f"finding subject {finding.subject} is not a subject here")
        if finding.severity == "halt" and not any(
            c.stat.startswith("flag:") for c in finding.cites
        ):
            problems.append(f"halt finding on {finding.subject} cites no flag")
    allowed = PROPOSAL_KINDS[stats.checkpoint]
    for proposal in parsed.proposals:
        if proposal.kind not in allowed:
            problems.append(f"{proposal.kind} is not a proposal kind at {stats.checkpoint}")
            continue
        problems.extend(_citations(proposal.cites, subjects))
        problems.extend(_proposal(proposal, subjects, stats))
    return "; ".join(problems[:12]) or None


def _citations(cites: Sequence[Citation], subjects: dict[str, SubjectStats]) -> list[str]:
    """Problems with citations naming unknown subjects or statistics."""
    problems = []
    for cite in cites:
        row = subjects.get(cite.subject)
        if row is None:
            problems.append(f"cited subject {cite.subject} is not a subject here")
        elif not row.has(cite.stat):
            problems.append(f"{cite.subject} has no statistic or flag {cite.stat}")
    return problems


def _proposal(
    proposal: ProposalAnswer, subjects: dict[str, SubjectStats], stats: CheckpointStats
) -> list[str]:
    """Problems with one proposal's targets and supporting evidence."""
    kinds = TARGET_KINDS[proposal.kind]
    rows = [subjects.get(target) for target in proposal.targets]
    problems = [
        f"{proposal.kind} target {target} is not a subject of kind {sorted(kinds)}"
        for target, row in zip(proposal.targets, rows, strict=True)
        if row is None or row.kind not in kinds
    ]
    if problems:
        return problems
    found = [row for row in rows if row is not None]
    if proposal.kind in ("retire_question", "revise_bar_rule", "flag_drift"):
        problems.extend(
            f"{proposal.kind} target {row.subject} carries no flag"
            for row in found
            if not row.flags
        )
    if proposal.kind == "rewrite_question" and not proposal.rewrite:
        problems.append("rewrite_question gives no rewritten text")
    if proposal.kind == "merge_questions":
        left = found[0]
        if len(found) != 2 or found[1].subject not in left.refs:
            problems.append("merge_questions must name two questions of one near-duplicate pair")
        if proposal.keep not in proposal.targets:
            problems.append("merge_questions must keep one of its targets")
    if proposal.kind == "split_family":
        family = found[0]
        if len(found) != 1 or "mixed" not in family.flags:
            problems.append("split_family must target one family flagged mixed")
        elif not proposal.refs or not set(proposal.refs) <= set(family.refs):
            problems.append(
                f"split_family must name separating tags from {family.subject}'s refs "
                f"{list(family.refs)}"
            )
    if proposal.kind == "merge_families":
        pair = {row.subject for row in found}
        indistinct = any(
            row.kind == "family_pair" and set(row.refs) == pair and "indistinct" in row.flags
            for row in stats.subjects
        )
        if len(found) != 2 or not indistinct:
            problems.append("merge_families must name the two families of an indistinct pair")
    return problems


def run_review(
    project: ProjectStore,
    settings: AstraSettings,
    stats: CheckpointStats,
    spec: MonitorSpec,
    *,
    code_revision: str,
) -> tuple[ReviewRecord, list[ProposalAnswer]]:
    """Run, or replay, every stance's draw and pool the accepted answers.

    Returns:
        The review record (findings accepted, proposals left empty for routing) and the accepted
        proposal answers, deduplicated by kind and targets.
    """
    documents = review_documents(stats, spec)
    overview: JsonObject = {"checkpoint": stats.checkpoint, "scope": stats.scope}
    schema = ToolSchema(
        name=REVIEW_TOOL,
        description="Submit cited findings and typed proposals.",
        input_schema=ReviewAnswer.model_json_schema(),
    )
    transcripts = []
    failures: dict[str, str] = {}
    findings: dict[tuple[str, str], Finding] = {}
    proposals: dict[tuple[str, tuple[str, ...]], ProposalAnswer] = {}
    for stance in settings.stances:
        reference, transcript = run_role(
            project,
            settings.author,
            settings.profile,
            role="tag-monitor-review",
            instruction=INSTRUCTION,
            stance=(stance, REVIEW_STANCES[stance]),
            documents=documents,
            overview=overview,
            answer=schema,
            sources=stats.sources,
            code_revision=code_revision,
            check=lambda answer: check_review(answer, stats),
        )
        transcripts.append(reference)
        answer = transcript.answer if transcript.stop == "answered" else None
        problem = (transcript.failure or "no answer") if answer is None else None
        problem = problem or check_review(answer or {}, stats)
        if problem is not None:
            failures[stance] = problem
            continue
        parsed = ReviewAnswer.model_validate(answer)
        for row in parsed.findings:
            finding = Finding.model_validate(row.model_dump(mode="json"))
            findings.setdefault((finding.subject, finding.statement), finding)
        for row in parsed.proposals:
            proposals.setdefault((row.kind, tuple(row.targets)), row)
    record = ReviewRecord(
        model_alias=settings.author.model_alias,
        transcripts=tuple(transcripts),
        accepted=len(failures) < len(settings.stances),
        failures=failures,
        findings=tuple(findings.values()),
    )
    return record, list(proposals.values())
