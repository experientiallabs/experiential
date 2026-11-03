"""Grade rubric criteria over finished episodes with a decision model, keeping doubt explicit.

Every criterion is a yes/no proposition, so it is asked as a question of family ``criterion`` on
the typed decisions surface (:class:`~exp.simulation.mining.decisions.DecisionsTraceTagger`), which
returns one probability per proposition and carries up to
:data:`~exp.runtime.gateway.decisions_contracts.MAX_DECISION_QUESTIONS` propositions per call. All
of an episode's propositions go into one bank, so an episode costs as few calls as the bank's size
allows, and answers are bought through the tagger's content-addressed decision cache.

For each criterion the bank holds:

- ``draws`` copies of the exact criterion text under distinct question identities, so each copy is
  a separate decision with its own cache key (repeated grading draws);
- one question per supplied paraphrase of the criterion;
- one evidence-presence proposition (family ``criterion_evidence``): "Does the episode show enough
  to decide ...?".

Each episode-criterion result gets exactly one classification, chosen in this order:

1. ``insufficient_evidence``: the evidence-presence probability is below the band, so the view
   lacks what the criterion needs. The criterion probability is not trusted.
2. ``ambiguous_criterion``: the draws disagree among themselves, the paraphrases disagree with the
   exact text, or the criterion is uncertain across independent episodes (in-band on at least
   ``episode_fraction`` of the episodes that showed enough evidence, over at least
   ``minimum_episodes`` distinct episode texts) and this episode is one of the uncertain ones.
3. ``disagreement``: a supplied reference verdict (an LLM judge or a second grader) is confident
   the other way from this grader's confident verdict.
4. ``pass`` or ``fail``: the mean probability is above or below the band.
5. ``uncertain``: the mean probability is inside the band and no test above applies. A single
   probability near one half is not, alone, evidence that the criterion is ambiguous.

Every test that fired is listed in the result's reasons, so a lower-precedence finding is never
lost. A criterion that is ambiguous anywhere goes back to rubric authoring instead of being
trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, sha256_json
from exp.optimize.evaluation.variance.contracts import Criterion
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.questions import Question, QuestionBank
from exp.simulation.mining.tagging import TagMatrix
from exp.simulation.mining.views import Unit

GRADER_PROTOCOL = "variance-grader-v1"
"""Version of the grading propositions and classification rules."""

CRITERION_FAMILY = "criterion"
"""Question family of every criterion proposition, its draws and its paraphrases."""

EVIDENCE_FAMILY = "criterion_evidence"
"""Question family of the evidence-presence propositions."""

EVIDENCE_TEMPLATE = (
    "Does the episode show enough to decide whether the following holds, either way? {criterion}"
)
"""Evidence-presence proposition asked beside each criterion."""

Classification = Literal[
    "pass", "fail", "uncertain", "ambiguous_criterion", "insufficient_evidence", "disagreement"
]
"""One graded criterion's outcome; only ``pass`` and ``fail`` are confident verdicts."""


class Band(ContractModel):
    """Probabilities inside ``[low, high]`` are not a confident verdict.

    Attributes:
        low: Probabilities below this are a confident no.
        high: Probabilities above this are a confident yes.
    """

    low: float = Field(default=0.3, gt=0.0, lt=1.0)
    high: float = Field(default=0.7, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Require a non-empty band."""
        if self.low >= self.high:
            raise ValueError("the band's low edge must be below its high edge")
        return self

    def verdict(self, probability: float) -> Literal["pass", "fail", "uncertain"]:
        """Classify one probability against the band.

        Args:
            probability: Probability that a proposition holds.

        Returns:
            ``pass`` above the band, ``fail`` below it, ``uncertain`` inside it.
        """
        if probability > self.high:
            return "pass"
        if probability < self.low:
            return "fail"
        return "uncertain"


class GraderSettings(ContractModel):
    """Frozen grading controls.

    Attributes:
        band: Confidence band.
        draws: Repeated grading draws of each criterion's exact text.
        spread: Largest range across draws, or between paraphrases and the exact text, that still
            counts as agreement. Draws that straddle the band disagree at any spread.
        episode_fraction: Share of evidence-bearing episodes in the band that makes a criterion
            ambiguous across episodes.
        minimum_episodes: Fewest distinct evidence-bearing episodes the cross-episode test needs.
    """

    band: Band = Band()
    draws: int = Field(default=2, ge=1, le=8, strict=True)
    spread: float = Field(default=0.4, gt=0.0, le=1.0)
    episode_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    minimum_episodes: int = Field(default=3, ge=2, le=1000, strict=True)


class GradedCriterion(ContractModel):
    """One criterion and the paraphrases it is cross-checked with.

    Attributes:
        criterion: The rubric proposition.
        paraphrases: Other wordings of the same proposition; their answers must agree with it.
    """

    criterion: Criterion
    paraphrases: tuple[str, ...] = Field(default=(), max_length=4)


class EpisodeView(ContractModel):
    """One finished episode as the decision model reads it.

    Attributes:
        episode_id: Stable identity of the episode.
        text: The rendered view.
    """

    episode_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ReferenceVerdict(ContractModel):
    """A confident verdict on one criterion from another grader.

    Attributes:
        source: Who gave it, such as ``llm-judge`` or ``second-grader``.
        episode_id: The episode graded.
        criterion_id: The criterion graded.
        passed: The verdict.
    """

    source: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    criterion_id: str = Field(min_length=1)
    passed: bool


class CriterionGrade(ContractModel):
    """One criterion graded on one episode.

    Attributes:
        episode_id: The episode graded.
        criterion_id: The criterion graded.
        probability: Mean probability over every draw and paraphrase.
        draws: Probability per draw of the exact text.
        paraphrases: Probability per paraphrase.
        evidence: Probability that the episode shows enough to decide.
        classification: The one outcome chosen by precedence.
        reasons: Every test that fired, in precedence order.
        disagreeing_sources: Reference sources whose verdict opposes this grader's.
    """

    episode_id: str
    criterion_id: str
    probability: float = Field(ge=0.0, le=1.0)
    draws: tuple[float, ...] = Field(min_length=1)
    paraphrases: tuple[float, ...] = ()
    evidence: float = Field(ge=0.0, le=1.0)
    classification: Classification
    reasons: tuple[str, ...] = ()
    disagreeing_sources: tuple[str, ...] = ()


class CriterionSummary(ContractModel):
    """One criterion across every graded episode.

    Attributes:
        criterion_id: The criterion.
        counts: Results per classification.
        ambiguous: Whether the criterion goes back to rubric authoring.
        reasons: Why it is ambiguous, when it is.
    """

    criterion_id: str
    counts: dict[str, int]
    ambiguous: bool
    reasons: tuple[str, ...] = ()


class GradeReport(ContractModel):
    """Every criterion graded on every episode.

    Attributes:
        protocol: Grader protocol the report was produced under.
        settings: The grading controls.
        bank_id: Identity of the bank asked.
        model: Identity of the decision model that answered.
        grades: One result per episode and criterion, episodes in input order.
        summaries: One summary per criterion, in criterion order.
    """

    protocol: str = GRADER_PROTOCOL
    settings: GraderSettings
    bank_id: str
    model: JsonObject
    grades: tuple[CriterionGrade, ...]
    summaries: tuple[CriterionSummary, ...]


def draw_id(criterion_id: str, draw: int) -> str:
    """Return the question identity of one draw of a criterion's exact text."""
    return f"{criterion_id}#draw{draw}"


def paraphrase_id(criterion_id: str, index: int) -> str:
    """Return the question identity of one paraphrase of a criterion."""
    return f"{criterion_id}#paraphrase{index}"


def evidence_id(criterion_id: str) -> str:
    """Return the question identity of a criterion's evidence-presence proposition."""
    return f"{criterion_id}#evidence"


def criterion_bank(criteria: Sequence[GradedCriterion], settings: GraderSettings) -> QuestionBank:
    """Build the one bank asked about every episode.

    Args:
        criteria: Criteria with distinct identities.
        settings: The grading controls.

    Returns:
        A content-addressed bank: draws, paraphrases and the evidence proposition per criterion.

    Raises:
        ValueError: There are no criteria or their identities repeat.
    """
    identities = [item.criterion.criterion_id for item in criteria]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("grading needs one or more criteria with distinct identities")
    questions: list[Question] = []
    for item in criteria:
        cid, text = item.criterion.criterion_id, item.criterion.text
        questions.extend(
            Question(question_id=draw_id(cid, draw), family=CRITERION_FAMILY, text=text)
            for draw in range(settings.draws)
        )
        questions.extend(
            Question(question_id=paraphrase_id(cid, index), family=CRITERION_FAMILY, text=other)
            for index, other in enumerate(item.paraphrases)
        )
        questions.append(
            Question(
                question_id=evidence_id(cid),
                family=EVIDENCE_FAMILY,
                text=EVIDENCE_TEMPLATE.format(criterion=text),
            )
        )
    digest = sha256_json([question.model_dump(mode="json") for question in questions])
    return QuestionBank(bank_id=f"criteria-{digest[:20]}", version=1, questions=tuple(questions))


def grade_episodes(
    grader: DecisionsTraceTagger,
    episodes: Sequence[EpisodeView],
    criteria: Sequence[GradedCriterion],
    *,
    settings: GraderSettings | None = None,
    references: Sequence[ReferenceVerdict] = (),
) -> GradeReport:
    """Grade every criterion on every episode in as few decision calls as the bank allows.

    Args:
        grader: Decision client; its batch limit bounds the propositions per call.
        episodes: Finished episodes with distinct identities.
        criteria: Criteria with distinct identities.
        settings: The grading controls; defaults when omitted.
        references: Verdicts from an LLM judge or a second grader, when supplied.

    Returns:
        The report.

    Raises:
        ValueError: There are no episodes, or their identities repeat.
        TraceTaggingError: A decision answer is missing or not a probability.
    """
    settings = settings or GraderSettings()
    identities = [episode.episode_id for episode in episodes]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("grading needs one or more episodes with distinct identities")
    bank = criterion_bank(criteria, settings)
    units = [Unit(episode.episode_id, "observations", episode.text) for episode in episodes]
    matrix = grader.tag(units, bank)
    hashes = {unit.trace_id: unit.unit_hash for unit in units}
    grades: list[CriterionGrade] = []
    summaries: list[CriterionSummary] = []
    by_criterion: dict[str, list[CriterionGrade]] = {}
    for item in criteria:
        raw = [_raw(matrix, episode.episode_id, item, settings) for episode in episodes]
        across = _across_episodes(raw, hashes, settings)
        rows = [_classify(row, across, references, settings) for row in raw]
        by_criterion[item.criterion.criterion_id] = rows
        summaries.append(_summary(item.criterion.criterion_id, rows, across))
    for index in range(len(episodes)):
        grades.extend(rows[index] for rows in by_criterion.values())
    return GradeReport(
        settings=settings,
        bank_id=bank.bank_id,
        model=dict(grader.model_identity),
        grades=tuple(grades),
        summaries=tuple(summaries),
    )


def references_from(report: GradeReport, source: str) -> tuple[ReferenceVerdict, ...]:
    """Turn another grader's confident verdicts into references for disagreement checks.

    Args:
        report: The other grader's report.
        source: Name recorded on each reference.

    Returns:
        One reference per ``pass`` or ``fail`` result; doubtful results give none.
    """
    return tuple(
        ReferenceVerdict(
            source=source,
            episode_id=grade.episode_id,
            criterion_id=grade.criterion_id,
            passed=grade.classification == "pass",
        )
        for grade in report.grades
        if grade.classification in ("pass", "fail")
    )


class _Raw(ContractModel):
    """One episode-criterion's answers before classification."""

    episode_id: str
    criterion_id: str
    draws: tuple[float, ...]
    paraphrases: tuple[float, ...]
    evidence: float
    mean: float
    reasons: tuple[str, ...]


def _raw(
    matrix: TagMatrix, episode_id: str, item: GradedCriterion, settings: GraderSettings
) -> _Raw:
    """Read one episode-criterion's answers and the per-episode tests.

    Args:
        matrix: The decision answers.
        episode_id: The episode.
        item: The criterion.
        settings: The grading controls.

    Returns:
        Answers, mean, and the reasons from evidence and draw or paraphrase agreement.
    """
    cid = item.criterion.criterion_id
    row = matrix.row(episode_id)
    value = dict(zip(matrix.question_ids, row.values, strict=True))
    draws = tuple(value[draw_id(cid, draw)] for draw in range(settings.draws))
    paraphrases = tuple(value[paraphrase_id(cid, i)] for i in range(len(item.paraphrases)))
    evidence = value[evidence_id(cid)]
    band = settings.band
    reasons: list[str] = []
    if evidence < band.low:
        reasons.append(f"insufficient_evidence: evidence-presence probability {evidence:.3f}")
    elif evidence <= band.high:
        reasons.append(f"evidence presence uncertain at {evidence:.3f}")
    if _disagree(draws, settings):
        reasons.append(f"ambiguous_criterion: repeated draws disagree {list(draws)}")
    if paraphrases and _disagree((fmean(draws), *paraphrases), settings):
        reasons.append(
            f"ambiguous_criterion: paraphrases {list(paraphrases)} disagree with the exact "
            f"text at {fmean(draws):.3f}"
        )
    return _Raw(
        episode_id=episode_id,
        criterion_id=cid,
        draws=draws,
        paraphrases=paraphrases,
        evidence=evidence,
        mean=fmean((*draws, *paraphrases)),
        reasons=tuple(reasons),
    )


def _disagree(values: tuple[float, ...], settings: GraderSettings) -> bool:
    """Return whether values range too widely or straddle the band."""
    if len(values) < 2:
        return False
    verdicts = {settings.band.verdict(value) for value in values}
    return max(values) - min(values) > settings.spread or {"pass", "fail"} <= verdicts


def _across_episodes(
    raw: Sequence[_Raw], hashes: dict[str, str], settings: GraderSettings
) -> str | None:
    """Return why a criterion is uncertain across independent episodes, or None.

    Episodes lacking evidence are left out, since their doubt has another cause, and episodes
    with identical text count once, since they are not independent.

    Args:
        raw: One criterion's answers on every episode.
        hashes: Rendered-text identity per episode.
        settings: The grading controls.

    Returns:
        The reason, or None when the test does not apply.
    """
    distinct: dict[str, float] = {}
    for row in raw:
        if row.evidence >= settings.band.low:
            distinct.setdefault(hashes[row.episode_id], row.mean)
    if len(distinct) < settings.minimum_episodes:
        return None
    uncertain = sum(settings.band.verdict(mean) == "uncertain" for mean in distinct.values())
    if uncertain / len(distinct) < settings.episode_fraction:
        return None
    return (
        f"ambiguous_criterion: uncertain on {uncertain} of {len(distinct)} independent episodes "
        "with enough evidence"
    )


def _classify(
    raw: _Raw,
    across: str | None,
    references: Sequence[ReferenceVerdict],
    settings: GraderSettings,
) -> CriterionGrade:
    """Choose one classification by precedence and keep every reason.

    Args:
        raw: Answers and per-episode reasons.
        across: The cross-episode ambiguity reason, if the test fired for this criterion.
        references: Verdicts from other graders.
        settings: The grading controls.

    Returns:
        The graded result.
    """
    base = settings.band.verdict(raw.mean)
    reasons = list(raw.reasons)
    if across is not None and base == "uncertain":
        reasons.append(across)
    opposing = sorted(
        {
            ref.source
            for ref in references
            if ref.episode_id == raw.episode_id
            and ref.criterion_id == raw.criterion_id
            and base != "uncertain"
            and ref.passed != (base == "pass")
        }
    )
    if opposing:
        reasons.append(f"disagreement: {base} here, the opposite from {opposing}")
    classification: Classification = base
    for label in ("insufficient_evidence", "ambiguous_criterion", "disagreement"):
        if any(reason.startswith(label + ":") for reason in reasons):
            classification = label
            break
    return CriterionGrade(
        episode_id=raw.episode_id,
        criterion_id=raw.criterion_id,
        probability=raw.mean,
        draws=raw.draws,
        paraphrases=raw.paraphrases,
        evidence=raw.evidence,
        classification=classification,
        reasons=tuple(reasons),
        disagreeing_sources=tuple(opposing),
    )


def _summary(
    criterion_id: str, rows: Sequence[CriterionGrade], across: str | None
) -> CriterionSummary:
    """Summarize one criterion over every episode.

    Args:
        criterion_id: The criterion.
        rows: Its results on every episode.
        across: The cross-episode ambiguity reason, if the test fired.

    Returns:
        Counts per classification and whether the criterion returns to rubric authoring.
    """
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.classification] = counts.get(row.classification, 0) + 1
    reasons = sorted(
        {
            reason
            for row in rows
            for reason in row.reasons
            if reason.startswith("ambiguous_criterion:")
        }
        | ({across} if across is not None else set())
    )
    return CriterionSummary(
        criterion_id=criterion_id, counts=counts, ambiguous=bool(reasons), reasons=tuple(reasons)
    )
