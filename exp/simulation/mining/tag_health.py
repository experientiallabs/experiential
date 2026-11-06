"""Tag-health statistics over saved tag matrices: what each question and bank says about a corpus.

Every statistic here is computed from answers the decision model already gave (one
:class:`~exp.simulation.mining.tagging.TagMatrix` per bank and view); nothing asks a model. The
statistics flag shapes no geometry can use, and leave every judgment of meaning to a reviewer:

- **per question**: the fire rate (share of answered traces whose probability is strictly above
  the fire threshold), the spread of its probabilities, the share of answers inside the ambiguity
  band (abstain or low confidence), and the flags ``always_yes`` / ``always_no`` (fires on at least
  ``always_rate`` of the traces, or on at most ``1 - always_rate``), ``near_constant`` (outside the
  selection's fire-rate band or with too small a spread) and ``low_confidence``;
- **per pair** of questions answered over the same traces: answer agreement (share of shared
  traces where both fire or both do not), the Jaccard overlap of the traces they fire on, and
  absolute correlation. Agreement alone is high for any two rarely firing questions, so a pair is
  a ``near_duplicate`` candidate only when its agreement is at least ``duplicate_agreement`` and
  its fire sets overlap by at least ``duplicate_jaccard`` over at least ``min_overlap`` traces;
- **per bank** (one matrix): coverage (share of traces where at least one question fires), the
  mean number of fired questions per trace, the share of its questions carrying any flag, and the
  traces the tagger excluded at the credential boundary, when the caller knows them.

The same thresholds as the bank selection apply by default (``tag_bank_selection``), so a monitor
reading a finished corpus and the selection that built its banks agree on what "near-constant"
means.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.simulation.mining.tag_selection import DEFAULT_AMBIGUITY_BAND, DEFAULT_FIRE_THRESHOLD
from exp.simulation.mining.tagging import TagMatrix

TAG_HEALTH_PROTOCOL = "tag-health-v1"
"""Version of the corpus statistics and their flags; bump when any changes."""


class HealthThresholds(ContractModel):
    """Where a statistic becomes a flag.

    Attributes:
        threshold: Probability strictly above which a question fires.
        ambiguity_band: Closed band of answers counted as abstain or low confidence.
        always_rate: Fire rate at or above which a question is ``always_yes`` (and at or below
            ``1 - always_rate`` ``always_no``).
        min_fire_rate: Smallest fire rate of a usable question (the selection's band).
        max_fire_rate: Largest fire rate of a usable question.
        min_spread: Smallest standard deviation of a usable question's probabilities.
        max_ambiguous: Largest share of answers inside the band before ``low_confidence``.
        duplicate_agreement: Answer agreement at or above which a pair may be ``near_duplicate``.
        duplicate_jaccard: Fire-set overlap at or above which such a pair is ``near_duplicate``.
        min_overlap: Fewest traces either question fires on before a pair is read.
        max_pairs: Most near-duplicate pairs reported per bank, highest overlap first.
    """

    threshold: float = Field(default=DEFAULT_FIRE_THRESHOLD, gt=0.0, lt=1.0)
    ambiguity_band: tuple[float, float] = DEFAULT_AMBIGUITY_BAND
    always_rate: float = Field(default=0.99, gt=0.5, le=1.0)
    min_fire_rate: float = Field(default=0.05, ge=0.0, lt=0.5)
    max_fire_rate: float = Field(default=0.95, gt=0.5, le=1.0)
    min_spread: float = Field(default=0.1, ge=0.0, le=0.5)
    max_ambiguous: float = Field(default=0.25, ge=0.0, le=1.0)
    duplicate_agreement: float = Field(default=0.95, gt=0.5, le=1.0)
    duplicate_jaccard: float = Field(default=0.8, gt=0.0, le=1.0)
    min_overlap: int = Field(default=3, ge=2)
    max_pairs: int = Field(default=200, ge=1, le=10_000)

    @model_validator(mode="after")
    def _band(self) -> HealthThresholds:
        """Require an ordered band inside the unit interval."""
        low, high = self.ambiguity_band
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("ambiguity_band must be an ordered pair inside [0, 1]")
        return self


class QuestionHealth(ContractModel):
    """One question's statistics over one matrix.

    Attributes:
        bank_id: The bank.
        view: The view it was asked over.
        question_id: The question.
        answered: Traces with an answer.
        fired: Traces where it fired.
        fire_rate: ``fired / answered``.
        spread: Standard deviation of its probabilities.
        ambiguous: Share of answers inside the ambiguity band.
        flags: Threshold crossings, sorted.
    """

    bank_id: str
    view: str
    question_id: str
    answered: int = Field(ge=0)
    fired: int = Field(ge=0)
    fire_rate: float = Field(ge=0.0, le=1.0)
    spread: float = Field(ge=0.0)
    ambiguous: float = Field(ge=0.0, le=1.0)
    flags: tuple[str, ...] = ()


class PairHealth(ContractModel):
    """Two questions of one matrix whose answers nearly coincide.

    Attributes:
        bank_id: The bank.
        view: The view.
        left: One question.
        right: The other.
        shared: Traces both answered.
        agreement: Share of shared traces where both fire or neither does.
        jaccard: Traces both fire on over traces either fires on.
        correlation: Absolute Pearson correlation of their probabilities.
    """

    bank_id: str
    view: str
    left: str
    right: str
    shared: int = Field(ge=0)
    agreement: float = Field(ge=0.0, le=1.0)
    jaccard: float = Field(ge=0.0, le=1.0)
    correlation: float = Field(ge=0.0, le=1.0)


class BankHealth(ContractModel):
    """One bank's statistics over one view.

    Attributes:
        bank_id: The bank.
        bank_version: Its revision.
        view: The view.
        traces: Rows in the matrix.
        questions: Columns in the matrix.
        coverage: Share of traces where at least one question fires.
        mean_fired: Mean fired questions per trace.
        flagged: Questions carrying any flag.
        flagged_share: ``flagged / questions``.
        duplicates: Near-duplicate pairs found (before the report cap).
        excluded: Traces excluded at the credential boundary, when known.
    """

    bank_id: str
    bank_version: int = Field(ge=1)
    view: str
    traces: int = Field(ge=0)
    questions: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    mean_fired: float = Field(ge=0.0)
    flagged: int = Field(ge=0)
    flagged_share: float = Field(ge=0.0, le=1.0)
    duplicates: int = Field(ge=0)
    excluded: int | None = Field(default=None, ge=0)


class CorpusHealth(ContractModel):
    """Every matrix's statistics.

    Attributes:
        protocol: Statistics protocol.
        thresholds: The thresholds applied.
        banks: One row per matrix, in input order.
        questions: Every question, matrix by matrix, in column order.
        pairs: Near-duplicate pairs, matrix by matrix, highest overlap first.
    """

    protocol: str = TAG_HEALTH_PROTOCOL
    thresholds: HealthThresholds
    banks: tuple[BankHealth, ...]
    questions: tuple[QuestionHealth, ...]
    pairs: tuple[PairHealth, ...]


def matrix_key(matrix: TagMatrix) -> str:
    """Return ``<bank_id>.<view>``, the name a matrix's statistics are reported under."""
    return f"{matrix.bank_id}.{matrix.view}"


def corpus_health(
    matrices: Sequence[TagMatrix],
    thresholds: HealthThresholds | None = None,
    *,
    excluded: Mapping[str, int] | None = None,
) -> CorpusHealth:
    """Compute question, pair and bank statistics for every matrix.

    Args:
        matrices: Saved matrices, one per bank and view.
        thresholds: Flag thresholds; defaults apply when omitted.
        excluded: Traces excluded at the credential boundary, by :func:`matrix_key` or by view.

    Returns:
        The statistics.
    """
    spec = thresholds or HealthThresholds()
    banks: list[BankHealth] = []
    questions: list[QuestionHealth] = []
    pairs: list[PairHealth] = []
    for matrix in matrices:
        values = np.array([row.values for row in matrix.rows], dtype=np.float64)
        if values.size == 0:
            values = np.zeros((0, len(matrix.question_ids)))
        rows = _questions(matrix, values, spec)
        found = _pairs(matrix, values, spec)
        fired = values > spec.threshold
        flagged = sum(1 for row in rows if row.flags)
        known = None
        if excluded is not None:
            known = excluded.get(matrix_key(matrix), excluded.get(matrix.view))
        banks.append(
            BankHealth(
                bank_id=matrix.bank_id,
                bank_version=matrix.bank_version,
                view=matrix.view,
                traces=int(values.shape[0]),
                questions=len(matrix.question_ids),
                coverage=_round(float(fired.any(axis=1).mean()) if values.shape[0] else 0.0),
                mean_fired=_round(float(fired.sum(axis=1).mean()) if values.shape[0] else 0.0),
                flagged=flagged,
                flagged_share=_round(flagged / max(1, len(matrix.question_ids))),
                duplicates=len(found),
                excluded=known,
            )
        )
        questions.extend(rows)
        pairs.extend(found[: spec.max_pairs])
    return CorpusHealth(
        thresholds=spec, banks=tuple(banks), questions=tuple(questions), pairs=tuple(pairs)
    )


def _questions(
    matrix: TagMatrix, values: np.ndarray, spec: HealthThresholds
) -> list[QuestionHealth]:
    """Return every column's statistics and flags."""
    low, high = spec.ambiguity_band
    rows: list[QuestionHealth] = []
    count = values.shape[0]
    for index, question in enumerate(matrix.question_ids):
        column = values[:, index] if count else np.zeros(0)
        fired = int((column > spec.threshold).sum())
        rate = fired / count if count else 0.0
        spread = float(column.std()) if count else 0.0
        ambiguous = float(((column >= low) & (column <= high)).mean()) if count else 0.0
        flags: list[str] = []
        if count and rate >= spec.always_rate:
            flags.append("always_yes")
        if count and rate <= 1.0 - spec.always_rate:
            flags.append("always_no")
        if count and (
            not spec.min_fire_rate <= rate <= spec.max_fire_rate or spread < spec.min_spread
        ):
            flags.append("near_constant")
        if count and ambiguous > spec.max_ambiguous:
            flags.append("low_confidence")
        rows.append(
            QuestionHealth(
                bank_id=matrix.bank_id,
                view=matrix.view,
                question_id=question,
                answered=count,
                fired=fired,
                fire_rate=_round(rate),
                spread=_round(spread),
                ambiguous=_round(ambiguous),
                flags=tuple(sorted(flags)),
            )
        )
    return rows


def _pairs(matrix: TagMatrix, values: np.ndarray, spec: HealthThresholds) -> list[PairHealth]:
    """Return the matrix's near-duplicate pairs, highest overlap first."""
    count, width = values.shape if values.ndim == 2 else (0, 0)
    if count < spec.min_overlap or width < 2:
        return []
    fired = (values > spec.threshold).astype(np.float64)
    both = fired.T @ fired
    agreement = (both + (1.0 - fired).T @ (1.0 - fired)) / count
    totals = fired.sum(axis=0)
    deviation = values.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.abs(np.corrcoef(values, rowvar=False))
    found: list[PairHealth] = []
    for left in range(width):
        for right in range(left + 1, width):
            share = float(agreement[left, right])
            union = float(totals[left] + totals[right] - both[left, right])
            if share < spec.duplicate_agreement or union < spec.min_overlap:
                continue
            jaccard = float(both[left, right]) / union
            if jaccard < spec.duplicate_jaccard:
                continue
            if deviation[left] == 0.0 or deviation[right] == 0.0:
                value = 1.0 if deviation[left] == deviation[right] == 0.0 else 0.0
            else:
                value = float(np.nan_to_num(correlation[left, right]))
            found.append(
                PairHealth(
                    bank_id=matrix.bank_id,
                    view=matrix.view,
                    left=matrix.question_ids[left],
                    right=matrix.question_ids[right],
                    shared=count,
                    agreement=_round(share),
                    jaccard=_round(jaccard),
                    correlation=_round(min(1.0, value)),
                )
            )
    found.sort(key=lambda row: (-row.jaccard, -row.agreement, row.left, row.right))
    return found


def _round(value: float) -> float:
    """Round a statistic for a stable record."""
    return round(value, 6)
