"""Select traces by the propositions a decision model found them to contain.

A :class:`~exp.simulation.mining.tagging.TagMatrix` answers one question bank about one rendered
view of every trace. Several consumers need a subset of those traces rather than a geometry: the
traces a world model is built from, the traces its retrieval index holds, and the held-out traces
a council probes. Each subset is a rule over fired propositions, and this module is the one reading
of that rule, so two consumers given the same matrix and rule always choose the same traces.

A proposition fires when its probability is strictly above the threshold, the same strict reading
the decision-tag clustering uses. Probabilities inside the ambiguity band are reported per trace,
because a selection that hinges on an answer the model was unsure of deserves a second look rather
than silent inclusion.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tagging import TagMatrix, TraceTaggingError

DEFAULT_FIRE_THRESHOLD = 0.5
"""Probability above which a proposition counts as fired, matching the clustering default."""

DEFAULT_AMBIGUITY_BAND = (0.3, 0.7)
"""Answers inside this closed band are reported as uncertain rather than silently trusted."""


class TagRule(ContractModel):
    """Which fired propositions put a trace into one selection.

    Attributes:
        name: Stable name of the selection, such as ``world-model`` or ``retrieval``.
        require_all: Every one of these propositions must fire.
        require_any: At least one of these must fire, when any are named.
        exclude: None of these may fire.
        threshold: Probability strictly above which a proposition fires, inside (0, 1).
    """

    name: str = Field(min_length=1, max_length=64)
    require_all: tuple[str, ...] = ()
    require_any: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    threshold: float = Field(default=DEFAULT_FIRE_THRESHOLD, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> TagRule:
        """Refuse a rule that names nothing or both requires and excludes one proposition."""
        named = (*self.require_all, *self.require_any, *self.exclude)
        if not named:
            raise ValueError(f"tag rule {self.name!r} names no proposition")
        if set(self.exclude) & (set(self.require_all) | set(self.require_any)):
            raise ValueError(f"tag rule {self.name!r} both requires and excludes a proposition")
        return self

    def propositions(self) -> frozenset[str]:
        """Return every proposition the rule reads."""
        return frozenset((*self.require_all, *self.require_any, *self.exclude))


class TagSelection(ContractModel):
    """The traces one rule chose from one matrix, with the evidence that chose them.

    Attributes:
        rule: The rule that was applied.
        bank_id: Identity of the bank the matrix answered.
        bank_version: Revision of that bank.
        view: Rendered view the matrix was tagged over.
        trace_ids: Selected traces in matrix row order.
        uncertain: Selected traces whose decision read an answer inside the ambiguity band,
            mapped to the propositions that were uncertain.
    """

    rule: TagRule
    bank_id: str
    bank_version: int
    view: str
    trace_ids: tuple[str, ...]
    uncertain: dict[str, tuple[str, ...]]


def fired_tags(
    matrix: TagMatrix,
    *,
    threshold: float = DEFAULT_FIRE_THRESHOLD,
    question_ids: Iterable[str] | None = None,
) -> dict[str, frozenset[str]]:
    """Return the propositions that fired for every trace in a matrix.

    Args:
        matrix: Tagged answers to one bank over one view.
        threshold: Probability strictly above which a proposition fires.
        question_ids: Restrict the reading to these columns; every column when omitted.

    Returns:
        Each trace identity mapped to its fired propositions.

    Raises:
        TraceTaggingError: A named column is not in the matrix.
    """
    columns = _columns(matrix, question_ids)
    return {
        row.trace_id: frozenset(
            question for question, index in columns.items() if row.values[index] > threshold
        )
        for row in matrix.rows
    }


def select_traces(
    matrix: TagMatrix,
    rule: TagRule,
    *,
    bank: QuestionBank,
    ambiguity_band: tuple[float, float] = DEFAULT_AMBIGUITY_BAND,
) -> TagSelection:
    """Apply one rule to a matrix, refusing a matrix answered against a different bank.

    Args:
        matrix: Tagged answers to one bank over one view.
        rule: Which fired propositions select a trace.
        bank: The bank the rule was written against; its identity and revision must match.
        ambiguity_band: Closed probability band reported as uncertain for selected traces.

    Returns:
        The selected traces and any uncertain answers they were selected on.

    Raises:
        TraceTaggingError: The matrix answered another bank or revision, or the rule names a
            proposition the matrix does not carry.
    """
    if (matrix.bank_id, matrix.bank_version) != (bank.bank_id, bank.version):
        raise TraceTaggingError(
            f"tag matrix answers bank {matrix.bank_id} v{matrix.bank_version}, but rule "
            f"{rule.name!r} was written against {bank.bank_id} v{bank.version}"
        )
    low, high = ambiguity_band
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("ambiguity band must satisfy 0 <= low <= high <= 1")
    columns = _columns(matrix, rule.propositions())
    selected: list[str] = []
    uncertain: dict[str, tuple[str, ...]] = {}
    for row in matrix.rows:
        fired = {q for q, index in columns.items() if row.values[index] > rule.threshold}
        if not set(rule.require_all) <= fired:
            continue
        if rule.require_any and not fired & set(rule.require_any):
            continue
        if fired & set(rule.exclude):
            continue
        selected.append(row.trace_id)
        unsure = tuple(
            question for question in sorted(columns) if low <= row.values[columns[question]] <= high
        )
        if unsure:
            uncertain[row.trace_id] = unsure
    return TagSelection(
        rule=rule,
        bank_id=matrix.bank_id,
        bank_version=matrix.bank_version,
        view=matrix.view,
        trace_ids=tuple(selected),
        uncertain=uncertain,
    )


def _columns(matrix: TagMatrix, question_ids: Iterable[str] | None) -> dict[str, int]:
    """Map requested question identities to their matrix column indexes.

    Args:
        matrix: Matrix whose columns are read.
        question_ids: Identities to resolve, or None for every column.

    Returns:
        Question identity to column index.

    Raises:
        TraceTaggingError: A requested identity is not a column of the matrix.
    """
    index = {question: position for position, question in enumerate(matrix.question_ids)}
    if question_ids is None:
        return index
    wanted = set(question_ids)
    missing = sorted(wanted - index.keys())
    if missing:
        raise TraceTaggingError(
            f"tag matrix for bank {matrix.bank_id} has no column {missing}; "
            "tag with the bank the rule names"
        )
    return {question: index[question] for question in wanted}
