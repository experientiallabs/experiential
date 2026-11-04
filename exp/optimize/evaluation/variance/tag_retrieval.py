"""Turn decision tags into a retrieval index's tag record and a probe task's query tags.

Tags reach retrieval twice. A tag selection already narrows which world-learning traces enter the
retrieval corpus (see :func:`~exp.optimize.evaluation.variance.exposure.world_learning_scopes`).
This module prepares the second use, in which tags shape what a query retrieves:

- :func:`index_trace_tags` reads a :class:`~exp.simulation.mining.tagging.TagMatrix` per view into
  the :class:`~exp.simulation.retrieval.tags.RAGIndexTags` record passed as ``trace_tags`` to
  ``persist_trace_rag``. It keeps only the traces the exposure plan reserves for world learning,
  the role retrieval may read, so probe and final-evaluation tags never enter an index.
- :func:`query_task_tags` reads a probe task's task-view row into the
  :class:`~exp.simulation.retrieval.tags.TaskQueryTags` a tag-aware retrieval is given, with the
  index's exact bank, revision, columns and threshold.

Both refuse matrices that mix banks, bank revisions or views. Fired propositions are read with
:func:`~exp.simulation.mining.tag_selection.fired_tags`, the one strict reading every tag consumer
shares.
"""

from __future__ import annotations

from exp.common.core.artifacts import sha256_json
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tag_selection import DEFAULT_FIRE_THRESHOLD, fired_tags
from exp.simulation.mining.tagging import TagMatrix, TraceTaggingError
from exp.simulation.retrieval.tags import (
    RAGIndexTags,
    RAGTagQuestion,
    RAGTagView,
    RAGTraceTags,
    TagView,
    TaskQueryTags,
)


def index_tag_view(
    matrix: TagMatrix,
    *,
    bank: QuestionBank,
    plan: ExposurePlan,
    view: TagView,
    threshold: float = DEFAULT_FIRE_THRESHOLD,
) -> RAGTagView:
    """Read one view's world-learning rows into an index tag record.

    Args:
        matrix: Tagged answers to ``bank`` over ``view``.
        bank: The bank the matrix answered; supplies each column's family.
        plan: Frozen exposure plan whose world-learning traces retrieval may read.
        view: The view the matrix must have been tagged over.
        threshold: Probability strictly above which a proposition fires.

    Returns:
        The view record over every world-learning trace the matrix tags.

    Raises:
        TraceTaggingError: The matrix answers another bank, revision or view, carries a column
            the bank lacks, or tags no world-learning trace.
    """
    _require_bank(matrix, bank, view)
    families = {question.question_id: question.family for question in bank.questions}
    fired = fired_tags(matrix, threshold=threshold)
    rows = sorted(
        trace
        for trace in fired
        if trace in plan.traces and plan.roles[plan.traces[trace]] == "world_learning"
    )
    if not rows:
        raise TraceTaggingError(
            f"{view} tag matrix for bank {matrix.bank_id} tags no world-learning trace; tag the "
            "traces the retrieval index holds"
        )
    return RAGTagView(
        bank_id=matrix.bank_id,
        bank_version=matrix.bank_version,
        view=view,
        threshold=threshold,
        questions=tuple(
            RAGTagQuestion(question_id=question, family=families[question])
            for question in matrix.question_ids
        ),
        matrix_sha256=sha256_json(matrix),
        traces=tuple(
            RAGTraceTags(trace_id=trace, fired=tuple(sorted(fired[trace]))) for trace in rows
        ),
    )


def index_trace_tags(
    plan: ExposurePlan,
    *,
    bank: QuestionBank,
    task: TagMatrix,
    observations: TagMatrix | None = None,
    threshold: float = DEFAULT_FIRE_THRESHOLD,
) -> RAGIndexTags:
    """Build the index tag record from a task-view matrix and an optional observations matrix.

    Pass the result as ``trace_tags`` to ``persist_trace_rag``, which restricts it to the traces
    the index holds and requires each view to cover all of them.

    Args:
        plan: Frozen exposure plan; only world-learning traces are recorded.
        bank: The one bank both matrices answered.
        task: Task-view matrix. Only this view may shape query-time ranking.
        observations: Optional observations-view matrix, recorded as annotation only.
        threshold: Fire threshold shared by both views.

    Returns:
        The tag record, one entry per supplied view.

    Raises:
        TraceTaggingError: A matrix answers another bank or revision, was tagged over the wrong
            view, or tags no world-learning trace.
    """
    views = [index_tag_view(task, bank=bank, plan=plan, view="task", threshold=threshold)]
    if observations is not None:
        views.append(
            index_tag_view(
                observations, bank=bank, plan=plan, view="observations", threshold=threshold
            )
        )
    return RAGIndexTags(views=tuple(sorted(views, key=lambda item: item.view)))


def query_task_tags(
    matrix: TagMatrix,
    trace_id: str,
    *,
    index_tags: RAGIndexTags,
    plan: ExposurePlan,
) -> TaskQueryTags:
    """Read one probe task's fired task-view propositions as query tags for an index.

    Args:
        matrix: Task-view matrix holding the probe trace's row.
        trace_id: Source trace of the probe task.
        index_tags: The tag record of the index the query will run against.
        plan: Frozen exposure plan the probe trace must belong to.

    Returns:
        Query tags with the index's bank, revision and threshold.

    Raises:
        TraceTaggingError: The matrix was tagged over a view other than ``task``, answers another
            bank or revision than the index's task record, has different columns, or holds no
            row for the trace.
        ValueError: The index records no task-view tags, or the trace is not in the plan.
    """
    if matrix.view != "task":
        raise TraceTaggingError(
            f"query tags must come from a task-view matrix, not {matrix.view!r}; the "
            f"{matrix.view} view describes the episode's outcome and would leak it"
        )
    record = index_tags.view("task")
    if record is None:
        raise ValueError("the index records no task-view tags to rank a query against")
    plan.role_of(trace_id)
    if (matrix.bank_id, matrix.bank_version) != (record.bank_id, record.bank_version):
        raise TraceTaggingError(
            f"query matrix answers bank {matrix.bank_id} v{matrix.bank_version}, but the index "
            f"was tagged with {record.bank_id} v{record.bank_version}"
        )
    columns = tuple(question.question_id for question in record.questions)
    if matrix.question_ids != columns:
        raise TraceTaggingError(
            f"query matrix for bank {matrix.bank_id} carries different columns than the index's "
            "task record; tag the probe task with the same question selection"
        )
    matrix.row(trace_id)
    fired = sorted(fired_tags(matrix, threshold=record.threshold)[trace_id])
    return TaskQueryTags(
        bank_id=matrix.bank_id,
        bank_version=matrix.bank_version,
        view=matrix.view,
        threshold=record.threshold,
        matrix_sha256=sha256_json(matrix),
        fired=tuple(fired),
    )


def _require_bank(matrix: TagMatrix, bank: QuestionBank, view: TagView) -> None:
    """Refuse a matrix of another bank, revision or view, or with a column the bank lacks.

    Args:
        matrix: Matrix about to be recorded.
        bank: The bank it must have answered.
        view: The view it must have been tagged over.

    Raises:
        TraceTaggingError: Any identity differs.
    """
    if (matrix.bank_id, matrix.bank_version) != (bank.bank_id, bank.version):
        raise TraceTaggingError(
            f"{view} tag matrix answers bank {matrix.bank_id} v{matrix.bank_version}, not "
            f"{bank.bank_id} v{bank.version}; record views tagged with one bank revision"
        )
    if matrix.view != view:
        raise TraceTaggingError(
            f"tag matrix was tagged over the {matrix.view!r} view, not {view!r}"
        )
    missing = sorted(set(matrix.question_ids) - set(bank.question_ids()))
    if missing:
        raise TraceTaggingError(f"tag matrix columns {missing} are not in bank {bank.bank_id}")
