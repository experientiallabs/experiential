"""Tests for building index tag records and probe query tags from tag matrices."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.optimize.evaluation.variance.exposure import ExposurePlan, plan_exposure
from exp.optimize.evaluation.variance.tag_retrieval import (
    index_tag_view,
    index_trace_tags,
    query_task_tags,
)
from exp.simulation.mining.bindings import TaskTraceLineageBinding
from exp.simulation.mining.questions import Question, QuestionBank
from exp.simulation.mining.tagging import TagMatrix, TagRow, TraceTaggingError
from exp.simulation.retrieval.tags import RAGIndexTags

_REF = ArtifactInput(artifact_id="bindings", sha256="0" * 64)
_BANK = QuestionBank(
    bank_id="shape-v1",
    version=1,
    questions=(
        Question(question_id="goal.repair", family="goal", text="Is the goal to repair?"),
        Question(question_id="env.gui", family="env", text="Is a GUI driven?"),
    ),
)


def _plan() -> ExposurePlan:
    """Twenty fit lineages and two held-out lineages, one trace each."""
    rows = [
        TaskTraceLineageBinding(
            trace_id=f"t{index}",
            lineage_id=f"lineage-{index}",
            partition="held_out" if index >= 20 else "fit",
        )
        for index in range(22)
    ]
    return plan_exposure(_REF, rows, seed=3, probe_fraction=0.3)


def _matrix(view: str = "task", *, bank_version: int = 1) -> TagMatrix:
    """Tag every plan trace; even traces fire goal.repair, multiples of three fire env.gui."""
    return TagMatrix(
        bank_id="shape-v1",
        bank_version=bank_version,
        question_ids=_BANK.question_ids(),
        view=view,
        rows=tuple(
            TagRow(
                trace_id=f"t{index}",
                unit_hash=f"h-{index}",
                values=(0.9 if index % 2 == 0 else 0.1, 0.8 if index % 3 == 0 else 0.5),
            )
            for index in range(22)
        ),
    )


def _expected(index: int) -> frozenset[str]:
    """Return the propositions fixture trace ``t<index>`` fires above 0.5 or 0.6."""
    return frozenset(
        name
        for name, fires in (("goal.repair", index % 2 == 0), ("env.gui", index % 3 == 0))
        if fires
    )


def _role_traces(plan: ExposurePlan, role: str) -> list[str]:
    """Return the plan's traces reserved for one role, sorted."""
    return sorted(t for t in plan.traces if plan.role_of(t) == role)


def test_index_record_holds_only_world_learning_traces() -> None:
    """Probe and final-evaluation rows never enter the index tag record."""
    plan = _plan()
    record = index_trace_tags(
        plan, bank=_BANK, task=_matrix(), observations=_matrix("observations")
    )

    assert [view.view for view in record.views] == ["observations", "task"]
    task = record.view("task")
    assert task is not None
    assert [row.trace_id for row in task.traces] == _role_traces(plan, "world_learning")
    assert task.matrix_sha256 == sha256_json(_matrix())
    assert task.families() == {"goal.repair": "goal", "env.gui": "env"}
    # A value of exactly 0.5 is not strictly above the threshold, so it never fires.
    assert task.fired_by_trace() == {
        trace: _expected(int(trace[1:])) for trace in _role_traces(plan, "world_learning")
    }


def test_index_record_refuses_mixed_banks_and_views() -> None:
    """A matrix of another revision or view, or with a foreign column, is refused."""
    plan = _plan()
    with pytest.raises(TraceTaggingError, match="not shape-v1 v1"):
        index_trace_tags(plan, bank=_BANK, task=_matrix(bank_version=2))
    with pytest.raises(TraceTaggingError, match="not 'task'"):
        index_trace_tags(plan, bank=_BANK, task=_matrix("observations"))
    with pytest.raises(TraceTaggingError, match="not 'observations'"):
        index_trace_tags(plan, bank=_BANK, task=_matrix(), observations=_matrix())
    narrow = _BANK.model_copy(update={"questions": _BANK.questions[:1]})
    with pytest.raises(TraceTaggingError, match="not in bank"):
        index_tag_view(_matrix(), bank=narrow, plan=plan, view="task")


def test_query_tags_match_the_index_and_refuse_leaking_views() -> None:
    """A probe's query tags use the index's threshold and must be task-view, same bank."""
    plan = _plan()
    record = index_trace_tags(plan, bank=_BANK, task=_matrix(), threshold=0.6)
    probe = _role_traces(plan, "mining_probes")[0]

    tags = query_task_tags(_matrix(), probe, index_tags=record, plan=plan)
    assert frozenset(tags.fired) == _expected(int(probe[1:]))
    assert tags.threshold == 0.6
    assert tags.view == "task"

    with pytest.raises(TraceTaggingError, match="would leak"):
        query_task_tags(_matrix("observations"), probe, index_tags=record, plan=plan)
    with pytest.raises(TraceTaggingError, match="but the index"):
        query_task_tags(_matrix(bank_version=2), probe, index_tags=record, plan=plan)
    with pytest.raises(ValueError, match="no lineage"):
        query_task_tags(_matrix(), "unknown", index_tags=record, plan=plan)
    obs_only = RAGIndexTags(
        views=(index_tag_view(_matrix("observations"), bank=_BANK, plan=plan, view="observations"),)
    )
    with pytest.raises(ValueError, match="no task-view tags"):
        query_task_tags(_matrix(), probe, index_tags=obs_only, plan=plan)
