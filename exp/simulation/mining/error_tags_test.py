"""Error tags over rendered views, their uncertainty, rankings, step points and replay."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exp.common.project import ProjectStore
from exp.simulation.mining.error_tags import (
    ERROR_STEPS_TYPE,
    ERROR_TAGS_TYPE,
    bank_for_view,
    error_bank,
    error_patterns,
    error_points,
    load_error_steps,
    load_error_tags,
    persist_error_tags,
    rank_error_tasks,
    rank_error_traces,
    require_error_bank,
    tag_error_steps,
    tag_errors,
    trajectory_unit,
)
from exp.simulation.mining.questions import Question, QuestionBank, QuestionBankError
from exp.simulation.mining.tagging import TagMatrix, TagRow, TraceTaggingError
from exp.simulation.mining.traceml import TraceStep, TraceTrajectory
from exp.simulation.mining.views import Unit

BANK = QuestionBank(
    bank_id="error-test-v1",
    version=1,
    questions=(
        Question(question_id="error.check.unverified", family="error.check", text="Unverified?"),
        Question(question_id="error.feedback.ignored", family="error.feedback", text="Ignored?"),
        Question(question_id="error.safety.destructive", family="error.safety", text="Destroy?"),
        Question(question_id="risk.destructive", family="risk", text="Destructive request?"),
    ),
)

Answer = Callable[[str, str], float]


@dataclass
class ScriptedTagger:
    """A tagger answering from a function of (row identity, question), refusing named rows.

    Attributes:
        answer: Probability per row identity and question.
        refuse: Row identities whose tagging is refused.
        calls: Every batch of row identities asked, in order.
    """

    answer: Answer
    refuse: frozenset[str] = frozenset()
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        """Answer every unit, or refuse the batch when it holds a refused row."""
        self.calls.append(tuple(unit.trace_id for unit in units))
        if any(unit.trace_id in self.refuse for unit in units):
            raise TraceTaggingError("scripted refusal")
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            question_ids=bank.question_ids(),
            view=units[0].view,
            rows=tuple(
                TagRow(
                    trace_id=unit.trace_id,
                    unit_hash=unit.unit_hash,
                    values=tuple(self.answer(unit.trace_id, q) for q in bank.question_ids()),
                )
                for unit in units
            ),
        )


def _table(rows: Mapping[str, tuple[float, float, float]]) -> Answer:
    """Answer the three error columns from a per-trace table; risk is always no."""
    columns = ("error.check.unverified", "error.feedback.ignored", "error.safety.destructive")

    def answer(trace: str, question: str) -> float:
        """Read one answer."""
        return rows[trace][columns.index(question)] if question in columns else 0.0

    return answer


def _units(*traces: str) -> tuple[Unit, ...]:
    """Observations units with distinct text."""
    return tuple(Unit(trace, "observations", f"episode {trace}") for trace in traces)


ROWS = {
    "a": (0.95, 0.9, 0.05),  # two confident errors
    "b": (0.6, 0.05, 0.05),  # one uncertain error
    "c": (0.02, 0.03, 0.01),  # clean
    "d": (0.95, 0.4, 0.97),  # two confident, one uncertain non-fired
}


def test_error_tags_keep_uncertain_answers_and_rank_with_bounds() -> None:
    """Scores carry confident and doubtful bounds; uncertain rows are kept, not dropped."""
    tagger = ScriptedTagger(_table(ROWS))
    tags = tag_errors(_units("a", "b", "c", "d"), tagger=tagger, bank=BANK)

    assert tags.matrix.question_ids == (
        "error.check.unverified",
        "error.feedback.ignored",
        "error.safety.destructive",
    )
    assert tags.planned == 4 and not tags.failed and len(tags.matrix.rows) == 4
    b = tags.summary("b")
    assert b is not None
    assert b.fired == ("error.check.unverified",) and b.uncertain == ("error.check.unverified",)
    assert (b.errors_low, b.errors_high) == (0, 1)
    d = tags.summary("d")
    assert d is not None and (d.errors_low, d.errors_high) == (2, 3)

    ranked = rank_error_traces(tags)
    assert [row.trace_id for row in ranked] == ["d", "a", "b", "c"]
    assert ranked[0].rank == 1 and "uncertain: error.feedback.ignored" in ranked[0].reason
    confident = rank_error_traces(tags, include_uncertain=False, trace_ids={"a", "b", "c"})
    assert [row.trace_id for row in confident] == ["a", "b", "c"]
    assert confident[1].score == 0.0
    focused = rank_error_traces(tags, question_ids={"error.safety.destructive"}, limit=1)
    assert [row.trace_id for row in focused] == ["d"]
    with pytest.raises(TraceTaggingError, match="no column"):
        rank_error_traces(tags, question_ids={"error.nope"})


def test_a_refused_unit_is_recorded_and_the_rest_are_kept() -> None:
    """A refused pass is retried unit by unit; only the refused unit is missing."""
    tagger = ScriptedTagger(_table(ROWS), refuse=frozenset({"b"}))
    tags = tag_errors(_units("a", "b", "c"), tagger=tagger, bank=BANK, unit_concurrency=2)

    assert {row.trace_id for row in tags.matrix.rows} == {"a", "c"}
    assert set(tags.failed) == {"b"} and "scripted refusal" in tags.failed["b"]
    assert tagger.calls[0] == ("a", "b", "c")
    everyone = ScriptedTagger(_table(ROWS), refuse=frozenset({"a"}))
    with pytest.raises(TraceTaggingError, match="every observations unit"):
        tag_errors(_units("a"), tagger=everyone, bank=BANK)


def test_each_family_stays_on_its_view() -> None:
    """Risk propositions go to the task view and error propositions everywhere else."""
    assert bank_for_view(BANK, "task").question_ids() == ("risk.destructive",)
    assert "risk.destructive" not in bank_for_view(BANK, "step").question_ids()
    task = tag_errors(
        (Unit("a", "task", "request a"),), tagger=ScriptedTagger(_table(ROWS)), bank=BANK
    )
    assert task.matrix.question_ids == ("risk.destructive",)
    stray = BANK.model_copy(
        update={"questions": (Question(question_id="obs.x", family="obs", text="x?"),)}
    )
    with pytest.raises(QuestionBankError, match="not an error bank"):
        require_error_bank(stray)


def test_the_local_error_bank_resolves_by_family(tmp_path: Path) -> None:
    """A local bank file resolves, is checked and can be cut to families."""
    lines = [f'bank_id = "{BANK.bank_id}"', "version = 1"]
    for question in BANK.questions:
        lines += [
            "[[questions]]",
            f'question_id = "{question.question_id}"',
            f'family = "{question.family}"',
            f'text = "{question.text}"',
        ]
    (tmp_path / f"{BANK.bank_id}.toml").write_text("\n".join(lines))
    bank = error_bank(BANK.bank_id, root=tmp_path, families=("risk",))
    assert bank.question_ids() == ("risk.destructive",)


def test_tasks_rank_by_their_worst_source_trace_and_untagged_tasks_last() -> None:
    """A task inherits its most error-laden tagged source trace."""
    tags = tag_errors(_units("a", "b", "c"), tagger=ScriptedTagger(_table(ROWS)), bank=BANK)
    ranked = rank_error_tasks(
        tags, {"t-clean": ("c",), "t-mixed": ("c", "a"), "t-none": ("zz",), "t-b": ("b",)}
    )
    assert [row.task_id for row in ranked] == ["t-mixed", "t-b", "t-clean", "t-none"]
    assert ranked[0].trace_id == "a" and ranked[-1].trace_id is None
    assert "no source trace" in ranked[-1].reason


def _trajectory(trace: str) -> TraceTrajectory:
    """A three-step trajectory: read, a failing write, a success claim."""
    return TraceTrajectory(
        trace_id=trace,
        tier="actor",
        task="fix the bug",
        steps=(
            TraceStep(index=0, locator="span:0", boundary=0, kind="tool", tool="read", result="x"),
            TraceStep(
                index=1,
                locator="span:2",
                boundary=1,
                kind="tool",
                tool="write",
                arguments={"path": "a.py"},
                result={"error": "denied"},
                failed=True,
            ),
            TraceStep(index=2, locator="msg:3", boundary=2, kind="message", message="done"),
        ),
    )


def test_step_error_tags_rank_actor_critic_points() -> None:
    """Step rows map back to their boundaries; points rank by the errors decided there."""
    steps = {
        "t1::span:0": (0.02, 0.02, 0.02),
        "t1::span:2": (0.45, 0.92, 0.55),
        "t1::msg:3": (0.97, 0.02, 0.85),
    }
    tagger = ScriptedTagger(_table(steps), refuse=frozenset({"t2::span:0"}))
    record = tag_error_steps(
        (_trajectory("t1"), _trajectory("t2"), TraceTrajectory(trace_id="t3", tier="actor")),
        tagger=tagger,
        bank=BANK,
    )
    one, two, three = record.traces
    assert one.status == "tagged" and two.status == "failed" and three.status == "empty"
    assert [step.boundary for step in one.steps] == [0, 1, 2]
    points = error_points(one)
    assert [point.point for point in points] == [1, 2]
    assert points[0].uncertain and points[0].locators == ("span:2",)
    assert [p.point for p in error_points(one, include_uncertain=False)] == [2, 1]
    assert [p.point for p in error_points(one, offered={2})] == [2]


def test_error_patterns_name_attempts_and_trajectories_render_as_units() -> None:
    """Patterns cite attempts by coalition identity; a whole attempt renders as one unit."""
    unit = trajectory_unit(_trajectory("attempt-1"))
    assert unit.view == "observations" and unit.trace_id == "attempt-1"
    assert "call write" in unit.text and '"error": "denied"' in unit.text
    tags = tag_errors(_units("a", "b", "c"), tagger=ScriptedTagger(_table(ROWS)), bank=BANK)
    patterns = error_patterns(tags, BANK, attempts={"a": "att-a", "b": "att-b"})
    assert [p.pattern_id for p in patterns] == ["error.check.unverified", "error.feedback.ignored"]
    first = patterns[0]
    assert first.attempts == ("att-a",) and first.supported_by == ("attempt:att-a",)
    assert "uncertain in 1" in first.description
    widened = error_patterns(tags, BANK, include_uncertain=True)
    assert widened[0].attempts == ("a", "b")


def test_error_records_persist_and_replay(tmp_path: Path) -> None:
    """Records are content-addressed: saving twice adopts the first, and loads verify type."""
    store = ProjectStore(tmp_path, "case").artifacts
    tags = tag_errors(_units("a", "b"), tagger=ScriptedTagger(_table(ROWS)), bank=BANK)
    reference = persist_error_tags(store, tags, code_revision="test")
    assert persist_error_tags(store, tags, code_revision="test") == reference
    assert load_error_tags(store, reference) == tags
    assert store.read(reference.artifact_id).manifest.artifact_type == ERROR_TAGS_TYPE
    steps = tag_error_steps(
        (_trajectory("t1"),),
        tagger=ScriptedTagger(lambda trace, question: 0.1),
        bank=BANK,
    )
    saved = persist_error_tags(store, steps, code_revision="test")
    assert store.read(saved.artifact_id).manifest.artifact_type == ERROR_STEPS_TYPE
    assert load_error_steps(store, saved) == steps
    with pytest.raises(ValueError, match="pinned error-tags"):
        load_error_tags(store, saved)
