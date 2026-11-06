"""Discrimination, answerability and redundancy filtering of authored questions, and publishing."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.simulation.mining.questions import (
    Question,
    QuestionBank,
    QuestionBankError,
    resolve_question_bank,
)
from exp.simulation.mining.tag_authoring import TagAuthor, author_questions
from exp.simulation.mining.tag_authoring_test import SNAPSHOT, ScriptedAuthor, corpus
from exp.simulation.mining.tag_bank_selection import (
    HISTORY_DIRECTORY,
    SELECTION_TYPE,
    BankSelectionSpec,
    corpus_tag_bank,
    publish_question_bank,
    select_tag_bank,
    selected_tag_bank,
)
from exp.simulation.mining.tagging import TagMatrix, TagRow, TraceTaggingError
from exp.simulation.mining.views import Unit

TAGGER: JsonObject = {"provider": "fixture", "model_id": "decision-fixture"}


@dataclass
class PlantedTagger:
    """Answers each question from the planted property its text names in brackets.

    ``[A]``, ``[B]`` and ``[C]`` read whether the unit text carries that property; ``[E]`` is a
    softer copy of ``[A]``; ``[D]`` splits like ``[A]`` but always inside the ambiguity band.

    Attributes:
        refuse: Raise for every call.
        calls: Banks asked, in order.
    """

    refuse: bool = False
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        """Answer every question about every unit."""
        if self.refuse:
            raise TraceTaggingError("scripted refusal")
        self.calls.append(bank.question_ids())
        rows = []
        for unit in units:
            values = []
            for question in bank.questions:
                marker = re.match(r"\[(\w)\]", question.text)
                name = marker.group(1) if marker else "D"
                held = f"PROP_{'A' if name in 'DE' else name}" in unit.text
                values.append(
                    (0.65 if held else 0.35)
                    if name == "D"
                    else (0.85 if held else 0.15)
                    if name == "E"
                    else (0.95 if held else 0.05)
                )
            rows.append(
                TagRow(trace_id=unit.trace_id, unit_hash=unit.unit_hash, values=tuple(values))
            )
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            question_ids=bank.question_ids(),
            view=units[0].view,
            rows=tuple(rows),
        )


def candidate() -> QuestionBank:
    """The planted authored candidate bank."""
    record = author_questions(
        TagAuthor(draw=ScriptedAuthor(), model=SNAPSHOT), scope="family-x", units=corpus()
    )
    return record.candidate_bank("authored-family-x")


def test_selection_keeps_discriminating_answerable_distinct_questions() -> None:
    """Constant, ambiguous and duplicate questions are dropped with their reasons."""
    tagger = PlantedTagger()
    selection = select_tag_bank(candidate(), tagger, units=corpus(), tagger_identity=TAGGER)

    assert selection.kept() == ("task_shape.makes_file", "environment_state.tabular")
    reasons = {score.question_id: score.reason for score in selection.scores}
    assert reasons["task_shape.always"].startswith("near-constant")
    assert reasons["environment_state.vague"].startswith("unanswerable")
    assert reasons["task_shape.makes_output"].startswith("redundant")
    assert reasons["behavior_error.checks_first"].startswith("redundant")
    duplicate = next(s for s in selection.scores if s.question_id == "task_shape.makes_output")
    assert duplicate.correlated_with == "task_shape.makes_file"
    assert len(tagger.calls) == 2  # one pass per view, each asking only its families
    assert set(selection.sample) == {"task", "observations"}


def test_a_lower_correlation_ceiling_and_question_limit_change_the_bank() -> None:
    """The limit keeps the most balanced questions first."""
    selection = select_tag_bank(
        candidate(),
        PlantedTagger(),
        units=corpus(),
        tagger_identity=TAGGER,
        spec=BankSelectionSpec(max_questions=1),
    )
    assert selection.kept() == ("task_shape.makes_file",)
    tabular = next(s for s in selection.scores if s.question_id == "environment_state.tabular")
    assert "limit" in tabular.reason


def test_selection_refuses_missing_views_and_an_empty_result() -> None:
    """A view the sample does not render is refused; nothing surviving is refused."""
    only_task = tuple(u for u in corpus() if u.view == "task")
    with pytest.raises(TraceTaggingError, match="observations"):
        select_tag_bank(candidate(), PlantedTagger(), units=only_task, tagger_identity=TAGGER)
    constant = QuestionBank(
        bank_id="b",
        version=1,
        questions=(Question(question_id="task_shape.c", family="task_shape", text="[C] Any?"),),
    )
    with pytest.raises(QuestionBankError, match="discriminates"):
        select_tag_bank(constant, PlantedTagger(), units=only_task, tagger_identity=TAGGER)


def test_selection_replays_from_its_artifact(tmp_path: Path) -> None:
    """The saved selection replays without asking the tagger."""
    store = ProjectStore(tmp_path, "case").artifacts
    bank = candidate()
    first, reference = selected_tag_bank(
        store, bank, PlantedTagger(), units=corpus(), tagger_identity=TAGGER, code_revision="t"
    )
    assert store.read(reference.artifact_id).manifest.artifact_type == SELECTION_TYPE
    again, same = selected_tag_bank(
        store,
        bank,
        PlantedTagger(refuse=True),
        units=corpus(),
        tagger_identity=TAGGER,
        code_revision="t",
    )
    assert again == first and same == reference


def test_publishing_versions_banks_and_never_reuses_an_identity(tmp_path: Path) -> None:
    """Same content keeps its version; changed content bumps it; a reworded id is suffixed."""
    one = QuestionBank(
        bank_id="authored-x",
        version=7,
        questions=(Question(question_id="task_shape.a", family="task_shape", text="Is A?"),),
    )
    first, path = publish_question_bank(one, root=tmp_path)
    assert first.version == 1 and path == tmp_path / "authored-x.toml"
    assert resolve_question_bank("authored-x", root=tmp_path) == first
    assert publish_question_bank(one, root=tmp_path)[0] == first
    reworded = one.model_copy(
        update={
            "questions": (
                Question(question_id="task_shape.a", family="task_shape", text="Is A true?"),
                Question(question_id="task_shape.b", family="task_shape", text="Is B?"),
            )
        }
    )
    second, _ = publish_question_bank(reworded, root=tmp_path)
    assert second.version == 2
    assert second.question_ids() == ("task_shape.a.r2", "task_shape.b")
    assert resolve_question_bank("authored-x", root=tmp_path) == second
    history = sorted(p.name for p in (tmp_path / HISTORY_DIRECTORY / "authored-x").iterdir())
    assert history == ["v1.toml", "v2.toml"]


def test_corpus_tag_bank_authors_selects_publishes_and_replays(tmp_path: Path) -> None:
    """The whole chain runs once, then replays with refusing models and the same bank."""
    store = ProjectStore(tmp_path, "case").artifacts
    banks = tmp_path / "banks"
    result = corpus_tag_bank(
        store,
        TagAuthor(draw=ScriptedAuthor(), model=SNAPSHOT),
        PlantedTagger(),
        scope="family-x",
        bank_id="authored-family-x",
        units=corpus(),
        tagger_identity=TAGGER,
        root=banks,
        code_revision="t",
    )
    assert result.bank.version == 1
    assert result.bank.question_ids() == ("task_shape.makes_file", "environment_state.tabular")
    assert resolve_question_bank("authored-family-x", root=banks) == result.bank
    again = corpus_tag_bank(
        store,
        TagAuthor(draw=ScriptedAuthor(refuse=True), model=SNAPSHOT),
        PlantedTagger(refuse=True),
        scope="family-x",
        bank_id="authored-family-x",
        units=corpus(),
        tagger_identity=TAGGER,
        root=banks,
        code_revision="t",
    )
    assert again == result
