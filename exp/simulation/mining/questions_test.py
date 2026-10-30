"""Tests for packaged and local decision-question banks."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS
from exp.simulation.mining.questions import (
    DEFAULT_BANK_ID,
    Question,
    QuestionBank,
    QuestionBankError,
    load_question_bank,
    packaged_bank_ids,
    packaged_question_bank,
)

_BANK = """
bank_id = "fixture-v1"
version = 2

[[questions]]
question_id = "sit.clean"
family = "situation"
text = "Was the case straightforward?"

[[questions]]
question_id = "req.retrieve"
family = "requirement"
text = "Must the agent find facts?"
"""


def _write(tmp_path: Path, text: str, name: str = "bank.toml") -> Path:
    """Write bank text to a temporary file and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_packaged_situation_bank_loads_without_any_project_setup() -> None:
    """An out-of-repo harness imports the bank and gets its questions, with no store or network."""
    bank = packaged_question_bank()
    assert bank.bank_id == DEFAULT_BANK_ID
    assert bank.version == 1
    assert len(bank.questions) == 28
    assert {question.family for question in bank.questions} == {"situation"}
    assert bank.question_ids()[0] == "sit.clean"


def test_packaged_situation_bank_fits_one_decision_call() -> None:
    """The bank is sized so one unit costs one call; chunking exists for larger banks."""
    assert len(packaged_question_bank().questions) <= MAX_DECISION_QUESTIONS


def test_packaged_bank_ids_lists_what_can_be_loaded() -> None:
    """An unknown bank id must be answerable with the set that does exist."""
    assert DEFAULT_BANK_ID in packaged_bank_ids()


def test_packaged_bank_question_text_is_the_full_instruction() -> None:
    """The text is sent verbatim to the model, so it must not be an abbreviated label."""
    bank = packaged_question_bank()
    ambiguous = next(q for q in bank.questions if q.question_id == "sit.ambiguous_matches")
    assert ambiguous.text == (
        "Was the task completed in a single step, with no intermediate "
        "lookups?"
    )


def test_unknown_packaged_bank_names_the_available_banks() -> None:
    """A typo in a bank id must say what to type instead."""
    with pytest.raises(QuestionBankError, match="no packaged question bank"):
        packaged_question_bank("situation-v9")


def test_bank_id_cannot_escape_the_packaged_directory() -> None:
    """A bank id becomes a resource path, so traversal is refused before any read."""
    with pytest.raises(QuestionBankError, match="lowercase components"):
        packaged_question_bank("../../../etc/passwd")


def test_families_filters_and_preserves_bank_order(tmp_path: Path) -> None:
    """Column order is the matrix contract, so filtering must not reorder what remains."""
    bank = load_question_bank(_write(tmp_path, _BANK))
    situation = bank.families("situation")
    assert situation.question_ids() == ("sit.clean",)
    assert situation.bank_id == bank.bank_id
    assert situation.version == bank.version
    assert bank.families().question_ids() == ("sit.clean", "req.retrieve")
    assert bank.families("requirement", "situation").question_ids() == (
        "sit.clean",
        "req.retrieve",
    )


def test_families_refuses_a_family_the_bank_does_not_declare(tmp_path: Path) -> None:
    """Silently returning an empty bank would buy decisions about nothing."""
    bank = load_question_bank(_write(tmp_path, _BANK))
    with pytest.raises(QuestionBankError, match="has no family"):
        bank.families("risk")


def test_load_question_bank_reports_a_missing_file(tmp_path: Path) -> None:
    """A wrong path is a user error and must say so, not raise an opaque OSError."""
    with pytest.raises(QuestionBankError, match="could not be read"):
        load_question_bank(tmp_path / "absent.toml")


def test_load_question_bank_reports_malformed_toml(tmp_path: Path) -> None:
    """A half-edited bank must fail at load, not at the first decision call."""
    with pytest.raises(QuestionBankError, match="not valid UTF-8 TOML"):
        load_question_bank(_write(tmp_path, "bank_id = "))


def test_load_question_bank_reports_a_missing_field(tmp_path: Path) -> None:
    """A question without text has no instruction to send, so the bank is invalid."""
    text = '\nbank_id = "x-v1"\nversion = 1\n\n[[questions]]\nquestion_id = "a"\nfamily = "f"\n'
    with pytest.raises(QuestionBankError, match="does not declare"):
        load_question_bank(_write(tmp_path, text))


def test_duplicate_question_ids_are_refused(tmp_path: Path) -> None:
    """Two columns with one identity would make a cached answer ambiguous."""
    repeated = _BANK.replace('question_id = "req.retrieve"', 'question_id = "sit.clean"')
    with pytest.raises(QuestionBankError, match="repeats question ids"):
        load_question_bank(_write(tmp_path, repeated))


def test_bank_is_frozen() -> None:
    """A bank enters cache keys, so a caller must not be able to edit it in place."""
    bank = QuestionBank(
        bank_id="x-v1",
        version=1,
        questions=(Question(question_id="a", family="f", text="Is it so?"),),
    )
    with pytest.raises(ValueError, match="frozen"):
        bank.version = 2
