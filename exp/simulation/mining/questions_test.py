"""Tests for local decision-question banks and the packaged example."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS
from exp.simulation.mining.questions import (
    EXAMPLE_BANK_ID,
    LOCAL_BANK_DIRECTORY,
    LOCAL_BANK_ROOT_VARIABLE,
    Question,
    QuestionBank,
    QuestionBankError,
    load_question_bank,
    local_bank_root,
    packaged_bank_ids,
    packaged_question_bank,
    resolve_question_bank,
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


def test_the_packaged_example_loads_without_any_project_setup() -> None:
    """Tests and documentation need one bank that is always there, with no store or network."""
    bank = packaged_question_bank()
    assert bank.bank_id == EXAMPLE_BANK_ID
    assert bank.version == 1
    assert bank.questions
    assert bank.question_ids()[0] == "example.single_step"


def test_the_packaged_example_is_the_only_bank_in_the_distribution() -> None:
    """A real bank's wording is the asset and is local configuration, so it never ships."""
    assert packaged_bank_ids() == (EXAMPLE_BANK_ID,)


def test_the_packaged_example_fits_one_decision_call() -> None:
    """One unit costs one call at this size; chunking exists for a bank that outgrows it."""
    assert len(packaged_question_bank().questions) <= MAX_DECISION_QUESTIONS


def test_packaged_bank_question_text_is_the_full_instruction() -> None:
    """The text is sent verbatim to the model, so it must not be an abbreviated label."""
    bank = packaged_question_bank()
    single = next(q for q in bank.questions if q.question_id == "example.single_step")
    assert single.text.endswith("?")
    assert len(single.text.split()) > 5


def test_a_local_bank_is_preferred_over_a_packaged_one(tmp_path: Path) -> None:
    """A real bank lives outside the distribution, so a local file must win by identity alone."""
    local = _BANK.replace('bank_id = "fixture-v1"', f'bank_id = "{EXAMPLE_BANK_ID}"')
    _write(tmp_path, local, f"{EXAMPLE_BANK_ID}.toml")

    bank = resolve_question_bank(EXAMPLE_BANK_ID, root=tmp_path)

    assert bank.question_ids() == ("sit.clean", "req.retrieve")
    assert packaged_question_bank(EXAMPLE_BANK_ID).question_ids() != bank.question_ids()


def test_resolution_falls_back_to_the_packaged_example(tmp_path: Path) -> None:
    """The example stays loadable with nothing installed, which is what tests rely on."""
    assert resolve_question_bank(EXAMPLE_BANK_ID, root=tmp_path).bank_id == EXAMPLE_BANK_ID


def test_an_unresolvable_bank_names_both_places_it_looked(tmp_path: Path) -> None:
    """The fix is either installing the bank or fixing the id, and the message cannot tell which."""
    with pytest.raises(QuestionBankError) as caught:
        resolve_question_bank("situation-v1", root=tmp_path)

    message = str(caught.value)
    assert str(tmp_path / "situation-v1.toml") in message
    assert EXAMPLE_BANK_ID in message
    assert LOCAL_BANK_ROOT_VARIABLE in message


def test_resolution_never_substitutes_a_bank_for_the_one_asked_for(tmp_path: Path) -> None:
    """Tagging with the wrong propositions still returns a matrix, so it must fail instead."""
    _write(tmp_path, _BANK, "fixture-v1.toml")

    with pytest.raises(QuestionBankError, match="no question bank named"):
        resolve_question_bank("requirement-v1", root=tmp_path)


def test_a_resolved_bank_id_cannot_escape_the_bank_directory(tmp_path: Path) -> None:
    """A bank id becomes a path component, so traversal is refused before any read."""
    with pytest.raises(QuestionBankError, match="lowercase components"):
        resolve_question_bank("../../../etc/passwd", root=tmp_path)


def test_the_local_bank_root_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bank kept outside the working directory still has to be readable."""
    monkeypatch.delenv(LOCAL_BANK_ROOT_VARIABLE, raising=False)
    assert local_bank_root() == LOCAL_BANK_DIRECTORY

    monkeypatch.setenv(LOCAL_BANK_ROOT_VARIABLE, "/srv/banks")
    assert local_bank_root() == Path("/srv/banks")

    monkeypatch.setenv(LOCAL_BANK_ROOT_VARIABLE, "   ")
    assert local_bank_root() == LOCAL_BANK_DIRECTORY


def test_unknown_packaged_bank_names_the_available_banks() -> None:
    """A typo in a bank id must say what to type instead."""
    with pytest.raises(QuestionBankError, match="no packaged question bank"):
        packaged_question_bank("example-v9")


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
