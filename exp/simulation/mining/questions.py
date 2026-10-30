"""Question banks asked of a decision model about one rendered trace view.

A bank is a frozen, ordered list of questions with stable identities. The identity and the text
travel together because a decision is cached against the exact text that produced it: rewording a
question is a new question, and reusing its identity would silently replay an answer to a question
nobody asked.

Banks ship as package data in TOML, one file per bank, so a caller gets the same questions on every
machine without a project store, a network call, or a configuration file. The packaged
``situation-v1`` bank asks what one case turned out to contain, as revealed by the environment:
whether sources disagreed, whether a lookup returned namesakes, whether no option satisfied the
requirements. Those are properties of the case rather than of the agent that worked it, which is
what makes them useful for grouping traces that one rubric covers.

Every question is a proposition, so an answer is a probability in ``[0, 1]`` and a bank's answers
form a vector of fixed width and fixed column order. Families partition a bank, because the
families need different views and are batched separately.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import ContractModel, validate_artifact_id

PACKAGED_BANK_DIRECTORY = "question_banks"
"""Package-data directory holding one TOML file per packaged bank, named ``<bank_id>.toml``."""

DEFAULT_BANK_ID = "situation-v1"
"""The packaged bank asked about the observations view: 28 propositions about the case."""


class QuestionBankError(ValueError):
    """A bank file is absent, malformed, or declares an inconsistent identity."""


class Question(ContractModel):
    """One proposition a decision model answers about a rendered view.

    Attributes:
        question_id: Stable column identity, such as ``sit.clean``. It names the proposition
            across banks, matrices, and caches, so it never changes meaning once published.
        family: The group this question belongs to, such as ``situation``. A family is asked
            about one view and is batched as a unit.
        text: The exact instruction sent to the decision model. It is part of the cache key, so
            editing it invalidates every answer previously bought for this column.
    """

    question_id: str = Field(min_length=1, max_length=256)
    family: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1)


class QuestionBank(ContractModel):
    """A frozen, ordered set of questions with one identity.

    Order is the contract: a tag vector's columns are this tuple's order, so two matrices built
    from one bank identity are directly comparable.

    Attributes:
        bank_id: Stable bank identity, also the packaged file name.
        version: Monotonic revision of this bank's content. A changed question text gets a new
            version, so a stored matrix can state which revision it was tagged against.
        questions: The questions in column order, with unique identities.
    """

    bank_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    questions: tuple[Question, ...] = Field(min_length=1)

    def families(self, *names: str) -> QuestionBank:
        """Return a copy holding only the named families, in the original order.

        Args:
            *names: Family names to keep. Passing none keeps the whole bank.

        Returns:
            A bank with the same identity and version, carrying the selected questions.

        Raises:
            QuestionBankError: A named family has no questions in this bank, or the selection
                is empty.
        """
        if not names:
            return self
        wanted = frozenset(names)
        missing = sorted(wanted - {question.family for question in self.questions})
        if missing:
            raise QuestionBankError(
                f"question bank {self.bank_id!r} has no family {missing}; "
                f"it declares {sorted({question.family for question in self.questions})}"
            )
        return self.model_copy(
            update={
                "questions": tuple(
                    question for question in self.questions if question.family in wanted
                )
            }
        )

    def question_ids(self) -> tuple[str, ...]:
        """Return the column identities in bank order."""
        return tuple(question.question_id for question in self.questions)


def _validated(bank: QuestionBank, source: str) -> QuestionBank:
    """Reject a bank whose columns are not unique, naming the file that declared it.

    Args:
        bank: Parsed bank awaiting its uniqueness check.
        source: Human-readable origin used in the error message.

    Returns:
        The same bank when every question identity is distinct.

    Raises:
        QuestionBankError: Two questions share one identity.
    """
    identities = bank.question_ids()
    duplicates = sorted({name for name in identities if identities.count(name) > 1})
    if duplicates:
        raise QuestionBankError(f"question bank {source} repeats question ids {duplicates}")
    return bank


def load_question_bank(path: Path) -> QuestionBank:
    """Read one bank from a TOML file.

    The file declares ``bank_id``, ``version``, and a ``questions`` array of tables, each with
    ``question_id``, ``family``, and ``text``.

    Args:
        path: Local TOML bank file.

    Returns:
        The parsed bank, in file order.

    Raises:
        QuestionBankError: The file is missing, is not valid TOML, does not match the bank
            contract, or repeats a question identity.
    """
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QuestionBankError(
            f"question bank {path} could not be read ({exc}); check the path"
        ) from exc
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise QuestionBankError(
            f"question bank {path} is not valid UTF-8 TOML ({exc}); fix the file"
        ) from exc
    try:
        bank = QuestionBank.model_validate(payload)
    except ValueError as exc:
        raise QuestionBankError(
            f"question bank {path} does not declare bank_id, version, and questions "
            f"with question_id, family, and text ({exc})"
        ) from exc
    return _validated(bank, str(path))


def packaged_question_bank(bank_id: str = DEFAULT_BANK_ID) -> QuestionBank:
    """Read a bank that ships inside this package.

    Args:
        bank_id: Packaged bank identity, which is also its file name. It must be a lowercase
            stable identifier so it can never escape the packaged directory.

    Returns:
        The parsed bank, in file order.

    Raises:
        QuestionBankError: The identity is not a stable identifier, no packaged bank carries it,
            or the packaged file does not declare that same identity.
    """
    try:
        validate_artifact_id(bank_id)
    except ValueError as exc:
        raise QuestionBankError(
            f"question bank id {bank_id!r} must be lowercase components separated by "
            "'.', '_', or '-'"
        ) from exc
    resource = resources.files(__package__).joinpath(
        f"{PACKAGED_BANK_DIRECTORY}/{bank_id}.toml",
    )
    try:
        text = resource.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError) as exc:
        raise QuestionBankError(
            f"no packaged question bank named {bank_id!r}; "
            f"available banks: {list(packaged_bank_ids())}"
        ) from exc
    try:
        bank = QuestionBank.model_validate(tomllib.loads(text))
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise QuestionBankError(
            f"packaged question bank {bank_id!r} is malformed ({exc}); "
            "reinstall the experiential distribution"
        ) from exc
    if bank.bank_id != bank_id:
        raise QuestionBankError(
            f"packaged question bank file {bank_id!r} declares bank_id {bank.bank_id!r}"
        )
    return _validated(bank, repr(bank_id))


def packaged_bank_ids() -> tuple[str, ...]:
    """Return every packaged bank identity, sorted.

    Returns:
        The stable identities accepted by :func:`packaged_question_bank`.
    """
    directory = resources.files(__package__).joinpath(PACKAGED_BANK_DIRECTORY)
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            entry.name.removesuffix(".toml")
            for entry in directory.iterdir()
            if entry.name.endswith(".toml")
        )
    )
