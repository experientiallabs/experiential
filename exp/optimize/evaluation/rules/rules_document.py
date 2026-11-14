"""Build, save and load the ``rules-document`` artifact from saved rule artifacts only.

The document collates a saved rule book and, when one exists, its saved tag-admission run into
one Markdown file (:func:`~exp.simulation.rules.rules_document.render_rules_document`). Building
it makes no model call: it reads two immutable artifacts and renders them. Its identity is a
``stable_id`` over the protocol and the two input manifest digests, so the same inputs always
name the same artifact.

Run as a module to write the document to a file without touching the store::

    python -m exp.optimize.evaluation.rules.rules_document ROOT PROJECT \
        --rule-book RULE_BOOK_ID --admission ADMISSION_ID --out rules-document.md
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, stable_id
from exp.common.core.files import write_text_atomic
from exp.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactStore,
    ProjectStore,
    artifact_input,
)
from exp.optimize.evaluation.rules.rule_tags import ADMISSION_TYPE, RuleAdmissionReport
from exp.simulation.rules.admission import AdmissionVerdict
from exp.simulation.rules.model import RuleBook
from exp.simulation.rules.rulebook import RULES_JSON
from exp.simulation.rules.rules_document import render_rules_document

RuntimeRules = Literal["recall", "document"]
"""How served worlds read learned rules: per-step decision-model recall, or the whole document."""

RULES_DOCUMENT_TYPE = "rules-document"
"""Artifact type of a saved rules document."""

RULES_DOCUMENT_FILE = "rules.md"
"""File name of the document inside its artifact."""

RULES_DOCUMENT_PROTOCOL = "rules-document-v1"
"""Version of the document layout; bound into every document identity."""

_RULE_BOOK_TYPE = "rule-book"
_ADMISSION_FILE = "admission.json"

logger = logging.getLogger(__name__)


def rules_document_identity(rule_book: ArtifactInput, admission: ArtifactInput | None) -> str:
    """Stable identity of the document over one rule book and an optional admission run.

    Args:
        rule_book: The saved rule book, with its manifest digest.
        admission: The saved tag-admission run, or None when the book was never tagged.

    Returns:
        The artifact identity.
    """
    return stable_id(
        RULES_DOCUMENT_TYPE,
        {
            "protocol": RULES_DOCUMENT_PROTOCOL,
            "rule_book": rule_book.model_dump(mode="json"),
            "admission": None if admission is None else admission.model_dump(mode="json"),
        },
    )


def compose_rules_document(
    store: ArtifactStore, rule_book: ArtifactInput, admission: ArtifactInput | None
) -> str:
    """Render the document from saved artifacts, reading the store only.

    Args:
        store: Artifact store holding both inputs; never written.
        rule_book: The saved ``rule-book`` artifact.
        admission: The saved ``rule-tag-admission`` artifact, or None.

    Returns:
        The Markdown document.

    Raises:
        ValueError: An input is not the artifact type it should be, or its manifest digest
            differs from the reference.
    """
    book = RuleBook.model_validate_json(_payload(store, rule_book, _RULE_BOOK_TYPE, RULES_JSON))
    verdicts: tuple[AdmissionVerdict, ...] = ()
    if admission is not None:
        raw = _payload(store, admission, ADMISSION_TYPE, _ADMISSION_FILE)
        verdicts = RuleAdmissionReport.model_validate_json(raw).verdicts()
    return render_rules_document(book, verdicts)


def save_rules_document(
    project: ProjectStore,
    rule_book: ArtifactInput,
    admission: ArtifactInput | None,
    *,
    code_revision: str,
) -> ArtifactInput:
    """Build and save the document, or adopt the identical one already saved.

    Args:
        project: Project holding the inputs and receiving the document.
        rule_book: The saved rule book.
        admission: The saved tag-admission run, or None.
        code_revision: Producing source identity.

    Returns:
        The saved ``rules-document`` artifact.
    """
    store = project.artifacts
    artifact_id = rules_document_identity(rule_book, admission)
    if store.exists(artifact_id):
        return artifact_input(store.read(artifact_id).manifest)
    text = compose_rules_document(store, rule_book, admission)
    inputs = (rule_book,) if admission is None else (rule_book, admission)
    try:
        manifest = store.write(
            artifact_id=artifact_id,
            artifact_type=RULES_DOCUMENT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={RULES_DOCUMENT_FILE: text.encode()},
        )
    except ArtifactAlreadyExistsError:
        manifest = store.read(artifact_id).manifest
    return artifact_input(manifest)


def load_rules_document(store: ArtifactStore, reference: ArtifactInput) -> str:
    """Read the text of one exact saved rules document.

    Args:
        store: Project artifact store.
        reference: The document artifact and its manifest digest.

    Returns:
        The Markdown document.

    Raises:
        ValueError: The artifact is not a rules document or its manifest digest changed.
    """
    return _payload(store, reference, RULES_DOCUMENT_TYPE, RULES_DOCUMENT_FILE).decode()


def _payload(store: ArtifactStore, reference: ArtifactInput, kind: str, name: str) -> bytes:
    """One file of an exact saved artifact of the expected type."""
    stored = store.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(
            f"artifact {reference.artifact_id} manifest digest changed; pass its current reference"
        )
    if stored.manifest.artifact_type != kind:
        raise ValueError(
            f"artifact {reference.artifact_id} is a {stored.manifest.artifact_type}, not a {kind}"
        )
    payload = stored.payloads.get(name)
    if payload is None:
        raise ValueError(f"artifact {reference.artifact_id} holds no {name}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Write a rules document to a file from saved artifacts, without writing the store.

    Args:
        argv: Command arguments; the process arguments when None.

    Returns:
        The process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="python -m exp.optimize.evaluation.rules.rules_document",
        description="Render a saved rule book and its tag admission as one rules document.",
    )
    parser.add_argument("root", type=Path, help="project store root")
    parser.add_argument("project", help="project name")
    parser.add_argument("--rule-book", required=True, help="saved rule-book artifact id")
    parser.add_argument("--admission", help="saved rule-tag-admission artifact id")
    parser.add_argument("--out", type=Path, required=True, help="Markdown file to write")
    args = parser.parse_args(argv)
    store = ProjectStore(args.root, args.project).artifacts
    rule_book = artifact_input(store.read(args.rule_book).manifest)
    admission = (
        None if args.admission is None else artifact_input(store.read(args.admission).manifest)
    )
    text = compose_rules_document(store, rule_book, admission)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.out, text)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(
        "Wrote %s (%d characters); the store artifact would be %s",
        args.out,
        len(text),
        rules_document_identity(rule_book, admission),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
