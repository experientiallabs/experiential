"""Tests for building, saving, loading and printing the rules-document artifact."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.rules.rule_tags import (
    ADMISSION_TYPE,
    CompiledRule,
    RuleAdmissionReport,
    admission_record,
)
from exp.optimize.evaluation.rules.rules_document import (
    RULES_DOCUMENT_TYPE,
    compose_rules_document,
    load_rules_document,
    main,
    rules_document_identity,
    save_rules_document,
)
from exp.simulation.rules.admission import (
    AdmissionPolicy,
    AdmissionVerdict,
    RuleQuestion,
    RuleTagMatrix,
    question_id_for,
)
from exp.simulation.rules.model import Rule, RuleBook
from exp.simulation.rules.rulebook import RULES_JSON, rulebook_json


def _envelope() -> ArtifactEnvelope:
    """Fixture provenance."""
    return ArtifactEnvelope(schema_version=1, created_at=datetime.now(UTC), code_revision="test")


def _inputs(project: ProjectStore) -> tuple[ArtifactInput, ArtifactInput]:
    """Save a two-rule book and an admission run confirming one of its rules."""
    book = RuleBook(
        contracts={},
        rules=tuple(
            Rule(
                rule_id=rule_id,
                tool="bash",
                kind="observation",
                statement=statement,
                status="unknown",
                support=1,
                contradiction=0,
            )
            for rule_id, statement in (
                ("observation.bash.a", "Exit codes are shown."),
                ("observation.bash.b", "Output is never truncated."),
            )
        ),
    )
    store = project.artifacts
    rule_book = store.write(
        artifact_id="rule-book-fixture",
        artifact_type="rule-book",
        envelope=_envelope(),
        files={RULES_JSON: rulebook_json(book).encode()},
    )
    verdict = AdmissionVerdict(
        rule_id="observation.bash.a",
        admitted=True,
        status="supported",
        supporting=("t1", "t2"),
        supporting_lineages=("l1", "l2"),
        reason="2 independent lineages agree with the rule and none contradicts it.",
    )
    text = "Does a bash reply end with its exit code?"
    question = RuleQuestion(
        question_id=question_id_for(verdict.rule_id, "applies", text), role="applies", text=text
    )
    record = admission_record(
        CompiledRule(
            rule_id=verdict.rule_id, truism="Exit codes are shown.", questions=(question,)
        ),
        RuleTagMatrix(rule_id=verdict.rule_id, question_ids=(question.question_id,)),
        verdict,
    )
    report = RuleAdmissionReport(
        policy=AdmissionPolicy(),
        trajectories=("t1", "t2"),
        lineages={"t1": "l1", "t2": "l2"},
        rules=(record,),
    )
    admission = store.write_json(
        artifact_id="admission-fixture",
        artifact_type=ADMISSION_TYPE,
        envelope=_envelope(),
        files={"admission.json": report},
    )
    return artifact_input(rule_book), artifact_input(admission)


def test_saved_document_is_keyed_by_its_inputs_and_reads_back(tmp_path: Path) -> None:
    """The document renders verdicts, saves once under a stable id, and loads exactly."""
    project = ProjectStore(tmp_path, "rules")
    rule_book, admission = _inputs(project)
    text = compose_rules_document(project.artifacts, rule_book, admission)
    assert "## Confirmed rules (strong guidance)\n\n### Tool `bash`" in text
    assert "`observation.bash.b`" in text and "not tagged across trajectories" in text
    saved = save_rules_document(project, rule_book, admission, code_revision="test")
    assert saved.artifact_id == rules_document_identity(rule_book, admission)
    assert project.artifacts.read(saved.artifact_id).manifest.artifact_type == RULES_DOCUMENT_TYPE
    assert load_rules_document(project.artifacts, saved) == text
    assert save_rules_document(project, rule_book, admission, code_revision="other") == saved
    untagged = save_rules_document(project, rule_book, None, code_revision="test")
    assert untagged.artifact_id != saved.artifact_id
    assert "## Confirmed rules (strong guidance)\n\nNone." in load_rules_document(
        project.artifacts, untagged
    )
    with pytest.raises(ValueError, match="manifest digest changed"):
        load_rules_document(project.artifacts, saved.model_copy(update={"sha256": "0" * 64}))
    with pytest.raises(ValueError, match="not a rules-document"):
        load_rules_document(project.artifacts, rule_book)
    with pytest.raises(ValueError, match="not a rule-book"):
        compose_rules_document(project.artifacts, admission, None)


def test_module_writes_the_document_to_a_file_without_writing_the_store(tmp_path: Path) -> None:
    """The command renders from saved artifacts to the chosen path and saves nothing."""
    project = ProjectStore(tmp_path, "rules")
    rule_book, admission = _inputs(project)
    before = project.artifacts.list_ids()
    out = tmp_path / "demo" / "rules-document.md"
    argv = [str(tmp_path), "rules", "--rule-book", rule_book.artifact_id]
    assert main([*argv, "--admission", admission.artifact_id, "--out", str(out)]) == 0
    assert out.read_text() == compose_rules_document(project.artifacts, rule_book, admission)
    assert project.artifacts.list_ids() == before
