"""Offline tests for executing rule challenges in a bound world."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.rules.challenges import (
    ChallengeCase,
    ProgramChallenges,
    RuntimeChallenges,
    run_challenges,
)
from exp.optimize.evaluation.rules.stage import RULE_BOOK_TYPE, rulebook_files
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    INITIAL,
    TOOLS,
    grounded_fixture,
)
from exp.runtime.environments.world_contracts import ToolEffectPolicy
from exp.simulation.induction.contracts import TransitionProgram
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.rules.model import (
    Operand,
    Predicate,
    Rule,
    RuleBook,
    ToolContract,
    Transition,
    Witness,
)
from exp.simulation.rules.workloads import boundary_workloads

CLOSE = TransitionProgram(
    tool="close",
    source="""def transition(state, args, facts):
    ticket = state["tickets"]
    if ticket.get("lock") != "none":
        raise ToolError({"error": "ticket is held by another agent"})
    ticket["status"] = "closed"
    return {"status": "closed"}
""",
)
"""A close program that refuses a held ticket."""

POLICIES = (ToolEffectPolicy(name="close", reads=("tickets",), writes=("tickets",)),)


def _lock_rule(rule_id: str, members: list[str], **update: object) -> Rule:
    """A close precondition requiring the lock to be one of ``members``."""
    rule = Rule(
        rule_id=rule_id,
        tool="close",
        kind="precondition",
        statement=f"close needs the lock in {members}",
        predicate=Predicate(
            op="in",
            left=Operand(scope="before", path="/tickets/lock"),
            right=Operand(value=list(members)),
        ),
        status="unknown",
        support=1,
        contradiction=0,
    )
    return rule.model_copy(update=update)


def _save(project: ProjectStore, book: RuleBook, name: str) -> ArtifactInput:
    """Save a rule book artifact."""
    return artifact_input(
        project.artifacts.write(
            artifact_id=name,
            artifact_type=RULE_BOOK_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test"
            ),
            files=rulebook_files(book),
        )
    )


def _cases(book: RuleBook) -> tuple[ChallengeCase, ...]:
    """Workloads of the book over the ticket state, with ``held`` as an observed lock."""
    return tuple(
        ChallengeCase(workload=workload)
        for workload in boundary_workloads(
            book, INITIAL, observed_values={"before:/tickets/lock": ("held",)}
        )
    )


def test_program_challenges_confirm_contradict_and_replay_without_executing(
    tmp_path: Path,
) -> None:
    """The free-lock rule is supported, the held-only rule contradicted and kept diagnostic."""
    project = ProjectStore(tmp_path, "challenges")
    witness = Witness(kind="native", reference="trace:native:1", supports=True)
    book = RuleBook(
        contracts={"close": ToolContract(name="close")},
        rules=(
            _lock_rule("free-lock", ["none"], status="supported"),
            _lock_rule("held-only", ["held"]),
            _lock_rule("held-witnessed", ["held"], witnesses=(witness,)),
        ),
    )
    reference = _save(project, book, "challenged-book")
    cases = _cases(book)
    with ProgramSandbox() as sandbox:
        executor = ProgramChallenges((CLOSE,), POLICIES, sandbox)
        report = run_challenges(
            project,
            book=book,
            rulebook=reference,
            cases=cases,
            executor=executor,
            code_revision="test",
        )
    replay = run_challenges(
        project,
        book=book,
        rulebook=reference,
        cases=cases,
        executor=executor,
        code_revision="test",
    )

    assert replay == report, "the closed sandbox would refuse any execution"
    assert {run.status for run in report.runs} == {"executed"}
    rows = {row.rule_id: row for row in report.rules}
    assert (rows["free-lock"].status, len(rows["free-lock"].consistent)) == ("supported", 2)
    assert rows["held-only"].status == "contradicted" and len(rows["held-only"].contradicting) == 2
    assert rows["held-witnessed"].status == "unknown" and rows["held-witnessed"].witnessed
    assert report.diagnostic == ("held-only",)
    assert report.world_defects == ("held-witnessed",)
    updated = {rule.rule_id: rule for rule in report.book.rules}
    assert updated["held-only"].status == "contradicted"
    assert all(ref.startswith("challenge:") for ref in updated["free-lock"].evidence)
    assert updated["free-lock"].witnesses == ()


def test_a_tool_without_a_program_stays_unexecuted_unless_observed(tmp_path: Path) -> None:
    """A predicted tool's workload uses a supplied observation or stays unexecuted."""
    project = ProjectStore(tmp_path, "challenges")
    book = RuleBook(
        contracts={"close": ToolContract(name="close")},
        rules=(_lock_rule("free-lock", ["none"], status="supported"),),
    )
    reference = _save(project, book, "predicted-book")
    cases = _cases(book)
    observed = {
        cases[0].key: Transition(
            before=INITIAL, after=INITIAL, action=cases[0].workload.action, observation={"ok": 1}
        )
    }
    with ProgramSandbox() as sandbox:
        report = run_challenges(
            project,
            book=book,
            rulebook=reference,
            cases=cases,
            executor=ProgramChallenges((), POLICIES, sandbox, observed=observed),
            code_revision="test",
        )

    assert [run.status for run in report.runs] == ["executed", "unexecuted"]
    (row,) = report.rules
    assert row.status == "supported" and row.unexecuted == (cases[1].key,)


def test_runtime_challenges_execute_in_the_grounded_world_and_replay_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grounded model answers each workload state; a replay makes no prediction."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    rule = Rule(
        rule_id="sla.free",
        tool="check_sla",
        kind="precondition",
        statement="The SLA reads only a ticket nobody holds.",
        predicate=Predicate(
            op="in",
            left=Operand(scope="before", path="/tickets/lock"),
            right=Operand(value=["none"]),
        ),
        status="unknown",
        support=1,
        contradiction=0,
    )
    book = RuleBook(
        contracts={tool.name: ToolContract(name=tool.name) for tool in TOOLS}, rules=(rule,)
    )
    reference = _save(fixture.project, book, "grounded-book")
    cases = _cases(book)
    with RuntimeChallenges(fixture.project, fixture.definition) as executor:
        report = run_challenges(
            fixture.project,
            book=book,
            rulebook=reference,
            cases=cases,
            executor=executor,
            code_revision="test",
        )
    calls = len(fixture.predictor.requests)
    with RuntimeChallenges(fixture.project, fixture.definition) as executor:
        replay = run_challenges(
            fixture.project,
            book=book,
            rulebook=reference,
            cases=cases,
            executor=executor,
            code_revision="test",
        )

    assert calls == 2 and len(fixture.predictor.requests) == calls
    assert replay == report
    assert [run.case.workload.expected for run in report.runs] == ["permitted", "refused"]
    assert [run.triage.verdict for run in report.runs if run.triage] == [
        "consistent_with_rule",
        "consistent_with_rule",
    ]
    (row,) = report.rules
    assert (row.prior, row.status) == ("unknown", "supported")
    held = report.runs[1].transition
    assert held is not None and held.observation == {"error": "ticket is held by another agent"}
