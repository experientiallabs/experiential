"""Offline tests for condensing the weighed rule book and research into world-model memory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.condense import (
    MEMORY_TYPE,
    WorldModelMemory,
    condense_memory,
    load_memory,
)
from exp.optimize.evaluation.rules.research import (
    ResearchDraw,
    TraceResearch,
    TraceResearchAnswer,
)
from exp.optimize.evaluation.rules.research_test import (
    SMALL,
    AgentScript,
    agent_setup,
    research_answer,
    wire,
)
from exp.optimize.evaluation.rules.stage import RULE_BOOK_TYPE, rulebook_files
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.contracts_test import refund_trace
from exp.simulation.rules.merge import apply_weighings, merge_rules
from exp.simulation.rules.model import (
    ResearchQuestion,
    RuleBook,
    RuleProposal,
    RuleWeighing,
    TrajectoryRules,
)


def _book() -> RuleBook:
    """A supported rule, a contradicted rule with both sides, and an unknown rule."""
    contracts = parse_tool_contracts((refund_trace("alpha"), refund_trace("beta")))

    def proposal(statement: str, trace: str, counter: tuple[str, ...] = ()) -> RuleProposal:
        """One refund proposal citing the trace's refused refund."""
        return RuleProposal(
            tool="refund",
            kind="precondition" if "paid" in statement else "observation",
            statement=statement,
            evidence=(f"trace:{trace}:span:r2",),
            counter_evidence=counter,
            confidence=0.5,
        )

    merged = merge_rules(
        contracts,
        [
            TrajectoryRules(
                trace_id="alpha",
                rules=(
                    proposal("A refund above the paid amount is refused.", "alpha"),
                    proposal("A refund needs an amount.", "alpha", ("trace:alpha:span:r3",)),
                ),
                questions=(
                    ResearchQuestion(
                        tool="refund",
                        question="Is a partial refund allowed?",
                        why_it_matters="Bounds the amount precondition.",
                        evidence=("trace:alpha:span:c3",),
                    ),
                ),
                limitations=("x",),
            ),
            TrajectoryRules(
                trace_id="beta",
                rules=(proposal("A refund reports ok.", "beta"),),
                limitations=("x",),
            ),
        ],
    )
    by = {rule.statement: rule for rule in merged.rules}
    paid = by["A refund above the paid amount is refused."]
    return apply_weighings(
        merged,
        [
            RuleWeighing(
                rule_id=paid.rule_id,
                status="supported",
                rationale="Directly observed refusal.",
                supporting=paid.evidence,
            )
        ],
    )


def _research() -> tuple[TraceResearch, ...]:
    """Saved-looking research histories for both traces."""
    return tuple(
        TraceResearch(
            trace_id=trace,
            draws=(
                ResearchDraw(
                    trace_id=trace,
                    variant="contract_first",
                    identity=f"draw-{trace}",
                    answer=TraceResearchAnswer.model_validate(research_answer(trace)),
                ),
            ),
        )
        for trace in ("alpha", "beta")
    )


def _save_book(project: ProjectStore, book: RuleBook) -> ArtifactInput:
    """Save the book as a rule-book artifact."""
    return artifact_input(
        project.artifacts.write(
            artifact_id="book",
            artifact_type=RULE_BOOK_TYPE,
            envelope=ArtifactEnvelope.model_validate(
                {"schema_version": 1, "created_at": "2026-01-01T00:00:00Z", "code_revision": "t"}
            ),
            files=rulebook_files(book),
        )
    )


def _condense(
    project: ProjectStore, settings: AuthorSettings, book: RuleBook, reference: ArtifactInput
) -> tuple[ArtifactInput, WorldModelMemory]:
    """Condense the book and research."""
    return condense_memory(
        project,
        settings,
        SMALL,
        book,
        reference,
        _research(),
        identity="condense",
        code_revision="t",
    )


def test_condensation_keeps_contradictions_and_unknowns_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A condenser that writes only settled behavior still leaves both other kinds kept."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    book = _book()
    reference = _save_book(project, book)
    supported = next(rule for rule in book.rules if rule.status == "supported")

    def memory(role: str, opening: JsonObject) -> JsonObject:
        """Write the supported rule only."""
        assert role == "condensation" and opening["rules_by_status"] == {
            "supported": 1,
            "contradicted": 1,
            "unknown": 1,
        }
        return {
            "summary": "Refunds are bounded by the paid amount.",
            "entries": [
                {
                    "kind": "behavior",
                    "tool": "refund",
                    "text": "A refund above the paid amount is refused.",
                    "rule_ids": [supported.rule_id],
                    "evidence": list(supported.evidence),
                }
            ],
        }

    transport = AgentScript(memory)
    wire(monkeypatch, transport)

    saved, condensed = _condense(project, settings, book, reference)

    contradicted = next(rule for rule in book.rules if rule.status == "contradicted")
    unknown = next(rule for rule in book.rules if rule.status == "unknown")
    assert condensed.condenser_failure is None and condensed.condenser is not None
    kept = {(entry.kind, entry.rule_ids, entry.kept_by) for entry in condensed.entries}
    assert ("contradiction", (contradicted.rule_id,), "host") in kept
    assert ("unknown", (unknown.rule_id,), "host") in kept
    assert [note.rule_id for note in condensed.contradictions] == [contradicted.rule_id]
    assert condensed.contradictions[0].counter_evidence == ("trace:alpha:span:r3",)
    sources = {note.source for note in condensed.unknowns}
    assert sources == {"rule", "question", "research"}
    assert [digest.unresolved for digest in condensed.research] == [
        ("What happens at exactly the paid amount?",)
    ] * 2
    stored = project.artifacts.read(saved.artifact_id)
    assert stored.manifest.artifact_type == MEMORY_TYPE
    markdown = stored.payloads["memory.md"].decode()
    assert "## Contradictions" in markdown and "Against: `trace:alpha:span:r3`" in markdown
    assert "(kept by host)" in markdown and "## Research history" in markdown
    assert load_memory(project, saved) == condensed
    assert json.loads(stored.payloads["memory.json"])["rulebook"] == reference.model_dump(
        mode="json"
    )
    draft_check = [
        row["host_validation"]
        for row in json.loads(
            project.artifacts.read_bytes(condensed.condenser.artifact_id, "transcript.json")
        )["outputs"]
        if row["phase"] == "draft"
    ]
    assert "not kept by a matching entry" in draft_check[0]

    assert _condense(project, settings, book, reference) == (saved, condensed)
    assert len(transport.requests) == 5


def test_a_condenser_citing_outside_the_evidence_leaves_host_sections_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uncited locators refuse the draft; the memory still keeps every hard matter."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    book = _book()
    reference = _save_book(project, book)

    def invented(role: str, opening: JsonObject) -> JsonObject:
        """Cite a locator nothing supplied."""
        del role, opening
        return {
            "summary": "s",
            "entries": [{"kind": "behavior", "text": "t", "evidence": ["trace:zeta:span:1"]}],
        }

    wire(monkeypatch, AgentScript(invented))

    _, condensed = _condense(project, settings, book, reference)

    assert condensed.condenser_failure is not None
    assert "trace:zeta:span:1" in condensed.condenser_failure
    assert all(entry.kept_by == "host" for entry in condensed.entries)
    assert {entry.kind for entry in condensed.entries} == {"contradiction", "unknown"}
