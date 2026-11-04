"""Offline tests for agentic evidence weighing of merged rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.traces import Trace
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.research import ResearchMaterial
from exp.optimize.evaluation.rules.research_test import SMALL, AgentScript, agent_setup, wire
from exp.optimize.evaluation.rules.weighing import WeighingDraw, weigh_rules_in_parallel
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.contracts_test import refund_trace
from exp.simulation.rules.merge import apply_weighings, merge_rules
from exp.simulation.rules.model import RuleBook, RuleProposal, TrajectoryRules

_STRONG = "A refund larger than the paid amount is refused."
_WEAK = "A lookup always reports the order as paid."


def _book(traces: tuple[Trace, ...]) -> RuleBook:
    """One rule with one direct witness, one rule with two correlated incidental witnesses."""
    strong = RuleProposal(
        tool="refund",
        kind="precondition",
        statement=_STRONG,
        evidence=("trace:alpha:span:r2",),
        confidence=0.9,
    )

    def weak(trace: str) -> RuleProposal:
        """The incidental lookup observation of one trace."""
        return RuleProposal(
            tool="lookup",
            kind="observation",
            statement=_WEAK,
            evidence=(f"trace:{trace}:span:r1",),
            confidence=0.3,
        )

    return merge_rules(
        parse_tool_contracts(traces),
        [
            TrajectoryRules(trace_id="alpha", rules=(strong,), limitations=("x",)),
            TrajectoryRules(trace_id="beta", rules=(weak("beta"),), limitations=("x",)),
            TrajectoryRules(trace_id="gamma", rules=(weak("gamma"),), limitations=("x",)),
        ],
    )


def _verdicts(role: str, opening: JsonObject) -> JsonObject:
    """The weigher's judgment: the direct refusal decides; the correlated lookups do not."""
    assert role == "weighing"
    rule = opening["rule"]
    assert isinstance(rule, dict)
    if rule["statement"] == _STRONG:
        return {
            "rule_id": rule["rule_id"],
            "status": "supported",
            "rationale": "The 400 refund against a 40 payment was refused: a direct observation.",
            "supporting": ["trace:alpha:span:r2"],
            "independence": "One trajectory, but it observes the boundary itself.",
        }
    return {
        "rule_id": rule["rule_id"],
        "status": "unknown",
        "rationale": "Both lookups come from the same scripted task; nothing tests unpaid orders.",
        "supporting": list(rule["evidence"]),
        "independence": "beta and gamma replay one task, so they are one weak observation.",
    }


def _weigh(
    project: ProjectStore, settings: AuthorSettings, book: RuleBook, traces: tuple[Trace, ...]
) -> tuple[WeighingDraw, ...]:
    """Weigh every rule of the book."""
    return weigh_rules_in_parallel(
        project,
        settings,
        SMALL,
        book,
        ResearchMaterial(traces, book.contracts),
        {},
        identity="weigh-run",
        excluded=frozenset({"held"}),
        code_revision="test",
        concurrency=2,
    )


def test_one_strong_witness_supports_and_two_correlated_weak_ones_stay_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weigher decides strength and independence; code only applies valid weighings."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    traces = (refund_trace("alpha"), refund_trace("beta"), refund_trace("gamma"))
    book = _book(traces)
    transport = AgentScript(_verdicts)
    wire(monkeypatch, transport)

    draws = _weigh(project, settings, book, traces)

    assert all(rule.status == "unknown" for rule in book.rules)
    weak, strong = book.rules
    assert (strong.support, weak.support) == (1, 2)
    assert [draw.failure for draw in draws] == [None, None]
    weighed = apply_weighings(book, [draw.weighing for draw in draws if draw.weighing])
    after = {rule.statement: rule for rule in weighed.rules}
    assert after[_STRONG].status == "supported"
    assert after[_WEAK].status == "unknown"
    weak_weighing = after[_WEAK].weighing
    assert weak_weighing is not None
    assert "one weak observation" in weak_weighing.independence
    assert len(transport.requests) == 10

    assert _weigh(project, settings, book, traces) == draws
    assert len(transport.requests) == 10


def test_a_weighing_citing_outside_the_rules_evidence_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A citation of another rule's or trace's locator fails the draw; the rule stays unknown."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    traces = (refund_trace("alpha"), refund_trace("beta"), refund_trace("gamma"))
    book = _book(traces)

    def invented(role: str, opening: JsonObject) -> JsonObject:
        """Support every rule with a step none of them cites."""
        answer = _verdicts(role, opening)
        answer["status"], answer["supporting"] = "supported", ["trace:beta:span:c3"]
        return answer

    wire(monkeypatch, AgentScript(invented))

    draws = _weigh(project, settings, book, traces)

    assert all(draw.weighing is None and draw.transcript is not None for draw in draws)
    assert all("outside the rule's evidence" in str(draw.failure) for draw in draws)
    weighed = apply_weighings(book, [draw.weighing for draw in draws if draw.weighing])
    assert weighed == book
