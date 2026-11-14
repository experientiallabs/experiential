"""Tests for collating a rule book and its tag admission into one rules document."""

from __future__ import annotations

import pytest

from exp.simulation.rules.admission import AdmissionVerdict
from exp.simulation.rules.model import ResearchQuestion, Rule, RuleBook, RuleWeighing
from exp.simulation.rules.rules_document import (
    EVIDENCE_LIMIT,
    RULES_DOCUMENT_HEADER,
    render_rules_document,
    tag_verdict,
)


def _rule(rule_id: str, tool: str | None, statement: str, **fields: object) -> Rule:
    """A rule with five evidence locators and the given overrides."""
    values: dict[str, object] = {
        "rule_id": rule_id,
        "tool": tool,
        "kind": "observation",
        "statement": statement,
        "status": "unknown",
        "support": 1,
        "contradiction": 0,
        "evidence": tuple(f"trace:t{i}:span:s{i}" for i in range(5)),
    }
    return Rule.model_validate(values | fields)


def _book() -> RuleBook:
    """Four rules over two tools and the environment, with questions on both."""
    return RuleBook(
        contracts={},
        rules=(
            _rule(
                "observation.bash.ok",
                "bash",
                "A bash call shows its exit code.",
                status="supported",
                weighing=RuleWeighing(
                    rule_id="observation.bash.ok",
                    status="supported",
                    rationale="Every recorded call ends with an exit line.",
                ),
            ),
            _rule("effect.bash.pipe", "bash", "A pipe masks the exit code.", kind="effect"),
            _rule("observation.read.bad", "read", "Reads never truncate."),
            _rule("invariant.env.untagged", None, "The workspace is writable.", kind="invariant"),
        ),
        questions=(
            ResearchQuestion(
                tool="bash",
                question="Does pipefail change the exit code?",
                why_it_matters="Decides whether the pipe rule holds.",
                evidence=("trace:t1:span:s1",),
            ),
            ResearchQuestion(
                tool=None,
                question="Is /tmp writable?",
                why_it_matters="Scratch writes depend on it.",
            ),
        ),
    )


def _verdicts() -> tuple[AdmissionVerdict, ...]:
    """Confirmed, unconfirmed and contradicted verdicts; the environment rule is untagged."""
    return (
        AdmissionVerdict(
            rule_id="observation.bash.ok",
            admitted=True,
            status="supported",
            supporting=("a", "b", "c", "d"),
            supporting_lineages=("la", "lb", "lc"),
            uncertain=("e",),
            reason="3 independent lineages agree with the rule and none contradicts it.",
        ),
        AdmissionVerdict(
            rule_id="effect.bash.pipe",
            admitted=False,
            status="unknown",
            supporting=("a",),
            supporting_lineages=("la",),
            not_applicable=("b", "c"),
            reason="Only 1 independent lineage agrees; 2 are required.",
        ),
        AdmissionVerdict(
            rule_id="observation.read.bad",
            admitted=False,
            status="contradicted",
            supporting=("a",),
            supporting_lineages=("la",),
            contradicting=("x", "y"),
            reason="2 trajectories contradict the rule.",
        ),
    )


def test_document_sections_rules_by_tag_verdict_with_counts_and_research() -> None:
    """Rules land under their verdict, by tool, with counts, reasoning, evidence and research."""
    text = render_rules_document(_book(), _verdicts())
    assert text.startswith(RULES_DOCUMENT_HEADER)
    assert "strong guidance" in text and "diagnostic context" in text
    assert "Never refuse for missing context." in text
    assert "- Rules: 4 (3 tagged across trajectories)" in text
    confirmed = text.index("## Confirmed rules")
    unconfirmed = text.index("## Unconfirmed rules")
    contradicted = text.index("## Contradicted rules")
    research = text.index("## Research questions")
    assert confirmed < text.index("`observation.bash.ok`") < unconfirmed
    assert unconfirmed < text.index("`effect.bash.pipe`") < contradicted
    assert unconfirmed < text.index("`invariant.env.untagged`") < contradicted
    assert contradicted < text.index("`observation.read.bad`") < research
    assert (
        "- Tag support: 4 supporting (3 independent lineages), 0 contradicting, 1 uncertain, "
        "0 not applicable, of 5 trajectories"
    ) in text
    assert "- Tag verdict: unconfirmed (not tagged across trajectories)" in text
    assert "- Reasoning: Every recorded call ends with an exit line." in text
    assert "- Book status: supported (1 supporting, 0 contradicting" in text
    assert "  - `x`" in text and "  - `y`" in text
    assert f"  - and {5 - EVIDENCE_LIMIT} more" in text
    assert "`trace:t4:span:s4`" not in text
    tail = text[research:]
    assert tail.index("### Tool `bash`") < tail.index("### Environment")
    assert "  - Why it matters: Decides whether the pipe rule holds." in tail
    assert render_rules_document(_book(), _verdicts()) == text


def test_untagged_book_is_all_diagnostic_and_duplicate_verdicts_refuse() -> None:
    """Without admission every rule is unconfirmed; one rule cannot have two verdicts."""
    text = render_rules_document(_book())
    assert "## Confirmed rules (strong guidance)\n\nNone." in text
    assert "- Unconfirmed rules: 4" in text
    assert tag_verdict(None) == "unconfirmed"
    assert [tag_verdict(v) for v in _verdicts()] == ["confirmed", "unconfirmed", "contradicted"]
    with pytest.raises(ValueError, match="more than one admission verdict"):
        render_rules_document(_book(), (*_verdicts(), _verdicts()[0]))
