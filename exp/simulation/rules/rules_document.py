"""Collate a rule book and its tag admission into one Markdown rules document.

The document is what a grounded simulator and the world authors read in place of per-step rule
recall. Each rule carries its statement, kind, tool, rule-book status, the reasoning that weighed
it and its tag-admission verdict, with supporting and contradicting trajectory counts and a few
evidence locators. The research questions follow, with why each matters, grouped by tool.

Rules the decision tags confirmed across the other trajectories are strong guidance. Every other
rule (unconfirmed, contradicted, or never tagged) is diagnostic context: it is not the only rule
that exists and never blocks a prediction. The header says so, and says that the reader must
still answer when no rule covers a case.

Rendering is deterministic: the same book and verdicts always give byte-identical text.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from exp.simulation.rules.admission import AdmissionVerdict
from exp.simulation.rules.merge import KIND_ORDER
from exp.simulation.rules.model import ResearchQuestion, Rule, RuleBook
from exp.simulation.rules.rulebook import _code, _line, _locators

TagVerdict = Literal["confirmed", "unconfirmed", "contradicted"]
"""How tag admission left one rule: confirmed guidance, or diagnostic context."""

EVIDENCE_LIMIT = 3
"""Most evidence locators and trajectory identities listed per rule and side."""

ENVIRONMENT = "environment"
"""Group name of rules and questions that belong to no single tool."""

RULES_DOCUMENT_HEADER = """# Environment rules

How this environment behaves, learned from recorded trajectories. Each rule was proposed by
research over the traces, weighed against its evidence, and then checked with decision tags
across the other recorded trajectories.

- **Confirmed** rules were supported by the tags in enough independent trajectories and
  contradicted by none. Treat them as strong guidance.
- **Unconfirmed** rules (the tags neither confirmed nor refuted them, or they were never tagged)
  and **contradicted** rules (some tagged trajectory went against them) are diagnostic context.
  They are not the only rules that exist, they may be wrong or incomplete, and they never block
  an answer.
- When no rule covers a case, still answer: predict the most plausible behavior from the
  recorded evidence, the tool contracts and these rules. Never refuse for missing context.
"""

_SECTIONS: tuple[tuple[TagVerdict, str], ...] = (
    ("confirmed", "Confirmed rules (strong guidance)"),
    ("unconfirmed", "Unconfirmed rules (diagnostic context)"),
    ("contradicted", "Contradicted rules (diagnostic context)"),
)


def tag_verdict(verdict: AdmissionVerdict | None) -> TagVerdict:
    """Classify one rule's admission verdict; an untagged rule is unconfirmed.

    Args:
        verdict: The rule's admission verdict, or None when tag admission never ran on it.

    Returns:
        ``confirmed`` when admitted, ``contradicted`` when a tagged trajectory contradicted it,
        otherwise ``unconfirmed``.
    """
    if verdict is None:
        return "unconfirmed"
    if verdict.status == "contradicted":
        return "contradicted"
    return "confirmed" if verdict.admitted else "unconfirmed"


def render_rules_document(book: RuleBook, verdicts: Sequence[AdmissionVerdict] = ()) -> str:
    """Render one Markdown rules document from a rule book and its tag-admission verdicts.

    Args:
        book: The rule book: rules, their reasoning and the open research questions.
        verdicts: Tag-admission verdicts by rule; a book rule without one is shown as never
            tagged and unconfirmed. A verdict for a rule absent from the book is ignored.

    Returns:
        The document, ending with a newline.

    Raises:
        ValueError: Two verdicts name the same rule.
    """
    by_rule: dict[str, AdmissionVerdict] = {}
    for verdict in verdicts:
        if verdict.rule_id in by_rule:
            raise ValueError(f"rule {verdict.rule_id} has more than one admission verdict")
        by_rule[verdict.rule_id] = verdict
    classified = [(rule, tag_verdict(by_rule.get(rule.rule_id))) for rule in book.rules]
    counts = {name: sum(1 for _, v in classified if v == name) for name, _ in _SECTIONS}
    tagged = sum(1 for rule in book.rules if rule.rule_id in by_rule)
    lines = [
        RULES_DOCUMENT_HEADER,
        f"- Rules: {len(book.rules)} ({tagged} tagged across trajectories)",
        *(f"- {title.split(' (')[0]}: {counts[name]}" for name, title in _SECTIONS),
        f"- Research questions: {len(book.questions)}",
        "",
    ]
    for name, title in _SECTIONS:
        members = [rule for rule, verdict in classified if verdict == name]
        lines += [f"## {title}", ""]
        if not members:
            lines += ["None.", ""]
            continue
        for tool in _groups(rule.tool for rule in members):
            lines += [f"### {_group_title(tool)}", ""]
            grouped = [rule for rule in members if rule.tool == tool]
            grouped.sort(key=lambda rule: (KIND_ORDER.index(rule.kind), rule.rule_id))
            for rule in grouped:
                lines += _document_rule(rule, name, by_rule.get(rule.rule_id))
    lines += _research(book.questions)
    return "\n".join(lines).rstrip("\n") + "\n"


def _groups(tools: Iterable[str | None]) -> list[str | None]:
    """Distinct tool groups, tools alphabetically and the environment last."""
    return sorted(set(tools), key=lambda tool: (tool is None, tool or ""))


def _group_title(tool: str | None) -> str:
    """Heading of one tool group."""
    return ENVIRONMENT.capitalize() if tool is None else f"Tool {_code(tool)}"


def _limited(title: str, values: Sequence[str]) -> list[str]:
    """A locator list cut to :data:`EVIDENCE_LIMIT`, saying how many were left out."""
    lines = _locators(title, values[:EVIDENCE_LIMIT])
    if len(values) > EVIDENCE_LIMIT:
        lines.append(f"  - and {len(values) - EVIDENCE_LIMIT} more")
    return lines


def _document_rule(
    rule: Rule, verdict: TagVerdict, admission: AdmissionVerdict | None
) -> list[str]:
    """One rule with its kind, tool, statuses, reasoning, tag counts and evidence."""
    lines = [
        f"#### {_code(rule.rule_id)}",
        "",
        _line(rule.statement),
        "",
        f"- Kind: {rule.kind}",
        f"- Tool: {ENVIRONMENT if rule.tool is None else _code(rule.tool)}",
        f"- Book status: {rule.status} ({rule.support} supporting, "
        f"{rule.contradiction} contradicting trajectories in research)",
    ]
    if admission is None:
        lines.append(f"- Tag verdict: {verdict} (not tagged across trajectories)")
    else:
        tagged = (
            len(admission.supporting)
            + len(admission.contradicting)
            + len(admission.uncertain)
            + len(admission.not_applicable)
        )
        lines += [
            f"- Tag verdict: {verdict}",
            f"- Tag support: {len(admission.supporting)} supporting "
            f"({len(admission.supporting_lineages)} independent lineages), "
            f"{len(admission.contradicting)} contradicting, {len(admission.uncertain)} uncertain, "
            f"{len(admission.not_applicable)} not applicable, of {tagged} trajectories",
            f"- Tag reading: {_line(admission.reason)}",
        ]
    if rule.weighing is not None:
        lines.append(f"- Reasoning: {_line(rule.weighing.rationale)}")
    lines += _limited("Evidence", rule.evidence)
    if rule.counter_evidence:
        lines += _limited("Counter-evidence", rule.counter_evidence)
    if admission is not None:
        if admission.supporting:
            lines += _limited("Supporting trajectories", admission.supporting)
        if admission.contradicting:
            lines += _limited("Contradicting trajectories", admission.contradicting)
    return [*lines, ""]


def _research(questions: Sequence[ResearchQuestion]) -> list[str]:
    """Research questions with why they matter, grouped by tool.

    Evidence locators are left out: they repeat the rules' own and only lengthen the prompt.
    """
    lines = ["## Research questions", ""]
    if not questions:
        return [*lines, "None.", ""]
    for tool in _groups(question.tool for question in questions):
        lines += [f"### {_group_title(tool)}", ""]
        for question in questions:
            if question.tool == tool:
                lines += [
                    f"- {_line(question.question)}",
                    f"  - Why it matters: {_line(question.why_it_matters)}",
                ]
        lines.append("")
    return lines
