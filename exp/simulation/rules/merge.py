"""Merge per-trajectory rule proposals into one rule book without averaging anything away.

Many independent analyses each propose rules about the same environment. Merging is purely
mechanical, with no model call and no fuzzy matching, so the result depends only on its inputs
and never on their order:

- Two proposals are the same rule when they share a tool and a kind and either carry identical
  predicates (compared as canonical JSON) or, where at least one of them has no predicate, state
  the same normalized sentence (lowercase, punctuation stripped, whitespace collapsed). Groups
  are the connected components of that relation.
- Contradictions are preserved. Any counter-evidence, or a predicate in one group that is the
  simple logical opposite of a predicate in another group for the same tool and kind (``not``
  of it, or ``eq``/``ne``, ``exists``/``absent``, ``lt``/``ge``, ``le``/``gt`` over the same
  operands), marks the rule contradicted and keeps every locator on both sides.
- Support and contradiction count distinct trajectories, so one long trace that repeats a rule
  many times still counts once. The counts are recorded, never compared to a threshold: a merged
  rule without counter-evidence is ``unknown`` until an evidence weighing decides it.
- Whether the evidence supports a rule is a judgment about strength and independence, not a
  count. :func:`apply_weighings` applies weighings made elsewhere (a model that read the cited
  evidence) after checking only their structure: the rule exists, every citation is one of the
  rule's own locators, no citation reaches an excluded (held-out) trace, and a decided status
  cites evidence on its side.
- Independent witnesses attach afterwards by rule identity. A refuting witness contradicts the
  rule, and a rule without a supporting witness stays diagnostic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from exp.common.core.artifacts import canonical_json_bytes, sha256_json
from exp.simulation.rules.model import (
    Predicate,
    PredicateOp,
    ResearchQuestion,
    Rule,
    RuleBook,
    RuleKind,
    RuleProposal,
    RuleStatus,
    RuleWeighing,
    ToolContract,
    TrajectoryRules,
    Witness,
)

KIND_ORDER: tuple[RuleKind, ...] = ("precondition", "effect", "observation", "invariant")
"""Stable presentation order of rule kinds."""

_COMPLEMENTS: dict[PredicateOp, PredicateOp] = {
    "eq": "ne",
    "ne": "eq",
    "exists": "absent",
    "absent": "exists",
    "lt": "ge",
    "ge": "lt",
    "le": "gt",
    "gt": "le",
}
_WHY_LIMIT = 600
_SLUG_LIMIT = 48


@dataclass(frozen=True)
class _Item:
    """One proposal together with the trajectory that made it.

    Attributes:
        trace_id: Trajectory whose analysis proposed the rule.
        proposal: The proposal itself.
        predicate_key: Canonical JSON of its predicate, or None without one.
        statement_key: Normalized statement.
    """

    trace_id: str
    proposal: RuleProposal
    predicate_key: str | None
    statement_key: str

    @property
    def identity(self) -> str:
        """The grouping key this proposal contributes to its rule's identity."""
        if self.predicate_key is not None:
            return "p:" + self.predicate_key
        return "s:" + self.statement_key


def normalize_text(text: str) -> str:
    """Normalize a sentence for exact matching.

    Args:
        text: A statement or question.

    Returns:
        The text lowercased, with punctuation removed and whitespace collapsed to single spaces.
    """
    stripped = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(stripped.split())


def canonical_predicate(predicate: Predicate) -> str:
    """Render a predicate as canonical JSON, the form used to compare predicates.

    Args:
        predicate: The predicate.

    Returns:
        Sorted-key, whitespace-free JSON.
    """
    return canonical_json_bytes(predicate).decode("utf-8")


def locator_trace(locator: str) -> str | None:
    """Return the trace a ``trace:<id>`` or ``trace:<id>:span:<span>`` locator names.

    Args:
        locator: Any evidence locator.

    Returns:
        The trace identity, or None for a locator that names no trace.
    """
    if not locator.startswith("trace:"):
        return None
    body = locator.removeprefix("trace:")
    return body.rsplit(":span:", 1)[0] if ":span:" in body else body


def merge_rules(
    contracts: dict[str, ToolContract],
    analyses: Sequence[TrajectoryRules],
    *,
    excluded: frozenset[str] = frozenset(),
) -> RuleBook:
    """Merge many per-trajectory analyses into one rule book.

    Several analyses may share a trajectory (different research stances over one trace); support
    still counts that trajectory once.

    Args:
        contracts: Tool contracts by tool name, carried into the book unchanged.
        analyses: Per-trajectory analyses, in any order.
        excluded: Trace identities no rule may be learned from or cite (held-out traces).

    Returns:
        A rule book whose rules, questions and sources are in a stable order that does not
        depend on the order of ``analyses`` or of the proposals inside them. A rule is
        ``contradicted`` when counter-evidence or an opposite predicate exists and ``unknown``
        otherwise, until :func:`apply_weighings` decides it.

    Raises:
        ValueError: An analysis comes from, or cites, an excluded trace.
    """
    for analysis in analyses:
        cited = {
            locator
            for proposal in analysis.rules
            for locator in (*proposal.evidence, *proposal.counter_evidence)
        } | {locator for question in analysis.questions for locator in question.evidence}
        reached = {analysis.trace_id} | {
            trace for trace in map(locator_trace, cited) if trace is not None
        }
        if reached & excluded:
            raise ValueError(
                f"analysis of {analysis.trace_id} reaches excluded traces "
                f"{sorted(reached & excluded)}; held-out evidence never enters the rule book"
            )
    items = [
        _Item(
            trace_id=analysis.trace_id,
            proposal=proposal,
            predicate_key=None
            if proposal.predicate is None
            else canonical_predicate(proposal.predicate),
            statement_key=normalize_text(proposal.statement),
        )
        for analysis in analyses
        for proposal in analysis.rules
    ]
    groups = _group(items)
    rules = [_rule(group, groups) for group in groups]
    rules.sort(key=_rule_order)
    return RuleBook(
        contracts=dict(sorted(contracts.items())),
        rules=tuple(rules),
        questions=_merge_questions(
            [question for analysis in analyses for question in analysis.questions]
        ),
        sources=tuple(sorted({analysis.trace_id for analysis in analyses})),
    )


def attach_witnesses(book: RuleBook, witnesses: Mapping[str, Sequence[Witness]]) -> RuleBook:
    """Attach independent witnesses to rules by rule identity.

    Witnesses are merged with any the rule already has, deduplicated and sorted. A witness that
    does not support a rule makes it contradicted; a supporting witness never lifts a
    contradiction, since the contradicting evidence is still there.

    Args:
        book: The rule book.
        witnesses: Witnesses keyed by ``rule_id``.

    Returns:
        A new rule book with the witnesses attached.

    Raises:
        ValueError: A key names no rule in the book.
    """
    known = {rule.rule_id for rule in book.rules}
    unknown = sorted(set(witnesses) - known)
    if unknown:
        raise ValueError(
            f"witnesses name rules that are not in the rule book: {', '.join(unknown)}. "
            "Key witnesses by the rule_id of a rule produced by merge_rules for this book."
        )
    rules: list[Rule] = []
    for rule in book.rules:
        attached = (*rule.witnesses, *witnesses.get(rule.rule_id, ()))
        merged = tuple(
            sorted(
                set(attached),
                key=lambda witness: (witness.kind, witness.reference, witness.supports),
            )
        )
        status: RuleStatus = rule.status
        if any(not witness.supports for witness in merged):
            status = "contradicted"
        rules.append(rule.model_copy(update={"witnesses": merged, "status": status}))
    return book.model_copy(update={"rules": tuple(rules)})


def check_weighing(rule: Rule, weighing: RuleWeighing, excluded: frozenset[str]) -> None:
    """Check only the structure of one weighing against the rule it weighs.

    Args:
        rule: The merged rule.
        weighing: The weighing proposed for it.
        excluded: Trace identities no citation may reach.

    Raises:
        ValueError: The weighing names another rule, cites a locator that is not one of the
            rule's own, cites an excluded trace, or decides a status without citing evidence on
            that side.
    """
    if weighing.rule_id != rule.rule_id:
        raise ValueError(f"weighing names rule {weighing.rule_id!r}, not {rule.rule_id!r}")
    own = {*rule.evidence, *rule.counter_evidence}
    cited = {*weighing.supporting, *weighing.contradicting}
    outside = sorted(cited - own)
    if outside:
        raise ValueError(f"weighing cites locators outside the rule's evidence: {outside}")
    reached = sorted(t for t in {locator_trace(loc) for loc in cited} if t in excluded)
    if reached:
        raise ValueError(f"weighing cites excluded traces: {reached}")
    unsupported = sorted(set(weighing.supporting) - set(rule.evidence))
    if unsupported:
        raise ValueError(f"supporting citations are not supporting evidence: {unsupported}")
    if weighing.status == "supported" and not weighing.supporting:
        raise ValueError("a supported weighing must cite at least one supporting locator")
    if weighing.status == "contradicted" and not weighing.contradicting:
        raise ValueError("a contradicted weighing must cite at least one contradicting locator")


def apply_weighings(
    book: RuleBook,
    weighings: Sequence[RuleWeighing],
    *,
    excluded: frozenset[str] = frozenset(),
) -> RuleBook:
    """Set each weighed rule's status from its structurally valid weighing.

    A rule without a weighing keeps its merged status. Every locator on both sides stays in the
    book whatever the weighing decided.

    Args:
        book: The merged rule book.
        weighings: At most one weighing per rule.
        excluded: Trace identities no citation may reach.

    Returns:
        A new rule book with the weighings attached.

    Raises:
        ValueError: A weighing names an unknown rule, a rule is weighed twice, or a weighing
            fails :func:`check_weighing`.
    """
    rules = {rule.rule_id: rule for rule in book.rules}
    by_rule: dict[str, RuleWeighing] = {}
    for weighing in weighings:
        rule = rules.get(weighing.rule_id)
        if rule is None:
            raise ValueError(f"weighing names a rule that is not in the book: {weighing.rule_id}")
        if weighing.rule_id in by_rule:
            raise ValueError(f"rule {weighing.rule_id} is weighed twice")
        check_weighing(rule, weighing, excluded)
        by_rule[weighing.rule_id] = weighing
    return book.model_copy(
        update={
            "rules": tuple(
                rule
                if rule.rule_id not in by_rule
                else rule.model_copy(
                    update={
                        "status": by_rule[rule.rule_id].status,
                        "weighing": by_rule[rule.rule_id],
                    }
                )
                for rule in book.rules
            )
        }
    )


def diagnostic_rules(book: RuleBook) -> tuple[Rule, ...]:
    """Rules that no independent witness supports.

    An environment built on any of these stays diagnostic: trace agreement alone does not make a
    rule a decision boundary.

    Args:
        book: The rule book.

    Returns:
        Rules without a supporting witness, in book order.
    """
    return tuple(
        rule for rule in book.rules if not any(witness.supports for witness in rule.witnesses)
    )


def _group(items: Sequence[_Item]) -> list[tuple[_Item, ...]]:
    """Partition proposals into rules: connected components of the same-rule relation."""
    parent = list(range(len(items)))

    def find(index: int) -> int:
        """Return the representative of an index's component."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        """Join two components."""
        parent[find(left)] = find(right)

    by_predicate: dict[tuple[str | None, str, str], list[int]] = {}
    by_statement: dict[tuple[str | None, str, str], list[int]] = {}
    for index, item in enumerate(items):
        scope = (item.proposal.tool, item.proposal.kind)
        if item.predicate_key is not None:
            by_predicate.setdefault((*scope, item.predicate_key), []).append(index)
        by_statement.setdefault((*scope, item.statement_key), []).append(index)
    for members in by_predicate.values():
        for index in members[1:]:
            union(members[0], index)
    for members in by_statement.values():
        if any(items[index].predicate_key is None for index in members):
            for index in members[1:]:
                union(members[0], index)
    components: dict[int, list[_Item]] = {}
    for index, item in enumerate(items):
        components.setdefault(find(index), []).append(item)
    return [tuple(sorted(members, key=_item_order)) for members in components.values()]


def _item_order(item: _Item) -> tuple[str, str, str, str]:
    """Total order on proposals, independent of input order."""
    return (
        item.trace_id,
        item.identity,
        item.proposal.statement,
        canonical_json_bytes(item.proposal).decode("utf-8"),
    )


def _negations(predicate_key: str) -> frozenset[str]:
    """Canonical forms that are the simple logical opposite of a predicate."""
    predicate = Predicate.model_validate_json(predicate_key)
    opposites = {canonical_predicate(Predicate(op="not", args=(predicate,)))}
    if predicate.op == "not":
        opposites.add(canonical_predicate(predicate.args[0]))
    complement = _COMPLEMENTS.get(predicate.op)
    if complement is not None:
        flipped = Predicate(op=complement, left=predicate.left, right=predicate.right)
        opposites.add(canonical_predicate(flipped))
    return frozenset(opposites)


def _rule(group: tuple[_Item, ...], groups: Sequence[tuple[_Item, ...]]) -> Rule:
    """Build one merged rule from its group, looking at the other groups for opposites."""
    first = group[0].proposal
    tool, kind = first.tool, first.kind
    keys = sorted({item.predicate_key for item in group if item.predicate_key is not None})
    opposites = {opposite for key in keys for opposite in _negations(key)}
    opposing: list[tuple[_Item, bool]] = [
        (item, other is group)
        for other in groups
        if other[0].proposal.tool == tool and other[0].proposal.kind == kind
        for item in other
        if item.predicate_key is not None and item.predicate_key in opposites
    ]
    countering = [item for item in group if item.proposal.counter_evidence]
    contradicting = {item.trace_id for item in countering}
    contradicting.update(item.trace_id for item, _ in opposing)
    counter_evidence = {
        locator for item in countering for locator in item.proposal.counter_evidence
    }
    counter_evidence.update(
        locator for item, inside in opposing if not inside for locator in item.proposal.evidence
    )
    support = len({item.trace_id for item in group})
    status: RuleStatus = "contradicted" if contradicting else "unknown"
    identities = sorted({item.identity for item in group})
    return Rule(
        rule_id=_rule_id(tool, kind, identities),
        tool=tool,
        kind=kind,
        statement=_statement(group),
        predicate=Predicate.model_validate_json(keys[0]) if len(keys) == 1 else None,
        status=status,
        support=support,
        contradiction=len(contradicting),
        evidence=tuple(sorted({locator for item in group for locator in item.proposal.evidence})),
        counter_evidence=tuple(sorted(counter_evidence)),
    )


def _statement(group: tuple[_Item, ...]) -> str:
    """The most frequent statement in a group, ties broken alphabetically."""
    counts: dict[str, int] = {}
    for item in group:
        counts[item.proposal.statement] = counts.get(item.proposal.statement, 0) + 1
    return min(counts, key=lambda statement: (-counts[statement], statement))


def _rule_id(tool: str | None, kind: RuleKind, identities: Sequence[str]) -> str:
    """Stable, readable rule identity from tool, kind and the group's canonical content."""
    digest = sha256_json([tool, kind, list(identities)])[:16]
    slug = "env" if tool is None else _slug(tool)
    return f"{kind}.{slug}.{digest}"


def _slug(text: str) -> str:
    """Lowercase identifier-safe form of a tool name; the digest keeps distinct names apart."""
    slug = re.sub(r"[^a-z0-9_.-]+", "-", text.lower()).strip("-._")
    return slug[:_SLUG_LIMIT].strip("-._") or "tool"


def _rule_order(rule: Rule) -> tuple[bool, str, int, str]:
    """Stable book order: tools alphabetically, environment last, then kind, then identity."""
    return (rule.tool is None, rule.tool or "", KIND_ORDER.index(rule.kind), rule.rule_id)


def _merge_questions(questions: Sequence[ResearchQuestion]) -> tuple[ResearchQuestion, ...]:
    """Deduplicate questions by tool and normalized text, keeping all evidence and reasons."""
    grouped: dict[tuple[str | None, str], list[ResearchQuestion]] = {}
    for question in questions:
        key = (question.tool, normalize_text(question.question))
        grouped.setdefault(key, []).append(question)
    merged: list[ResearchQuestion] = []
    for members in grouped.values():
        reasons = sorted({member.why_it_matters for member in members})
        why = "; ".join(reasons)
        if len(why) > _WHY_LIMIT:
            why = why[: _WHY_LIMIT - 3].rstrip() + "..."
        merged.append(
            ResearchQuestion(
                tool=members[0].tool,
                question=min(member.question for member in members),
                why_it_matters=why,
                evidence=tuple(sorted({loc for member in members for loc in member.evidence})),
            )
        )
    merged.sort(
        key=lambda question: (question.tool is None, question.tool or "", question.question)
    )
    return tuple(merged)
