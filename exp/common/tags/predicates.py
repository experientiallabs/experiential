"""A small rule language over tag probabilities, parsed to a JSON-serializable tree.

Grammar, with keywords matched case-insensitively::

    expr  := or
    or    := and ("OR" and)*
    and   := unary ("AND" unary)*
    unary := "NOT" unary | atom
    atom  := "(" expr ")" | ["all" "("] tag [")"] [cmp number]
    cmp   := ">" | ">=" | "<" | "<="

A bare tag reads its highest score across windows (true in any window) and ``all(tag)`` reads
its lowest (true in every window). A tag with no comparison means ``>= 0.5``. Conversation tags
have a single opening window, so both forms read the same score. Example::

    rule = parse_rule("coding AND (tool_error OR all(model_mistake) > 0.8) AND NOT model_refused")
    matched = evaluate_rule(rule, summarize(tags, opening))

The parsed ``TagRule`` names each tag's scope and score index, so another engine can compile it
to a query without reparsing the text.
"""

from __future__ import annotations

import difflib
import operator
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, Sha256
from exp.common.tags.bank import DEFAULT_TAG_BANK, TagBank, TagId, TagScope
from exp.common.tags.records import TagSummary

DEFAULT_THRESHOLD = 0.5
MAX_RULE_CHARS = 4096

Comparison = Literal[">", ">=", "<", "<="]
Aggregate = Literal["max", "min"]

_TOKEN = re.compile(
    r"(?P<number>\d+(?:\.\d*)?|\.\d+)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<cmp>>=|<=|>|<)"
    r"|(?P<paren>[()])"
)
_KEYWORDS = frozenset({"and", "or", "not"})
_OPERATORS: dict[str, Comparison] = {">": ">", ">=": ">=", "<": "<", "<=": "<="}
_COMPARE: dict[Comparison, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


class TagRuleError(ValueError):
    """Raised when a rule cannot be parsed or evaluated; the message says how to fix it."""


class TagCondition(ContractModel):
    """Compare one tag's aggregated score with a threshold.

    Attributes:
        kind: Node discriminator, always ``tag``.
        tag: The tag id.
        scope: The tag's scope, which selects the conversation or turn score array.
        index: The tag's position in its scope's score array.
        aggregate: ``max`` for a bare tag, ``min`` for ``all(tag)``.
        op: Comparison operator.
        threshold: Probability the aggregated score is compared with.
    """

    kind: Literal["tag"] = "tag"
    tag: TagId
    scope: TagScope
    index: int = Field(ge=0)
    aggregate: Aggregate
    op: Comparison
    threshold: float = Field(ge=0.0, le=1.0)


class NotRule(ContractModel):
    """Negate one rule node.

    Attributes:
        kind: Node discriminator, always ``not``.
        operand: The negated node.
    """

    kind: Literal["not"] = "not"
    operand: RuleNode


class AndRule(ContractModel):
    """Require every operand to hold.

    Attributes:
        kind: Node discriminator, always ``and``.
        operands: Two or more nodes, in source order.
    """

    kind: Literal["and"] = "and"
    operands: tuple[RuleNode, ...] = Field(min_length=2)


class OrRule(ContractModel):
    """Require at least one operand to hold.

    Attributes:
        kind: Node discriminator, always ``or``.
        operands: Two or more nodes, in source order.
    """

    kind: Literal["or"] = "or"
    operands: tuple[RuleNode, ...] = Field(min_length=2)


RuleNode = Annotated[TagCondition | NotRule | AndRule | OrRule, Field(discriminator="kind")]

NotRule.model_rebuild()
AndRule.model_rebuild()
OrRule.model_rebuild()


class TagRule(ContractModel):
    """A parsed rule bound to the bank whose score order it indexes.

    Attributes:
        source: The rule text as written.
        bank_id: Bank the tag indexes refer to.
        bank_version: Version of that bank.
        bank_digest: ``TagBank.digest`` of that bank, so scores from edited wording or order
            under the same version are rejected.
        conversation_tags: Number of conversation tags, the exact conversation score count.
        turn_tags: Number of turn tags, the exact turn score count.
        root: The rule tree.
    """

    source: str = Field(min_length=1, max_length=MAX_RULE_CHARS)
    bank_id: ArtifactId
    bank_version: int = Field(ge=1)
    bank_digest: Sha256
    conversation_tags: int = Field(ge=0)
    turn_tags: int = Field(ge=0)
    root: RuleNode

    @model_validator(mode="after")
    def _indexes_in_range(self) -> TagRule:
        """Require every condition to index inside its scope's score array.

        A rule read back from JSON skips the parser, so this keeps a hand-edited or foreign
        tree from reading past the scores at evaluation time.

        Returns:
            The validated rule.

        Raises:
            ValueError: A condition's index is not below its scope's tag count.
        """
        for condition in self.conditions():
            count = (
                self.conversation_tags
                if condition.scope is TagScope.CONVERSATION
                else self.turn_tags
            )
            if condition.index >= count:
                raise ValueError(
                    f"tag {condition.tag!r} has {condition.scope.value} index {condition.index} "
                    f"but the bank has {count} {condition.scope.value} tags; parse the rule again"
                )
        return self

    def conditions(self) -> tuple[TagCondition, ...]:
        """Return every tag condition in source order.

        Returns:
            The rule's leaves.
        """
        return tuple(_walk(self.root))


def _walk(node: RuleNode) -> Iterator[TagCondition]:
    """Yield the tag conditions under one node in source order.

    Args:
        node: Any rule node.

    Yields:
        Each leaf condition.
    """
    if isinstance(node, TagCondition):
        yield node
    elif isinstance(node, NotRule):
        yield from _walk(node.operand)
    else:
        for operand in node.operands:
            yield from _walk(operand)


@dataclass(frozen=True)
class _Token:
    """One lexical token.

    Attributes:
        kind: ``number``, ``name``, ``cmp``, ``paren`` or ``end``.
        text: The exact source text.
        position: Zero-based character offset in the rule.
    """

    kind: str
    text: str
    position: int


def _tokenize(source: str) -> tuple[_Token, ...]:
    """Split a rule into tokens.

    Args:
        source: The rule text.

    Returns:
        The tokens followed by one ``end`` token.

    Raises:
        TagRuleError: A character cannot start any token.
    """
    tokens: list[_Token] = []
    position = 0
    while position < len(source):
        if source[position].isspace():
            position += 1
            continue
        match = _TOKEN.match(source, position)
        if match is None or match.lastgroup is None:
            raise _error(source, position, f"unexpected character {source[position]!r}")
        tokens.append(_Token(match.lastgroup, match.group(), position))
        position = match.end()
    tokens.append(_Token("end", "", len(source)))
    return tuple(tokens)


def _error(source: str, position: int, message: str) -> TagRuleError:
    """Build an error that points at the offending position.

    Args:
        source: The rule text.
        position: Offset of the problem.
        message: What went wrong and, when useful, what to write instead.

    Returns:
        The error, with the rule and a caret under the position.
    """
    return TagRuleError(f"position {position}: {message}\n  {source}\n  {' ' * position}^")


def _describe(token: _Token) -> str:
    """Name a token for an error message.

    Args:
        token: The token found.

    Returns:
        A short description such as ``'AND'`` or ``end of rule``.
    """
    return "end of rule" if token.kind == "end" else repr(token.text)


class _Parser:
    """Recursive descent parser for one rule against one bank."""

    def __init__(self, source: str, bank: TagBank) -> None:
        """Tokenize the rule.

        Args:
            source: The rule text.
            bank: Bank that resolves tag ids to scopes and indexes.
        """
        self._source = source
        self._bank = bank
        self._tokens = _tokenize(source)
        self._position = 0

    def parse(self) -> RuleNode:
        """Parse the whole rule.

        Returns:
            The rule tree.

        Raises:
            TagRuleError: The rule is malformed or names an unknown tag.
        """
        node = self._or()
        token = self._peek()
        if token.kind != "end":
            raise self._fail(
                token, f"expected AND, OR or the end of the rule, found {_describe(token)}"
            )
        return node

    def _peek(self) -> _Token:
        """Return the current token without consuming it."""
        return self._tokens[self._position]

    def _advance(self) -> _Token:
        """Consume and return the current token."""
        token = self._tokens[self._position]
        self._position += 1
        return token

    def _keyword(self, word: str) -> bool:
        """Report whether the current token is the given keyword.

        Args:
            word: Lowercase keyword.

        Returns:
            True when the current token is that keyword in any letter case.
        """
        token = self._peek()
        return token.kind == "name" and token.text.lower() == word

    def _fail(self, token: _Token, message: str) -> TagRuleError:
        """Build an error positioned at a token.

        Args:
            token: The offending token.
            message: What went wrong.

        Returns:
            The positioned error.
        """
        return _error(self._source, token.position, message)

    def _or(self) -> RuleNode:
        """Parse ``and ("OR" and)*``.

        Returns:
            One node, or an ``OrRule`` over two or more.
        """
        operands = [self._and()]
        while self._keyword("or"):
            self._advance()
            operands.append(self._and())
        return operands[0] if len(operands) == 1 else OrRule(operands=tuple(operands))

    def _and(self) -> RuleNode:
        """Parse ``unary ("AND" unary)*``.

        Returns:
            One node, or an ``AndRule`` over two or more.
        """
        operands = [self._unary()]
        while self._keyword("and"):
            self._advance()
            operands.append(self._unary())
        return operands[0] if len(operands) == 1 else AndRule(operands=tuple(operands))

    def _unary(self) -> RuleNode:
        """Parse ``"NOT" unary | atom``.

        Returns:
            The parsed node.
        """
        if self._keyword("not"):
            self._advance()
            return NotRule(operand=self._unary())
        return self._atom()

    def _atom(self) -> RuleNode:
        """Parse a parenthesized expression or one tag condition.

        Returns:
            The parsed node.

        Raises:
            TagRuleError: The atom is malformed.
        """
        token = self._peek()
        if token.kind == "paren" and token.text == "(":
            self._advance()
            node = self._or()
            self._expect_close(f"to match the '(' at position {token.position}")
            return node
        aggregate: Aggregate = "max"
        nxt = self._tokens[self._position + 1] if token.kind != "end" else token
        if token.kind == "name" and token.text.lower() == "all":
            if not (nxt.kind == "paren" and nxt.text == "("):
                raise self._fail(token, "write all(tag), for example all(tool_error)")
            self._advance()
            self._advance()
            aggregate = "min"
        tag_token = self._advance()
        tag_id, scope, index = self._resolve(tag_token)
        if aggregate == "min":
            self._expect_close("after the tag in all(...)")
        op: Comparison = ">="
        threshold = DEFAULT_THRESHOLD
        if self._peek().kind == "cmp":
            op = _OPERATORS[self._advance().text]
            threshold = self._threshold(self._advance())
        return TagCondition(
            tag=tag_id, scope=scope, index=index, aggregate=aggregate, op=op, threshold=threshold
        )

    def _expect_close(self, context: str) -> None:
        """Consume a closing parenthesis.

        Args:
            context: Where the parenthesis belongs, for the error message.

        Raises:
            TagRuleError: The current token is not ``)``.
        """
        token = self._peek()
        if not (token.kind == "paren" and token.text == ")"):
            raise self._fail(token, f"expected ')' {context}, found {_describe(token)}")
        self._advance()

    def _resolve(self, token: _Token) -> tuple[str, TagScope, int]:
        """Resolve a tag token against the bank.

        Args:
            token: Token expected to name a tag.

        Returns:
            The tag id, its scope and its score index.

        Raises:
            TagRuleError: The token is not a name, is a keyword, or names no tag in the bank.
        """
        if token.kind != "name" or token.text.lower() in _KEYWORDS:
            raise self._fail(token, f"expected a tag id, found {_describe(token)}")
        ids = self._bank.ids
        if token.text not in ids:
            close = difflib.get_close_matches(token.text.lower(), ids, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise self._fail(
                token,
                f"unknown tag {token.text!r}{hint}; valid tags in {self._bank.bank_id} "
                f"v{self._bank.version}: {', '.join(ids)}",
            )
        tag = self._bank.get(token.text)
        return tag.id, tag.scope, self._bank.index(tag.id)

    def _threshold(self, token: _Token) -> float:
        """Read a probability threshold.

        Args:
            token: Token expected to be a number.

        Returns:
            The threshold.

        Raises:
            TagRuleError: The token is not a number in ``[0, 1]``.
        """
        if token.kind != "number":
            raise self._fail(
                token, f"expected a threshold between 0 and 1, found {_describe(token)}"
            )
        value = float(token.text)
        if not 0.0 <= value <= 1.0:
            raise self._fail(token, f"threshold {token.text} must be between 0 and 1")
        return value


def parse_rule(source: str, bank: TagBank = DEFAULT_TAG_BANK) -> TagRule:
    """Parse rule text against a bank.

    Args:
        source: Rule text, for example ``"coding AND NOT all(tool_error) < 0.2"``.
        bank: Bank whose tag ids the rule may use.

    Returns:
        The parsed rule, bound to the bank's id and version.

    Raises:
        TagRuleError: The rule is empty, too long, malformed, or names an unknown tag.
    """
    if not source.strip():
        raise TagRuleError("a rule needs at least one tag, for example: coding AND tool_error")
    if len(source) > MAX_RULE_CHARS:
        raise TagRuleError(f"a rule may hold at most {MAX_RULE_CHARS} characters")
    root = _Parser(source, bank).parse()
    return TagRule(
        source=source,
        bank_id=bank.bank_id,
        bank_version=bank.version,
        bank_digest=bank.digest,
        conversation_tags=len(bank.for_scope(TagScope.CONVERSATION)),
        turn_tags=len(bank.for_scope(TagScope.TURN)),
        root=root,
    )


def _score_array(condition: TagCondition, summary: TagSummary) -> Sequence[float]:
    """Select the score array one condition reads.

    Args:
        condition: The tag condition.
        summary: The exchange's summary.

    Returns:
        The conversation scores or the turn maximum or minimum array.

    Raises:
        TagRuleError: A conversation tag is used without conversation scores.
    """
    if condition.scope is TagScope.CONVERSATION:
        if summary.conversation is None:
            raise TagRuleError(
                f"the rule reads conversation tag {condition.tag!r} but the summary has no "
                "conversation scores; summarize with the exchange that tagged the opening"
            )
        return summary.conversation
    return summary.turn_max if condition.aggregate == "max" else summary.turn_min


def _evaluate(node: RuleNode, summary: TagSummary) -> bool:
    """Evaluate one node.

    Args:
        node: The rule node.
        summary: The exchange's summary.

    Returns:
        Whether the node holds.
    """
    if isinstance(node, TagCondition):
        return _COMPARE[node.op](_score_array(node, summary)[node.index], node.threshold)
    if isinstance(node, NotRule):
        return not _evaluate(node.operand, summary)
    results = (_evaluate(operand, summary) for operand in node.operands)
    return all(results) if isinstance(node, AndRule) else any(results)


def evaluate_rule(rule: TagRule, summary: TagSummary) -> bool:
    """Decide whether one exchange matches a rule.

    Args:
        rule: A rule from ``parse_rule``.
        summary: The exchange's summary from ``summarize``; it must carry conversation scores
            when the rule uses a conversation tag.

    Returns:
        True when the exchange matches.

    Raises:
        TagRuleError: The rule was parsed for another bank, version or bank content, a score
            array does not hold exactly one score per tag of its scope, or a conversation tag
            has no conversation scores.
    """
    if (rule.bank_id, rule.bank_version) != (summary.bank_id, summary.bank_version):
        raise TagRuleError(
            f"rule was parsed for {rule.bank_id} v{rule.bank_version} but the scores come from "
            f"{summary.bank_id} v{summary.bank_version}; parse the rule against that bank"
        )
    if rule.bank_digest != summary.bank_digest:
        raise TagRuleError(
            f"the scores come from different {rule.bank_id} v{rule.bank_version} wording or "
            "order than the rule was parsed for; parse the rule against the bank that tagged them"
        )
    for name, scores, expected in (
        ("turn", summary.turn_max, rule.turn_tags),
        ("conversation", summary.conversation, rule.conversation_tags),
    ):
        if scores is not None and len(scores) != expected:
            raise TagRuleError(
                f"expected {expected} {name} scores, one per {name} tag, but got {len(scores)}"
            )
    for condition in rule.conditions():
        _score_array(condition, summary)
    return _evaluate(rule.root, summary)


__all__ = [
    "DEFAULT_THRESHOLD",
    "MAX_RULE_CHARS",
    "AndRule",
    "NotRule",
    "OrRule",
    "RuleNode",
    "TagCondition",
    "TagRule",
    "TagRuleError",
    "evaluate_rule",
    "parse_rule",
]
