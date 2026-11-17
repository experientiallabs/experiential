"""Tag contracts: question banks, per-exchange tag results and the tag rule language.

Tags are true/false questions a decision model answers with a probability about captured
traffic. This package holds only pure contracts; asking the questions lives in the runtime.
"""

from exp.common.tags.bank import (
    DEFAULT_TAG_BANK,
    MAX_TAG_TEXT_CHARS,
    MAX_TAGS_PER_SCOPE,
    RESERVED_TAG_IDS,
    TagBank,
    TagDefinition,
    TagId,
    TagScope,
)
from exp.common.tags.predicates import (
    DEFAULT_THRESHOLD,
    AndRule,
    NotRule,
    OrRule,
    RuleNode,
    TagCondition,
    TagRule,
    TagRuleError,
    evaluate_rule,
    parse_rule,
)
from exp.common.tags.records import ExchangeTags, TagFacts, TagSummary, WindowScores, summarize

__all__ = [
    "DEFAULT_TAG_BANK",
    "DEFAULT_THRESHOLD",
    "MAX_TAGS_PER_SCOPE",
    "MAX_TAG_TEXT_CHARS",
    "RESERVED_TAG_IDS",
    "AndRule",
    "ExchangeTags",
    "NotRule",
    "OrRule",
    "RuleNode",
    "TagBank",
    "TagCondition",
    "TagDefinition",
    "TagFacts",
    "TagId",
    "TagRule",
    "TagRuleError",
    "TagScope",
    "TagSummary",
    "WindowScores",
    "evaluate_rule",
    "parse_rule",
    "summarize",
]
