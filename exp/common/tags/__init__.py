"""Tag contracts: question banks, exchanges, windows, tag results and the tag rule language.

Tags are true/false questions a decision model answers with a probability about captured
traffic. This package holds only pure contracts and rendering; asking the questions lives in
the runtime.
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
from exp.common.tags.exchange import (
    TagExchange,
    TagFunction,
    TagImagePart,
    TagMessage,
    TagTextPart,
    TagTool,
    TagToolCall,
    opening_key,
    tag_facts,
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
from exp.common.tags.windows import (
    TagWindow,
    TagWindowBudget,
    opening_window,
    render_message,
    turn_windows,
)

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
    "TagExchange",
    "TagFacts",
    "TagFunction",
    "TagId",
    "TagImagePart",
    "TagMessage",
    "TagRule",
    "TagRuleError",
    "TagScope",
    "TagSummary",
    "TagTextPart",
    "TagTool",
    "TagToolCall",
    "TagWindow",
    "TagWindowBudget",
    "WindowScores",
    "evaluate_rule",
    "opening_key",
    "opening_window",
    "parse_rule",
    "render_message",
    "summarize",
    "tag_facts",
    "turn_windows",
]
