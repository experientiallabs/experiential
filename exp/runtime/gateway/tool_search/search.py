"""BM25 and regular-expression search over deferred tool definitions.

Small on purpose: the corpus is one request's tool list (at most a few
thousand short documents), so a plain in-memory BM25 with a lowercase
alphanumeric tokenizer is exact enough and needs no dependency. Regular
expressions run on RE2 (linear time, bounded memory, the engine the
guardrails already use), so a model-supplied pattern cannot backtrack a
worker thread into a denial of service; length and document bounds apply too.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import re2

from exp.runtime.gateway.contracts import GatewayToolDefinition
from exp.runtime.gateway.tool_search.contracts import (
    DEFAULT_TOOL_SEARCH_LIMIT,
    MAXIMUM_TOOL_SEARCH_LIMIT,
    MAXIMUM_TOOL_SEARCH_PATTERN_CHARACTERS,
    MAXIMUM_TOOL_SEARCH_QUERY_CHARACTERS,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or of to in on for with is are be was were do does did what which who whom "
    "whose how when where why i me my we our you your it its this that these those there here "
    "can could should would will shall may might must please want need use using".split()
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_MAXIMUM_DOCUMENT_CHARACTERS = 4000
_BM25_K1 = 1.2
_BM25_B = 0.75
_RE2_MAX_MEMORY = 262_144


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, splitting camelCase and snake_case names.

    Args:
        text: Free text or an identifier.

    Returns:
        The token list (duplicates kept for term frequency).
    """
    spaced = _CAMEL.sub(" ", text.replace("_", " ").replace("-", " ").replace(".", " "))
    return _TOKEN.findall(spaced.lower())


def tool_document(tool: GatewayToolDefinition) -> str:
    """Render the searchable text of one tool: name, description, parameter names.

    Args:
        tool: A deferred tool definition.

    Returns:
        The bounded document text.
    """
    parts = [tool.name]
    if tool.description:
        parts.append(tool.description)
    properties = tool.parameters.get("properties") if isinstance(tool.parameters, dict) else None
    if isinstance(properties, dict):
        parts.extend(str(key) for key in properties)
        for value in properties.values():
            if isinstance(value, dict) and isinstance(value.get("description"), str):
                parts.append(str(value["description"]))
    return " ".join(parts)[:_MAXIMUM_DOCUMENT_CHARACTERS]


def clamp_limit(limit: object) -> int:
    """Bound a caller-supplied result limit.

    Args:
        limit: The raw ``limit`` argument the model sent.

    Returns:
        An integer between 1 and the ceiling; the default when absent or invalid.
    """
    if isinstance(limit, bool) or not isinstance(limit, int):
        return DEFAULT_TOOL_SEARCH_LIMIT
    return max(1, min(MAXIMUM_TOOL_SEARCH_LIMIT, limit))


def bm25_search(
    query: str, tools: Sequence[GatewayToolDefinition], *, limit: int
) -> list[GatewayToolDefinition]:
    """Rank tools by BM25 relevance to ``query``.

    Args:
        query: Natural-language query.
        tools: Deferred tool definitions (the corpus).
        limit: Most results to return.

    Returns:
        Tools with a positive score, best first, at most ``limit``.
    """
    terms = [
        term
        for term in tokenize(query[:MAXIMUM_TOOL_SEARCH_QUERY_CHARACTERS])
        if term not in _STOPWORDS
    ]
    if not terms or not tools:
        return []
    documents = [tokenize(tool_document(tool)) for tool in tools]
    lengths = [len(document) for document in documents]
    average = max(1.0, sum(lengths) / len(lengths))
    frequencies = [Counter(document) for document in documents]
    document_count = len(documents)
    scored: list[tuple[float, int]] = []
    for index, counts in enumerate(frequencies):
        score = 0.0
        for term in set(terms):
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            containing = sum(1 for other in frequencies if term in other)
            idf = math.log(1.0 + (document_count - containing + 0.5) / (containing + 0.5))
            normalized = frequency * (_BM25_K1 + 1.0)
            denominator = frequency + _BM25_K1 * (
                1.0 - _BM25_B + _BM25_B * lengths[index] / average
            )
            score += idf * normalized / denominator
        if score > 0.0:
            scored.append((score, index))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [tools[index] for _score, index in scored[:limit]]


class ToolSearchPatternError(ValueError):
    """The model's regular expression is unusable."""


def regex_search(
    pattern: str, tools: Sequence[GatewayToolDefinition], *, limit: int
) -> list[GatewayToolDefinition]:
    """Return tools whose name or description matches ``pattern`` (case-insensitive).

    Args:
        pattern: Regular expression from the model.
        tools: Deferred tool definitions.
        limit: Most results to return, in declaration order.

    Returns:
        Matching tools in declaration order.

    Raises:
        ToolSearchPatternError: The pattern is too long or is not valid RE2.
    """
    if not pattern or len(pattern) > MAXIMUM_TOOL_SEARCH_PATTERN_CHARACTERS:
        raise ToolSearchPatternError("pattern must be 1-200 characters")
    options = re2.Options()
    options.max_mem = _RE2_MAX_MEMORY
    options.log_errors = False
    options.case_sensitive = False
    try:
        compiled = re2.compile(pattern, options=options)
    except re2.error:
        raise ToolSearchPatternError(
            "invalid pattern: RE2 syntax only (no backreferences or lookaround)"
        ) from None
    matched: list[GatewayToolDefinition] = []
    for tool in tools:
        haystack = f"{tool.name}\n{tool.description or ''}"[:_MAXIMUM_DOCUMENT_CHARACTERS]
        if compiled.search(haystack):
            matched.append(tool)
            if len(matched) >= limit:
                break
    return matched
