"""Bounded RE2 detection with deterministic, in-memory text redaction."""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import Literal, Protocol, cast

import re2
from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCheck,
    GuardrailCompletion,
)


class BuiltinPattern(StrEnum):
    """Deterministic detector families; these do not cover contextual personal data."""

    EMAIL = "email"
    CREDIT_CARD = "credit_card"
    API_KEY = "api_key"


_BUILTINS = {
    BuiltinPattern.EMAIL: r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",
    BuiltinPattern.CREDIT_CARD: r"\b[0-9](?:[ -]?[0-9]){12,18}\b",
    BuiltinPattern.API_KEY: (
        r"\b(?:sk-(?:proj-|ant-api[0-9]+-)?[A-Za-z0-9_-]{20,}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        r"|xpl_[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b"
    ),
}
_MAX_MATCHES = 4096
_MAX_TEXT_BYTES = 1_048_576


class RegexAdapterDocument(ContractModel):
    """Author a local rule using custom RE2 expressions and optional built-in families.

    All overlapping matches are replaced by one literal replacement. Capture
    expansion is intentionally unavailable, so replacements cannot echo secrets.
    """

    adapter_id: ArtifactId
    kind: Literal["regex"] = "regex"
    patterns: tuple[str, ...] = Field(default=(), max_length=32)
    builtin_patterns: tuple[BuiltinPattern, ...] = ()
    replacement: str = Field(default="[REDACTED]", min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_patterns(self) -> RegexAdapterDocument:
        """Reject empty rules and excessive or duplicate pattern definitions."""
        if not self.patterns and not self.builtin_patterns:
            raise ValueError("regex requires patterns or builtin_patterns; add at least one")
        if len(set(self.builtin_patterns)) != len(self.builtin_patterns):
            raise ValueError("builtin_patterns must be unique; remove duplicate families")
        if any(not pattern or len(pattern.encode("utf-8")) > 1024 for pattern in self.patterns):
            raise ValueError("each regex pattern must contain 1 to 1024 UTF-8 bytes")
        return self


class _Match(Protocol):
    """The RE2 match operations consumed at the untyped library boundary."""

    def span(self) -> tuple[int, int]:
        """Return Python string offsets for the matched text."""
        ...


class _Pattern(Protocol):
    """The compiled RE2 operations consumed by this detector."""

    def finditer(self, text: str) -> Iterator[_Match]:
        """Iterate matches without materializing their text."""
        ...


def _compile(pattern: str) -> _Pattern:
    """Compile with bounded RE2 memory and content-free validation failures."""
    options = re2.Options()
    options.max_mem = 262_144
    options.log_errors = False
    try:
        # RE2 has no type annotations; Unicode span behavior is regression tested.
        return cast(_Pattern, re2.compile(pattern, options=options))
    except re2.error:
        raise ValueError("invalid RE2 expression; check syntax and simplify the pattern") from None


def _valid_card(text: str, start: int, end: int) -> bool:
    """Validate a whole card candidate with Luhn, rejecting slices of longer numbers."""
    if (start and text[start - 1].isdigit()) or (end < len(text) and text[end].isdigit()):
        return False
    if start >= 2 and text[start - 1] in " -" and text[start - 2].isdigit():
        return False
    if end + 1 < len(text) and text[end] in " -" and text[end + 1].isdigit():
        return False
    digits = [int(char) for char in text[start:end] if char in "0123456789"]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        doubled = digit * 2 if index % 2 else digit
        total += doubled - 9 if doubled > 9 else doubled
    return total % 10 == 0


class RegexClassifier:
    """Inspect text and tool arguments with compiled, bounded deterministic patterns."""

    def __init__(self, document: RegexAdapterDocument) -> None:
        """Compile the authored rule once, outside the gateway request path.

        Args:
            document: Validated expressions, built-in families, and literal replacement.
        """
        self._patterns = tuple(
            [(_compile(pattern), False) for pattern in document.patterns]
            + [
                (_compile(_BUILTINS[kind]), kind is BuiltinPattern.CREDIT_CARD)
                for kind in document.builtin_patterns
            ]
        )
        self._replacement = document.replacement

    def _redact(self, text: str) -> tuple[bool, str]:
        """Union matched spans before replacement so overlapping rules cannot leak tails."""
        if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
            raise ValueError("regex subject exceeds the 1 MiB inspection limit")
        spans: list[tuple[int, int]] = []
        matches = 0
        for pattern, is_card in self._patterns:
            for match in pattern.finditer(text):
                matches += 1
                if matches > _MAX_MATCHES:
                    raise ValueError("regex subject exceeds the match limit")
                start, end = match.span()
                if start == end:
                    continue
                if not is_card or _valid_card(text, start, end):
                    spans.append((start, end))
        if not spans:
            return False, text
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        pieces: list[str] = []
        position = 0
        for start, end in merged:
            pieces.extend((text[position:start], self._replacement))
            position = end
        pieces.append(text[position:])
        return True, "".join(pieces)

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Redact message text; a match in tool arguments cannot be safely rewritten."""
        messages = []
        flagged = False
        tool_match = False
        for message in request.messages:
            found, text = self._redact(message.content or "")
            flagged |= found
            messages.append(message.model_copy(update={"content": text}) if found else message)
            for call in message.tool_calls:
                found, _ = self._redact(call.arguments_json())
                flagged |= found
                tool_match |= found
        if flagged and check.action is GuardrailAction.MODIFY and not tool_match:
            return ClassifierVerdict(flagged=True, replacement_messages=tuple(messages))
        # A flagged modify without replacement is refused by the engine, even
        # under a fail-open policy. Tool arguments must never pass unredacted.
        return ClassifierVerdict(flagged=flagged)

    async def inspect_output(
        self, *, completion: GuardrailCompletion, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Redact completion text; the engine blocks modifications of tool completions."""
        flagged, text = self._redact(completion.text)
        for call in completion.tool_calls:
            found, _ = self._redact(call.arguments)
            flagged |= found
        if flagged and check.action is GuardrailAction.MODIFY:
            return ClassifierVerdict(flagged=True, replacement_text=text)
        return ClassifierVerdict(flagged=flagged)
