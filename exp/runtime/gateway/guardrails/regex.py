"""Bounded RE2 detection with deterministic, in-memory text redaction.

The detector is deterministic: its verdict on a prefix of a subject cannot be
changed by text arriving later, except through a match that straddles the
prefix boundary. That property lets a streamed completion be redacted
incrementally, so :class:`RegexClassifier` also serves as the streaming
redactor described in :mod:`exp.runtime.gateway.guardrails.streaming`.
"""

from __future__ import annotations

import json
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
from exp.runtime.gateway.guardrails.streaming import StreamingRedactor


class BuiltinPattern(StrEnum):
    """Deterministic detector families; these do not cover contextual personal data."""

    EMAIL = "email"
    CREDIT_CARD = "credit_card"
    API_KEY = "api_key"


_BUILTINS = {
    BuiltinPattern.EMAIL: r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",
    BuiltinPattern.CREDIT_CARD: r"\b[0-9](?:[ -]?[0-9]){12,}\b",
    BuiltinPattern.API_KEY: (
        r"\b(?:sk-(?:proj-|ant-api[0-9]+-)?[A-Za-z0-9_-]{20,}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        r"|xpl_[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b"
    ),
}
_MAX_MATCHES = 4096
_MAX_TEXT_BYTES = 1_048_576
_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"
_EMAIL_ALPHABET = frozenset(_LETTERS + _DIGITS + ".!#$%&'*+/=?^_`{|}~-@")
_CARD_ALPHABET = frozenset(_DIGITS + " -")
_API_KEY_ALPHABET = frozenset(_LETTERS + _DIGITS + "_-")
_HOLDS: dict[BuiltinPattern, frozenset[str]] = {
    # One match of a family can only be built from that family's own
    # characters, so a character outside the alphabet ends every candidate
    # that could still grow. The expressions themselves are not length
    # bounded, so no character inside the alphabet ever settles.
    BuiltinPattern.EMAIL: _EMAIL_ALPHABET,
    BuiltinPattern.CREDIT_CARD: _CARD_ALPHABET,
    BuiltinPattern.API_KEY: _API_KEY_ALPHABET,
}


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
    stream_window_characters: int = Field(default=512, ge=64, le=65_536)
    """How far back a stream looks for the character that settles a release.

    A built-in family holds its trailing run of candidate characters, and
    the character before that run is what proves the rest settled. This is
    how far back that character is looked for: a longer unbroken run holds
    the whole tail instead, so a long match is buffered rather than released
    in pieces. It never bounds a match, and a rule carrying an authored
    expression is not streamed at all, because an RE2 expression declares
    neither the characters nor the length one of its matches can span.
    """

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


def _run_start(text: str, alphabet: frozenset[str], window: int) -> int:
    """Return where the trailing run of one family's characters begins.

    The run is searched for over the trailing window only. A run that fills
    the window has no proven start, so the whole tail stays buffered: the
    family's expressions are not length bounded, and releasing inside a run
    could cut a match that later text completes.

    Args:
        text: Buffered completion tail.
        alphabet: Every character one match of the family can contain.
        window: Most characters searched back for the run's first character.

    Returns:
        The offset of the first character that must stay buffered.
    """
    start = len(text)
    floor = max(0, len(text) - window)
    while start > floor and text[start - 1] in alphabet:
        start -= 1
    return 0 if start == floor and floor > 0 else start


def _valid_card(digits: list[int]) -> bool:
    """Require a nonuniform 13 to 19 digit candidate with a valid Luhn checksum."""
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        doubled = digit * 2 if index % 2 else digit
        total += doubled - 9 if doubled > 9 else doubled
    return total % 10 == 0


def _card_candidates(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    """Inspect card spans at digit-group boundaries, including adjacent cards.

    Spaces and hyphens can separate cards or groups within a card. Check all
    13 to 19 digit spans at those boundaries. Never split an uninterrupted
    digit group into smaller candidates.
    """
    for first in range(start, end):
        if text[first] not in "0123456789":
            continue
        if first > start and text[first - 1] in "0123456789":
            continue
        digits = 0
        for last in range(first, min(end, first + 38)):
            if text[last] not in "0123456789":
                continue
            digits += 1
            if digits > 19:
                break
            if digits >= 13 and (last + 1 == end or text[last + 1] in " -"):
                yield first, last + 1


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
        self._document = document
        self._stream_window = document.stream_window_characters
        self._holds = tuple(_HOLDS[kind] for kind in document.builtin_patterns)
        self._authored = bool(document.patterns)

    def stream_redactor(self) -> StreamingRedactor | None:
        """Return this detector as its own redactor when every match is bounded.

        Releasing a prefix early is safe only against a match whose span is
        known in advance, which is true of the built-in families and of no
        authored expression, so an authored rule keeps the buffered path.

        Returns:
            The streaming redactor, or ``None`` for an authored rule.
        """
        return None if self._authored else self

    def release_boundary(self, text: str) -> int:
        """Return how many leading characters of a buffered tail are settled.

        A match that later text can still grow into must end at the end of
        the tail and is written entirely in one family's own characters. The
        boundary is therefore the start of the trailing run of such
        characters, and everything before it is settled: no later delta can
        reach back across a character the family cannot match. A run longer
        than the configured window settles nothing, so an unbroken run of
        candidate characters buffers instead of releasing a prefix a later
        delta could turn into one long match. Luhn validation is not applied
        here, so a card candidate that is not yet a valid card still holds
        the boundary back. An authored rule never reaches this method: it is
        not streamable, so its completions stay buffered.

        Args:
            text: Buffered completion tail, oldest character first.

        Returns:
            The count of leading characters no later text can change.
        """
        boundary = len(text)
        for alphabet in self._holds:
            boundary = min(boundary, _run_start(text, alphabet, self._stream_window))
        return max(boundary, 0)

    def redact(self, text: str) -> tuple[bool, str]:
        """Return whether ``text`` matched and its fully redacted form.

        Args:
            text: One complete subject, or one settled prefix of a completion.

        Returns:
            The flag and the redacted text.

        Raises:
            ValueError: The subject breached an inspection bound.
        """
        return self._redact(text)

    def native_specification(self) -> str:
        """Return the JSON rule the native deterministic detector compiles.

        The data plane compiles this once per policy load and then enforces
        matching output chains without a Python callback. The document is
        content-free: authored expressions, built-in families, and the
        literal replacement.
        """
        return json.dumps(
            {
                "patterns": list(self._document.patterns),
                "builtin_patterns": [kind.value for kind in self._document.builtin_patterns],
                "replacement": self._document.replacement,
            },
            separators=(",", ":"),
        )

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
                if not is_card:
                    spans.append((start, end))
                    continue
                for first, last in _card_candidates(text, start, end):
                    matches += 1
                    if matches > _MAX_MATCHES:
                        raise ValueError("regex subject exceeds the match limit")
                    digits = [int(char) for char in text[first:last] if char in "0123456789"]
                    if _valid_card(digits):
                        spans.append((first, last))
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
