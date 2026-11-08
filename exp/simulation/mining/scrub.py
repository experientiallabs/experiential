"""Deterministic credential scrub applied to rendered views before the credential boundary.

A rendered view that carries anything credential-shaped is refused at the decision boundary
(:func:`exp.common.project.request_budget.validate_provider_response`) and its trace is dropped
from tagging. The scrub replaces each detected span with a typed placeholder such as
``[REDACTED:aws_key:32]``: the kind of secret and its length class (the smallest power of two,
at least 8, holding it), never any of its characters. The boundary still runs on the scrubbed
text, so only a trace whose secret the scrub cannot recognize stays refused.

The scrub is a pure function of the text. The same input always yields the same output, so a
view rendered twice scrubs identically and every content key derived from it stays content
based. Text with nothing to redact is returned unchanged, byte for byte.

Detected kinds, applied in this order:

- ``private_key``: a PEM ``-----BEGIN ... PRIVATE KEY-----`` block through its END line, or
  through the end of its base64 body when the view truncated the block.
- ``openai_key``, ``aws_key``, ``slack_token``, ``bearer_token``: the secret-value shapes the
  boundary rejects (``sk-``, ``AKIA``, ``xox?-`` and ``Bearer`` tokens).
- ``assignment``: a credential term followed by assignment punctuation or an assignment verb
  (``password: ...``, ``API_KEY=...``, ``"secret": "..."``), together with the assigned value,
  since the boundary rejects the term-plus-assignment regardless of what follows. A term
  right after a literal ``\\n``, ``\\r`` or ``\\t`` counts as a word start, because the
  boundary also checks JSON-encoded text after decoding those escapes.
"""

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

from exp.simulation.mining.views import Unit

_PLACEHOLDER_PREFIX = "[REDACTED:"
_PEM_LABEL = r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
_PEM_BODY = r"(?:[A-Za-z0-9+/=\s]|\\[nr])"
_TERM = (
    r"access[_ -]?token|api[_ -]?key(?:[_ -]?env)?|auth[_ -]?token|authorization|"
    r"credential(?:[_ -]?(?:env|ref))?|password|refresh[_ -]?token|"
    r"secret(?:[_ -]?(?:env|ref))?|token[_ -]?env"
)
_ENV_NAME = (
    r"(?:[A-Z][A-Z0-9]*_)*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|REFRESH_TOKEN|SECRET|CREDENTIAL)"
    r"(?:_[A-Z0-9]+)*"
)
_START = r"(?:\b|(?<=\\[nrt]))"
"""A word start, or the end of an escape such as ``\\t`` that decodes to whitespace."""
_VALUE = r"""(?:\\?"(?:[^"\\]|\\[^"])*\\?"|'[^'\n]*'|(?!\[REDACTED:)[^\s,;}\]"'<>\\]+)"""

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            rf"{_PEM_LABEL}(?:{_PEM_BODY}*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
            rf"|{_PEM_BODY}*)"
        ),
    ),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE)),
    (
        "assignment",
        re.compile(
            rf"(?:{_START}(?i:{_TERM})\b|{_START}{_ENV_NAME}\b)"
            rf"(?:[\s\"'`\\]*[:=]|\s+(?i:is|was|equals|set\s+to)\b)"
            rf"(?:[ \t]*{_VALUE})?"
        ),
    ),
)

SCRUB_KINDS: tuple[str, ...] = tuple(kind for kind, _ in _RULES)
"""Every placeholder kind the scrub may write, in the order the rules apply."""


def _length_class(length: int) -> int:
    """Return the smallest power of two, at least 8, that holds ``length`` characters."""
    size = 8
    while size < length:
        size *= 2
    return size


def scrub_text(text: str) -> tuple[str, Counter[str]]:
    """Replace every detected secret span with a typed, length-classed placeholder.

    Args:
        text: Rendered view text.

    Returns:
        The scrubbed text (the same object when nothing matched) and the number of spans
        replaced per kind. The counts never carry any matched content.
    """
    counts: Counter[str] = Counter()
    scrubbed = text
    for kind, pattern in _RULES:

        def placeholder(match: re.Match[str], kind: str = kind) -> str:
            """Name the kind and length class of one span, dropping its content."""
            counts[kind] += 1
            return f"{_PLACEHOLDER_PREFIX}{kind}:{_length_class(len(match.group(0)))}]"

        scrubbed = pattern.sub(placeholder, scrubbed)
    return (text if not counts else scrubbed), counts


def scrub_unit(unit: Unit) -> tuple[Unit, Counter[str]]:
    """Scrub one rendered view, keeping its identity and provenance.

    Args:
        unit: Rendered view of one trace.

    Returns:
        The unit itself when nothing matched, otherwise a copy with scrubbed text whose
        ``meta`` records the per-kind redaction counts; and those counts.
    """
    text, counts = scrub_text(unit.text)
    if not counts:
        return unit, counts
    meta = {**unit.meta, "redactions": dict(sorted(counts.items()))}
    return replace(unit, text=text, meta=meta), counts


def scrub_units(units: Mapping[str, Unit]) -> dict[str, Unit]:
    """Scrub every view of one trace, as :func:`views.build_units` returns them.

    Args:
        units: Rendered views by view name.

    Returns:
        The scrubbed views by the same names.
    """
    return {name: scrub_unit(unit)[0] for name, unit in units.items()}
