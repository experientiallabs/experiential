"""Redact credentials and home-directory user names from imported agent session text.

Coding-agent session files record whatever the agent read, ran, and printed, so they routinely
contain pasted tokens, ``.env`` contents, authorization headers, and absolute paths that name the
local account. :class:`SessionRedactor` rewrites every string of a session before it becomes
canonical trace evidence. It reuses the shared secret patterns of
:func:`exp.common.core.artifacts.redact_secret_text` and adds the session-specific shapes those
patterns do not cover: private-key blocks, provider token prefixes, JSON web tokens,
authorization headers, credential assignments, and home-directory user names.

The redactor counts its replacements so the importer can record, per trace, that the policy
ran and how much it changed. Redaction is best-effort pattern matching: it removes the common
credential shapes, not every possible secret.
"""

from __future__ import annotations

import re

from pydantic import JsonValue

from exp.common.core.artifacts import (
    SECRET_REDACTION_PLACEHOLDER,
    JsonObject,
    redact_secret_text,
)

REDACTION_POLICY = "agent-sessions-redaction-v1"
"""Stable name of the redaction rules applied to every imported session string."""

HOME_USER_PLACEHOLDER = "[USER]"
"""Replacement for the account name in a home-directory path."""

_SECRET_PATTERNS = (
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
        re.DOTALL,
    ),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\b(?:gsk|npm)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxai-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
)
_VALUE_PATTERNS = (
    re.compile(
        r"(?P<prefix>\bauthorization\b[\"']?\s*[:=]\s*[\"']?(?:basic|bearer|token)\s+)"
        r"(?P<value>[A-Za-z0-9._~+/=-]{8,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>\b(?:[A-Z0-9]+_)*(?:API_KEY|ACCESS_KEY|PRIVATE_KEY|SECRET_KEY|AUTH_KEY|"
        r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)\s*[=:]\s*[\"']?)"
        r"(?P<value>(?![$<\[{]|process\.|os\.)[^\s\"'`,;]{6,})"
    ),
    re.compile(
        r"(?P<prefix>[\"']?\b(?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|"
        r"secret[_-]?key|client[_-]?secret|aws_secret_access_key|password|passwd)\b[\"']?"
        r"\s*[:=]\s*[\"'])(?P<value>(?![$<\[{])[^\s\"'`,;]{6,})(?=[\"'])",
        re.IGNORECASE,
    ),
)
_HOME_PATTERNS = (
    re.compile(r"(?P<prefix>(?<![A-Za-z0-9_])/(?:home|Users)/)(?P<user>[^/\\\s\"'`<>:;,()\[\]]+)"),
    re.compile(
        r"(?P<prefix>\b[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/))"
        r"(?P<user>[^/\\\s\"'`<>:;,()\[\]]+)",
        re.IGNORECASE,
    ),
)
_SHARED_HOME_NAMES = frozenset({"public", "default", "shared", "all users", HOME_USER_PLACEHOLDER})


class SessionRedactor:
    """Redact session strings and count what each rule replaced.

    Attributes:
        secret_replacements: Credential-shaped substrings replaced so far.
        home_path_replacements: Home-directory account names replaced so far.
    """

    def __init__(self) -> None:
        """Start with zero recorded replacements."""
        self.secret_replacements = 0
        self.home_path_replacements = 0

    def text(self, value: str) -> str:
        """Return one string with credentials and home-directory user names replaced.

        Args:
            value: Session text, such as a prompt, tool argument, or tool output.

        Returns:
            The redacted text. Text without a match is returned unchanged.
        """
        redacted = value
        for pattern in _SECRET_PATTERNS:
            redacted, count = pattern.subn(SECRET_REDACTION_PLACEHOLDER, redacted)
            self.secret_replacements += count
        for pattern in _VALUE_PATTERNS:
            redacted, count = pattern.subn(_keep_prefix, redacted)
            self.secret_replacements += count
        redacted, count = redact_secret_text(redacted)
        self.secret_replacements += count
        for pattern in _HOME_PATTERNS:
            redacted = pattern.sub(self._home_user, redacted)
        return redacted

    def json(self, value: JsonValue) -> JsonValue:
        """Redact every string leaf of a JSON value; object keys are kept.

        Args:
            value: JSON-safe session content.

        Returns:
            A structurally equal value whose string leaves are redacted.
        """
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.json(item) for item in value]
        if isinstance(value, dict):
            return {key: self.json(item) for key, item in value.items()}
        return value

    def provenance(self) -> JsonObject:
        """Describe the applied policy and its replacement counts.

        Returns:
            The policy name and the secret and home-path replacement counts.
        """
        return {
            "policy": REDACTION_POLICY,
            "secret_replacements": self.secret_replacements,
            "home_path_replacements": self.home_path_replacements,
        }

    def _home_user(self, match: re.Match[str]) -> str:
        """Replace the account name of one home-directory path match.

        Args:
            match: A home-path match with ``prefix`` and ``user`` groups.

        Returns:
            The prefix followed by the placeholder, or the match unchanged for shared folders.
        """
        if match.group("user").casefold() in {name.casefold() for name in _SHARED_HOME_NAMES}:
            return match.group(0)
        self.home_path_replacements += 1
        return match.group("prefix") + HOME_USER_PLACEHOLDER


def _keep_prefix(match: re.Match[str]) -> str:
    """Keep an assignment's name and separator while replacing its value.

    Args:
        match: A credential-assignment match with ``prefix`` and ``value`` groups.

    Returns:
        The prefix followed by the shared redaction placeholder.
    """
    return match.group("prefix") + SECRET_REDACTION_PLACEHOLDER
