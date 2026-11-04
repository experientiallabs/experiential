"""Session redaction removes credential shapes and home-directory account names."""

from __future__ import annotations

import pytest

from exp.common.traces.ingest.session_redaction import SessionRedactor


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("token ghp_" + "a" * 36, "token [REDACTED]"),
        ("hf_" + "B" * 34, "[REDACTED]"),
        ("AKIAABCDEFGHIJKLMNOP", "[REDACTED]"),
        ("Authorization: Bearer abcdefghijklmnop", "Authorization: Bearer [REDACTED]"),
        ("DATABASE_PASSWORD=hunter2hunter2", "DATABASE_PASSWORD=[REDACTED]"),
        ('{"password": "correct-horse"}', '{"password": "[REDACTED]"}'),
        ("STRIPE_KEY=${STRIPE_KEY}", "STRIPE_KEY=${STRIPE_KEY}"),
        ('const STATUS_KEY = "pi-turn";', 'const STATUS_KEY = "pi-turn";'),
        ("MAX_TOKENS = 8_000_000", "MAX_TOKENS = 8_000_000"),
        ("apiKey: process.env.KEY", "apiKey: process.env.KEY"),
        (
            "-H 'Authorization: Bearer gsk_" + "Q" * 40 + "'",
            "-H 'Authorization: Bearer [REDACTED]'",
        ),
        (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----",
            "[REDACTED]",
        ),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N", "[REDACTED]"),
    ],
)
def test_credentials_are_replaced(text: str, expected: str) -> None:
    """Each supported credential shape is replaced, while references are kept."""
    assert SessionRedactor().text(text) == expected


def test_shared_env_names_are_redacted_by_the_common_rules() -> None:
    """The shared artifact rules still apply, including credential variable names."""
    redactor = SessionRedactor()

    assert redactor.text("export OPENAI_API_KEY=sk-" + "x" * 30) == "export [REDACTED]=[REDACTED]"
    assert redactor.secret_replacements >= 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cd /home/alice/src && ls", "cd /home/[USER]/src && ls"),
        ("/Users/bob", "/Users/[USER]"),
        ("C:\\Users\\carol\\repo", "C:\\Users\\[USER]\\repo"),
        ("C:\\\\Users\\\\carol\\\\repo", "C:\\\\Users\\\\[USER]\\\\repo"),
        ("C:\\Users\\Public\\data", "C:\\Users\\Public\\data"),
        ("/home/[USER]/x", "/home/[USER]/x"),
    ],
)
def test_home_directory_names_are_replaced(text: str, expected: str) -> None:
    """Home-directory account names are replaced; shared folders and placeholders are kept."""
    assert SessionRedactor().text(text) == expected


def test_json_redaction_counts_and_keeps_keys() -> None:
    """JSON leaves are redacted, keys are kept, and provenance reports the counts."""
    redactor = SessionRedactor()

    redacted = redactor.json({"path": "/home/dave/a", "items": ["sk-" + "y" * 24, 3]})

    assert redacted == {"path": "/home/[USER]/a", "items": ["[REDACTED]", 3]}
    assert redactor.provenance() == {
        "policy": "agent-sessions-redaction-v1",
        "secret_replacements": 1,
        "home_path_replacements": 1,
    }
