"""Suite-wide fixtures for explicitly interactive CLI tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def interactive_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present a terminal stdin for a test that must reach an interactive prompt."""
    from wmo.cli import consent

    monkeypatch.setattr(consent, "_stdin_is_terminal", lambda: True)
