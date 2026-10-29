"""Tests for the reasoning display opt-out and its kill switch."""

from __future__ import annotations

import pytest

from exp.common.models.model import ModelCapabilities
from exp.runtime.gateway.reasoning_display import reasoning_output_hidden
from exp.runtime.models.providers.reasoning_compat import (
    REASONING_DISPLAY_ENVIRONMENT,
    reasoning_display_enabled,
)


def test_display_is_on_unless_the_rung_opts_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undeclared and default rungs display; a stamped rung withholds."""
    monkeypatch.delenv(REASONING_DISPLAY_ENVIRONMENT, raising=False)
    assert reasoning_output_hidden(None) is False
    assert reasoning_output_hidden(ModelCapabilities()) is False
    assert reasoning_output_hidden(ModelCapabilities(reasoning_output_hidden=True)) is True


def test_kill_switch_withholds_every_rung(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the switch to 0 hides reasoning everywhere; other values change nothing."""
    monkeypatch.setenv(REASONING_DISPLAY_ENVIRONMENT, "0")
    assert reasoning_output_hidden(None) is True
    assert reasoning_output_hidden(ModelCapabilities()) is True
    monkeypatch.setenv(REASONING_DISPLAY_ENVIRONMENT, "1")
    assert reasoning_output_hidden(ModelCapabilities()) is False


def test_kill_switch_restores_provider_reasoning_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With display off no request builder asks a provider for readable reasoning."""
    monkeypatch.setenv(REASONING_DISPLAY_ENVIRONMENT, "0")
    assert reasoning_display_enabled() is False
    monkeypatch.delenv(REASONING_DISPLAY_ENVIRONMENT)
    assert reasoning_display_enabled() is True
