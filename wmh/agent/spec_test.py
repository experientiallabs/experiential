"""Tests for HarnessSpec validation (the evolvable artifact must stay well-formed)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmh.agent.spec import HarnessSpec


def test_default_spec_is_valid_and_has_submit() -> None:
    spec = HarnessSpec()
    assert "submit" in spec.tools
    assert spec.max_turns >= 1


def test_spec_requires_submit_tool() -> None:
    with pytest.raises(ValidationError):
        HarnessSpec(tools=["bash", "read_file"])  # no submit


def test_spec_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        HarnessSpec(tools=["bash", "submit", "teleport"])


def test_spec_dedupes_tools_preserving_order() -> None:
    spec = HarnessSpec(tools=["bash", "bash", "submit", "read_file", "submit"])
    assert spec.tools == ["bash", "submit", "read_file"]


def test_spec_rejects_bad_turns_and_temperature() -> None:
    with pytest.raises(ValidationError):
        HarnessSpec(max_turns=0)
    with pytest.raises(ValidationError):
        HarnessSpec(temperature=3.0)
