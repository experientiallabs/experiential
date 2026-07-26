"""Tests for the public package API surface."""

from __future__ import annotations

import wmo
from wmo.core.types import ActionKind


def test_public_api_matches_quickstart() -> None:
    # README/docstring quickstart imports ActionKind from the package root.
    assert "ActionKind" in wmo.__all__
    assert wmo.ActionKind is ActionKind
