"""Launcher preflight must reject absent authentication before importing GPU engines."""

from pathlib import Path

import pytest

from exp.optimize.claas.service.execution_test import configuration
from exp.optimize.claas.service.launcher import run_configuration


def test_missing_key_precedes_runtime_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process rejects a missing credential before any heavyweight runtime construction."""
    monkeypatch.delenv("EXPERIENTIAL_CLAAS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="EXPERIENTIAL_CLAAS_TOKEN"):
        run_configuration(configuration(tmp_path, mode="run"))
