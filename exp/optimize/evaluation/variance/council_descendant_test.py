"""Real preparation ancestry is insufficient to invent an unproduced full study."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.optimize.evaluation.variance.council_descendant import register_full_council_from_prelude
from exp.optimize.evaluation.variance.council_prelude_test import (
    _prelude_fixture,
    _register_fixture,
)


def test_native_descendant_refuses_missing_actual_studies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved native prelude never substitutes empty registrations for five actual studies."""
    project, prepared, resume, contract, wire = _prelude_fixture(tmp_path, monkeypatch)
    registration = _register_fixture(project, prepared, resume, contract)
    calls = len(wire.requests)
    with pytest.raises(ValidationError):
        register_full_council_from_prelude(
            project, registration.qualification, (), code_revision="council-executor"
        )
    assert len(wire.requests) == calls
