"""Known excluded calibration ancestry cannot qualify a Harbor delivery."""

from pathlib import Path

import pytest

from exp.optimize.evaluation.analysis.source_access import exclude_source_from_qualification
from exp.optimize.evaluation.analysis.source_access_test import _source
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.harbor.export import export_harbor_task


def test_export_refuses_excluded_judgment_before_task_content(tmp_path: Path) -> None:
    """Excluded calibration ancestors refuse even if a derived report claims qualification."""
    project, _ = source_project(tmp_path)
    judgment = _source(project.artifacts, "unsafe-judgment")
    calibration = _source(project.artifacts, "calibration", (judgment,))
    exclude_source_from_qualification(
        project.artifacts,
        judgment,
        audit=_source(project.artifacts, "audit"),
        rationale="Proven unsafe redraw",
        code_revision="fixture",
    )
    with pytest.raises(ValueError, match="excluded from qualification"):
        export_harbor_task(
            project,
            calibration,
            tmp_path / "export",
            name="test/excluded",
            calibration=calibration,
        )
    assert not (tmp_path / "export").exists()
