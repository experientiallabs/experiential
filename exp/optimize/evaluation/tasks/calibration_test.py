"""Qualification exclusions fence judge evidence before calibration content is read."""

from pathlib import Path

import pytest

from exp.optimize.evaluation.analysis.source_access import exclude_source_from_qualification
from exp.optimize.evaluation.analysis.source_access_test import _source
from exp.optimize.evaluation.native_artifacts_test import TextAdapter
from exp.optimize.evaluation.tasks.admission_test import (
    calibration,
    native_label,
    specification,
    witnessed_cases,
)
from exp.optimize.evaluation.tasks.calibration import SavedCalibrationJudge, execute_calibration
from exp.simulation.evaluation import admission as witness_admission


@pytest.fixture(autouse=True)
def trusted_text_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trust the text fixture adapter so native receipts re-derive without a workbook."""
    monkeypatch.setattr(witness_admission, "maintained_native_adapters", lambda: (TextAdapter(),))


def test_calibration_refuses_excluded_judge_ancestor(tmp_path: Path) -> None:
    """A valid witnessed pack cannot launder a known excluded judge dependency."""
    project, _, _ = native_label(tmp_path)
    spec = calibration(project, witnessed_cases(project, specification(project)))
    ancestor = _source(project.artifacts, "unsafe-response")
    report = _source(project.artifacts, "judge-report", (ancestor,))
    exclude_source_from_qualification(
        project.artifacts,
        ancestor,
        audit=_source(project.artifacts, "audit"),
        rationale="Proven unsafe redraw",
        code_revision="fixture",
    )
    spec = spec.model_copy(
        update={
            "judgments": (
                SavedCalibrationJudge(case_id=spec.pack.cases[0].challenge.id, report=report),
            )
        }
    )
    with pytest.raises(ValueError, match="excluded from qualification"):
        execute_calibration(project, spec, code_revision="fixture")


@pytest.mark.parametrize("kind", ["witness", "source"])
def test_calibration_refuses_direct_excluded_label_input(tmp_path: Path, kind: str) -> None:
    """Pack witnesses and direct challenge sources need not be task ancestors to be fenced."""
    project, _, _ = native_label(tmp_path)
    spec = calibration(project, witnessed_cases(project, specification(project)))
    challenge = spec.pack.cases[0].challenge
    if kind == "source":
        source = _source(project.artifacts, "direct-label-source")
        changed = spec.pack.cases[0].model_copy(
            update={"challenge": challenge.model_copy(update={"source_refs": (source,)})}
        )
        spec = spec.model_copy(
            update={"pack": spec.pack.model_copy(update={"cases": (changed, *spec.pack.cases[1:])})}
        )
    else:
        source = challenge.witness
    assert source is not None
    exclude_source_from_qualification(
        project.artifacts,
        source,
        audit=_source(project.artifacts, "direct-audit"),
        rationale="Proven invalid qualification input",
        code_revision="fixture",
    )
    with pytest.raises(ValueError, match="excluded from qualification"):
        execute_calibration(project, spec, code_revision="fixture")
