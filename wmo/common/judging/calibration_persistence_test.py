"""Persistence-bound calibration regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.judging import CalibrationError, JudgeCalibrationService
from wmo.common.judging.calibration_test import _TIME, _build, _entries, _write_graph


def test_unpersisted_report_cannot_produce_or_store_a_calibration(tmp_path: Path) -> None:
    """Require exact persisted reports before calibration approval or persistence."""
    graph = _write_graph(tmp_path, _entries())
    service = JudgeCalibrationService()
    report = _build(graph)

    with pytest.raises(CalibrationError, match="unavailable"):
        service.approve(graph.store, report, approved_at=_TIME)
    service.write_report(graph.store, report)
    calibration = service.approve(graph.store, report, approved_at=_TIME)
    with pytest.raises(CalibrationError, match="exact persisted"):
        service.write_calibration(
            graph.store,
            report=report,
            calibration=calibration.model_copy(update={"out_of_fold_report_sha256": "d" * 64}),
        )
    unpersisted_report = report.model_copy(update={"report_id": "unpersisted-report"})
    with pytest.raises(CalibrationError, match="unavailable"):
        service.write_calibration(
            graph.store,
            report=unpersisted_report,
            calibration=calibration,
        )
