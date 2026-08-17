"""Persistence-bound calibration regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.judging import CalibrationError, JudgeCalibrationService
from wmo.common.judging.calibration_contracts import JudgeScoreObservation
from wmo.common.judging.calibration_test import _TIME, _build, _entries, _write_graph
from wmo.common.judging.judgment import Judgment


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


def test_persisted_observations_and_judgments_omit_citation_fields(tmp_path: Path) -> None:
    """Calibration reports store raw scores without span citations or required rationales."""
    graph = _write_graph(tmp_path, _entries())
    report = _build(graph)
    observation = report.observations[0]
    judgment = Judgment.model_validate_json(
        graph.store.artifacts.read_bytes(observation.judgment.artifact_id, "judgment.json")
    )

    assert not hasattr(observation, "evidence_span_ids")
    dumped = observation.model_dump(mode="json")
    assert "evidence_span_ids" not in dumped
    assert dumped["raw_score"] == judgment.dimensions[0].raw_score
    assert "evidence_span_ids" not in judgment.dimensions[0].model_dump(mode="json")


def test_score_observation_loads_retired_citation_fields() -> None:
    """Citation-era calibration observations keep the raw score and drop span IDs."""
    loaded = JudgeScoreObservation.model_validate(
        {
            "judgment": {"artifact_id": "judgment-1", "sha256": "a" * 64},
            "source_rollout": {"artifact_id": "rollout-1", "sha256": "a" * 64},
            "dimension_id": "task-success",
            "raw_score": 1,
            "evidence_span_ids": ["span-1"],
        }
    )

    assert loaded.raw_score == 1
    assert "evidence_span_ids" not in loaded.model_dump(mode="json")
