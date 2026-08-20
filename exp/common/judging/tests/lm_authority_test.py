"""End-to-end eligibility coverage for store-backed LM judging."""

from __future__ import annotations

import json
from pathlib import Path

from exp.common.judging import JudgeCalibrationService, LMJudge
from exp.common.judging.calibration_test import (
    _TIME,
    _entries,
    _FakeJudgeClient,
    _model,
    _prompt,
    _write_graph,
)


def test_store_backed_lm_judge_permits_fully_calibrated_evidence(tmp_path: Path) -> None:
    """A completed OOF-approved calibration may authoritatively score a later rollout."""
    graph = _write_graph(tmp_path, _entries())
    service = JudgeCalibrationService()
    report = service.build_report(
        graph.store,
        rubric_id=graph.rubric.rubric_id,
        label_set_id=graph.label_set.label_set_id,
        router_lineage_split_id=graph.split.split_id,
        observations=graph.observations,
        created_at=_TIME,
        code_revision="calibration-revision",
    )
    service.write_report(graph.store, report)
    calibration = service.write_calibration(
        graph.store,
        report=report,
        calibration=service.approve(graph.store, report, approved_at=_TIME),
    )
    source_observation = graph.observations[0]
    client = _FakeJudgeClient(
        _model(),
        json.dumps(
            {
                "dimensions": [
                    {
                        "dimension_id": "task-success",
                        "raw_score": 4,
                        "rationale": "The rollout has sufficient evidence.",
                    }
                ]
            }
        ),
    )
    judgment = LMJudge(
        client,
        _prompt(),
        code_revision="judging-revision",
        clock=lambda: _TIME,
    ).judge_persisted(
        graph.store,
        rollout_artifact_id=source_observation.source_rollout.artifact_id,
        rubric_artifact_id=graph.rubric.rubric_id,
        calibration_artifact_id=calibration.calibration_id,
    )

    assert judgment.calibration_id == calibration.calibration_id
    assert len(client.requests) == 1
