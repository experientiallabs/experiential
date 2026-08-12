"""Plan-bound fidelity fixture shared by SFT builder regression tests."""

from __future__ import annotations

from datetime import datetime

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.evaluations import FidelityPair, FidelityReport


def approved_fidelity_report(
    *, inputs: tuple[ArtifactInput, ...], created_at: datetime, digest: str
) -> FidelityReport:
    """Return one approved report with eight exact usable overlap pairs."""
    return FidelityReport(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision="w12-test",
        fidelity_report_id="fidelity-teacher",
        evaluation_plan_id="plan-teacher",
        evaluation_plan_sha256=digest,
        protocol_sha256=digest,
        overlap_cell_ids=tuple(f"fidelity-cell-{index}" for index in range(8)),
        planned_overlap_count=8,
        usable_overlap_count=8,
        failed_overlap_count=0,
        score_mae=0.05,
        pairs=tuple(
            FidelityPair(
                fidelity_cell_id=f"fidelity-cell-{index}",
                observed_cell_id=f"observed-cell-{index}",
                observed_rollout_id=f"observed-rollout-{index}",
                simulated_rollout_id=f"simulated-rollout-{index}",
                observed_score=0.8,
                simulated_score=0.8,
                absolute_error=0.0,
                status="usable",
            )
            for index in range(8)
        ),
        gate_id="fidelity-gate-teacher",
        gate_sha256=digest,
        status="approved",
        approved_at=created_at,
    )
