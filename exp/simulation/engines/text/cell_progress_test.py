"""Tests for phase-qualified evaluation-cell progress reporting."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from exp.common.progress import ProgressEvent
from exp.simulation.engines.text.cell_progress import cell_progress_reporter


class _Cell(BaseModel):
    """Minimal purposeful stand-in for one evaluation-plan cell."""

    purpose: Literal["fit", "held_out", "fidelity"]


def test_reporter_emits_exact_counts_with_the_shared_phase() -> None:
    """Each call reports the live completed count under the run's one purpose."""
    seen: list[ProgressEvent] = []
    completed: dict[str, str] = {}
    observe = cell_progress_reporter(
        seen.append,
        (_Cell(purpose="held_out"), _Cell(purpose="held_out")),
        completed,
    )
    observe()
    completed["a"] = "rollout"
    observe()
    assert seen == [
        ProgressEvent(stage="evaluation cells", completed=0, total=2, detail="held-out"),
        ProgressEvent(stage="evaluation cells", completed=1, total=2, detail="held-out"),
    ]


def test_mixed_purposes_carry_no_qualifier() -> None:
    """Cells spanning several purposes report without a misleading phase name."""
    seen: list[ProgressEvent] = []
    observe = cell_progress_reporter(
        seen.append,
        (_Cell(purpose="fit"), _Cell(purpose="fidelity")),
        {},
    )
    observe()
    assert seen == [ProgressEvent(stage="evaluation cells", completed=0, total=2)]


def test_reporter_without_observer_is_silent() -> None:
    """A missing hook produces no events."""
    observe = cell_progress_reporter(None, (_Cell(purpose="fit"),), {})
    observe()
