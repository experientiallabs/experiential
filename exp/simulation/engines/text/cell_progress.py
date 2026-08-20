"""Phase-qualified evaluation-cell progress reporting for the text simulator."""

from __future__ import annotations

from collections.abc import Callable, Sequence, Sized
from typing import Literal, Protocol

from exp.common.progress import ProgressHook, report


class PurposefulCell(Protocol):
    """One evaluation-plan cell exposing its frozen evaluation purpose."""

    purpose: Literal["fit", "held_out", "fidelity"]


def cell_progress_reporter(
    progress: ProgressHook | None,
    cells: Sequence[PurposefulCell],
    completed: Sized,
) -> Callable[[], None]:
    """Bind one phase-qualified reporter over a run's live completed-cell map.

    Args:
        progress: Optional observer of exact completed and total cell counts.
        cells: Exact evaluation-plan cells selected by the running specification.
        completed: Live mapping whose size is the current completed-cell count.

    Returns:
        A zero-argument callable reporting the current exact counts on each call.
    """
    detail = _purpose_detail(cells)
    total = len(cells)

    def observe() -> None:
        """Report the exact completed evaluation-cell count."""
        report(progress, "evaluation cells", completed=len(completed), total=total, detail=detail)

    return observe


def _purpose_detail(cells: Sequence[PurposefulCell]) -> str | None:
    """Name the one evaluation purpose shared by every selected cell.

    Args:
        cells: Exact evaluation-plan cells selected by the running specification.

    Returns:
        The shared purpose with underscores spelled as hyphens, or ``None`` when purposes differ.
    """
    purposes = {cell.purpose for cell in cells}
    if len(purposes) != 1:
        return None
    return purposes.pop().replace("_", "-")
