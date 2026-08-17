"""Observational progress contract shared by long-running WMO operations.

A progress hook is an optional caller-owned observer. Services emit truthful stage names and,
when a stage has countable units, exact completed and total counts. Emitting progress never
changes workflow behavior: hooks receive frozen events and own all presentation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEvent:
    """One truthful moment of a long-running operation.

    Args:
        stage: Short human-readable stage name.
        completed: Exact completed unit count, present only with ``total``.
        total: Exact total unit count, present only with ``completed``.
        detail: Optional short qualifier distinguishing repeated stages.
    """

    stage: str
    completed: int | None = None
    total: int | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        """Reject empty stages and partial or impossible counts.

        Raises:
            ValueError: The stage is blank, counts are partial, or completed exceeds total.
        """
        if not self.stage.strip():
            raise ValueError("progress stage must be a nonempty name")
        if (self.completed is None) != (self.total is None):
            raise ValueError("progress counts require both completed and total")
        if self.completed is not None and self.total is not None:
            if self.completed < 0 or self.total < 0 or self.completed > self.total:
                raise ValueError("progress counts must satisfy 0 <= completed <= total")


ProgressHook = Callable[[ProgressEvent], None]


def report(
    hook: ProgressHook | None,
    stage: str,
    *,
    completed: int | None = None,
    total: int | None = None,
    detail: str | None = None,
) -> None:
    """Emit one progress event to an optional observer.

    Args:
        hook: Caller-owned observer, or ``None`` when nobody is watching.
        stage: Short human-readable stage name.
        completed: Exact completed unit count, present only with ``total``.
        total: Exact total unit count, present only with ``completed``.
        detail: Optional short qualifier distinguishing repeated stages.
    """
    if hook is not None:
        hook(ProgressEvent(stage=stage, completed=completed, total=total, detail=detail))
