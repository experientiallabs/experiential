"""The offline Astra refinement's own cumulative attempt ledger, safe across worker processes.

By default the refinement reserves on its own ledger file
(``<project runtime>/astra-prompts/ledger.json``), separate from the 100,000-attempt pipeline
ledger, with :data:`DEFAULT_ALLOWANCE` attempts: enough for every one of the 168 targets at the
default calibration size. Passing a front-door ledger path instead shares the pipeline ledger.

:class:`LockedLedger` is a :class:`~.frontdoor_ledger.PhaseLedger` whose reservations and
settlements re-read the file under an exclusive ``flock``, so several refinement workers can
reserve on one file without overwriting each other's phases.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.frontdoor_ledger import LedgerState, Measure, PhaseLedger

DEFAULT_ALLOWANCE = 160_000
"""Attempts of the refinement's own ledger (168 targets at the default size need ~106k with
Astra on one Codex attempt per draw, ~156k at three)."""


def default_ledger_path(project: ProjectStore) -> Path:
    """The refinement's own ledger file."""
    return project.paths.runtime_directory / "astra-prompts" / "ledger.json"


class LockedLedger(PhaseLedger):
    """A phase ledger whose writes re-read the file under an exclusive lock."""

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the ledger's lock file and reload the saved phases."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with (self.path.parent / f"{self.path.name}.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                if self.path.exists():
                    saved = LedgerState.model_validate_json(self.path.read_bytes())
                    self.state = self.state.model_copy(update={"phases": saved.phases})
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def admit(self, phase: str, usd: float, attempts: int) -> str | None:
        """Reserve under the lock."""
        with self._locked():
            return super().admit(phase, usd, attempts)

    def settle(self, phase: str, before: Measure, after: Measure) -> None:
        """Settle under the lock."""
        with self._locked():
            super().settle(phase, before, after)


def refinement_ledger(
    project: ProjectStore, shared: Path | None = None, allowance: int | None = None
) -> LockedLedger:
    """The ledger a refinement reserves on: its own file, or the shared pipeline ledger.

    Args:
        project: The refinement project.
        shared: A front-door ledger to share instead of the refinement's own.
        allowance: A newly consented allowance; None keeps the saved one, or
            :data:`DEFAULT_ALLOWANCE` for a new own ledger.
    """
    if shared is not None:
        return LockedLedger(shared, unpriced_attempts=allowance)
    path = default_ledger_path(project)
    if allowance is None and not path.exists():
        allowance = DEFAULT_ALLOWANCE
    return LockedLedger(path, unpriced_attempts=allowance)
