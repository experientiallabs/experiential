"""The front door's one cumulative phase ledger: reserve before a phase, record what it used.

A front-door registration has exactly one ledger file
(``<project runtime>/front-door/<spec>/ledger.json``), never reset and never split per tier or
per start. Its ceiling is the consented dollar ceiling and unpriced-attempt allowance. Before a
phase starts, its bound (from its owner's plan of exactly the work it will run, priced once that
phase is reachable) is reserved; a bound that does not fit what is left is refused before the
phase starts. After the phase, its measured usage (recorded provider attempts and, when an
operation ceiling is installed, accounted dollars) replaces the reservation. Every rerun reads
the same file, so a resumed run continues the same ledger.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.core.files import write_text_atomic
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.pipeline import PHYSICAL_ATTEMPT_TYPE
from exp.runtime.models.spend_ceiling import current_spend_ceiling


class PhaseEntry(ContractModel):
    """One admitted phase.

    Attributes:
        phase: The phase name.
        reserved_usd: Its priced bound, reserved before it started.
        reserved_attempts: Its unpriced-attempt bound, reserved before it started.
        measured_usd: Dollars the operation ceiling accounted while it ran, when measured.
        measured_attempts: Provider attempts recorded while it ran, once it ended.
    """

    phase: str
    reserved_usd: float = Field(ge=0)
    reserved_attempts: int = Field(ge=0)
    measured_usd: float | None = Field(default=None, ge=0)
    measured_attempts: int | None = Field(default=None, ge=0)

    @property
    def charged_usd(self) -> float:
        """Measured dollars once known, else the reservation."""
        return self.reserved_usd if self.measured_usd is None else self.measured_usd

    @property
    def charged_attempts(self) -> int:
        """Measured attempts once known, else the reservation."""
        return self.reserved_attempts if self.measured_attempts is None else self.measured_attempts


class LedgerState(ContractModel):
    """The saved ledger.

    Attributes:
        ceiling_usd: The consented cumulative dollar ceiling.
        unpriced_attempts: The consented cumulative allowance of unpriced attempts.
        phases: Every admitted phase, in order, across every run.
    """

    ceiling_usd: float = Field(default=0, ge=0)
    unpriced_attempts: int = Field(default=0, ge=0)
    phases: tuple[PhaseEntry, ...] = ()


class PhaseLedger:
    """Reads, reserves against and records into one registration's cumulative ledger.

    Attributes:
        path: The ledger file.
        state: The current saved state.
    """

    def __init__(
        self, path: Path, *, ceiling_usd: float | None = None, unpriced_attempts: int | None = None
    ) -> None:
        """Open the ledger, adopting a newly consented ceiling without touching its phases.

        Args:
            path: The ledger file.
            ceiling_usd: The consented dollar ceiling, or None to keep the saved one.
            unpriced_attempts: The consented unpriced allowance, or None to keep the saved one.
        """
        self.path = path
        state = LedgerState()
        if path.exists():
            state = LedgerState.model_validate_json(path.read_bytes())
        update: dict[str, float | int] = {}
        if ceiling_usd is not None:
            update["ceiling_usd"] = ceiling_usd
        if unpriced_attempts is not None:
            update["unpriced_attempts"] = unpriced_attempts
        self.state = state.model_copy(update=update)

    @property
    def remaining_usd(self) -> float:
        """Dollars left of the cumulative ceiling."""
        return self.state.ceiling_usd - sum(row.charged_usd for row in self.state.phases)

    @property
    def remaining_attempts(self) -> int:
        """Unpriced attempts left of the cumulative allowance."""
        return self.state.unpriced_attempts - sum(row.charged_attempts for row in self.state.phases)

    def admit(self, phase: str, usd: float, attempts: int) -> str | None:
        """Reserve one phase's bound, or return why it is refused before it starts.

        Args:
            phase: The phase.
            usd: Its priced bound.
            attempts: Its unpriced-attempt bound.

        Returns:
            None when reserved and saved; otherwise the refusal reason.
        """
        if usd > self.remaining_usd + 1e-12:
            return (
                f"{phase} may cost up to ${usd:.6f}, more than the ${self.remaining_usd:.6f} "
                "left of the cumulative ceiling; rerun to review the now-priced plan"
            )
        if attempts > self.remaining_attempts:
            return (
                f"{phase} needs up to {attempts} unpriced attempts, more than the "
                f"{self.remaining_attempts} left of the cumulative allowance"
            )
        entry = PhaseEntry(phase=phase, reserved_usd=usd, reserved_attempts=attempts)
        self._save(self.state.phases + (entry,))
        return None

    def settle(self, phase: str, before: Measure, after: Measure) -> None:
        """Replace the latest reservation of a phase with what it measurably used.

        Args:
            phase: The phase.
            before: The reading taken when it started.
            after: The reading taken when it ended.
        """
        rows = list(self.state.phases)
        index = max(i for i, row in enumerate(rows) if row.phase == phase)
        usd = None if before.usd is None or after.usd is None else max(0.0, after.usd - before.usd)
        rows[index] = rows[index].model_copy(
            update={
                "measured_attempts": max(0, after.attempts - before.attempts),
                "measured_usd": usd,
            }
        )
        self._save(tuple(rows))

    def _save(self, phases: tuple[PhaseEntry, ...]) -> None:
        """Persist the state with these phases."""
        self.state = self.state.model_copy(update={"phases": phases})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.path, self.state.model_dump_json(indent=2))


class Measure(ContractModel):
    """One usage reading.

    Attributes:
        attempts: Provider attempts the admitted transports have recorded in the project.
        usd: Dollars the installed operation ceiling has accounted, when one is installed.
    """

    attempts: int = Field(ge=0)
    usd: float | None = None


def measure(project: ProjectStore) -> Measure:
    """Read the project's recorded provider attempts and the installed ceiling's accounting."""
    attempts = len(project.artifacts.list_ids(artifact_type=PHYSICAL_ATTEMPT_TYPE))
    ceiling = current_spend_ceiling()
    return Measure(
        attempts=attempts, usd=None if ceiling is None else ceiling.summary().accounted_usd
    )
