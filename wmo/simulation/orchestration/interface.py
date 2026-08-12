"""Small simulator interface and mode resolution errors shared across engines."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmo.common.rollouts import SimulationArtifactSet, SimulationMode
from wmo.simulation.specs import SimulationSpec


class SimulationModeUnsupportedError(ValueError):
    """A persisted simulation mode has no executable implementation in this WMO release."""


@runtime_checkable
class Simulator(Protocol):
    """Executes one frozen sparse simulation specification."""

    def run(self, spec: SimulationSpec) -> SimulationArtifactSet:
        """Run exactly the selected cells and return their immutable artifact-set envelope.

        Args:
            spec: Fully validated simulation specification to execute or resume.

        Returns:
            Immutable index of completed per-cell simulation artifacts.
        """


def require_implemented_mode(spec: SimulationSpec, *implemented: SimulationMode) -> None:
    """Reject a reserved or otherwise unsupported mode before any simulation side effect.

    Args:
        spec: Persisted recipe whose requested mode is being resolved.
        implemented: Modes the caller can execute in its current package.

    Raises:
        SimulationModeUnsupportedError: The requested mode is intentionally reserved or owned by
            another simulator implementation.
    """
    if spec.mode not in implemented:
        supported = ", ".join(mode.value for mode in implemented) or "none"
        raise SimulationModeUnsupportedError(
            f"simulation mode {spec.mode.value!r} is not implemented; supported modes: {supported}"
        )
