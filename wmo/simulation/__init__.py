"""Canonical text and executable simulation contracts and engines."""

from wmo.simulation.comparison import (
    PairedSimulationCell,
    PairedSimulationOutcome,
    SimulationComparisonError,
    SimulationComparisonReport,
    SimulationComparisonSpec,
    compare_text_and_sandbox,
    persist_comparison,
    persist_comparison_spec,
)
from wmo.simulation.orchestration.interface import (
    SimulationModeUnsupportedError,
    Simulator,
)
from wmo.simulation.specs import (
    MixedRealitySettings,
    SandboxSettings,
    SimulationSpec,
    WorldModelSettings,
)

__all__ = [
    "MixedRealitySettings",
    "PairedSimulationCell",
    "PairedSimulationOutcome",
    "SandboxSettings",
    "SimulationComparisonError",
    "SimulationComparisonReport",
    "SimulationComparisonSpec",
    "SimulationSpec",
    "Simulator",
    "SimulationModeUnsupportedError",
    "WorldModelSettings",
    "compare_text_and_sandbox",
    "persist_comparison",
    "persist_comparison_spec",
]
