"""Consolidation cannot run in an immutable evaluation checkpoint."""

import pytest

from exp.optimize.evaluation.fidelity.memory_policy import FidelityMemory
from exp.simulation.world_model.dreaming import DreamPolicy


def test_frozen_memory_refuses_consolidation() -> None:
    """A caller cannot enable post-feedback consolidation through a frozen run."""
    with pytest.raises(ValueError, match="never during frozen"):
        FidelityMemory(lineage="memory", mode="frozen", checkpoint=1, dreaming=DreamPolicy())
