"""Consolidation cannot run in an immutable evaluation checkpoint."""

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.fidelity.memory_policy import FidelityMemory
from exp.simulation.world_model.dreaming import DreamPolicy


def test_frozen_memory_refuses_consolidation() -> None:
    """A caller cannot enable post-feedback consolidation through a frozen run."""
    with pytest.raises(ValueError, match="never during frozen"):
        FidelityMemory(lineage="memory", mode="frozen", checkpoint=1, dreaming=DreamPolicy())


def test_a_lesson_set_restricts_only_a_frozen_checkpoint() -> None:
    """Learning always reads its whole head; lessons restrict a frozen evaluation only."""
    lessons = ArtifactInput(artifact_id="learning-lesson-set-x", sha256="0" * 64)
    assert FidelityMemory(lineage="memory", mode="frozen", checkpoint=1, lessons=lessons).lessons
    with pytest.raises(ValueError, match="never learning"):
        FidelityMemory(lineage="memory", mode="learn", lessons=lessons)
    unset = FidelityMemory(lineage="memory", mode="frozen", checkpoint=1)
    assert "lessons" not in unset.model_dump(mode="json"), "existing identities are unchanged"
