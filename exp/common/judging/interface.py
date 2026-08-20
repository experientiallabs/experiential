"""Behavioral extension point for scoring immutable rollout evidence."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from exp.common.core.artifacts import ArtifactId
from exp.common.judging.judgment import Judgment
from exp.common.project import ProjectStore


@runtime_checkable
class Judge(Protocol):
    """Scores persisted rollout evidence against persisted rubric and calibration artifacts."""

    def judge_persisted(
        self,
        store: ProjectStore,
        *,
        rollout_artifact_id: ArtifactId,
        rubric_artifact_id: ArtifactId,
        calibration_artifact_id: ArtifactId,
    ) -> Judgment:
        """Return a structured judgment from recursively verified persisted evidence.

        Args:
            store: Project store that owns immutable judging inputs.
            rollout_artifact_id: Completed rollout artifact to score.
            rubric_artifact_id: Completed rubric artifact that defines score dimensions.
            calibration_artifact_id: Completed calibration artifact to verify before the model
                call.

        Returns:
            The structured judgment over the persisted rollout.
        """
