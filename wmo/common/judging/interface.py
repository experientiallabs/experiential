"""Behavioral extension point for scoring immutable rollout evidence."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmo.common.judging.calibration_provenance import VerifiedJudgeCalibration
from wmo.common.judging.judgment import Judgment
from wmo.common.judging.rubric import Rubric
from wmo.common.rollouts import RolloutArtifact


@runtime_checkable
class Judge(Protocol):
    """Scores one existing rollout against a frozen rubric and calibration."""

    def judge(
        self,
        rollout: RolloutArtifact,
        rubric: Rubric,
        calibration: VerifiedJudgeCalibration,
    ) -> Judgment:
        """Return a structured evidence-cited judgment without rerunning the rollout.

        Args:
            rollout: Existing immutable rollout evidence to score.
            rubric: Immutable customer rubric that defines score dimensions.
            calibration: Recursively verified calibration authorization to apply.

        Returns:
            The structured judgment over the supplied rollout.
        """
