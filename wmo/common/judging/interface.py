"""Behavioral extension point for scoring immutable rollout evidence."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmo.common.judging.judgment import Judgment
from wmo.common.judging.rubric import JudgeCalibration, Rubric
from wmo.common.rollouts import RolloutArtifact


@runtime_checkable
class Judge(Protocol):
    """Scores one existing rollout against a frozen rubric and calibration."""

    def judge(
        self,
        rollout: RolloutArtifact,
        rubric: Rubric,
        calibration: JudgeCalibration,
    ) -> Judgment:
        """Return a structured evidence-cited judgment without rerunning the rollout."""
