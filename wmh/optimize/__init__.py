"""Pluggable optimizers (GEPA prompt evolution today) + the LLM judges that score predictions.

The switchable-optimizer interface lives in `wmh.optimize.base`; concrete optimizers implement
its `Optimizer` protocol and return `OptimizeResult`s whose `ArtifactRef`s say what they built.
"""

from wmh.optimize.base import ArtifactRef, OptimizeMetrics, Optimizer, OptimizeResult
from wmh.optimize.gepa import GEPAOptimizer
from wmh.optimize.judge import Judge, JudgeResult, RubricJudge
from wmh.optimize.numeric import NumericJudge
from wmh.optimize.reward import EpisodeRewardJudge, EpisodeScore

__all__ = [
    "ArtifactRef",
    "EpisodeRewardJudge",
    "EpisodeScore",
    "GEPAOptimizer",
    "OptimizeMetrics",
    "OptimizeResult",
    "Optimizer",
    "Judge",
    "JudgeResult",
    "NumericJudge",
    "RubricJudge",
]
