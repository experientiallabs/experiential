"""Pluggable optimizers (GEPA prompt evolution today) + the LLM judges that score predictions.

The switchable-optimizer interface lives in `wmo.optimize.base`; concrete optimizers implement
its `Optimizer` protocol and return `OptimizeResult`s whose `ArtifactRef`s say what they built.
"""

from wmo.optimize.base import ArtifactRef, OptimizeMetrics, Optimizer, OptimizeResult
from wmo.optimize.gepa import GEPAOptimizer
from wmo.optimize.judge import Judge, JudgeResult, RubricJudge
from wmo.optimize.numeric import NumericJudge
from wmo.optimize.reward import EpisodeRewardJudge, EpisodeScore

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
