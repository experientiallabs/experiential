"""Public scorecard API, including ablation ladder assembly.

The scoring implementation lives in `scorecard_core`; ladder assembly depends on that core.
This facade preserves the supported single import path without making the core and ladder import
each other.
"""

from __future__ import annotations

from wmo.optimize.routing.scorecard_core import (
    DEFAULT_COMPLETION,
    DOMINANCE_TOLERANCE,
    EFFECTIVE_COST_RULE,
    Arm,
    CompletionRule,
    ConditionLabel,
    EffectiveCost,
    Ladder,
    LadderRung,
    LatencyBlock,
    LatencyObjective,
    OperatingPoint,
    Provenance,
    QualityBlock,
    RowOverhead,
    Scorecard,
    build_scorecard,
    effective_cost_per_completed_task,
    rows_for_model,
)
from wmo.optimize.routing.scorecard_ladder import build_ladder, rows_for_policy

__all__ = (
    "DEFAULT_COMPLETION",
    "DOMINANCE_TOLERANCE",
    "EFFECTIVE_COST_RULE",
    "Arm",
    "CompletionRule",
    "ConditionLabel",
    "EffectiveCost",
    "Ladder",
    "LadderRung",
    "LatencyBlock",
    "LatencyObjective",
    "OperatingPoint",
    "Provenance",
    "QualityBlock",
    "RowOverhead",
    "Scorecard",
    "build_ladder",
    "build_scorecard",
    "effective_cost_per_completed_task",
    "rows_for_model",
    "rows_for_policy",
)
