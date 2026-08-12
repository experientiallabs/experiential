"""Offline guarded kNN fitting over canonical immutable evaluation datasets."""

from wmo.optimize.router.optimizer import RouterOptimizationError, RouterOptimizer
from wmo.optimize.router.spec import (
    RouterFitResult,
    RouterOptimizationResult,
    RouterOptimizationSpec,
)
from wmo.optimize.router.workflow import (
    EvaluationInputs,
    RouterOptimizationConfig,
    RouterWorkflowError,
    RouterWorkflowResult,
    optimize_router,
)

__all__ = [
    "RouterOptimizationError",
    "RouterFitResult",
    "RouterOptimizationResult",
    "RouterOptimizationSpec",
    "RouterOptimizer",
    "EvaluationInputs",
    "RouterOptimizationConfig",
    "RouterWorkflowError",
    "RouterWorkflowResult",
    "optimize_router",
]
