"""Offline guarded kNN fitting over canonical immutable evaluation datasets."""

from wmo.optimize.router.optimizer import RouterOptimizationError, RouterOptimizer
from wmo.optimize.router.spec import (
    RouterFitResult,
    RouterOptimizationResult,
    RouterOptimizationSpec,
)
from wmo.optimize.router.workflow import (
    EvaluationInputs,
    RouterFitConfig,
    RouterFitWorkflowResult,
    RouterOptimizationConfig,
    RouterReportConfig,
    RouterWorkflowError,
    RouterWorkflowResult,
    fit_router,
    optimize_router,
    report_router,
)

__all__ = [
    "RouterOptimizationError",
    "RouterFitResult",
    "RouterOptimizationResult",
    "RouterOptimizationSpec",
    "RouterOptimizer",
    "EvaluationInputs",
    "RouterFitConfig",
    "RouterFitWorkflowResult",
    "RouterOptimizationConfig",
    "RouterReportConfig",
    "RouterWorkflowError",
    "RouterWorkflowResult",
    "optimize_router",
    "fit_router",
    "report_router",
]
