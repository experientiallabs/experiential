"""Router composition, automatic evidence execution, fitting, and activation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.optimize.router.activation import load_project_router as load_project_router
    from wmo.optimize.router.activation import load_router as load_router
    from wmo.optimize.router.composition import ApprovedRouterReview as ApprovedRouterReview
    from wmo.optimize.router.composition import (
        RouterCandidateSetupPlan as RouterCandidateSetupPlan,
    )
    from wmo.optimize.router.composition import (
        RouterCompositionBudget as RouterCompositionBudget,
    )
    from wmo.optimize.router.composition import (
        RouterCompositionResult as RouterCompositionResult,
    )
    from wmo.optimize.router.composition import (
        RouterEvaluationSetup as RouterEvaluationSetup,
    )
    from wmo.optimize.router.composition import RouterWorkflowServices as RouterWorkflowServices
    from wmo.optimize.router.composition import compose_router as compose_router
    from wmo.optimize.router.fit.optimizer import RouterOptimizationError as RouterOptimizationError
    from wmo.optimize.router.fit.optimizer import RouterOptimizer as RouterOptimizer
    from wmo.optimize.router.fit.spec import RouterFitResult as RouterFitResult
    from wmo.optimize.router.fit.spec import RouterOptimizationResult as RouterOptimizationResult
    from wmo.optimize.router.fit.spec import RouterOptimizationSpec as RouterOptimizationSpec
    from wmo.optimize.router.fit.workflow import EvaluationInputs as EvaluationInputs
    from wmo.optimize.router.fit.workflow import RouterFitConfig as RouterFitConfig
    from wmo.optimize.router.fit.workflow import (
        RouterFitWorkflowResult as RouterFitWorkflowResult,
    )
    from wmo.optimize.router.fit.workflow import RouterReportConfig as RouterReportConfig
    from wmo.optimize.router.fit.workflow import RouterWorkflowError as RouterWorkflowError
    from wmo.optimize.router.fit.workflow import RouterWorkflowResult as RouterWorkflowResult
    from wmo.optimize.router.fit.workflow import fit_router as fit_router
    from wmo.optimize.router.fit.workflow import report_router as report_router

_EXPORT_MODULES = {
    "RouterOptimizationError": "wmo.optimize.router.fit.optimizer",
    "RouterFitResult": "wmo.optimize.router.fit.spec",
    "RouterOptimizationResult": "wmo.optimize.router.fit.spec",
    "RouterOptimizationSpec": "wmo.optimize.router.fit.spec",
    "RouterOptimizer": "wmo.optimize.router.fit.optimizer",
    "EvaluationInputs": "wmo.optimize.router.fit.workflow",
    "RouterFitConfig": "wmo.optimize.router.fit.workflow",
    "RouterFitWorkflowResult": "wmo.optimize.router.fit.workflow",
    "RouterReportConfig": "wmo.optimize.router.fit.workflow",
    "RouterWorkflowError": "wmo.optimize.router.fit.workflow",
    "RouterWorkflowResult": "wmo.optimize.router.fit.workflow",
    "fit_router": "wmo.optimize.router.fit.workflow",
    "report_router": "wmo.optimize.router.fit.workflow",
    "ApprovedRouterReview": "wmo.optimize.router.composition",
    "RouterCandidateSetupPlan": "wmo.optimize.router.composition",
    "RouterCompositionBudget": "wmo.optimize.router.composition",
    "RouterCompositionResult": "wmo.optimize.router.composition",
    "RouterEvaluationSetup": "wmo.optimize.router.composition",
    "RouterWorkflowServices": "wmo.optimize.router.composition",
    "compose_router": "wmo.optimize.router.composition",
    "load_project_router": "wmo.optimize.router.activation",
    "load_router": "wmo.optimize.router.activation",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    """Resolve one supported router optimization export on first access.

    Args:
        name: Package attribute requested by Python.

    Returns:
        The supported object loaded from its owning router module.

    Raises:
        AttributeError: The name is not part of the supported package API.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus supported lazy exports for introspection."""
    return sorted(set(globals()) | set(__all__))
