"""World Model Optimizer public customer services with lazy package-root exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.optimize.router.activation import load_project_router as load_project_router
    from wmo.optimize.router.activation import load_router as load_router
    from wmo.optimize.router.composition import (
        ApprovedRouterReview as ApprovedRouterReview,
    )
    from wmo.optimize.router.composition import FidelityApprovalDecision as FidelityApprovalDecision
    from wmo.optimize.router.composition import LocalTraceSource as LocalTraceSource
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
    from wmo.optimize.router.workflow import EvaluationInputs as EvaluationInputs
    from wmo.optimize.router.workflow import RouterFitConfig as RouterFitConfig
    from wmo.optimize.router.workflow import (
        RouterFitWorkflowResult as RouterFitWorkflowResult,
    )
    from wmo.optimize.router.workflow import (
        RouterOptimizationConfig as RouterOptimizationConfig,
    )
    from wmo.optimize.router.workflow import RouterReportConfig as RouterReportConfig
    from wmo.optimize.router.workflow import RouterWorkflowResult as RouterWorkflowResult
    from wmo.optimize.router.workflow import fit_router as fit_router
    from wmo.optimize.router.workflow import optimize_router as optimize_router
    from wmo.optimize.router.workflow import report_router as report_router
    from wmo.runtime.router.application import (
        create_project_router_app as create_project_router_app,
    )
    from wmo.runtime.router.runtime import RouterRuntime as RouterRuntime
    from wmo.simulation.build import BuildReviewReadiness as BuildReviewReadiness
    from wmo.simulation.build import ProjectBuild as ProjectBuild
    from wmo.simulation.build import TaskSetBuild as TaskSetBuild
    from wmo.simulation.build import build_project as build_project
    from wmo.simulation.build import build_task_set as build_task_set
    from wmo.simulation.world_model.application import WorldModel as WorldModel
    from wmo.simulation.world_model.application import (
        WorldModelLoadError as WorldModelLoadError,
    )
    from wmo.simulation.world_model.application import (
        WorldModelObservation as WorldModelObservation,
    )
    from wmo.simulation.world_model.application import (
        WorldModelSession as WorldModelSession,
    )
    from wmo.simulation.world_model.application import (
        WorldModelSessionError as WorldModelSessionError,
    )
    from wmo.simulation.world_model.application import (
        WorldModelSessionLimits as WorldModelSessionLimits,
    )
    from wmo.simulation.world_model.application import load_world_model as load_world_model

_EXPORT_MODULES = {
    "BuildReviewReadiness": "wmo.simulation.build",
    "ProjectBuild": "wmo.simulation.build",
    "TaskSetBuild": "wmo.simulation.build",
    "build_project": "wmo.simulation.build",
    "build_task_set": "wmo.simulation.build",
    "WorldModel": "wmo.simulation.world_model.application",
    "WorldModelLoadError": "wmo.simulation.world_model.application",
    "WorldModelObservation": "wmo.simulation.world_model.application",
    "WorldModelSession": "wmo.simulation.world_model.application",
    "WorldModelSessionError": "wmo.simulation.world_model.application",
    "WorldModelSessionLimits": "wmo.simulation.world_model.application",
    "load_world_model": "wmo.simulation.world_model.application",
    "EvaluationInputs": "wmo.optimize.router.workflow",
    "RouterFitConfig": "wmo.optimize.router.workflow",
    "RouterFitWorkflowResult": "wmo.optimize.router.workflow",
    "RouterOptimizationConfig": "wmo.optimize.router.workflow",
    "RouterReportConfig": "wmo.optimize.router.workflow",
    "RouterWorkflowResult": "wmo.optimize.router.workflow",
    "fit_router": "wmo.optimize.router.workflow",
    "optimize_router": "wmo.optimize.router.workflow",
    "report_router": "wmo.optimize.router.workflow",
    "RouterRuntime": "wmo.runtime.router.runtime",
    "create_project_router_app": "wmo.runtime.router.application",
    "load_project_router": "wmo.optimize.router.activation",
    "load_router": "wmo.optimize.router.activation",
    "ApprovedRouterReview": "wmo.optimize.router.composition",
    "LocalTraceSource": "wmo.optimize.router.composition",
    "FidelityApprovalDecision": "wmo.optimize.router.composition",
    "RouterCompositionBudget": "wmo.optimize.router.composition",
    "RouterCompositionResult": "wmo.optimize.router.composition",
    "RouterEvaluationSetup": "wmo.optimize.router.composition",
    "RouterWorkflowServices": "wmo.optimize.router.composition",
    "compose_router": "wmo.optimize.router.composition",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    """Resolve one supported package-root service on first access.

    Args:
        name: Package attribute requested by Python.

    Returns:
        The supported public object loaded from its owning domain module.

    Raises:
        AttributeError: The name is not part of the supported public API.
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
