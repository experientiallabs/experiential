"""World Model Optimizer public customer services with lazy package-root exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.optimize.router.workflow import EvaluationInputs as EvaluationInputs
    from wmo.optimize.router.workflow import (
        RouterOptimizationConfig as RouterOptimizationConfig,
    )
    from wmo.optimize.router.workflow import RouterWorkflowResult as RouterWorkflowResult
    from wmo.optimize.router.workflow import optimize_router as optimize_router
    from wmo.runtime.router.application import (
        create_project_router_app as create_project_router_app,
    )
    from wmo.runtime.router.application import load_project_router as load_project_router
    from wmo.runtime.router.runtime import RouterRuntime as RouterRuntime
    from wmo.simulation.build import BuildReviewReadiness as BuildReviewReadiness
    from wmo.simulation.build import ProjectBuild as ProjectBuild
    from wmo.simulation.build import TaskSetBuild as TaskSetBuild
    from wmo.simulation.build import build_project as build_project
    from wmo.simulation.build import build_task_set as build_task_set
    from wmo.workflow.router import (
        ApprovedRouterReview as ApprovedRouterReview,
    )
    from wmo.workflow.router import LocalTraceSource as LocalTraceSource
    from wmo.workflow.router import (
        RouterCompositionBudget as RouterCompositionBudget,
    )
    from wmo.workflow.router import (
        RouterCompositionResult as RouterCompositionResult,
    )
    from wmo.workflow.router import (
        RouterEvaluationSetup as RouterEvaluationSetup,
    )
    from wmo.workflow.router import RouterWorkflowServices as RouterWorkflowServices
    from wmo.workflow.router import compose_router as compose_router

_EXPORT_MODULES = {
    "BuildReviewReadiness": "wmo.simulation.build",
    "ProjectBuild": "wmo.simulation.build",
    "TaskSetBuild": "wmo.simulation.build",
    "build_project": "wmo.simulation.build",
    "build_task_set": "wmo.simulation.build",
    "EvaluationInputs": "wmo.optimize.router.workflow",
    "RouterOptimizationConfig": "wmo.optimize.router.workflow",
    "RouterWorkflowResult": "wmo.optimize.router.workflow",
    "optimize_router": "wmo.optimize.router.workflow",
    "RouterRuntime": "wmo.runtime.router.runtime",
    "create_project_router_app": "wmo.runtime.router.application",
    "load_project_router": "wmo.runtime.router.application",
    "ApprovedRouterReview": "wmo.workflow.router",
    "LocalTraceSource": "wmo.workflow.router",
    "RouterCompositionBudget": "wmo.workflow.router",
    "RouterCompositionResult": "wmo.workflow.router",
    "RouterEvaluationSetup": "wmo.workflow.router",
    "RouterWorkflowServices": "wmo.workflow.router",
    "compose_router": "wmo.workflow.router",
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
