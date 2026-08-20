"""Router composition, automatic evidence execution, fitting, and activation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exp.optimize.router.activation import load_project_router as load_project_router
    from exp.optimize.router.activation import load_router as load_router
    from exp.optimize.router.attempt_authority import (
        FileHostedAttemptAuthorityStore as FileHostedAttemptAuthorityStore,
    )
    from exp.optimize.router.attempt_authority import (
        HostedAttemptAuthority as HostedAttemptAuthority,
    )
    from exp.optimize.router.attempt_authority import (
        HostedAttemptAuthorityStore as HostedAttemptAuthorityStore,
    )
    from exp.optimize.router.attempt_authority import HostedAttemptState as HostedAttemptState
    from exp.optimize.router.attempt_authority import HostedProviderHazard as HostedProviderHazard
    from exp.optimize.router.attempt_authority import HostedStageCommit as HostedStageCommit
    from exp.optimize.router.attempt_authority import (
        create_hosted_attempt_authority as create_hosted_attempt_authority,
    )
    from exp.optimize.router.composition import ApprovedRouterReview as ApprovedRouterReview
    from exp.optimize.router.composition import (
        RouterCandidateSetupPlan as RouterCandidateSetupPlan,
    )
    from exp.optimize.router.composition import (
        RouterCompositionBudget as RouterCompositionBudget,
    )
    from exp.optimize.router.composition import (
        RouterCompositionResult as RouterCompositionResult,
    )
    from exp.optimize.router.composition import (
        RouterEvaluationSetup as RouterEvaluationSetup,
    )
    from exp.optimize.router.composition import RouterWorkflowServices as RouterWorkflowServices
    from exp.optimize.router.composition import compose_router as compose_router
    from exp.optimize.router.fit.optimizer import RouterOptimizationError as RouterOptimizationError
    from exp.optimize.router.fit.optimizer import RouterOptimizer as RouterOptimizer
    from exp.optimize.router.fit.spec import RouterFitResult as RouterFitResult
    from exp.optimize.router.fit.spec import RouterOptimizationResult as RouterOptimizationResult
    from exp.optimize.router.fit.spec import RouterOptimizationSpec as RouterOptimizationSpec
    from exp.optimize.router.fit.workflow import EvaluationInputs as EvaluationInputs
    from exp.optimize.router.fit.workflow import RouterFitConfig as RouterFitConfig
    from exp.optimize.router.fit.workflow import (
        RouterFitWorkflowResult as RouterFitWorkflowResult,
    )
    from exp.optimize.router.fit.workflow import (
        RouterOptimizationConfig as RouterOptimizationConfig,
    )
    from exp.optimize.router.fit.workflow import RouterReportConfig as RouterReportConfig
    from exp.optimize.router.fit.workflow import RouterWorkflowError as RouterWorkflowError
    from exp.optimize.router.fit.workflow import RouterWorkflowResult as RouterWorkflowResult
    from exp.optimize.router.fit.workflow import fit_router as fit_router
    from exp.optimize.router.fit.workflow import optimize_router as optimize_router
    from exp.optimize.router.fit.workflow import report_router as report_router
    from exp.optimize.router.hosted import HostedRouterWorkflowError as HostedRouterWorkflowError
    from exp.optimize.router.hosted import (
        HostedRouterWorkflowOptions as HostedRouterWorkflowOptions,
    )
    from exp.optimize.router.hosted import HostedRouterWorkflowResult as HostedRouterWorkflowResult
    from exp.optimize.router.hosted import HostedRouterWorkflowSetup as HostedRouterWorkflowSetup
    from exp.optimize.router.hosted import HostedStageBundle as HostedStageBundle
    from exp.optimize.router.hosted import (
        restore_hosted_project_bundle as restore_hosted_project_bundle,
    )
    from exp.optimize.router.hosted import run_hosted_router_workflow as run_hosted_router_workflow
    from exp.optimize.router.hosted_preflight import (
        HostedRouterPreflightError as HostedRouterPreflightError,
    )
    from exp.optimize.router.spend import ProviderSpendComponent as ProviderSpendComponent
    from exp.optimize.router.spend import ProviderSpendEntry as ProviderSpendEntry
    from exp.optimize.router.spend import ProviderSpendLedger as ProviderSpendLedger
    from exp.optimize.router.spend import ProviderSpendStatus as ProviderSpendStatus

_EXPORT_MODULES = {
    "FileHostedAttemptAuthorityStore": "exp.optimize.router.attempt_authority",
    "HostedAttemptAuthority": "exp.optimize.router.attempt_authority",
    "HostedAttemptAuthorityStore": "exp.optimize.router.attempt_authority",
    "HostedAttemptState": "exp.optimize.router.attempt_authority",
    "HostedProviderHazard": "exp.optimize.router.attempt_authority",
    "HostedStageCommit": "exp.optimize.router.attempt_authority",
    "create_hosted_attempt_authority": "exp.optimize.router.attempt_authority",
    "RouterOptimizationError": "exp.optimize.router.fit.optimizer",
    "RouterFitResult": "exp.optimize.router.fit.spec",
    "RouterOptimizationResult": "exp.optimize.router.fit.spec",
    "RouterOptimizationSpec": "exp.optimize.router.fit.spec",
    "RouterOptimizer": "exp.optimize.router.fit.optimizer",
    "EvaluationInputs": "exp.optimize.router.fit.workflow",
    "RouterFitConfig": "exp.optimize.router.fit.workflow",
    "RouterFitWorkflowResult": "exp.optimize.router.fit.workflow",
    "RouterOptimizationConfig": "exp.optimize.router.fit.workflow",
    "RouterReportConfig": "exp.optimize.router.fit.workflow",
    "RouterWorkflowError": "exp.optimize.router.fit.workflow",
    "RouterWorkflowResult": "exp.optimize.router.fit.workflow",
    "optimize_router": "exp.optimize.router.fit.workflow",
    "fit_router": "exp.optimize.router.fit.workflow",
    "report_router": "exp.optimize.router.fit.workflow",
    "ApprovedRouterReview": "exp.optimize.router.composition",
    "RouterCandidateSetupPlan": "exp.optimize.router.composition",
    "RouterCompositionBudget": "exp.optimize.router.composition",
    "RouterCompositionResult": "exp.optimize.router.composition",
    "RouterEvaluationSetup": "exp.optimize.router.composition",
    "RouterWorkflowServices": "exp.optimize.router.composition",
    "compose_router": "exp.optimize.router.composition",
    "load_project_router": "exp.optimize.router.activation",
    "load_router": "exp.optimize.router.activation",
    "HostedRouterWorkflowError": "exp.optimize.router.hosted",
    "HostedRouterWorkflowOptions": "exp.optimize.router.hosted",
    "HostedRouterWorkflowResult": "exp.optimize.router.hosted",
    "HostedRouterWorkflowSetup": "exp.optimize.router.hosted",
    "HostedStageBundle": "exp.optimize.router.hosted",
    "restore_hosted_project_bundle": "exp.optimize.router.hosted",
    "run_hosted_router_workflow": "exp.optimize.router.hosted",
    "HostedRouterPreflightError": "exp.optimize.router.hosted_preflight",
    "ProviderSpendComponent": "exp.optimize.router.spend",
    "ProviderSpendEntry": "exp.optimize.router.spend",
    "ProviderSpendLedger": "exp.optimize.router.spend",
    "ProviderSpendStatus": "exp.optimize.router.spend",
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
