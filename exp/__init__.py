"""World Model Optimizer public customer services with lazy package-root exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exp.common.evaluations import FidelityReport as FidelityReport
    from exp.common.evaluations import (
        build_fidelity_evaluation_plan as build_fidelity_evaluation_plan,
    )
    from exp.common.evaluations import build_fidelity_report as build_fidelity_report
    from exp.common.models import BillingSource as BillingSource
    from exp.common.models import ConnectionConfig as ConnectionConfig
    from exp.common.models import DiscoveredModel as DiscoveredModel
    from exp.common.models import ModelCapabilities as ModelCapabilities
    from exp.common.models import ModelCatalog as ModelCatalog
    from exp.common.models import ModelRecord as ModelRecord
    from exp.common.models import ModelRoles as ModelRoles
    from exp.common.models import ResolvedDiscoveredModel as ResolvedDiscoveredModel
    from exp.common.models import resolve_discovered_model as resolve_discovered_model
    from exp.common.project import ExportedProjectBundle as ExportedProjectBundle
    from exp.common.project import ProjectBudgetConfiguration as ProjectBudgetConfiguration
    from exp.common.project import ProjectModelConfiguration as ProjectModelConfiguration
    from exp.common.project import ProjectProviderFreeStage as ProjectProviderFreeStage
    from exp.common.project import ProjectRetrievalConfiguration as ProjectRetrievalConfiguration
    from exp.common.project import ProjectStage as ProjectStage
    from exp.common.project import ProjectStageEvent as ProjectStageEvent
    from exp.common.project import ProjectStore as ProjectStore
    from exp.common.project import ProjectSystemConfiguration as ProjectSystemConfiguration
    from exp.common.project import (
        ProjectTracePreparationSettings as ProjectTracePreparationSettings,
    )
    from exp.common.project import export_project_bundle as export_project_bundle
    from exp.common.project import restore_project_bundle as restore_project_bundle
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
    from exp.optimize.router.composition import (
        ApprovedRouterReview as ApprovedRouterReview,
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
    from exp.optimize.router.fit.workflow import EvaluationInputs as EvaluationInputs
    from exp.optimize.router.fit.workflow import RouterFitConfig as RouterFitConfig
    from exp.optimize.router.fit.workflow import (
        RouterFitWorkflowResult as RouterFitWorkflowResult,
    )
    from exp.optimize.router.fit.workflow import (
        RouterOptimizationConfig as RouterOptimizationConfig,
    )
    from exp.optimize.router.fit.workflow import RouterReportConfig as RouterReportConfig
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
    from exp.runtime.gateway.composition import GatewayRuntime as GatewayRuntime
    from exp.runtime.gateway.composition import GatewayRuntimeConfig as GatewayRuntimeConfig
    from exp.runtime.gateway.composition import create_gateway_runtime as create_gateway_runtime
    from exp.runtime.models import RuntimeModelCatalog as RuntimeModelCatalog
    from exp.runtime.router.economics import (
        BillingSourceEconomics as BillingSourceEconomics,
    )
    from exp.runtime.router.economics import (
        RoutedCompletionEconomics as RoutedCompletionEconomics,
    )
    from exp.runtime.router.economics import (
        RoutedProviderComponent as RoutedProviderComponent,
    )
    from exp.runtime.router.economics import (
        RoutedProviderOperation as RoutedProviderOperation,
    )
    from exp.runtime.router.economics import (
        RoutedSpendDisposition as RoutedSpendDisposition,
    )
    from exp.runtime.router.economics import RoutedSpendLedger as RoutedSpendLedger
    from exp.runtime.router.runtime import RoutedModelResponse as RoutedModelResponse
    from exp.runtime.router.runtime import RouterRuntime as RouterRuntime
    from exp.simulation.build import BuildReviewReadiness as BuildReviewReadiness
    from exp.simulation.build import ProjectBuild as ProjectBuild
    from exp.simulation.build import TaskSetBuild as TaskSetBuild
    from exp.simulation.build import build_project as build_project
    from exp.simulation.build import build_task_set as build_task_set
    from exp.simulation.build import (
        load_project_provider_free_stage as load_project_provider_free_stage,
    )
    from exp.simulation.build import prepare_project_traces as prepare_project_traces
    from exp.simulation.world_model.application import WorldModel as WorldModel
    from exp.simulation.world_model.application import (
        WorldModelLoadError as WorldModelLoadError,
    )
    from exp.simulation.world_model.application import (
        WorldModelObservation as WorldModelObservation,
    )
    from exp.simulation.world_model.application import (
        WorldModelSession as WorldModelSession,
    )
    from exp.simulation.world_model.application import (
        WorldModelSessionError as WorldModelSessionError,
    )
    from exp.simulation.world_model.application import (
        WorldModelSessionLimits as WorldModelSessionLimits,
    )
    from exp.simulation.world_model.application import load_world_model as load_world_model

_EXPORT_MODULES = {
    "BillingSource": "exp.common.models",
    "ConnectionConfig": "exp.common.models",
    "DiscoveredModel": "exp.common.models",
    "ModelCapabilities": "exp.common.models",
    "ModelCatalog": "exp.common.models",
    "ModelRecord": "exp.common.models",
    "ModelRoles": "exp.common.models",
    "ResolvedDiscoveredModel": "exp.common.models",
    "resolve_discovered_model": "exp.common.models",
    "RuntimeModelCatalog": "exp.runtime.models",
    "GatewayRuntime": "exp.runtime.gateway.composition",
    "GatewayRuntimeConfig": "exp.runtime.gateway.composition",
    "create_gateway_runtime": "exp.runtime.gateway.composition",
    "ExportedProjectBundle": "exp.common.project",
    "export_project_bundle": "exp.common.project",
    "restore_project_bundle": "exp.common.project",
    "ProjectProviderFreeStage": "exp.common.project",
    "ProjectBudgetConfiguration": "exp.common.project",
    "ProjectModelConfiguration": "exp.common.project",
    "ProjectRetrievalConfiguration": "exp.common.project",
    "ProjectStage": "exp.common.project",
    "ProjectStageEvent": "exp.common.project",
    "ProjectStore": "exp.common.project",
    "ProjectSystemConfiguration": "exp.common.project",
    "ProjectTracePreparationSettings": "exp.common.project",
    "BuildReviewReadiness": "exp.simulation.build",
    "ProjectBuild": "exp.simulation.build",
    "TaskSetBuild": "exp.simulation.build",
    "build_project": "exp.simulation.build",
    "build_task_set": "exp.simulation.build",
    "load_project_provider_free_stage": "exp.simulation.build",
    "prepare_project_traces": "exp.simulation.build",
    "WorldModel": "exp.simulation.world_model.application",
    "WorldModelLoadError": "exp.simulation.world_model.application",
    "WorldModelObservation": "exp.simulation.world_model.application",
    "WorldModelSession": "exp.simulation.world_model.application",
    "WorldModelSessionError": "exp.simulation.world_model.application",
    "WorldModelSessionLimits": "exp.simulation.world_model.application",
    "load_world_model": "exp.simulation.world_model.application",
    "FidelityReport": "exp.common.evaluations",
    "build_fidelity_evaluation_plan": "exp.common.evaluations",
    "build_fidelity_report": "exp.common.evaluations",
    "EvaluationInputs": "exp.optimize.router.fit.workflow",
    "RouterFitConfig": "exp.optimize.router.fit.workflow",
    "RouterFitWorkflowResult": "exp.optimize.router.fit.workflow",
    "RouterOptimizationConfig": "exp.optimize.router.fit.workflow",
    "RouterReportConfig": "exp.optimize.router.fit.workflow",
    "RouterWorkflowResult": "exp.optimize.router.fit.workflow",
    "fit_router": "exp.optimize.router.fit.workflow",
    "optimize_router": "exp.optimize.router.fit.workflow",
    "report_router": "exp.optimize.router.fit.workflow",
    "RouterRuntime": "exp.runtime.router.runtime",
    "load_project_router": "exp.optimize.router.activation",
    "load_router": "exp.optimize.router.activation",
    "ApprovedRouterReview": "exp.optimize.router.composition",
    "RouterCompositionBudget": "exp.optimize.router.composition",
    "RouterCompositionResult": "exp.optimize.router.composition",
    "RouterEvaluationSetup": "exp.optimize.router.composition",
    "RouterWorkflowServices": "exp.optimize.router.composition",
    "compose_router": "exp.optimize.router.composition",
    "HostedRouterWorkflowError": "exp.optimize.router.hosted",
    "HostedRouterWorkflowOptions": "exp.optimize.router.hosted",
    "HostedRouterWorkflowResult": "exp.optimize.router.hosted",
    "HostedRouterWorkflowSetup": "exp.optimize.router.hosted",
    "HostedStageBundle": "exp.optimize.router.hosted",
    "restore_hosted_project_bundle": "exp.optimize.router.hosted",
    "run_hosted_router_workflow": "exp.optimize.router.hosted",
    "FileHostedAttemptAuthorityStore": "exp.optimize.router.attempt_authority",
    "HostedAttemptAuthority": "exp.optimize.router.attempt_authority",
    "HostedAttemptAuthorityStore": "exp.optimize.router.attempt_authority",
    "HostedAttemptState": "exp.optimize.router.attempt_authority",
    "HostedProviderHazard": "exp.optimize.router.attempt_authority",
    "HostedStageCommit": "exp.optimize.router.attempt_authority",
    "create_hosted_attempt_authority": "exp.optimize.router.attempt_authority",
    "HostedRouterPreflightError": "exp.optimize.router.hosted_preflight",
    "ProviderSpendComponent": "exp.optimize.router.spend",
    "ProviderSpendEntry": "exp.optimize.router.spend",
    "ProviderSpendLedger": "exp.optimize.router.spend",
    "ProviderSpendStatus": "exp.optimize.router.spend",
    "BillingSourceEconomics": "exp.runtime.router.economics",
    "RoutedCompletionEconomics": "exp.runtime.router.economics",
    "RoutedProviderComponent": "exp.runtime.router.economics",
    "RoutedProviderOperation": "exp.runtime.router.economics",
    "RoutedSpendDisposition": "exp.runtime.router.economics",
    "RoutedSpendLedger": "exp.runtime.router.economics",
    "RoutedModelResponse": "exp.runtime.router.runtime",
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
