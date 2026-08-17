"""World Model Optimizer public customer services with lazy package-root exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.common.evaluations import FidelityReport as FidelityReport
    from wmo.common.evaluations import (
        build_fidelity_evaluation_plan as build_fidelity_evaluation_plan,
    )
    from wmo.common.evaluations import build_fidelity_report as build_fidelity_report
    from wmo.common.models import ConnectionConfig as ConnectionConfig
    from wmo.common.models import DiscoveredModel as DiscoveredModel
    from wmo.common.models import ModelCapabilities as ModelCapabilities
    from wmo.common.models import ModelCatalog as ModelCatalog
    from wmo.common.models import ModelRecord as ModelRecord
    from wmo.common.models import ModelRoles as ModelRoles
    from wmo.common.models import ResolvedDiscoveredModel as ResolvedDiscoveredModel
    from wmo.common.models import resolve_discovered_model as resolve_discovered_model
    from wmo.common.project import ExportedProjectBundle as ExportedProjectBundle
    from wmo.common.project import ProjectBudgetConfiguration as ProjectBudgetConfiguration
    from wmo.common.project import ProjectModelConfiguration as ProjectModelConfiguration
    from wmo.common.project import ProjectProviderFreeStage as ProjectProviderFreeStage
    from wmo.common.project import ProjectRetrievalConfiguration as ProjectRetrievalConfiguration
    from wmo.common.project import ProjectStage as ProjectStage
    from wmo.common.project import ProjectStageEvent as ProjectStageEvent
    from wmo.common.project import ProjectStore as ProjectStore
    from wmo.common.project import ProjectSystemConfiguration as ProjectSystemConfiguration
    from wmo.common.project import (
        ProjectTracePreparationSettings as ProjectTracePreparationSettings,
    )
    from wmo.common.project import export_project_bundle as export_project_bundle
    from wmo.common.project import restore_project_bundle as restore_project_bundle
    from wmo.optimize.router.activation import load_project_router as load_project_router
    from wmo.optimize.router.activation import load_router as load_router
    from wmo.optimize.router.attempt_authority import (
        FileHostedAttemptAuthorityStore as FileHostedAttemptAuthorityStore,
    )
    from wmo.optimize.router.attempt_authority import (
        HostedAttemptAuthority as HostedAttemptAuthority,
    )
    from wmo.optimize.router.attempt_authority import (
        HostedAttemptAuthorityStore as HostedAttemptAuthorityStore,
    )
    from wmo.optimize.router.attempt_authority import HostedAttemptState as HostedAttemptState
    from wmo.optimize.router.attempt_authority import HostedProviderHazard as HostedProviderHazard
    from wmo.optimize.router.attempt_authority import HostedStageCommit as HostedStageCommit
    from wmo.optimize.router.attempt_authority import (
        create_hosted_attempt_authority as create_hosted_attempt_authority,
    )
    from wmo.optimize.router.composition import (
        ApprovedRouterReview as ApprovedRouterReview,
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
    from wmo.optimize.router.fit.workflow import EvaluationInputs as EvaluationInputs
    from wmo.optimize.router.fit.workflow import RouterFitConfig as RouterFitConfig
    from wmo.optimize.router.fit.workflow import (
        RouterFitWorkflowResult as RouterFitWorkflowResult,
    )
    from wmo.optimize.router.fit.workflow import (
        RouterOptimizationConfig as RouterOptimizationConfig,
    )
    from wmo.optimize.router.fit.workflow import RouterReportConfig as RouterReportConfig
    from wmo.optimize.router.fit.workflow import RouterWorkflowResult as RouterWorkflowResult
    from wmo.optimize.router.fit.workflow import fit_router as fit_router
    from wmo.optimize.router.fit.workflow import optimize_router as optimize_router
    from wmo.optimize.router.fit.workflow import report_router as report_router
    from wmo.optimize.router.hosted import HostedRouterWorkflowError as HostedRouterWorkflowError
    from wmo.optimize.router.hosted import (
        HostedRouterWorkflowOptions as HostedRouterWorkflowOptions,
    )
    from wmo.optimize.router.hosted import HostedRouterWorkflowResult as HostedRouterWorkflowResult
    from wmo.optimize.router.hosted import HostedRouterWorkflowSetup as HostedRouterWorkflowSetup
    from wmo.optimize.router.hosted import HostedStageBundle as HostedStageBundle
    from wmo.optimize.router.hosted import (
        restore_hosted_project_bundle as restore_hosted_project_bundle,
    )
    from wmo.optimize.router.hosted import run_hosted_router_workflow as run_hosted_router_workflow
    from wmo.optimize.router.hosted_preflight import (
        HostedRouterPreflightError as HostedRouterPreflightError,
    )
    from wmo.optimize.router.spend import ProviderSpendComponent as ProviderSpendComponent
    from wmo.optimize.router.spend import ProviderSpendEntry as ProviderSpendEntry
    from wmo.optimize.router.spend import ProviderSpendLedger as ProviderSpendLedger
    from wmo.optimize.router.spend import ProviderSpendStatus as ProviderSpendStatus
    from wmo.runtime.models import RuntimeModelCatalog as RuntimeModelCatalog
    from wmo.runtime.router.application import (
        create_project_router_app as create_project_router_app,
    )
    from wmo.runtime.router.runtime import (
        RoutedCompletionEconomics as RoutedCompletionEconomics,
    )
    from wmo.runtime.router.runtime import RoutedModelResponse as RoutedModelResponse
    from wmo.runtime.router.runtime import RouterRuntime as RouterRuntime
    from wmo.simulation.build import BuildReviewReadiness as BuildReviewReadiness
    from wmo.simulation.build import ProjectBuild as ProjectBuild
    from wmo.simulation.build import TaskSetBuild as TaskSetBuild
    from wmo.simulation.build import build_project as build_project
    from wmo.simulation.build import build_task_set as build_task_set
    from wmo.simulation.build import (
        load_project_provider_free_stage as load_project_provider_free_stage,
    )
    from wmo.simulation.build import prepare_project_traces as prepare_project_traces
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
    "ConnectionConfig": "wmo.common.models",
    "DiscoveredModel": "wmo.common.models",
    "ModelCapabilities": "wmo.common.models",
    "ModelCatalog": "wmo.common.models",
    "ModelRecord": "wmo.common.models",
    "ModelRoles": "wmo.common.models",
    "ResolvedDiscoveredModel": "wmo.common.models",
    "resolve_discovered_model": "wmo.common.models",
    "RuntimeModelCatalog": "wmo.runtime.models",
    "ExportedProjectBundle": "wmo.common.project",
    "export_project_bundle": "wmo.common.project",
    "restore_project_bundle": "wmo.common.project",
    "ProjectProviderFreeStage": "wmo.common.project",
    "ProjectBudgetConfiguration": "wmo.common.project",
    "ProjectModelConfiguration": "wmo.common.project",
    "ProjectRetrievalConfiguration": "wmo.common.project",
    "ProjectStage": "wmo.common.project",
    "ProjectStageEvent": "wmo.common.project",
    "ProjectStore": "wmo.common.project",
    "ProjectSystemConfiguration": "wmo.common.project",
    "ProjectTracePreparationSettings": "wmo.common.project",
    "BuildReviewReadiness": "wmo.simulation.build",
    "ProjectBuild": "wmo.simulation.build",
    "TaskSetBuild": "wmo.simulation.build",
    "build_project": "wmo.simulation.build",
    "build_task_set": "wmo.simulation.build",
    "load_project_provider_free_stage": "wmo.simulation.build",
    "prepare_project_traces": "wmo.simulation.build",
    "WorldModel": "wmo.simulation.world_model.application",
    "WorldModelLoadError": "wmo.simulation.world_model.application",
    "WorldModelObservation": "wmo.simulation.world_model.application",
    "WorldModelSession": "wmo.simulation.world_model.application",
    "WorldModelSessionError": "wmo.simulation.world_model.application",
    "WorldModelSessionLimits": "wmo.simulation.world_model.application",
    "load_world_model": "wmo.simulation.world_model.application",
    "FidelityReport": "wmo.common.evaluations",
    "build_fidelity_evaluation_plan": "wmo.common.evaluations",
    "build_fidelity_report": "wmo.common.evaluations",
    "EvaluationInputs": "wmo.optimize.router.fit.workflow",
    "RouterFitConfig": "wmo.optimize.router.fit.workflow",
    "RouterFitWorkflowResult": "wmo.optimize.router.fit.workflow",
    "RouterOptimizationConfig": "wmo.optimize.router.fit.workflow",
    "RouterReportConfig": "wmo.optimize.router.fit.workflow",
    "RouterWorkflowResult": "wmo.optimize.router.fit.workflow",
    "fit_router": "wmo.optimize.router.fit.workflow",
    "optimize_router": "wmo.optimize.router.fit.workflow",
    "report_router": "wmo.optimize.router.fit.workflow",
    "RouterRuntime": "wmo.runtime.router.runtime",
    "create_project_router_app": "wmo.runtime.router.application",
    "load_project_router": "wmo.optimize.router.activation",
    "load_router": "wmo.optimize.router.activation",
    "ApprovedRouterReview": "wmo.optimize.router.composition",
    "RouterCompositionBudget": "wmo.optimize.router.composition",
    "RouterCompositionResult": "wmo.optimize.router.composition",
    "RouterEvaluationSetup": "wmo.optimize.router.composition",
    "RouterWorkflowServices": "wmo.optimize.router.composition",
    "compose_router": "wmo.optimize.router.composition",
    "HostedRouterWorkflowError": "wmo.optimize.router.hosted",
    "HostedRouterWorkflowOptions": "wmo.optimize.router.hosted",
    "HostedRouterWorkflowResult": "wmo.optimize.router.hosted",
    "HostedRouterWorkflowSetup": "wmo.optimize.router.hosted",
    "HostedStageBundle": "wmo.optimize.router.hosted",
    "restore_hosted_project_bundle": "wmo.optimize.router.hosted",
    "run_hosted_router_workflow": "wmo.optimize.router.hosted",
    "FileHostedAttemptAuthorityStore": "wmo.optimize.router.attempt_authority",
    "HostedAttemptAuthority": "wmo.optimize.router.attempt_authority",
    "HostedAttemptAuthorityStore": "wmo.optimize.router.attempt_authority",
    "HostedAttemptState": "wmo.optimize.router.attempt_authority",
    "HostedProviderHazard": "wmo.optimize.router.attempt_authority",
    "HostedStageCommit": "wmo.optimize.router.attempt_authority",
    "create_hosted_attempt_authority": "wmo.optimize.router.attempt_authority",
    "HostedRouterPreflightError": "wmo.optimize.router.hosted_preflight",
    "ProviderSpendComponent": "wmo.optimize.router.spend",
    "ProviderSpendEntry": "wmo.optimize.router.spend",
    "ProviderSpendLedger": "wmo.optimize.router.spend",
    "ProviderSpendStatus": "wmo.optimize.router.spend",
    "RoutedCompletionEconomics": "wmo.runtime.router.runtime",
    "RoutedModelResponse": "wmo.runtime.router.runtime",
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
