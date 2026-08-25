"""Tests for the public package API surface."""

from __future__ import annotations

import subprocess
import sys
from inspect import signature

import exp
import exp.cli.optimize.router as router_cli
import exp.common.evaluations as evaluations
from exp.common.evaluations import (
    FidelityReport,
    build_fidelity_evaluation_plan,
    build_fidelity_report,
)
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    DiscoveredModel,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    ModelSnapshot,
    ResolvedDiscoveredModel,
    resolve_discovered_model,
)
from exp.common.project import (
    ExportedProjectBundle,
    ProjectBudgetConfiguration,
    ProjectModelConfiguration,
    ProjectProviderFreeStage,
    ProjectRetrievalConfiguration,
    ProjectStage,
    ProjectStageEvent,
    ProjectStore,
    ProjectSystemConfiguration,
    ProjectTracePreparationSettings,
    export_project_bundle,
    restore_project_bundle,
)
from exp.optimize.router.activation import load_project_router, load_router
from exp.optimize.router.attempt_authority import (
    FileHostedAttemptAuthorityStore,
    HostedAttemptAuthority,
    HostedAttemptAuthorityStore,
    HostedAttemptState,
    HostedProviderHazard,
    HostedStageCommit,
    create_hosted_attempt_authority,
)
from exp.optimize.router.automatic import service as automatic_router
from exp.optimize.router.composition import compose_router
from exp.optimize.router.fit.workflow import fit_router, optimize_router, report_router
from exp.optimize.router.hosted import (
    HostedRouterWorkflowOptions,
    HostedRouterWorkflowResult,
    HostedRouterWorkflowSetup,
    restore_hosted_project_bundle,
    run_hosted_router_workflow,
)
from exp.optimize.router.spend import ProviderSpendEntry, ProviderSpendLedger
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.router.economics import (
    BillingSourceEconomics,
    RoutedCompletionEconomics,
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
    RoutedSpendLedger,
)
from exp.runtime.router.runtime import (
    RoutedModelResponse,
    RouterRuntime,
)
from exp.simulation.build import (
    build_project,
    load_project_provider_free_stage,
    prepare_project_traces,
)
from exp.simulation.world_model.application import (
    WorldModel,
    WorldModelLoadError,
    WorldModelObservation,
    WorldModelSession,
    WorldModelSessionError,
    WorldModelSessionLimits,
    load_world_model,
)


def test_public_api_matches_quickstart() -> None:
    """The quickstart uses only deliberate package-root services.

    Every supported build, optimization, router, and world-model entrypoint resolves to its owning
    implementation while unsupported conveniences remain absent.
    """
    assert exp.build_project is build_project
    assert exp.prepare_project_traces is prepare_project_traces
    assert exp.load_project_provider_free_stage is load_project_provider_free_stage
    assert exp.ProjectProviderFreeStage is ProjectProviderFreeStage
    assert exp.ProjectTracePreparationSettings is ProjectTracePreparationSettings
    assert exp.export_project_bundle is export_project_bundle
    assert exp.restore_project_bundle is restore_project_bundle
    assert exp.ExportedProjectBundle is ExportedProjectBundle
    assert {
        "export_project_bundle",
        "restore_project_bundle",
        "ExportedProjectBundle",
    }.issubset(exp.__all__)
    assert not {
        "ProjectBundleManifest",
        "ProjectBundleMember",
        "ProjectBundleError",
        "ProjectModelCatalog",
    }.intersection(exp.__all__)
    assert exp.optimize_router is optimize_router
    assert exp.fit_router is fit_router
    assert exp.report_router is report_router
    assert exp.RouterRuntime is RouterRuntime
    assert exp.compose_router is compose_router
    assert exp.run_hosted_router_workflow is run_hosted_router_workflow
    assert exp.HostedRouterWorkflowSetup is HostedRouterWorkflowSetup
    assert exp.HostedRouterWorkflowOptions is HostedRouterWorkflowOptions
    assert exp.HostedRouterWorkflowResult is HostedRouterWorkflowResult
    assert exp.restore_hosted_project_bundle is restore_hosted_project_bundle
    assert exp.HostedAttemptAuthority is HostedAttemptAuthority
    assert exp.HostedAttemptAuthorityStore is HostedAttemptAuthorityStore
    assert exp.HostedAttemptState is HostedAttemptState
    assert exp.HostedProviderHazard is HostedProviderHazard
    assert exp.HostedStageCommit is HostedStageCommit
    assert exp.FileHostedAttemptAuthorityStore is FileHostedAttemptAuthorityStore
    assert exp.create_hosted_attempt_authority is create_hosted_attempt_authority
    assert exp.ProjectBudgetConfiguration is ProjectBudgetConfiguration
    assert exp.ProjectModelConfiguration is ProjectModelConfiguration
    assert exp.ProjectRetrievalConfiguration is ProjectRetrievalConfiguration
    assert exp.ProjectSystemConfiguration is ProjectSystemConfiguration
    assert exp.ProjectStage is ProjectStage
    assert exp.ProjectStageEvent is ProjectStageEvent
    assert exp.ProjectStore is ProjectStore
    assert exp.BillingSource is BillingSource
    assert exp.ConnectionConfig is ConnectionConfig
    assert exp.DiscoveredModel is DiscoveredModel
    assert exp.ModelCapabilities is ModelCapabilities
    assert exp.ModelCatalog is ModelCatalog
    assert exp.ModelRecord is ModelRecord
    assert exp.ModelRoles is ModelRoles
    assert exp.ResolvedDiscoveredModel is ResolvedDiscoveredModel
    assert exp.resolve_discovered_model is resolve_discovered_model
    assert exp.RuntimeModelCatalog is RuntimeModelCatalog
    assert "ledger" in signature(FileHostedAttemptAuthorityStore.commit_stage).parameters
    assert exp.ProviderSpendLedger is ProviderSpendLedger
    assert exp.ProviderSpendEntry is ProviderSpendEntry
    assert exp.BillingSourceEconomics is BillingSourceEconomics
    assert exp.RoutedCompletionEconomics is RoutedCompletionEconomics
    assert exp.RoutedProviderComponent is RoutedProviderComponent
    assert exp.RoutedProviderOperation is RoutedProviderOperation
    assert exp.RoutedSpendDisposition is RoutedSpendDisposition
    assert exp.RoutedSpendLedger is RoutedSpendLedger
    assert exp.RoutedModelResponse is RoutedModelResponse
    assert ModelRecord.model_fields["billing_source"].is_required()
    assert ModelSnapshot.model_fields["billing_source"].is_required()
    assert ProviderSpendEntry.model_fields["billing_source"].is_required()
    assert set(RoutedCompletionEconomics.model_fields) == {
        "operations",
        "operation_count",
        "router_embedding",
        "selected_candidate",
        "by_billing_source",
        "total",
    }
    assert exp.FidelityReport is FidelityReport
    assert exp.build_fidelity_evaluation_plan is build_fidelity_evaluation_plan
    assert exp.build_fidelity_report is build_fidelity_report
    assert automatic_router.compose_router is compose_router
    assert router_cli.optimize_project_router is automatic_router.optimize_project_router
    assert exp.load_project_router is load_project_router
    assert exp.load_router is load_router
    assert "ghost" in signature(exp.load_router).parameters
    assert "create_project_router_app" not in exp.__all__
    assert exp.load_world_model is load_world_model
    assert exp.WorldModel is WorldModel
    assert exp.WorldModelLoadError is WorldModelLoadError
    assert exp.WorldModelSession is WorldModelSession
    assert exp.WorldModelSessionError is WorldModelSessionError
    assert exp.WorldModelSessionLimits is WorldModelSessionLimits
    assert exp.WorldModelObservation is WorldModelObservation
    assert "FidelityApproval" not in exp.__all__
    assert "FidelityApprovalDecision" not in exp.__all__
    assert "FidelityGate" not in evaluations.__all__
    assert "FidelityThresholds" not in evaluations.__all__
    assert "default_fidelity_thresholds" not in evaluations.__all__
    assert "persist_fidelity_thresholds" not in evaluations.__all__
    assert "fidelity_thresholds_id" not in signature(build_fidelity_evaluation_plan).parameters
    assert "overlap_count" in signature(build_fidelity_evaluation_plan).parameters
    assert not {"status", "approved_at", "gate_id", "gate_sha256"}.intersection(
        FidelityReport.model_fields
    )
    assert "ActionKind" not in exp.__all__


def test_runtime_router_import_isolated_from_simulation_and_offline_optimizer() -> None:
    """A fresh runtime import must not initialize simulation or optimizer owners."""
    code = """
import sys
import exp.runtime.router

offline = sorted(name for name in sys.modules if name.startswith("exp.optimize"))
gepa = sorted(name for name in sys.modules if name == "gepa" or name.startswith("gepa."))
simulation_model = sorted(
    name for name in sys.modules if name.startswith("exp.simulation.model")
)
assert not offline, offline
assert not gepa, gepa
assert not simulation_model, simulation_model
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
