"""Tests for the public package API surface."""

from __future__ import annotations

import subprocess
import sys
from inspect import signature

import wmo
import wmo.cli.router_app as router_cli
import wmo.common.evaluations as evaluations
from wmo.common.evaluations import (
    FidelityReport,
    build_fidelity_evaluation_plan,
    build_fidelity_report,
)
from wmo.common.models import (
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
from wmo.common.project import (
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
from wmo.optimize.router.activation import load_project_router, load_router
from wmo.optimize.router.attempt_authority import (
    FileHostedAttemptAuthorityStore,
    HostedAttemptAuthority,
    HostedAttemptAuthorityStore,
    HostedAttemptState,
    HostedProviderHazard,
    HostedStageCommit,
    create_hosted_attempt_authority,
)
from wmo.optimize.router.automatic import service as automatic_router
from wmo.optimize.router.composition import compose_router
from wmo.optimize.router.fit.workflow import fit_router, optimize_router, report_router
from wmo.optimize.router.hosted import (
    HostedRouterWorkflowOptions,
    HostedRouterWorkflowResult,
    HostedRouterWorkflowSetup,
    restore_hosted_project_bundle,
    run_hosted_router_workflow,
)
from wmo.optimize.router.spend import ProviderSpendEntry, ProviderSpendLedger
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.application import (
    create_project_router_app,
)
from wmo.runtime.router.economics import (
    BillingSourceEconomics,
    RoutedCompletionEconomics,
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
    RoutedSpendLedger,
)
from wmo.runtime.router.runtime import (
    RoutedModelResponse,
    RouterRuntime,
)
from wmo.simulation.build import (
    build_project,
    load_project_provider_free_stage,
    prepare_project_traces,
)
from wmo.simulation.world_model.application import (
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
    assert wmo.build_project is build_project
    assert wmo.prepare_project_traces is prepare_project_traces
    assert wmo.load_project_provider_free_stage is load_project_provider_free_stage
    assert wmo.ProjectProviderFreeStage is ProjectProviderFreeStage
    assert wmo.ProjectTracePreparationSettings is ProjectTracePreparationSettings
    assert wmo.export_project_bundle is export_project_bundle
    assert wmo.restore_project_bundle is restore_project_bundle
    assert wmo.ExportedProjectBundle is ExportedProjectBundle
    assert {
        "export_project_bundle",
        "restore_project_bundle",
        "ExportedProjectBundle",
    }.issubset(wmo.__all__)
    assert not {
        "ProjectBundleManifest",
        "ProjectBundleMember",
        "ProjectBundleError",
        "ProjectModelCatalog",
    }.intersection(wmo.__all__)
    assert wmo.optimize_router is optimize_router
    assert wmo.fit_router is fit_router
    assert wmo.report_router is report_router
    assert wmo.RouterRuntime is RouterRuntime
    assert wmo.compose_router is compose_router
    assert wmo.run_hosted_router_workflow is run_hosted_router_workflow
    assert wmo.HostedRouterWorkflowSetup is HostedRouterWorkflowSetup
    assert wmo.HostedRouterWorkflowOptions is HostedRouterWorkflowOptions
    assert wmo.HostedRouterWorkflowResult is HostedRouterWorkflowResult
    assert wmo.restore_hosted_project_bundle is restore_hosted_project_bundle
    assert wmo.HostedAttemptAuthority is HostedAttemptAuthority
    assert wmo.HostedAttemptAuthorityStore is HostedAttemptAuthorityStore
    assert wmo.HostedAttemptState is HostedAttemptState
    assert wmo.HostedProviderHazard is HostedProviderHazard
    assert wmo.HostedStageCommit is HostedStageCommit
    assert wmo.FileHostedAttemptAuthorityStore is FileHostedAttemptAuthorityStore
    assert wmo.create_hosted_attempt_authority is create_hosted_attempt_authority
    assert wmo.ProjectBudgetConfiguration is ProjectBudgetConfiguration
    assert wmo.ProjectModelConfiguration is ProjectModelConfiguration
    assert wmo.ProjectRetrievalConfiguration is ProjectRetrievalConfiguration
    assert wmo.ProjectSystemConfiguration is ProjectSystemConfiguration
    assert wmo.ProjectStage is ProjectStage
    assert wmo.ProjectStageEvent is ProjectStageEvent
    assert wmo.ProjectStore is ProjectStore
    assert wmo.BillingSource is BillingSource
    assert wmo.ConnectionConfig is ConnectionConfig
    assert wmo.DiscoveredModel is DiscoveredModel
    assert wmo.ModelCapabilities is ModelCapabilities
    assert wmo.ModelCatalog is ModelCatalog
    assert wmo.ModelRecord is ModelRecord
    assert wmo.ModelRoles is ModelRoles
    assert wmo.ResolvedDiscoveredModel is ResolvedDiscoveredModel
    assert wmo.resolve_discovered_model is resolve_discovered_model
    assert wmo.RuntimeModelCatalog is RuntimeModelCatalog
    assert "ledger" in signature(FileHostedAttemptAuthorityStore.commit_stage).parameters
    assert wmo.ProviderSpendLedger is ProviderSpendLedger
    assert wmo.ProviderSpendEntry is ProviderSpendEntry
    assert wmo.BillingSourceEconomics is BillingSourceEconomics
    assert wmo.RoutedCompletionEconomics is RoutedCompletionEconomics
    assert wmo.RoutedProviderComponent is RoutedProviderComponent
    assert wmo.RoutedProviderOperation is RoutedProviderOperation
    assert wmo.RoutedSpendDisposition is RoutedSpendDisposition
    assert wmo.RoutedSpendLedger is RoutedSpendLedger
    assert wmo.RoutedModelResponse is RoutedModelResponse
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
    assert wmo.FidelityReport is FidelityReport
    assert wmo.build_fidelity_evaluation_plan is build_fidelity_evaluation_plan
    assert wmo.build_fidelity_report is build_fidelity_report
    assert automatic_router.compose_router is compose_router
    assert router_cli.optimize_project_router is automatic_router.optimize_project_router
    assert wmo.load_project_router is load_project_router
    assert wmo.load_router is load_router
    assert "ghost" in signature(wmo.load_router).parameters
    assert wmo.create_project_router_app is create_project_router_app
    assert wmo.load_world_model is load_world_model
    assert wmo.WorldModel is WorldModel
    assert wmo.WorldModelLoadError is WorldModelLoadError
    assert wmo.WorldModelSession is WorldModelSession
    assert wmo.WorldModelSessionError is WorldModelSessionError
    assert wmo.WorldModelSessionLimits is WorldModelSessionLimits
    assert wmo.WorldModelObservation is WorldModelObservation
    assert "FidelityApproval" not in wmo.__all__
    assert "FidelityApprovalDecision" not in wmo.__all__
    assert "FidelityGate" not in evaluations.__all__
    assert "FidelityThresholds" not in evaluations.__all__
    assert "default_fidelity_thresholds" not in evaluations.__all__
    assert "persist_fidelity_thresholds" not in evaluations.__all__
    assert "fidelity_thresholds_id" not in signature(build_fidelity_evaluation_plan).parameters
    assert "overlap_count" in signature(build_fidelity_evaluation_plan).parameters
    assert not {"status", "approved_at", "gate_id", "gate_sha256"}.intersection(
        FidelityReport.model_fields
    )
    assert "ActionKind" not in wmo.__all__


def test_runtime_router_import_isolated_from_simulation_and_offline_optimizer() -> None:
    """A fresh runtime import must not initialize simulation or optimizer owners."""
    code = """
import sys
import wmo.runtime.router

offline = sorted(name for name in sys.modules if name.startswith("wmo.optimize"))
gepa = sorted(name for name in sys.modules if name == "gepa" or name.startswith("gepa."))
simulation_model = sorted(
    name for name in sys.modules if name.startswith("wmo.simulation.model")
)
assert not offline, offline
assert not gepa, gepa
assert not simulation_model, simulation_model
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
