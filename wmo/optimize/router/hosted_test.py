"""Hosted Project workflow acceptance tests."""

from __future__ import annotations

import math
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    canonical_json_bytes,
    sorted_unique_inputs,
)
from wmo.common.core.money import USD_ZERO
from wmo.common.evaluations import EvaluationDatasetManifest, fidelity
from wmo.common.judging import JudgeCalibration
from wmo.common.models import ModelCatalog, ModelClient, ModelRequest, ModelResponse
from wmo.common.project import (
    ExportedProjectBundle,
    ProjectBudgetConfiguration,
    ProjectBuildArtifacts,
    ProjectBundleError,
    ProjectConfig,
    ProjectHostedJudgeEvidence,
    ProjectModelConfiguration,
    ProjectProviderFreeStage,
    ProjectRetrievalConfiguration,
    ProjectRouterPolicyArtifacts,
    ProjectRouterReportArtifacts,
    ProjectStage,
    ProjectStageEventKind,
    ProjectStore,
    ProjectSystemConfiguration,
    ProjectTracePreparationSettings,
    artifact_input,
    export_project_bundle,
    write_project_config,
)
from wmo.common.project import (
    restore_project_bundle as restore_prepared_project_bundle,
)
from wmo.common.routing import KnnRouterPolicy
from wmo.common.routing.bank import KnnBankManifest
from wmo.optimize.router.attempt_authority import (
    FileHostedAttemptAuthorityStore,
    HostedAttemptAuthorityError,
    HostedProviderHazard,
    HostedStageCommit,
)
from wmo.optimize.router.automatic.service_test import (
    _REVISION,
    _TIME,
    _catalog,
    _ProviderState,
    _RuntimeCatalog,
    _trace,
)
from wmo.optimize.router.composition import RouterPolicyLock
from wmo.optimize.router.fit.report import HeldOutRouterReport
from wmo.optimize.router.hosted import (
    HostedRouterPreflightError,
    HostedRouterWorkflowError,
    HostedRouterWorkflowOptions,
    HostedRouterWorkflowSetup,
    _automatic_options,
    run_hosted_router_workflow,
)
from wmo.optimize.router.hosted import (
    restore_hosted_project_bundle as restore_project_bundle,
)
from wmo.optimize.router.hosted_spend import complete_component_entries
from wmo.optimize.router.spend import (
    ProviderSpendComponent,
    ProviderSpendEntry,
    ProviderSpendLedger,
    ProviderSpendStatus,
    load_provider_spend_ledger,
    persist_provider_spend_ledger,
)
from wmo.runtime.models import ResolvedModel, RuntimeModelCatalog
from wmo.simulation.build import build_project
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.mining.service import MiningSpec
from wmo.simulation.retrieval import (
    RAGEmbedderBinding,
    load_completed_build_rag_lineage_bindings,
    persist_trace_rag,
)
from wmo.simulation.world_model import persist_grounded_world_model


def test_hosted_workflow_runs_from_restored_bundle_and_replays_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build, optimize, report, export, restore, and replay without fidelity or repeat calls."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    state = _ProviderState()
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / "attempt-authority")
    authority = attempt_store.create()
    fidelity_calls = 0

    def reject_fidelity(*_args: object, **_kwargs: object) -> None:
        nonlocal fidelity_calls
        fidelity_calls += 1
        raise AssertionError("hosted router fitting touched fidelity")

    monkeypatch.setattr(fidelity, "build_fidelity_report", reject_fidelity)
    bundle_directory = tmp_path / "completed-bundles"
    bundle_directory.mkdir()
    result = run_hosted_router_workflow(
        prepared,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        attempt_store,
        bundle_directory=bundle_directory,
        attempt_id=authority.attempt_id,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
        options=_options(),
    )

    assert result.policy_id
    assert result.report_id
    assert fidelity_calls == 0
    assert [item.stage for item in result.bundles] == [
        ProjectStage.BUILDING_WORLD_MODEL,
        ProjectStage.OPTIMIZING_ROUTER,
        ProjectStage.COMPLETING_REPORT,
    ]
    assert [
        (event.stage, event.kind)
        for event in result.events
        if event.kind == ProjectStageEventKind.COMPLETED
    ] == [
        (ProjectStage.BUILDING_WORLD_MODEL, ProjectStageEventKind.COMPLETED),
        (ProjectStage.OPTIMIZING_ROUTER, ProjectStageEventKind.COMPLETED),
        (ProjectStage.COMPLETING_REPORT, ProjectStageEventKind.COMPLETED),
    ]
    policy_completed = next(
        event.sequence
        for event in result.events
        if event.stage == ProjectStage.OPTIMIZING_ROUTER
        and event.kind == ProjectStageEventKind.COMPLETED
    )
    report_started = next(
        event.sequence
        for event in result.events
        if event.stage == ProjectStage.COMPLETING_REPORT
        and event.kind == ProjectStageEventKind.STARTED
    )
    assert policy_completed < report_started
    config = prepared.load_project()
    assert config.build is not None
    assert config.hosted_judge is not None
    assert config.hosted_judge.status == "provisional"
    assert result.automatic.preflight.judgment_status == "provisional"
    assert config.router_policy is not None
    assert config.router_report is not None
    assert result.spend_ledger.total_usd == sum(
        (entry.amount_usd for entry in result.spend_ledger.entries),
        start=USD_ZERO,
    )
    assert result.spend_ledger.total_usd <= result.spend_ledger.ceiling_usd
    assert {entry.component for entry in result.spend_ledger.entries} == set(ProviderSpendComponent)
    judge_statuses = [
        entry.status
        for entry in result.spend_ledger.entries
        if entry.component == ProviderSpendComponent.JUDGE
    ]
    assert ProviderSpendStatus.LOCALLY_PRICED in judge_statuses
    assert any(
        entry.component == ProviderSpendComponent.ROUTER_EMBEDDING
        and entry.status == ProviderSpendStatus.RESERVED
        for entry in result.spend_ledger.entries
    )
    for index, stage_bundle in enumerate(result.bundles):
        payload = stage_bundle.bundle.path.read_bytes()
        assert b"FIXTURE_API_KEY" not in payload
        assert str(tmp_path).encode() not in payload
        restored = restore_project_bundle(
            stage_bundle.bundle.path,
            root=tmp_path / f"verified-stage-{index}",
            expected_sha256=stage_bundle.bundle.sha256,
        )
        assert restored.load_project().provider_free_stage is not None

    final_bundle = result.bundles[-1].bundle
    terminal_state = attempt_store.state(authority)
    assert terminal_state.terminal is True
    assert terminal_state.latest_commit is not None
    assert terminal_state.latest_commit.bundle_sha256 == final_bundle.sha256
    assert config.build_spend_ledger is not None
    build_spend = load_provider_spend_ledger(
        prepared.artifacts,
        config.build_spend_ledger,
    )
    stale_stage_store = FileHostedAttemptAuthorityStore(tmp_path / "stale-stage-authority")
    stale_stage_authority = stale_stage_store.create()
    stale_stage_store.bind(
        stale_stage_authority,
        project_id=prepared.paths.project_id,
        ceiling_usd=Decimal("25.000000"),
    )
    stale_stage_store.begin(
        HostedProviderHazard(
            project_id=prepared.paths.project_id,
            attempt_id=stale_stage_authority.attempt_id,
            authority_sha256=stale_stage_authority.authority_sha256,
            stage=ProjectStage.BUILDING_WORLD_MODEL,
            component=ProviderSpendComponent.RETRIEVAL_EMBEDDING,
            reserved_usd=Decimal("1.000000"),
        )
    )
    with pytest.raises(HostedAttemptAuthorityError, match="verified Project bundle"):
        stale_stage_store.commit_stage(
            HostedStageCommit(
                project_id=prepared.paths.project_id,
                attempt_id=stale_stage_authority.attempt_id,
                authority_sha256=stale_stage_authority.authority_sha256,
                stage=ProjectStage.BUILDING_WORLD_MODEL,
                bundle_sha256=final_bundle.sha256,
                bundle_size_bytes=final_bundle.size_bytes,
                spend_ledger=config.build_spend_ledger,
                spend_total_usd=build_spend.total_usd,
            ),
            final_bundle,
            build_spend,
        )
    assert stale_stage_store.unresolved(stale_stage_authority) is not None
    replay_store = restore_project_bundle(
        final_bundle.path,
        root=tmp_path / "final-replay-root",
        expected_sha256=final_bundle.sha256,
    )
    replay_state = _ProviderState()
    mismatched_state = _ProviderState()
    with pytest.raises(HostedRouterPreflightError, match="stage pointer"):
        run_hosted_router_workflow(
            replay_store,
            _setup(),
            catalog,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, mismatched_state)),
            FileHostedAttemptAuthorityStore(tmp_path / "attempt-authority"),
            bundle_directory=tmp_path / "mismatched-replay-bundles",
            attempt_id=authority.attempt_id,
            created_at=_TIME + timedelta(hours=1),
            code_revision=_REVISION,
            resume_bundle_sha256="0" * 64,
            options=_options(),
        )
    assert mismatched_state.credential_resolutions == 0
    replay = run_hosted_router_workflow(
        replay_store,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, replay_state)),
        FileHostedAttemptAuthorityStore(tmp_path / "attempt-authority"),
        bundle_directory=tmp_path / "replay-bundles",
        attempt_id=authority.attempt_id,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
        resume_bundle_sha256=final_bundle.sha256,
        options=_options(),
    )

    assert replay.policy_id == result.policy_id
    assert replay.report_id == result.report_id
    assert replay.bundles == ()
    assert replay_state.embedding_calls == []
    assert replay_state.completion_calls == []
    assert attempt_store.unresolved(authority) is None


@pytest.mark.parametrize("failed_alias", ["judge", "world"])
def test_hosted_optimization_failure_reserves_remaining_ceiling_and_blocks_local_retry(
    tmp_path: Path,
    failed_alias: str,
) -> None:
    """Judge and simulation ambiguity expose safe bundles and never reset unknown spend."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    state = _ProviderState()
    runtime = _FailingRuntimeCatalog(catalog, state, failed_alias=failed_alias)
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / f"attempt-{failed_alias}-authority")
    authority = attempt_store.create()
    bundle_directory = tmp_path / "failed-bundles"
    bundle_directory.mkdir()

    with pytest.raises(HostedRouterWorkflowError) as captured:
        run_hosted_router_workflow(
            prepared,
            _setup(),
            catalog,
            cast(RuntimeModelCatalog, runtime),
            attempt_store,
            bundle_directory=bundle_directory,
            attempt_id=authority.attempt_id,
            created_at=_TIME + timedelta(hours=2),
            code_revision=_REVISION,
            options=_options(),
        )

    error = captured.value
    assert str(error) == "hosted router workflow failed closed after a provider reservation"
    assert error.__cause__ is None
    assert error.ledger.outcome == "failed_closed"
    assert math.isclose(
        error.ledger.total_usd,
        error.ledger.ceiling_usd,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert [item.stage for item in error.bundles] == [ProjectStage.BUILDING_WORLD_MODEL]
    assert any(entry.status == ProviderSpendStatus.RESERVED for entry in error.ledger.entries)
    credentials_before_retry = state.credential_resolutions
    with pytest.raises(HostedRouterWorkflowError) as blocked:
        run_hosted_router_workflow(
            prepared,
            _setup(),
            catalog,
            cast(RuntimeModelCatalog, runtime),
            attempt_store,
            bundle_directory=bundle_directory,
            attempt_id=authority.attempt_id,
            created_at=_TIME + timedelta(hours=2),
            code_revision=_REVISION,
            options=_options(),
        )
    assert blocked.value.ledger.outcome == "failed_closed"
    assert state.credential_resolutions == credentials_before_retry

    build_bundle = error.bundles[0].bundle
    restored = restore_project_bundle(
        build_bundle.path,
        root=tmp_path / f"restored-{failed_alias}-build",
        expected_sha256=build_bundle.sha256,
    )
    restored_state = _ProviderState()
    with pytest.raises(HostedRouterWorkflowError) as restored_blocked:
        run_hosted_router_workflow(
            restored,
            _setup(),
            catalog,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, restored_state)),
            FileHostedAttemptAuthorityStore(tmp_path / f"attempt-{failed_alias}-authority"),
            bundle_directory=tmp_path / f"recovered-{failed_alias}-bundles",
            attempt_id=authority.attempt_id,
            created_at=_TIME + timedelta(hours=2),
            code_revision=_REVISION,
            options=_options(),
        )
    assert restored_blocked.value.ledger.outcome == "failed_closed"
    assert restored_state.credential_resolutions == 0
    assert restored_state.embedding_calls == []
    assert restored_state.completion_calls == []

    alternate_store = FileHostedAttemptAuthorityStore(
        tmp_path / f"alternate-{failed_alias}-authority"
    )
    alternate_store.create()
    alternate_state = _ProviderState()
    with pytest.raises(HostedRouterPreflightError, match="authority"):
        run_hosted_router_workflow(
            restored,
            _setup(),
            catalog,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, alternate_state)),
            alternate_store,
            bundle_directory=tmp_path / f"alternate-{failed_alias}-bundles",
            attempt_id=authority.attempt_id,
            created_at=_TIME + timedelta(hours=2),
            code_revision=_REVISION,
            options=_options(),
        )
    assert alternate_state.credential_resolutions == 0
    assert alternate_state.embedding_calls == []
    assert alternate_state.completion_calls == []


def test_hosted_preflight_rejects_missing_connection_before_project_write_or_dispatch(
    tmp_path: Path,
) -> None:
    """Resolve every credential-backed role before late setup or provider dispatch."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / "missing-connection-authority")
    authority = attempt_store.create()
    before_config = prepared.load_project()
    before_artifacts = prepared.artifacts.list_ids()

    with pytest.raises(HostedRouterPreflightError, match="transient provider clients"):
        run_hosted_router_workflow(
            prepared,
            _setup(),
            catalog,
            RuntimeModelCatalog(catalog, environment={}),
            attempt_store,
            bundle_directory=tmp_path / "unused-bundles",
            attempt_id=authority.attempt_id,
            created_at=_TIME + timedelta(hours=3),
            code_revision=_REVISION,
            options=_options(),
        )

    assert prepared.load_project() == before_config
    assert prepared.artifacts.list_ids() == before_artifacts


def test_hosted_preflight_reserves_full_simulation_before_build_dispatch(tmp_path: Path) -> None:
    """Reject a ceiling below full candidate/world execution with zero provider calls."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    state = _ProviderState()
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / "low-budget-authority")
    authority = attempt_store.create()
    before_config = prepared.load_project()
    before_artifacts = prepared.artifacts.list_ids()
    setup = _setup().model_copy(
        update={
            "budgets": ProjectBudgetConfiguration(
                maximum_build_cost_usd=Decimal("5.000000"),
                maximum_provider_cost_usd=Decimal("10.000000"),
            )
        }
    )

    with pytest.raises(HostedRouterPreflightError, match="full build"):
        run_hosted_router_workflow(
            prepared,
            setup,
            catalog,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
            attempt_store,
            bundle_directory=tmp_path / "unused-budget-bundles",
            attempt_id=authority.attempt_id,
            created_at=_TIME + timedelta(hours=4),
            code_revision=_REVISION,
            options=_options(),
        )

    assert state.credential_resolutions == 0
    assert state.embedding_calls == []
    assert state.completion_calls == []
    assert prepared.load_project() == before_config
    assert prepared.artifacts.list_ids() == before_artifacts


def test_hosted_automatic_ceiling_never_widens_a_large_one_microunit_boundary() -> None:
    """The legacy automatic float seam never rounds an exact hosted ceiling upward."""
    exact = Decimal("99999999999998.999999")
    assert Decimal.from_float(float(exact)) > exact

    automatic = _automatic_options(_setup(), _options(), exact)

    assert Decimal.from_float(automatic.maximum_provider_cost_usd) <= exact


def test_builtin_chat_system_contract_matches_platform_shape_and_bounds() -> None:
    """The shared setup shape normalizes prompt text and fixes model-call bounds."""
    system = ProjectSystemConfiguration(system_prompt="  Follow policy.  ")

    assert system.model_dump(mode="json") == {
        "kind": "builtin_chat",
        "system_prompt": "Follow policy.",
        "maximum_model_calls": 8,
    }
    with pytest.raises(ValueError, match="blank"):
        ProjectSystemConfiguration(system_prompt="   ")
    with pytest.raises(ValueError):
        ProjectSystemConfiguration(system_prompt="x" * 20_001)
    with pytest.raises(ValueError):
        ProjectSystemConfiguration(system_prompt="valid", maximum_model_calls=65)
    with pytest.raises(ValueError):
        ProjectSystemConfiguration.model_validate({"system_prompt": "valid", "unsupported": True})


def test_external_commit_failure_emits_no_completion_and_blocks_new_process(
    tmp_path: Path,
) -> None:
    """A crash-window bundle cannot clear spend authority before external pointer CAS."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    state = _ProviderState()
    authority_directory = tmp_path / "commit-failure-authority"
    attempt_store = _FailingCommitAuthorityStore(authority_directory)
    authority = attempt_store.create()
    emitted = []

    with pytest.raises(HostedRouterWorkflowError) as captured:
        run_hosted_router_workflow(
            prepared,
            _setup(),
            catalog,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
            attempt_store,
            bundle_directory=tmp_path / "uncommitted-bundles",
            attempt_id=authority.attempt_id,
            created_at=_TIME + timedelta(hours=5),
            code_revision=_REVISION,
            options=_options(),
            event_sink=emitted.append,
        )

    error = captured.value
    assert state.embedding_calls
    assert [item.stage for item in error.bundles] == [ProjectStage.BUILDING_WORLD_MODEL]
    assert all(event.kind != ProjectStageEventKind.COMPLETED for event in emitted)
    assert emitted[-1].kind == ProjectStageEventKind.FAILED
    local_bundle = error.bundles[0].bundle
    restored = restore_project_bundle(
        local_bundle.path,
        root=tmp_path / "uncommitted-restore",
        expected_sha256=local_bundle.sha256,
    )
    restarted_state = _ProviderState()

    with pytest.raises(HostedRouterWorkflowError):
        run_hosted_router_workflow(
            restored,
            _setup(),
            catalog,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, restarted_state)),
            FileHostedAttemptAuthorityStore(authority_directory),
            bundle_directory=tmp_path / "blocked-restart-bundles",
            attempt_id=authority.attempt_id,
            created_at=_TIME + timedelta(hours=5),
            code_revision=_REVISION,
            resume_bundle_sha256=local_bundle.sha256,
            options=_options(),
        )

    assert restarted_state.credential_resolutions == 0
    assert restarted_state.embedding_calls == []
    assert restarted_state.completion_calls == []


@pytest.mark.parametrize(
    ("selection_name", "message"),
    [
        ("hosted_judge", "provisional judge setup differs"),
        ("router_policy", "router policy differs"),
        ("router_report", "held-out report differs"),
    ],
)
def test_bundle_restore_rejects_same_project_semantic_pointer_swaps(
    tmp_path: Path,
    selection_name: str,
    message: str,
) -> None:
    """Authenticated same-type pointers cannot cross judge, policy, or report graphs."""
    primary_root = tmp_path / "primary"
    alternate_root = tmp_path / "alternate"
    primary_root.mkdir()
    alternate_root.mkdir()
    primary, catalog = _restored_prepared_project(primary_root)
    alternate, alternate_catalog = _restored_prepared_project(
        alternate_root,
        trace_offset=100,
    )
    primary_authority = FileHostedAttemptAuthorityStore(primary_root / "authority")
    alternate_authority = FileHostedAttemptAuthorityStore(alternate_root / "authority")
    primary_id = primary_authority.create().attempt_id
    alternate_id = alternate_authority.create().attempt_id
    run_hosted_router_workflow(
        primary,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, _ProviderState())),
        primary_authority,
        bundle_directory=primary_root / "bundles",
        attempt_id=primary_id,
        created_at=_TIME + timedelta(hours=6),
        code_revision=_REVISION,
        options=_options(),
    )
    run_hosted_router_workflow(
        alternate,
        _setup(),
        alternate_catalog,
        cast(
            RuntimeModelCatalog,
            _RuntimeCatalog(alternate_catalog, _ProviderState()),
        ),
        alternate_authority,
        bundle_directory=alternate_root / "bundles",
        attempt_id=alternate_id,
        created_at=_TIME + timedelta(hours=6),
        code_revision=_REVISION,
        options=_options(),
    )
    _copy_artifacts(alternate, primary)
    current = primary.load_project()
    other = alternate.load_project()
    update = {selection_name: getattr(other, selection_name)}
    if selection_name == "hosted_judge":
        update.update(router_policy=None, router_report=None)
    elif selection_name == "router_policy":
        update.update(router_report=None)
    malicious = ProjectConfig.model_validate({**current.model_dump(mode="python"), **update})
    write_project_config(primary.paths.project_toml, malicious)
    bundle = export_project_bundle(
        primary,
        tmp_path / f"swapped-{selection_name}.wmo.zip",
        producer_revision=_REVISION,
    )

    with pytest.raises(ProjectBundleError, match=message):
        restore_project_bundle(
            bundle.path,
            root=tmp_path / f"rejected-{selection_name}",
            expected_sha256=bundle.sha256,
        )


def test_bundle_restore_rejects_calibration_with_forged_semantic_identity(
    tmp_path: Path,
) -> None:
    """Canonical calibration verification rejects an authenticated wrong payload identity."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / "authority")
    authority = attempt_store.create()
    run_hosted_router_workflow(
        prepared,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, _ProviderState())),
        attempt_store,
        bundle_directory=tmp_path / "bundles",
        attempt_id=authority.attempt_id,
        created_at=_TIME + timedelta(hours=7),
        code_revision=_REVISION,
        options=_options(),
    )
    current = prepared.load_project()
    assert current.hosted_judge is not None
    original = current.hosted_judge.calibration
    calibration = JudgeCalibration.model_validate_json(
        prepared.artifacts.read_bytes(original.artifact_id, "calibration.json")
    )
    forged = calibration.model_copy(update={"calibration_id": "judge-calibration-forged-identity"})
    manifest = prepared.artifacts.write_or_verify_exact(
        artifact_id="judge-calibration-forged-pointer",
        artifact_type="judge-calibration",
        envelope=ArtifactEnvelope(
            schema_version=forged.schema_version,
            created_at=forged.created_at,
            inputs=forged.inputs,
            code_revision=forged.code_revision,
            source=forged.source,
        ),
        files={"calibration.json": canonical_json_bytes(forged)},
    )
    malicious = ProjectConfig.model_validate(
        {
            **current.model_dump(mode="python"),
            "hosted_judge": ProjectHostedJudgeEvidence(
                setup=current.hosted_judge.setup,
                calibration=artifact_input(manifest),
            ),
            "router_policy": None,
            "router_report": None,
        }
    )
    write_project_config(prepared.paths.project_toml, malicious)
    bundle = export_project_bundle(
        prepared,
        tmp_path / "forged-calibration.wmo.zip",
        producer_revision=_REVISION,
    )

    with pytest.raises(ProjectBundleError, match="calibration"):
        restore_project_bundle(
            bundle.path,
            root=tmp_path / "rejected-forged-calibration",
            expected_sha256=bundle.sha256,
        )


@pytest.mark.parametrize("variant", ["system_prompt", "world_model"])
def test_bundle_restore_rejects_policy_from_another_execution_contract(
    tmp_path: Path,
    variant: str,
) -> None:
    """A policy cannot move across system or world-model setup with a forged valid ledger."""
    primary_root = tmp_path / "primary"
    alternate_root = tmp_path / "alternate"
    primary_root.mkdir()
    alternate_root.mkdir()
    primary, catalog = _restored_prepared_project(primary_root)
    alternate, alternate_catalog = _restored_prepared_project(alternate_root)
    primary_setup = _setup()
    alternate_setup = _setup(
        system_prompt=(
            "Use the alternate support policy."
            if variant == "system_prompt"
            else "Resolve the support request accurately and safely."
        )
    )
    if variant == "world_model":
        alternate_catalog = _catalog_with_alternate_world(alternate_catalog)
    primary_authority = FileHostedAttemptAuthorityStore(primary_root / "authority")
    alternate_authority = FileHostedAttemptAuthorityStore(alternate_root / "authority")
    run_hosted_router_workflow(
        primary,
        primary_setup,
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, _ProviderState())),
        primary_authority,
        bundle_directory=primary_root / "bundles",
        attempt_id=primary_authority.create().attempt_id,
        created_at=_TIME + timedelta(hours=8),
        code_revision=_REVISION,
        options=_options(),
    )
    run_hosted_router_workflow(
        alternate,
        alternate_setup,
        alternate_catalog,
        cast(
            RuntimeModelCatalog,
            _RuntimeCatalog(alternate_catalog, _ProviderState()),
        ),
        alternate_authority,
        bundle_directory=alternate_root / "bundles",
        attempt_id=alternate_authority.create().attempt_id,
        created_at=_TIME + timedelta(hours=8),
        code_revision=_REVISION,
        options=_options(),
    )
    _copy_artifacts(alternate, primary)
    current = primary.load_project()
    other = alternate.load_project()
    assert current.router_policy is not None
    assert other.router_policy is not None
    replacement_ledger = _replacement_ledger(
        primary,
        source_pointer=current.router_policy.spend_ledger,
        stage=ProjectStage.OPTIMIZING_ROUTER,
        stage_outputs=(other.router_policy.policy_lock, other.router_policy.policy),
    )
    malicious = ProjectConfig.model_validate(
        {
            **current.model_dump(mode="python"),
            "router_policy": ProjectRouterPolicyArtifacts(
                policy_lock=other.router_policy.policy_lock,
                policy=other.router_policy.policy,
                spend_ledger=replacement_ledger,
            ),
            "router_report": None,
        }
    )
    write_project_config(primary.paths.project_toml, malicious)
    bundle = export_project_bundle(
        primary,
        tmp_path / f"swapped-execution-{variant}.wmo.zip",
        producer_revision=_REVISION,
    )

    with pytest.raises(ProjectBundleError, match="execution contract differs"):
        restore_project_bundle(
            bundle.path,
            root=tmp_path / f"rejected-execution-{variant}",
            expected_sha256=bundle.sha256,
        )


def test_bundle_restore_rejects_grounded_build_from_another_model_setup(
    tmp_path: Path,
) -> None:
    """A build cannot move across a different selected world-model snapshot."""
    primary_root = tmp_path / "primary"
    alternate_root = tmp_path / "alternate"
    primary_root.mkdir()
    alternate_root.mkdir()
    primary, catalog = _restored_prepared_project(primary_root)
    alternate, alternate_catalog = _restored_prepared_project(alternate_root)
    alternate_catalog = _catalog_with_alternate_world(alternate_catalog)
    primary_authority = FileHostedAttemptAuthorityStore(primary_root / "authority")
    alternate_authority = FileHostedAttemptAuthorityStore(alternate_root / "authority")
    run_hosted_router_workflow(
        primary,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, _ProviderState())),
        primary_authority,
        bundle_directory=primary_root / "bundles",
        attempt_id=primary_authority.create().attempt_id,
        created_at=_TIME + timedelta(hours=8),
        code_revision=_REVISION,
        options=_options(),
    )
    run_hosted_router_workflow(
        alternate,
        _setup(),
        alternate_catalog,
        cast(
            RuntimeModelCatalog,
            _RuntimeCatalog(alternate_catalog, _ProviderState()),
        ),
        alternate_authority,
        bundle_directory=alternate_root / "bundles",
        attempt_id=alternate_authority.create().attempt_id,
        created_at=_TIME + timedelta(hours=8),
        code_revision=_REVISION,
        options=_options(),
    )
    _copy_artifacts(alternate, primary)
    current = primary.load_project()
    other = alternate.load_project()
    assert current.build_spend_ledger is not None
    assert other.build is not None
    replacement_ledger = _replacement_ledger(
        primary,
        source_pointer=current.build_spend_ledger,
        stage=ProjectStage.BUILDING_WORLD_MODEL,
        stage_outputs=(
            other.build.trace_dataset,
            other.build.task_set,
            other.build.serving_rag,
            other.build.fit_rag,
            other.build.world_model,
        ),
    )
    malicious = ProjectConfig.model_validate(
        {
            **current.model_dump(mode="python"),
            "build": other.build,
            "build_spend_ledger": replacement_ledger,
            "hosted_judge": None,
            "router_policy": None,
            "router_report": None,
        }
    )
    write_project_config(primary.paths.project_toml, malicious)
    bundle = export_project_bundle(
        primary,
        tmp_path / "swapped-grounded-build.wmo.zip",
        producer_revision=_REVISION,
    )

    with pytest.raises(ProjectBundleError, match="grounded build"):
        restore_project_bundle(
            bundle.path,
            root=tmp_path / "rejected-grounded-build",
            expected_sha256=bundle.sha256,
        )


def test_bundle_restore_rejects_rag_with_an_alternate_task_partition(
    tmp_path: Path,
) -> None:
    """Canonical same-source RAGs cannot replace the selected task-set lineage split."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    state = _ProviderState()
    runtime = _RuntimeCatalog(catalog, state)
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / "authority")
    authority = attempt_store.create()
    run_hosted_router_workflow(
        prepared,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, runtime),
        attempt_store,
        bundle_directory=tmp_path / "bundles",
        attempt_id=authority.attempt_id,
        created_at=_TIME + timedelta(hours=9),
        code_revision=_REVISION,
        options=_options(),
    )
    current = prepared.load_project()
    assert current.build is not None
    assert current.build_spend_ledger is not None
    bindings = load_completed_build_rag_lineage_bindings(
        prepared.artifacts,
        current.build,
    )
    fit_trace = next(item.trace_id for item in bindings if item.partition == "fit")
    held_out_trace = next(item.trace_id for item in bindings if item.partition == "held_out")
    alternate_bindings = tuple(
        item.model_copy(
            update={
                "partition": (
                    "held_out"
                    if item.trace_id == fit_trace
                    else "fit"
                    if item.trace_id == held_out_trace
                    else item.partition
                )
            }
        )
        for item in bindings
    )
    resolved_embedder = runtime.resolve("embedder")
    assert resolved_embedder.embedding_client is not None
    embedder = RAGEmbedderBinding(
        client=resolved_embedder.embedding_client,
        snapshot=resolved_embedder.snapshot,
        maximum_attempts=3,
        input_usd_per_million_tokens=0.1,
    )
    serving = persist_trace_rag(
        prepared.artifacts,
        (current.build.trace_dataset,),
        alternate_bindings,
        created_at=_TIME + timedelta(hours=10),
        code_revision=_REVISION,
        embedder=embedder,
        default_top_k=2,
        included_partitions=frozenset({"fit", "held_out"}),
    )
    fit = persist_trace_rag(
        prepared.artifacts,
        (current.build.trace_dataset,),
        alternate_bindings,
        created_at=_TIME + timedelta(hours=10),
        code_revision=_REVISION,
        embedder=embedder,
        default_top_k=2,
        included_partitions=frozenset({"fit"}),
    )
    resolved_world = runtime.resolve("world")
    world = persist_grounded_world_model(
        prepared.artifacts,
        artifact_input(serving.manifest),
        model_alias=resolved_world.alias,
        model=resolved_world.snapshot,
        created_at=_TIME + timedelta(hours=10),
        code_revision=_REVISION,
        top_k=2,
    )
    alternate_build = ProjectBuildArtifacts(
        trace_dataset=current.build.trace_dataset,
        task_set=current.build.task_set,
        serving_rag=artifact_input(serving.manifest),
        fit_rag=artifact_input(fit.manifest),
        world_model=artifact_input(world.manifest),
    )
    replacement_ledger = _replacement_ledger(
        prepared,
        source_pointer=current.build_spend_ledger,
        stage=ProjectStage.BUILDING_WORLD_MODEL,
        stage_outputs=(
            alternate_build.trace_dataset,
            alternate_build.task_set,
            alternate_build.serving_rag,
            alternate_build.fit_rag,
            alternate_build.world_model,
        ),
    )
    malicious = ProjectConfig.model_validate(
        {
            **current.model_dump(mode="python"),
            "build": alternate_build,
            "build_spend_ledger": replacement_ledger,
            "hosted_judge": None,
            "router_policy": None,
            "router_report": None,
        }
    )
    write_project_config(prepared.paths.project_toml, malicious)
    bundle = export_project_bundle(
        prepared,
        tmp_path / "alternate-lineage-split.wmo.zip",
        producer_revision=_REVISION,
    )

    with pytest.raises(ProjectBundleError, match="RAG lineage differs"):
        restore_project_bundle(
            bundle.path,
            root=tmp_path / "rejected-alternate-lineage",
            expected_sha256=bundle.sha256,
        )


def test_bundle_restore_rejects_dropped_or_changed_prior_spend(
    tmp_path: Path,
) -> None:
    """Policy and report ledgers cannot omit or mutate earlier provider charges."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / "authority")
    authority = attempt_store.create()
    run_hosted_router_workflow(
        prepared,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, _ProviderState())),
        attempt_store,
        bundle_directory=tmp_path / "bundles",
        attempt_id=authority.attempt_id,
        created_at=_TIME + timedelta(hours=9),
        code_revision=_REVISION,
        options=_options(),
    )
    current = prepared.load_project()
    assert current.build_spend_ledger is not None
    assert current.router_policy is not None
    assert current.router_report is not None
    build_ledger = load_provider_spend_ledger(
        prepared.artifacts,
        current.build_spend_ledger,
    )
    policy_ledger = load_provider_spend_ledger(
        prepared.artifacts,
        current.router_policy.spend_ledger,
    )
    build_entry = next(
        item for item in build_ledger.entries if item.status != ProviderSpendStatus.NOT_INCURRED
    )
    prior_operation_id = build_entry.operation_id
    dropped_entries = complete_component_entries(
        tuple(item for item in policy_ledger.entries if item.operation_id != prior_operation_id)
    )
    dropped_pointer = _replacement_ledger(
        prepared,
        source_pointer=current.router_policy.spend_ledger,
        stage=ProjectStage.OPTIMIZING_ROUTER,
        stage_outputs=(
            current.router_policy.policy_lock,
            current.router_policy.policy,
        ),
        entries=dropped_entries,
    )
    dropped_config = ProjectConfig.model_validate(
        {
            **current.model_dump(mode="python"),
            "router_policy": ProjectRouterPolicyArtifacts(
                policy_lock=current.router_policy.policy_lock,
                policy=current.router_policy.policy,
                spend_ledger=dropped_pointer,
            ),
            "router_report": None,
        }
    )
    write_project_config(prepared.paths.project_toml, dropped_config)
    dropped_bundle = export_project_bundle(
        prepared,
        tmp_path / "dropped-build-spend.wmo.zip",
        producer_revision=_REVISION,
    )
    with pytest.raises(ProjectBundleError, match="drops or changes"):
        restore_project_bundle(
            dropped_bundle.path,
            root=tmp_path / "rejected-dropped-build-spend",
            expected_sha256=dropped_bundle.sha256,
        )

    write_project_config(prepared.paths.project_toml, current)
    report_ledger = load_provider_spend_ledger(
        prepared.artifacts,
        current.router_report.spend_ledger,
    )
    prior_entry = next(
        item for item in policy_ledger.entries if item.status != ProviderSpendStatus.NOT_INCURRED
    )
    changed = ProviderSpendEntry.model_validate(
        {
            **prior_entry.model_dump(mode="python"),
            "amount_usd": prior_entry.amount_usd + Decimal("0.000001"),
        }
    )
    changed_entries = tuple(
        changed if item.operation_id == changed.operation_id else item
        for item in report_ledger.entries
    )
    changed_pointer = _replacement_ledger(
        prepared,
        source_pointer=current.router_report.spend_ledger,
        stage=ProjectStage.COMPLETING_REPORT,
        stage_outputs=(current.router_report.report,),
        entries=changed_entries,
    )
    changed_config = ProjectConfig.model_validate(
        {
            **current.model_dump(mode="python"),
            "router_report": ProjectRouterReportArtifacts(
                report=current.router_report.report,
                spend_ledger=changed_pointer,
            ),
        }
    )
    write_project_config(prepared.paths.project_toml, changed_config)
    changed_bundle = export_project_bundle(
        prepared,
        tmp_path / "changed-fit-spend.wmo.zip",
        producer_revision=_REVISION,
    )
    with pytest.raises(ProjectBundleError, match="drops or changes"):
        restore_project_bundle(
            changed_bundle.path,
            root=tmp_path / "rejected-changed-fit-spend",
            expected_sha256=changed_bundle.sha256,
        )


def test_bundle_restore_rejects_same_policy_report_from_another_evaluation(
    tmp_path: Path,
) -> None:
    """A report cannot replace the locked plan's held-out evaluation under the same policy."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / "authority")
    authority = attempt_store.create()
    run_hosted_router_workflow(
        prepared,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, _ProviderState())),
        attempt_store,
        bundle_directory=tmp_path / "bundles",
        attempt_id=authority.attempt_id,
        created_at=_TIME + timedelta(hours=11),
        code_revision=_REVISION,
        options=_options(),
    )
    current = prepared.load_project()
    assert current.router_policy is not None
    assert current.router_report is not None
    report = HeldOutRouterReport.model_validate_json(
        prepared.artifacts.read_bytes(
            current.router_report.report.artifact_id,
            "report.json",
        )
    )
    evaluation_input = next(
        item
        for item in report.inputs
        if prepared.artifacts.read(item.artifact_id).manifest.artifact_type == "evaluation"
    )
    evaluation = EvaluationDatasetManifest.model_validate_json(
        prepared.artifacts.read_bytes(evaluation_input.artifact_id, "evaluation.json")
    )
    forged_evaluation = evaluation.model_copy(
        update={
            "evaluation_id": "evaluation-wrong-held-out-plan",
            "evaluation_plan_id": "evaluation-plan-wrong-held-out",
            "evaluation_plan_sha256": "f" * 64,
        }
    )
    evaluation_manifest = prepared.artifacts.write_or_verify_exact(
        artifact_id=forged_evaluation.evaluation_id,
        artifact_type="evaluation",
        envelope=ArtifactEnvelope(
            schema_version=forged_evaluation.schema_version,
            created_at=forged_evaluation.created_at,
            inputs=forged_evaluation.inputs,
            code_revision=forged_evaluation.code_revision,
            source=forged_evaluation.source,
        ),
        files={
            "evaluation.json": canonical_json_bytes(forged_evaluation),
            forged_evaluation.rows_path: prepared.artifacts.read_bytes(
                evaluation_input.artifact_id,
                evaluation.rows_path,
            ),
        },
    )
    forged_evaluation_input = artifact_input(evaluation_manifest)
    forged_report = report.model_copy(
        update={
            "report_id": "router-report-wrong-held-out-evaluation",
            "evaluation_id": forged_evaluation.evaluation_id,
            "inputs": sorted_unique_inputs(
                current.router_policy.policy,
                forged_evaluation_input,
            ),
        }
    )
    report_manifest = prepared.artifacts.write_or_verify_exact(
        artifact_id=forged_report.report_id,
        artifact_type="router-report",
        envelope=ArtifactEnvelope(
            schema_version=forged_report.schema_version,
            created_at=forged_report.created_at,
            inputs=forged_report.inputs,
            code_revision=forged_report.code_revision,
            source=forged_report.source,
        ),
        files={"report.json": canonical_json_bytes(forged_report)},
    )
    forged_report_input = artifact_input(report_manifest)
    replacement_ledger = _replacement_ledger(
        prepared,
        source_pointer=current.router_report.spend_ledger,
        stage=ProjectStage.COMPLETING_REPORT,
        stage_outputs=(forged_report_input,),
    )
    malicious = ProjectConfig.model_validate(
        {
            **current.model_dump(mode="python"),
            "router_report": ProjectRouterReportArtifacts(
                report=forged_report_input,
                spend_ledger=replacement_ledger,
            ),
        }
    )
    write_project_config(prepared.paths.project_toml, malicious)
    bundle = export_project_bundle(
        prepared,
        tmp_path / "wrong-held-out-evaluation.wmo.zip",
        producer_revision=_REVISION,
    )

    with pytest.raises(ProjectBundleError, match="selected policy evaluation"):
        restore_project_bundle(
            bundle.path,
            root=tmp_path / "rejected-wrong-held-out-evaluation",
            expected_sha256=bundle.sha256,
        )


def test_bundle_restore_rejects_fit_evaluation_with_held_out_scope(
    tmp_path: Path,
) -> None:
    """A frozen policy cannot authenticate a fit evaluation that exposes held-out scope."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / "authority")
    authority = attempt_store.create()
    run_hosted_router_workflow(
        prepared,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, _ProviderState())),
        attempt_store,
        bundle_directory=tmp_path / "bundles",
        attempt_id=authority.attempt_id,
        created_at=_TIME + timedelta(hours=12),
        code_revision=_REVISION,
        options=_options(),
    )
    current = prepared.load_project()
    assert current.router_policy is not None
    assert current.router_report is not None
    lock = RouterPolicyLock.model_validate_json(
        prepared.artifacts.read_bytes(
            current.router_policy.policy_lock.artifact_id,
            "lock.json",
        )
    )
    policy = KnnRouterPolicy.model_validate_json(
        prepared.artifacts.read_bytes(current.router_policy.policy.artifact_id, "policy.json")
    )
    bank = KnnBankManifest.model_validate_json(
        prepared.artifacts.read_bytes(lock.bank.artifact_id, "bank.json")
    )
    fit_evaluation = EvaluationDatasetManifest.model_validate_json(
        prepared.artifacts.read_bytes(lock.fit_evaluation.artifact_id, "evaluation.json")
    )
    report = HeldOutRouterReport.model_validate_json(
        prepared.artifacts.read_bytes(current.router_report.report.artifact_id, "report.json")
    )
    report_evaluation_input = _artifact_input_of_type(
        prepared,
        report.inputs,
        "evaluation",
    )
    report_evaluation = EvaluationDatasetManifest.model_validate_json(
        prepared.artifacts.read_bytes(report_evaluation_input.artifact_id, "evaluation.json")
    )
    forged_evaluation = fit_evaluation.model_copy(
        update={
            "evaluation_id": "evaluation-fit-with-held-out-scope",
            "held_out_task_ids": report_evaluation.held_out_task_ids,
        }
    )
    forged_evaluation_manifest = prepared.artifacts.write_or_verify_exact(
        artifact_id=forged_evaluation.evaluation_id,
        artifact_type="evaluation",
        envelope=forged_evaluation,
        files={
            "evaluation.json": canonical_json_bytes(forged_evaluation),
            forged_evaluation.rows_path: prepared.artifacts.read_bytes(
                lock.fit_evaluation.artifact_id,
                fit_evaluation.rows_path,
            ),
        },
    )
    forged_evaluation_input = artifact_input(forged_evaluation_manifest)
    forged_bank = bank.model_copy(
        update={
            "bank_artifact_id": "knn-bank-fit-with-held-out-scope",
            "fit_evaluation_id": forged_evaluation.evaluation_id,
            "inputs": sorted_unique_inputs(
                *(
                    forged_evaluation_input if item == lock.fit_evaluation else item
                    for item in bank.inputs
                )
            ),
        }
    )
    forged_bank_manifest = prepared.artifacts.write_or_verify_exact(
        artifact_id=forged_bank.bank_artifact_id,
        artifact_type="knn-bank",
        envelope=forged_bank,
        files={
            "bank.json": canonical_json_bytes(forged_bank),
            forged_bank.bank_path: prepared.artifacts.read_bytes(
                lock.bank.artifact_id,
                bank.bank_path,
            ),
        },
    )
    forged_bank_input = artifact_input(forged_bank_manifest)
    forged_policy = policy.model_copy(
        update={
            "policy_id": "router-policy-fit-with-held-out-scope",
            "inputs": sorted_unique_inputs(forged_evaluation_input, forged_bank_input),
            "fit_evaluation_id": forged_evaluation.evaluation_id,
            "bank_artifact_id": forged_bank.bank_artifact_id,
        }
    )
    forged_policy_manifest = prepared.artifacts.write_or_verify_exact(
        artifact_id=forged_policy.policy_id,
        artifact_type="router-policy",
        envelope=forged_policy,
        files={"policy.json": canonical_json_bytes(forged_policy)},
    )
    forged_policy_input = artifact_input(forged_policy_manifest)
    forged_lock = lock.model_copy(
        update={
            "lock_id": "router-policy-lock-fit-with-held-out-scope",
            "fit_evaluation": forged_evaluation_input,
            "bank": forged_bank_input,
            "policy": forged_policy_input,
            "inputs": sorted_unique_inputs(
                lock.plan,
                forged_evaluation_input,
                forged_bank_input,
                forged_policy_input,
            ),
        }
    )
    forged_lock_manifest = prepared.artifacts.write_or_verify_exact(
        artifact_id=forged_lock.lock_id,
        artifact_type="router-policy-lock",
        envelope=forged_lock,
        files={"lock.json": canonical_json_bytes(forged_lock)},
    )
    forged_lock_input = artifact_input(forged_lock_manifest)
    replacement_ledger = _replacement_ledger(
        prepared,
        source_pointer=current.router_policy.spend_ledger,
        stage=ProjectStage.OPTIMIZING_ROUTER,
        stage_outputs=(forged_lock_input, forged_policy_input),
    )
    malicious = ProjectConfig.model_validate(
        {
            **current.model_dump(mode="python"),
            "router_policy": ProjectRouterPolicyArtifacts(
                policy_lock=forged_lock_input,
                policy=forged_policy_input,
                spend_ledger=replacement_ledger,
            ),
            "router_report": None,
        }
    )
    write_project_config(prepared.paths.project_toml, malicious)
    bundle = export_project_bundle(
        prepared,
        tmp_path / "fit-with-held-out-scope.wmo.zip",
        producer_revision=_REVISION,
    )

    with pytest.raises(ProjectBundleError, match="router policy differs"):
        restore_project_bundle(
            bundle.path,
            root=tmp_path / "rejected-fit-with-held-out-scope",
            expected_sha256=bundle.sha256,
        )


def test_bundle_restore_rejects_report_evaluation_with_fit_scope(
    tmp_path: Path,
) -> None:
    """A held-out report cannot authenticate an evaluation that exposes fit scope."""
    prepared, catalog = _restored_prepared_project(tmp_path)
    attempt_store = FileHostedAttemptAuthorityStore(tmp_path / "authority")
    authority = attempt_store.create()
    run_hosted_router_workflow(
        prepared,
        _setup(),
        catalog,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, _ProviderState())),
        attempt_store,
        bundle_directory=tmp_path / "bundles",
        attempt_id=authority.attempt_id,
        created_at=_TIME + timedelta(hours=12),
        code_revision=_REVISION,
        options=_options(),
    )
    current = prepared.load_project()
    assert current.router_policy is not None
    assert current.router_report is not None
    lock = RouterPolicyLock.model_validate_json(
        prepared.artifacts.read_bytes(
            current.router_policy.policy_lock.artifact_id,
            "lock.json",
        )
    )
    fit_evaluation = EvaluationDatasetManifest.model_validate_json(
        prepared.artifacts.read_bytes(lock.fit_evaluation.artifact_id, "evaluation.json")
    )
    report = HeldOutRouterReport.model_validate_json(
        prepared.artifacts.read_bytes(current.router_report.report.artifact_id, "report.json")
    )
    report_evaluation_input = _artifact_input_of_type(
        prepared,
        report.inputs,
        "evaluation",
    )
    report_evaluation = EvaluationDatasetManifest.model_validate_json(
        prepared.artifacts.read_bytes(report_evaluation_input.artifact_id, "evaluation.json")
    )
    forged_evaluation = report_evaluation.model_copy(
        update={
            "evaluation_id": "evaluation-held-out-with-fit-scope",
            "fit_task_ids": fit_evaluation.fit_task_ids,
        }
    )
    forged_evaluation_manifest = prepared.artifacts.write_or_verify_exact(
        artifact_id=forged_evaluation.evaluation_id,
        artifact_type="evaluation",
        envelope=forged_evaluation,
        files={
            "evaluation.json": canonical_json_bytes(forged_evaluation),
            forged_evaluation.rows_path: prepared.artifacts.read_bytes(
                report_evaluation_input.artifact_id,
                report_evaluation.rows_path,
            ),
        },
    )
    forged_evaluation_input = artifact_input(forged_evaluation_manifest)
    forged_report = report.model_copy(
        update={
            "report_id": "router-report-held-out-with-fit-scope",
            "evaluation_id": forged_evaluation.evaluation_id,
            "inputs": sorted_unique_inputs(
                current.router_policy.policy,
                forged_evaluation_input,
            ),
        }
    )
    forged_report_manifest = prepared.artifacts.write_or_verify_exact(
        artifact_id=forged_report.report_id,
        artifact_type="router-report",
        envelope=forged_report,
        files={"report.json": canonical_json_bytes(forged_report)},
    )
    forged_report_input = artifact_input(forged_report_manifest)
    replacement_ledger = _replacement_ledger(
        prepared,
        source_pointer=current.router_report.spend_ledger,
        stage=ProjectStage.COMPLETING_REPORT,
        stage_outputs=(forged_report_input,),
    )
    malicious = ProjectConfig.model_validate(
        {
            **current.model_dump(mode="python"),
            "router_report": ProjectRouterReportArtifacts(
                report=forged_report_input,
                spend_ledger=replacement_ledger,
            ),
        }
    )
    write_project_config(prepared.paths.project_toml, malicious)
    bundle = export_project_bundle(
        prepared,
        tmp_path / "held-out-with-fit-scope.wmo.zip",
        producer_revision=_REVISION,
    )

    with pytest.raises(ProjectBundleError, match="held-out report differs"):
        restore_project_bundle(
            bundle.path,
            root=tmp_path / "rejected-held-out-with-fit-scope",
            expected_sha256=bundle.sha256,
        )


class _FailingClient:
    """Wrap a provider client and fail one selected completion role without exposing its text."""

    def __init__(self, delegate: ModelClient) -> None:
        self._delegate = delegate

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Fail one provider completion after accepting the standard request contract."""
        del request
        raise RuntimeError("provider-secret-error-text")


class _FailingCommitAuthorityStore(FileHostedAttemptAuthorityStore):
    """Simulate a worker failure before the external stage pointer CAS acknowledges."""

    def commit_stage(
        self,
        commit: HostedStageCommit,
        bundle: ExportedProjectBundle,
        ledger: ProviderSpendLedger,
    ) -> None:
        """Reject the external acknowledgment while leaving its hazard durable."""
        del commit, bundle, ledger
        raise HostedAttemptAuthorityError("external stage pointer commit failed")


class _FailingRuntimeCatalog(_RuntimeCatalog):
    """Retain the deterministic catalog while failing one completion role."""

    def __init__(
        self,
        catalog: ModelCatalog,
        state: _ProviderState,
        *,
        failed_alias: str,
    ) -> None:
        super().__init__(catalog, state)
        self._failed_alias = failed_alias

    def resolve(self, alias: str) -> ResolvedModel:
        resolved = super().resolve(alias)
        if alias != self._failed_alias:
            return resolved
        failed = _FailingClient(resolved.client)
        return ResolvedModel(
            resolved.alias,
            resolved.snapshot,
            resolved.capabilities,
            failed,
            resolved.embedding_client,
        )

    def with_catalog(self, catalog: ModelCatalog) -> _FailingRuntimeCatalog:
        return _FailingRuntimeCatalog(
            catalog,
            self._state,
            failed_alias=self._failed_alias,
        )


def _restored_prepared_project(
    tmp_path: Path,
    *,
    trace_offset: int = 0,
) -> tuple[ProjectStore, ModelCatalog]:
    """Return a PR2-style restored provider-free Project and its transient catalog."""
    catalog = _catalog()
    root = tmp_path / "prepared-source-root"
    root.mkdir()
    store = ProjectStore(root, "support")
    settings = ProjectTracePreparationSettings(
        source_kind="otlp",
        fit_task_budget=2,
        held_out_task_budget=1,
        descriptor_dimensions=64,
    )
    store.initialize(
        ProjectConfig(
            project_id="support",
            trace_preparation=settings,
            retrieval=None,
            budgets=None,
        )
    )
    candidate, _capabilities = RuntimeModelCatalog(catalog, environment={}).snapshot("candidate-a")
    built = build_project(
        TraceNormalizationResult(
            traces=tuple(_trace(index + trace_offset, candidate) for index in range(12)),
            issues=(),
        ),
        store,
        created_at=_TIME,
        code_revision=_REVISION,
        mining_spec=MiningSpec(
            fit_task_budget=2,
            held_out_task_budget=1,
            semantic_duplicate_threshold=1.0,
        ),
    )
    store.bind_provider_free_stage(
        ProjectProviderFreeStage(
            trace_dataset=artifact_input(built.artifacts.trace_dataset.manifest),
            task_set=artifact_input(
                store.artifacts.read(built.artifacts.task_set.task_set_id).manifest
            ),
        )
    )
    exported = export_project_bundle(
        store,
        tmp_path / "prepared.wmo.zip",
        producer_revision=_REVISION,
    )
    restored = restore_prepared_project_bundle(
        exported.path,
        root=tmp_path / "restored-prepared-root",
        expected_sha256=exported.sha256,
    )
    return restored, catalog


def _copy_artifacts(source: ProjectStore, target: ProjectStore) -> None:
    """Copy immutable artifacts exactly so a test bundle can authenticate swapped pointers."""
    for artifact_id in source.artifacts.list_ids():
        stored = source.artifacts.read(artifact_id)
        manifest = stored.manifest
        target.artifacts.write_or_verify_exact(
            artifact_id=artifact_id,
            artifact_type=manifest.artifact_type,
            envelope=ArtifactEnvelope(
                schema_version=manifest.schema_version,
                created_at=manifest.created_at,
                inputs=manifest.inputs,
                code_revision=manifest.code_revision,
                source=manifest.source,
            ),
            files={
                item.path: source.artifacts.read_bytes(artifact_id, item.path)
                for item in manifest.files
            },
        )


def _artifact_input_of_type(
    project: ProjectStore,
    inputs: tuple[ArtifactInput, ...],
    artifact_type: str,
) -> ArtifactInput:
    """Return the unique input with one expected immutable artifact type."""
    matches = tuple(
        item
        for item in inputs
        if project.artifacts.read(item.artifact_id).manifest.artifact_type == artifact_type
    )
    assert len(matches) == 1
    return matches[0]


def _replacement_ledger(
    project: ProjectStore,
    *,
    source_pointer: ArtifactInput,
    stage: ProjectStage,
    stage_outputs: tuple[ArtifactInput, ...],
    entries: tuple[ProviderSpendEntry, ...] | None = None,
) -> ArtifactInput:
    """Persist one authenticated ledger variant for semantic restore rejection tests."""
    source = load_provider_spend_ledger(project.artifacts, source_pointer)
    _ledger, pointer = persist_provider_spend_ledger(
        project.artifacts,
        project_id=source.project_id,
        stage=stage,
        attempt_id=source.attempt_id,
        attempt_authority_sha256=source.attempt_authority_sha256,
        ceiling_usd=source.ceiling_usd,
        entries=source.entries if entries is None else entries,
        stage_outputs=stage_outputs,
        outcome="completed",
        created_at=source.created_at,
        code_revision=source.code_revision,
    )
    return pointer


def _catalog_with_alternate_world(catalog: ModelCatalog) -> ModelCatalog:
    """Return a validated catalog with a distinct snapshot behind the world alias."""
    return ModelCatalog.model_validate(
        {
            **catalog.model_dump(mode="python"),
            "models": {
                **catalog.models,
                "world": catalog.models["world"].model_copy(update={"model": "alternate-world"}),
            },
        }
    )


def _setup(
    *,
    system_prompt: str = "Resolve the support request accurately and safely.",
    world_model: str = "world",
) -> HostedRouterWorkflowSetup:
    """Return one finite late hosted setup shared by acceptance tests."""
    return HostedRouterWorkflowSetup(
        system=ProjectSystemConfiguration(
            kind="builtin_chat",
            system_prompt=system_prompt,
            maximum_model_calls=1,
        ),
        models=ProjectModelConfiguration(
            world_model=world_model,
            judge="judge",
            embedder="embedder",
            candidates=("candidate-a", "candidate-b"),
            incumbent="candidate-a",
        ),
        retrieval=ProjectRetrievalConfiguration(top_k=2),
        budgets=ProjectBudgetConfiguration(
            maximum_build_cost_usd=Decimal("5.000000"),
            maximum_provider_cost_usd=Decimal("25.000000"),
        ),
    )


def _options() -> HostedRouterWorkflowOptions:
    """Return bounded controls small enough for the deterministic fixture."""
    return HostedRouterWorkflowOptions(
        maximum_judgments=20,
        maximum_router_feature_tokens=8_192,
        maximum_retrieval_query_tokens=32_768,
        maximum_judge_input_tokens=32_768,
        maximum_judge_output_tokens=4_096,
        simulation_maximum_output_tokens=8_000,
        maximum_concurrency=1,
    )
