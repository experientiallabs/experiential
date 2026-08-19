"""Automatic completed-build router composition tests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from click import unstyle
from rich.console import Console
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.cli.build_cmd import _build_grounded_artifacts
from wmo.cli.router_candidate_setup import collect_router_candidate_setup
from wmo.common.core.artifacts import SourceIdentity, canonical_json_bytes
from wmo.common.models import (
    AssistantAction,
    ConnectionConfig,
    Embedding,
    ModelCapabilities,
    ModelCatalog,
    ModelFinishReason,
    ModelRecord,
    ModelRequest,
    ModelResponse,
    ModelRoles,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ProviderConnection,
    ProviderModelSelection,
    RouterCandidateSelection,
    Usage,
    catalog_state_sha256,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.project import (
    AgentConfiguration,
    ArtifactCorruptionError,
    ProjectConfig,
    ProjectModelConfiguration,
    ProjectStore,
    artifact_input,
    write_project_config,
)
from wmo.common.project.manifests import file_digest
from wmo.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from wmo.optimize.router.activation import load_project_router
from wmo.optimize.router.automatic.attribution import (
    RouterObservedAttributionSet,
    persist_router_observed_attribution_set,
)
from wmo.optimize.router.automatic.preflight import (
    AutomaticRouterOptions,
    AutomaticRouterPreflightError,
    preflight_automatic_router,
)
from wmo.optimize.router.automatic.replay import find_completed_automatic_router_replay
from wmo.optimize.router.automatic.service import (
    AutomaticRouterError,
    optimize_project_router,
    persist_router_candidate_setup,
)
from wmo.optimize.router.composition import RouterCandidateSetupPlan
from wmo.optimize.router.judging.artifacts import read_audit, write_audit, write_review_state
from wmo.optimize.router.judging.contracts import (
    ManualJudgeAxisDecision,
    ManualJudgeError,
    ManualJudgeLabel,
    ManualJudgeReviewState,
)
from wmo.optimize.router.judging.labels import calibration_sample_digest
from wmo.optimize.router.judging.review import (
    ManualJudgeTraceProposal,
    completed_trace_review_count,
)
from wmo.optimize.router.judging.service import (
    calibrate_manual_judge,
    calibration_sample,
    commit_manual_judge_setup,
    estimate_manual_judge_budget,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from wmo.runtime.agents import ChatAgentRuntime
from wmo.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from wmo.runtime.router.application import RouterApplicationError
from wmo.simulation.build import build_project, select_completed_build
from wmo.simulation.ingest.model_identity import (
    normalized_capabilities_sha256,
    normalized_model_identity_evidence,
)
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.mining.service import MiningSpec

_TIME = datetime(2026, 8, 14, tzinfo=UTC)
_REVISION = "a" * 40
_RUNNER = CliRunner()
_CUSTOM_AGENT_CONSTRUCTIONS = 0


@dataclass
class _ProviderState:
    """Shared counters for deterministic provider and credential boundaries."""

    credential_resolutions: int = 0
    embedding_calls: list[tuple[str, ...]] = field(default_factory=list)
    completion_calls: list[tuple[str, ModelRequest]] = field(default_factory=list)


class _EmbeddingClient:
    """Return deterministic unit vectors while recording every provider-shaped call."""

    def __init__(self, state: _ProviderState) -> None:
        """Bind shared provider counters.

        Args:
            state: Mutable test-only provider call log.
        """
        self._state = state

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return stable four-dimensional unit vectors.

        Args:
            texts: Exact embedding batch.

        Returns:
            One deterministic unit vector for every input.
        """
        self._state.embedding_calls.append(tuple(texts))
        values = []
        for text in texts:
            raw = tuple(float(value + 1) for value in hashlib.sha256(text.encode()).digest()[:4])
            norm = math.sqrt(sum(value * value for value in raw))
            values.append(Embedding(values=tuple(value / norm for value in raw)))
        return tuple(values)


class _CompletionClient:
    """Return deterministic candidate, world-model, or judge protocol responses."""

    def __init__(self, alias: str, model: ModelSnapshot, state: _ProviderState) -> None:
        """Bind one alias, exact model identity, and shared counters.

        Args:
            alias: Catalog alias whose protocol response is produced.
            model: Exact response model identity.
            state: Mutable test-only provider call log.
        """
        self._alias = alias
        self._model = model
        self._state = state

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one schema-valid, zero-dollar observed response.

        Args:
            request: Exact provider-neutral request.

        Returns:
            Candidate text, terminal world transition, or structured judge JSON.
        """
        self._state.completion_calls.append((self._alias, request))
        if self._alias == "world":
            content = json.dumps({"message": "", "terminal": True})
        elif self._alias == "judge":
            visible = request.messages[-1].content or ""
            span_ids = re.findall(r'"span_id":\s*"([^"]+)"', visible)
            assert span_ids
            content = json.dumps(
                {
                    "dimensions": [
                        {
                            "dimension_id": "task-success",
                            "raw_score": 1,
                            "rationale": "The visible response resolves the task.",
                        }
                    ]
                }
            )
        else:
            content = "Resolved the support request."
        return ModelResponse(
            output=AssistantAction(content=content),
            model=self._model,
            economics=OperationEconomics(
                usage=Usage(input_tokens=8, output_tokens=4, cached_input_tokens=0),
                cost_usd=NumericMeasurement(value=0.0, provenance="observed"),
            ),
            finish_reason=ModelFinishReason.COMPLETED,
        )


class _RuntimeCatalog:
    """Resolve exact catalog identities to deterministic local provider clients."""

    def __init__(self, catalog: ModelCatalog, state: _ProviderState) -> None:
        """Bind current catalog metadata and shared call counters.

        Args:
            catalog: Current secret-free local catalog.
            state: Mutable test-only provider call log.
        """
        self._catalog_value = catalog
        self._static = RuntimeModelCatalog(catalog, environment={})
        self._state = state

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        """Return static model identity without credentials.

        Args:
            alias: Current catalog alias.

        Returns:
            Exact model snapshot and capabilities.
        """
        return self._static.snapshot(alias)

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        del role
        """Construct one deterministic runtime client after recording credential resolution.

        Args:
            alias: Current catalog alias.

        Returns:
            Exact resolved completion and optional embedding clients.
        """
        self._state.credential_resolutions += 1
        snapshot, capabilities = self.snapshot(alias)
        completion = _CompletionClient(alias, snapshot, self._state)
        embedding = _EmbeddingClient(self._state) if capabilities.supports_embeddings else None
        return ResolvedModel(alias, snapshot, capabilities, completion, embedding)

    def preflight(
        self,
        alias: str,
        _requirement: object | None = None,
        *,
        role: CatalogRoleName | None = None,
    ) -> ResolvedModel:
        del role
        """Reuse deterministic resolution for locally verified fixture capabilities.

        Args:
            alias: Current catalog alias.
            _requirement: Production capability requirement already represented by fixture data.

        Returns:
            Exact deterministic resolved model.
        """
        return self.resolve(alias)

    def with_catalog(self, catalog: ModelCatalog) -> _RuntimeCatalog:
        """Return an equivalent resolver over the post-confirmation catalog.

        Args:
            catalog: Confirmed candidate-inclusive catalog.

        Returns:
            New resolver sharing the same provider counters.
        """
        return _RuntimeCatalog(catalog, self._state)


@pytest.mark.parametrize(
    "candidate_order",
    [
        ("candidate-a", "candidate-b"),
        ("candidate-b", "candidate-a"),
    ],
)
def test_configless_automatic_router_composes_and_replays_without_dispatch(
    tmp_path: Path,
    candidate_order: tuple[str, str],
) -> None:
    """Run the real happy path, then replay it through the CLI without approval or calls.

    Args:
        tmp_path: Isolated local WMO root.
        candidate_order: User-selected aliases in lexical or reversed order.
    """
    store, catalog, state = _completed_project(tmp_path)
    _approve_manual_judge(store, catalog, state)
    plan = collect_router_candidate_setup(
        store.model_catalog_path,
        catalog,
        candidates=candidate_order,
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    options = AutomaticRouterOptions(
        maximum_provider_cost_usd=25.0,
        maximum_judgments=20,
        maximum_model_calls=1,
        maximum_router_feature_tokens=8_192,
        maximum_retrieval_query_tokens=32_768,
        simulation_maximum_output_tokens=8_000,
    )
    before_completion = len(state.completion_calls)
    before_embedding = len(state.embedding_calls)

    result = optimize_project_router(
        store,
        plan,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        options=options,
        provider_spend_consented=True,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
    )

    assert result.composition.optimization.optimization.policy.policy_id
    assert result.composition.optimization.optimization.report.report_id
    canonical_aliases = ("candidate-a", "candidate-b")
    assert tuple(item.alias for item in result.preflight.candidates) == canonical_aliases
    assert (
        tuple(item.candidate_alias for item in result.artifacts.pricing.candidate_prices)
        == canonical_aliases
    )
    assert (
        tuple(item.candidate_alias for item in result.artifacts.execution_contract.candidates)
        == canonical_aliases
    )
    assert tuple(item.alias for item in result.composition.plan.candidate_snapshots) == (
        canonical_aliases
    )
    assert (
        tuple(item.alias for item in result.composition.optimization.optimization.policy.candidates)
        == canonical_aliases
    )
    assert result.artifacts.execution_contract.maximum_provider_cost_usd == 25.0
    assert result.artifacts.attribution_input in result.artifacts.execution_contract.inputs
    assert result.artifacts.attribution_input in result.composition.plan.inputs
    assert result.preflight.observed_traces[0].attribution.match_kind == "strict_snapshot"
    assert result.composition.plan.inputs
    assert len(state.completion_calls) > before_completion
    assert len(state.embedding_calls) > before_embedding
    completed_completion = tuple(state.completion_calls)
    completed_embedding = tuple(state.embedding_calls)
    completed_credentials = state.credential_resolutions
    completed_artifacts = store.artifacts.list_ids()
    completed_catalog = store.model_catalog_path.read_bytes()
    completed_review = store.read_review()

    replay = find_completed_automatic_router_replay(
        store,
        result.preflight,
        options=options,
        code_revision=_REVISION,
    )

    assert replay is not None
    assert replay.policy_id == result.composition.optimization.optimization.policy.policy_id
    assert (
        find_completed_automatic_router_replay(
            store,
            replace(result.preflight, incumbent_alias="candidate-b"),
            options=options,
            code_revision=_REVISION,
        )
        is None
    )
    catalog_drift = replace(result.preflight, catalog_sha256="0" * 64)
    assert (
        find_completed_automatic_router_replay(
            store,
            catalog_drift,
            options=options,
            code_revision=_REVISION,
        )
        is None
    )
    assert tuple(state.completion_calls) == completed_completion
    assert tuple(state.embedding_calls) == completed_embedding
    assert state.credential_resolutions == completed_credentials

    cli = _RUNNER.invoke(
        app,
        [
            "optimize",
            "router",
            "support",
            "--root",
            str(store.paths.root),
            "--maximum-judgments",
            "20",
            "--maximum-model-calls",
            "1",
            "--simulation-maximum-output-tokens",
            "8000",
            "--non-interactive",
        ],
        env={"WMO_RELEASE_REVISION": _REVISION},
    )

    assert cli.exit_code == 0, cli.output
    assert "replay: verified completed optimization" in unstyle(cli.output)
    assert store.artifacts.list_ids() == completed_artifacts
    assert store.model_catalog_path.read_bytes() == completed_catalog
    assert store.read_review() == completed_review
    assert tuple(state.completion_calls) == completed_completion
    assert tuple(state.embedding_calls) == completed_embedding
    assert state.credential_resolutions == completed_credentials

    redacted_config = store.load_project().model_copy(update={"redacted_field_names": ("email",)})
    write_project_config(store.paths.project_toml, redacted_config)
    redacted_preflight = preflight_automatic_router(
        store,
        plan.selection,
        catalog_override=plan.prospective_catalog,
        options=options,
    )
    assert (
        redacted_preflight.simulation_configuration_sha256
        != result.preflight.simulation_configuration_sha256
    )
    assert (
        find_completed_automatic_router_replay(
            store,
            redacted_preflight,
            options=options,
            code_revision=_REVISION,
        )
        is None
    )

    changed_config = store.load_project().model_copy(
        update={
            "agent": AgentConfiguration(
                factory="wmo.optimize.router.automatic.service_test:_custom_agent_factory",
                code_revision="custom-agent-v1",
            )
        }
    )
    write_project_config(store.paths.project_toml, changed_config)
    custom_agent = preflight_automatic_router(
        store,
        plan.selection,
        catalog_override=plan.prospective_catalog,
        options=options,
    )
    assert custom_agent.agent_factory_sha256 != result.preflight.agent_factory_sha256
    assert (
        find_completed_automatic_router_replay(
            store,
            custom_agent,
            options=options,
            code_revision=_REVISION,
        )
        is None
    )
    write_project_config(
        store.paths.project_toml,
        changed_config.model_copy(
            update={
                "agent": changed_config.agent.model_copy(
                    update={"code_revision": "custom-agent-v2"}
                )
                if changed_config.agent is not None
                else None
            }
        ),
    )
    revised_agent = preflight_automatic_router(
        store,
        plan.selection,
        catalog_override=plan.prospective_catalog,
        options=options,
    )
    assert revised_agent.agent_factory_sha256 != custom_agent.agent_factory_sha256
    assert (
        find_completed_automatic_router_replay(
            store,
            revised_agent,
            options=options,
            code_revision=_REVISION,
        )
        is None
    )


def test_replay_restores_discovered_candidate_records_before_reporting_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified replay restores missing discovered provider records before returning.

    Args:
        tmp_path: Isolated local WMO root.
        monkeypatch: Test patching seam for the CLI's already-collected candidate plan.
    """
    import wmo.cli.router_app as router_app

    store, catalog, state = _completed_project(tmp_path)
    _approve_manual_judge(store, catalog, state)
    connection = ProviderConnection(
        name="discovered",
        provider="openai",
        api_key_env="DISCOVERED_API_KEY",
    )
    capabilities = catalog.models["candidate-a"].capabilities
    assert capabilities is not None
    model = ProviderModelSelection(
        alias="candidate-new",
        connection=connection.name,
        model="candidate-new",
        capabilities=capabilities,
    )
    prospective = catalog.model_copy(
        update={
            "connections": {
                **catalog.connections,
                connection.name: connection.catalog_config(),
            },
            "models": {**catalog.models, model.alias: model.catalog_record()},
            "roles": catalog.roles.model_copy(
                update={
                    "candidates": ("candidate-a", model.alias),
                    "incumbent": "candidate-a",
                }
            ),
        }
    )
    plan = RouterCandidateSetupPlan(
        selection=RouterCandidateSelection(
            candidates=("candidate-a", model.alias), incumbent="candidate-a"
        ),
        candidate_models=(model,),
        prospective_catalog=prospective,
        expected_catalog_sha256=catalog_state_sha256(store.model_catalog_path),
        candidate_connections=(connection,),
    )
    options = AutomaticRouterOptions(
        maximum_judgments=20,
        maximum_model_calls=1,
        simulation_maximum_output_tokens=8_000,
    )
    optimize_project_router(
        store,
        plan,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        options=options,
        provider_spend_consented=True,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
    )

    persisted = load_model_catalog(store.model_catalog_path)
    damaged = persisted.model_copy(
        update={
            "connections": {
                name: value
                for name, value in persisted.connections.items()
                if name != connection.name
            },
            "models": {
                alias: value for alias, value in persisted.models.items() if alias != model.alias
            },
            "roles": persisted.roles.model_copy(update={"candidates": (), "incumbent": None}),
        }
    )
    write_model_catalog(store.model_catalog_path, damaged)
    replay_plan = replace(
        plan,
        expected_catalog_sha256=catalog_state_sha256(store.model_catalog_path),
    )
    monkeypatch.setattr(
        router_app,
        "collect_router_candidate_setup",
        lambda *_args, **_kwargs: replay_plan,
    )
    before_credentials = state.credential_resolutions
    before_completion = tuple(state.completion_calls)
    before_embedding = tuple(state.embedding_calls)

    cli = _RUNNER.invoke(
        app,
        [
            "optimize",
            "router",
            "support",
            "--root",
            str(store.paths.root),
            "--maximum-judgments",
            "20",
            "--maximum-model-calls",
            "1",
            "--simulation-maximum-output-tokens",
            "8000",
            "--non-interactive",
        ],
        env={"WMO_RELEASE_REVISION": _REVISION},
    )

    assert cli.exit_code == 0, cli.output
    assert "replay: verified completed optimization" in unstyle(cli.output)
    restored = load_model_catalog(store.model_catalog_path)
    assert restored.connections[connection.name] == connection.catalog_config()
    assert restored.models[model.alias] == model.catalog_record()
    assert restored.roles.candidates == replay_plan.selection.candidates
    assert restored.roles.incumbent == replay_plan.selection.incumbent
    assert state.credential_resolutions == before_credentials
    assert tuple(state.completion_calls) == before_completion
    assert tuple(state.embedding_calls) == before_embedding


def test_automatic_router_refuses_spend_before_credentials_calls_or_writes(
    tmp_path: Path,
) -> None:
    """Keep a complete project unchanged when the shared spend ceiling is not approved.

    Args:
        tmp_path: Isolated local WMO root.
    """
    store, catalog, state = _completed_project(tmp_path)
    _approve_manual_judge(store, catalog, state)
    plan = collect_router_candidate_setup(
        store.model_catalog_path,
        catalog,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    before_credentials = state.credential_resolutions
    before_completion = tuple(state.completion_calls)
    before_embedding = tuple(state.embedding_calls)
    before_artifacts = store.artifacts.list_ids()
    before_catalog = store.model_catalog_path.read_bytes()
    before_review = store.read_review()

    with pytest.raises(AutomaticRouterError, match="explicit consent"):
        optimize_project_router(
            store,
            plan,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
            options=AutomaticRouterOptions(
                maximum_model_calls=1,
                simulation_maximum_output_tokens=8_000,
            ),
            provider_spend_consented=False,
            created_at=_TIME + timedelta(hours=1),
            code_revision=_REVISION,
        )

    assert state.credential_resolutions == before_credentials
    assert tuple(state.completion_calls) == before_completion
    assert tuple(state.embedding_calls) == before_embedding
    assert store.artifacts.list_ids() == before_artifacts
    assert store.model_catalog_path.read_bytes() == before_catalog
    assert store.read_review() == before_review


def test_discovered_candidate_provider_records_and_roles_persist_atomically(
    tmp_path: Path,
) -> None:
    """Persist newly discovered candidate metadata and router roles in one catalog write.

    Args:
        tmp_path: Isolated local WMO root.
    """
    catalog = _catalog()
    root = tmp_path / ".wmo"
    root.mkdir()
    path = root / "models.toml"
    write_model_catalog(path, catalog)
    project = ProjectStore(root, "support")
    connection = ProviderConnection(
        name="discovered",
        provider="openai",
        api_key_env="DISCOVERED_API_KEY",
    )
    capabilities = catalog.models["candidate-a"].capabilities
    assert capabilities is not None
    model = ProviderModelSelection(
        alias="candidate-new",
        connection=connection.name,
        model="candidate-new",
        capabilities=capabilities,
    )
    prospective = catalog.model_copy(
        update={
            "connections": {
                **catalog.connections,
                connection.name: connection.catalog_config(),
            },
            "models": {**catalog.models, model.alias: model.catalog_record()},
            "roles": catalog.roles.model_copy(
                update={"candidates": ("candidate-a", model.alias), "incumbent": "candidate-a"}
            ),
        }
    )
    plan = RouterCandidateSetupPlan(
        selection=RouterCandidateSelection(
            candidates=("candidate-a", model.alias), incumbent="candidate-a"
        ),
        candidate_models=(model,),
        prospective_catalog=prospective,
        expected_catalog_sha256=catalog_state_sha256(path),
        candidate_connections=(connection,),
    )

    configured = persist_router_candidate_setup(project, plan)
    saved = load_model_catalog(path)
    assert saved.connections[connection.name].provider == "openai"
    assert saved.models[model.alias] == model.catalog_record()
    assert saved.roles.candidates == plan.selection.candidates
    assert configured.roles.candidates == plan.selection.candidates
    assert configured.roles.candidates == ("candidate-a", model.alias)


def test_provider_model_only_telemetry_composes_with_inferred_unique_attribution(
    tmp_path: Path,
) -> None:
    """Normal provider/model OTLP evidence completes the configless production path.

    Args:
        tmp_path: Isolated local WMO root.
    """
    store, catalog, state = _completed_project(tmp_path, inferred_identity=True)
    _approve_manual_judge(store, catalog, state)
    plan = collect_router_candidate_setup(
        store.model_catalog_path,
        catalog,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )

    result = optimize_project_router(
        store,
        plan,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        options=AutomaticRouterOptions(
            maximum_judgments=20,
            maximum_model_calls=1,
            simulation_maximum_output_tokens=8_000,
        ),
        provider_spend_consented=True,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
    )

    attribution = result.preflight.observed_traces[0].attribution
    assert attribution.candidate_alias == "candidate-a"
    assert attribution.match_kind == "inferred_unique"
    assert result.artifacts.attribution_input in result.composition.plan.inputs


def test_duplicate_candidate_identities_fail_before_stateful_boundaries(tmp_path: Path) -> None:
    """Duplicate selected identities fail before writes, credentials, or providers.

    Args:
        tmp_path: Isolated local WMO root.
    """
    store, catalog, state = _completed_project(tmp_path, inferred_identity=True)
    _approve_manual_judge(store, catalog, state)
    ambiguous = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "candidate-b": catalog.models["candidate-b"].model_copy(
                    update={"model": "candidate-a"}
                ),
            }
        }
    )
    plan = collect_router_candidate_setup(
        store.model_catalog_path,
        ambiguous,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    before_artifacts = store.artifacts.list_ids()
    before_catalog = store.model_catalog_path.read_bytes()
    before_completion = tuple(state.completion_calls)
    before_embedding = tuple(state.embedding_calls)
    before_credentials = state.credential_resolutions

    with pytest.raises(AutomaticRouterPreflightError, match="distinct exact model identities"):
        optimize_project_router(
            store,
            plan,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
            options=AutomaticRouterOptions(
                maximum_judgments=20,
                maximum_model_calls=1,
                simulation_maximum_output_tokens=8_000,
            ),
            provider_spend_consented=True,
            created_at=_TIME + timedelta(hours=1),
            code_revision=_REVISION,
        )

    assert store.artifacts.list_ids() == before_artifacts
    assert store.model_catalog_path.read_bytes() == before_catalog
    assert tuple(state.completion_calls) == before_completion
    assert tuple(state.embedding_calls) == before_embedding
    assert state.credential_resolutions == before_credentials


@pytest.mark.parametrize("tamper", ["payload", "source", "extra-file", "lineage-record"])
def test_completed_replay_rejects_attribution_tamper_before_provider_access(
    tmp_path: Path,
    tamper: str,
) -> None:
    """A manifest-valid changed attribution payload poisons replay before any provider call.

    Args:
        tmp_path: Isolated local WMO root.
        tamper: Manifest-valid attribution mutation to apply.
    """
    store, catalog, state = _completed_project(tmp_path)
    _approve_manual_judge(store, catalog, state)
    plan = collect_router_candidate_setup(
        store.model_catalog_path,
        catalog,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    options = AutomaticRouterOptions(
        maximum_judgments=20,
        maximum_model_calls=1,
        simulation_maximum_output_tokens=8_000,
    )
    result = optimize_project_router(
        store,
        plan,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        options=options,
        provider_spend_consented=True,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
    )
    if tamper == "lineage-record":
        before_artifacts = store.artifacts.list_ids()
        record = result.preflight.observed_traces[0].attribution.model_copy(
            update={"lineage_id": "forged-lineage"}
        )
        with pytest.raises(ArtifactCorruptionError, match="partition, or lineage"):
            persist_router_observed_attribution_set(
                store.artifacts,
                trace_dataset=result.preflight.completed_build.trace_dataset,
                task_set=result.preflight.completed_build.task_set,
                catalog_sha256=result.preflight.catalog_sha256,
                candidates=result.preflight.candidates,
                records=(record,),
                created_at=_TIME + timedelta(hours=2),
                code_revision=_REVISION,
            )
        assert store.artifacts.list_ids() == before_artifacts
        return
    attribution_id = result.artifacts.attribution_input.artifact_id
    stored = store.artifacts.read(attribution_id)
    if tamper == "payload":
        value = RouterObservedAttributionSet.model_validate_json(
            store.artifacts.read_bytes(attribution_id, "attribution.json")
        ).model_copy(update={"catalog_sha256": "0" * 64})
        payload = canonical_json_bytes(value)
        manifest = stored.manifest.model_copy(
            update={"files": (file_digest("attribution.json", payload),)}
        )
        (stored.directory / "attribution.json").write_bytes(payload)
        match = "content identity differs"
    elif tamper == "source":
        manifest = stored.manifest.model_copy(
            update={
                "source": SourceIdentity(
                    kind="otlp",
                    source_id="forged",
                    sha256="0" * 64,
                )
            }
        )
        match = "must not have source"
    else:
        extra = b"{}"
        (stored.directory / "extra.json").write_bytes(extra)
        manifest = stored.manifest.model_copy(
            update={
                "files": tuple(
                    sorted(
                        (*stored.manifest.files, file_digest("extra.json", extra)),
                        key=lambda item: item.path,
                    )
                )
            }
        )
        match = "exact one-file shape"
    (stored.directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    before_completion = tuple(state.completion_calls)
    before_embedding = tuple(state.embedding_calls)
    before_credentials = state.credential_resolutions

    with pytest.raises(ArtifactCorruptionError, match=match):
        find_completed_automatic_router_replay(
            store,
            result.preflight,
            options=options,
            code_revision=_REVISION,
        )

    assert tuple(state.completion_calls) == before_completion
    assert tuple(state.embedding_calls) == before_embedding
    assert state.credential_resolutions == before_credentials


def test_completed_custom_agent_replay_does_not_import_or_construct_factory(
    tmp_path: Path,
) -> None:
    """Replay from identity-only preflight without executing customer agent code.

    Args:
        tmp_path: Isolated local WMO root.
    """
    global _CUSTOM_AGENT_CONSTRUCTIONS

    _CUSTOM_AGENT_CONSTRUCTIONS = 0
    store, catalog, state = _completed_project(
        tmp_path,
        agent=AgentConfiguration(
            factory="wmo.optimize.router.automatic.service_test:_custom_agent_factory",
            code_revision="custom-agent-v1",
        ),
    )
    _approve_manual_judge(store, catalog, state)
    plan = collect_router_candidate_setup(
        store.model_catalog_path,
        catalog,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    options = AutomaticRouterOptions(
        maximum_provider_cost_usd=25.0,
        maximum_judgments=20,
        maximum_model_calls=1,
        simulation_maximum_output_tokens=8_000,
    )
    optimize_project_router(
        store,
        plan,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        options=options,
        provider_spend_consented=True,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
    )
    completed_constructions = _CUSTOM_AGENT_CONSTRUCTIONS
    assert completed_constructions > 0

    cli = _RUNNER.invoke(
        app,
        [
            "optimize",
            "router",
            "support",
            "--root",
            str(store.paths.root),
            "--maximum-judgments",
            "20",
            "--maximum-model-calls",
            "1",
            "--simulation-maximum-output-tokens",
            "8000",
            "--non-interactive",
        ],
        env={"WMO_RELEASE_REVISION": _REVISION},
    )

    assert cli.exit_code == 0, cli.output
    assert "replay: verified completed optimization" in unstyle(cli.output)
    assert _CUSTOM_AGENT_CONSTRUCTIONS == completed_constructions


def test_automatic_router_rejects_confirmed_catalog_drift_before_credentials(
    tmp_path: Path,
) -> None:
    """Reject a stale candidate confirmation before resolving any provider client.

    Args:
        tmp_path: Isolated local WMO root.
    """
    store, catalog, state = _completed_project(tmp_path)
    _approve_manual_judge(store, catalog, state)
    plan = collect_router_candidate_setup(
        store.model_catalog_path,
        catalog,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    mutated = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "unrelated": ModelRecord(
                    connection="provider",
                    model="unrelated",
                    capabilities=catalog.models["candidate-a"].capabilities,
                ),
            }
        }
    )
    write_model_catalog(store.model_catalog_path, mutated)
    before_credentials = state.credential_resolutions
    before_completion = tuple(state.completion_calls)
    before_embedding = tuple(state.embedding_calls)
    before_artifacts = store.artifacts.list_ids()

    with pytest.raises(ValueError, match="models.toml changed"):
        optimize_project_router(
            store,
            plan,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
            options=AutomaticRouterOptions(
                maximum_model_calls=1,
                simulation_maximum_output_tokens=8_000,
            ),
            provider_spend_consented=True,
            created_at=_TIME + timedelta(hours=1),
            code_revision=_REVISION,
        )

    assert state.credential_resolutions == before_credentials
    assert tuple(state.completion_calls) == before_completion
    assert tuple(state.embedding_calls) == before_embedding
    assert store.artifacts.list_ids() == before_artifacts


def test_automatic_router_rejects_substituted_manual_judge_audit_before_calls(
    tmp_path: Path,
) -> None:
    """Reject a valid but unrelated audit budget before credentials, writes, or calls.

    Args:
        tmp_path: Isolated local WMO root.
    """
    store, catalog, state = _completed_project(tmp_path)
    _approve_manual_judge(store, catalog, state)
    review = store.read_review()
    assert isinstance(review, dict)
    selected = ManualJudgeReviewState.model_validate(review["manual_judge"])
    assert selected.audit is not None
    audit = read_audit(store, selected.audit)
    substituted_budget = audit.budget.model_copy(
        update={"maximum_input_tokens_per_call": 1, "estimated_cost_usd": 0.0}
    )
    expected_estimate = (
        (
            substituted_budget.maximum_input_tokens_per_call
            * substituted_budget.input_usd_per_million_tokens
            + substituted_budget.maximum_output_tokens_per_call
            * substituted_budget.output_usd_per_million_tokens
        )
        / 1_000_000
        * substituted_budget.maximum_attempts_per_call
        * substituted_budget.call_count
    )
    substituted_budget = substituted_budget.model_copy(
        update={"estimated_cost_usd": expected_estimate}
    )
    substituted = write_audit(
        store,
        setup_input=audit.setup,
        label_input=audit.human_labels,
        split_input=audit.lineage_split,
        provisional_input=audit.provisional_calibration,
        report_input=audit.report,
        budget=substituted_budget,
        judgments=audit.judgments,
        trace_reviews=audit.trace_reviews,
        positional_bias=(
            (audit.positional_bias_comparisons, audit.positional_bias_flips)
            if audit.positional_bias_comparisons is not None
            and audit.positional_bias_flips is not None
            else None
        ),
        created_at=_TIME + timedelta(minutes=5),
        code_revision=_REVISION,
    )
    substituted_input = artifact_input(store.artifacts.read(substituted.audit_id).manifest)
    write_review_state(store, selected.model_copy(update={"audit": substituted_input}))
    plan = collect_router_candidate_setup(
        store.model_catalog_path,
        catalog,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    before_credentials = state.credential_resolutions
    before_completion = tuple(state.completion_calls)
    before_embedding = tuple(state.embedding_calls)
    before_artifacts = store.artifacts.list_ids()

    with pytest.raises(
        ValueError,
        match=(
            "selected judge calibration audit differs from its setup, approved lineage, or budget"
        ),
    ):
        preflight_automatic_router(
            store,
            plan.selection,
            catalog_override=plan.prospective_catalog,
            options=AutomaticRouterOptions(
                maximum_provider_cost_usd=25.0,
                maximum_judgments=20,
                maximum_model_calls=1,
                router_embedding_maximum_attempts=1,
                completion_maximum_attempts=1,
                simulation_maximum_output_tokens=8_000,
            ),
        )

    assert state.credential_resolutions == before_credentials
    assert tuple(state.completion_calls) == before_completion
    assert tuple(state.embedding_calls) == before_embedding
    assert store.artifacts.list_ids() == before_artifacts


def test_preflight_accepts_calibration_resumed_after_a_failed_first_pass(
    tmp_path: Path,
) -> None:
    """Approve a calibration resumed from a failed first pass, then pass router preflight.

    Args:
        tmp_path: Isolated local WMO root.
    """
    store, catalog, state = _completed_project(tmp_path)
    setup_plan = prepare_manual_judge_setup(
        store,
        catalog,
        preview_count=1,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    commit_manual_judge_setup(store, setup_plan, confirmed=True)
    plan = prepare_manual_judge_calibration(store, sample_size=2)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=32_768,
        maximum_cost_usd=1.0,
    )
    reviewed_positions: list[int] = []

    def _accept(proposal: ManualJudgeTraceProposal) -> tuple[ManualJudgeAxisDecision, ...]:
        """Accept every proposed axis unchanged.

        Args:
            proposal: Persisted configured-judge proposal for the current trace.

        Returns:
            One explicit acceptance per proposed rubric axis.
        """
        return tuple(
            ManualJudgeAxisDecision(dimension_id=item.dimension_id, accepted=True)
            for item in proposal.judgment.dimensions
        )

    def _fail_on_second(
        proposal: ManualJudgeTraceProposal,
    ) -> tuple[ManualJudgeAxisDecision, ...]:
        """Accept the first trace, then fail like a correction missing its judgment.

        Args:
            proposal: Persisted configured-judge proposal for the current trace.

        Returns:
            One explicit acceptance per proposed rubric axis for the first trace only.

        Raises:
            ManualJudgeError: The reviewer reaches any trace after the first.
        """
        reviewed_positions.append(proposal.position)
        if len(reviewed_positions) > 1:
            raise ManualJudgeError(
                "a corrected score requires --judgment TRACE:dim=CORRECTED_JUDGMENT"
            )
        return _accept(proposal)

    with pytest.raises(ManualJudgeError, match="corrected score requires"):
        calibrate_manual_judge(
            store,
            runtime,
            plan,
            (),
            budget,
            spend_consented=True,
            approve=True,
            accept_insufficient_labels=True,
            created_at=_TIME,
            code_revision=_REVISION,
            reviewer=_fail_on_second,
        )

    plan = prepare_manual_judge_calibration(store, sample_size=2)
    reviewed = completed_trace_review_count(
        store,
        plan.setup,
        calibration_sample_digest(plan.setup, calibration_sample(plan)),
    )
    assert reviewed == 1
    resumed_budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=32_768,
        maximum_cost_usd=1.0,
        completed_review_count=reviewed,
    )
    result = calibrate_manual_judge(
        store,
        runtime,
        plan,
        (),
        resumed_budget,
        spend_consented=True,
        approve=True,
        accept_insufficient_labels=True,
        created_at=_TIME + timedelta(minutes=1),
        code_revision=_REVISION,
        reviewer=_accept,
    )
    assert result.approved_calibration is not None
    candidate_plan = collect_router_candidate_setup(
        store.model_catalog_path,
        catalog,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )

    preflight = preflight_automatic_router(
        store,
        candidate_plan.selection,
        catalog_override=candidate_plan.prospective_catalog,
        options=AutomaticRouterOptions(
            maximum_provider_cost_usd=25.0,
            maximum_judgments=20,
            maximum_model_calls=1,
            simulation_maximum_output_tokens=8_000,
        ),
    )

    assert preflight.approved_calibration_input == result.approved_calibration
    assert preflight.judge_audit.budget.call_count == 1
    assert sum(len(item.probes) for item in preflight.judge_audit.judgments) == 2


@pytest.mark.parametrize("tamper", ["execution", "policy"])
def test_runtime_activation_rejects_automatic_contract_tamper_before_credentials(
    tmp_path: Path,
    tamper: str,
) -> None:
    """Reject execution-content or policy-incumbent drift before client resolution.

    Args:
        tmp_path: Isolated local WMO root.
        tamper: Immutable automatic artifact modified after optimization.
    """
    store, catalog, state = _completed_project(tmp_path)
    _approve_manual_judge(store, catalog, state)
    plan = collect_router_candidate_setup(
        store.model_catalog_path,
        catalog,
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
        non_interactive=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    result = optimize_project_router(
        store,
        plan,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        options=AutomaticRouterOptions(
            maximum_judgments=20,
            maximum_model_calls=1,
            simulation_maximum_output_tokens=8_000,
        ),
        provider_spend_consented=True,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
    )
    policy = result.composition.optimization.optimization.policy
    if tamper == "execution":
        artifact_id = result.artifacts.execution_contract.execution_contract_id
        file_name = "execution-contract.json"
        value = result.artifacts.execution_contract.model_copy(
            update={"incumbent_alias": "candidate-b"}
        )
    else:
        artifact_id = policy.policy_id
        file_name = "policy.json"
        value = policy.model_copy(update={"baseline_alias": "candidate-b"})
    stored = store.artifacts.read(artifact_id)
    payload = canonical_json_bytes(value)
    manifest = stored.manifest.model_copy(update={"files": (file_digest(file_name, payload),)})
    artifact_directory = store.paths.artifact_directory(artifact_id)
    (artifact_directory / file_name).write_bytes(payload)
    (artifact_directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    before_credentials = state.credential_resolutions
    before_completion = tuple(state.completion_calls)
    before_embedding = tuple(state.embedding_calls)

    with pytest.raises(RouterApplicationError):
        load_project_router(
            "support",
            store.paths.root,
            policy_id=policy.policy_id,
            runtime_catalog=cast(
                RuntimeModelCatalog,
                _RuntimeCatalog(result.preflight.catalog, state),
            ),
        )

    assert state.credential_resolutions == before_credentials
    assert tuple(state.completion_calls) == before_completion
    assert tuple(state.embedding_calls) == before_embedding


def _completed_project(
    tmp_path: Path,
    *,
    agent: AgentConfiguration | None = None,
    inferred_identity: bool = False,
) -> tuple[ProjectStore, ModelCatalog, _ProviderState]:
    """Create one exact completed build with candidate-attributed real traces.

    Args:
        tmp_path: Isolated local WMO root.
        agent: Optional exact custom agent configuration frozen during build.
        inferred_identity: Whether source model digests use telemetry fallbacks rather than the
            selected catalog snapshot.

    Returns:
        Completed project, catalog, and shared provider counters.
    """
    catalog = _catalog()
    state = _ProviderState()
    root = tmp_path / ".wmo"
    root.mkdir()
    write_model_catalog(root / "models.toml", catalog)
    store = ProjectStore(root, "support")
    store.initialize(
        ProjectConfig(
            project_id="support",
            agent=agent,
            models=ProjectModelConfiguration(
                world_model="world",
                judge="judge",
                embedder="embedder",
            ),
        )
    )
    runtime = _RuntimeCatalog(catalog, state)
    candidate_model, _capabilities = runtime.snapshot("candidate-a")
    recorded_model = (
        candidate_model.model_copy(
            update={
                "capabilities_sha256": normalized_capabilities_sha256(
                    {},
                    candidate_model.provider,
                    candidate_model.model_id,
                    candidate_model.revision,
                    error_type=ValueError,
                ),
                "connection_sha256": ConnectionConfig(
                    provider=candidate_model.provider
                ).identity_sha256(),
            }
        )
        if inferred_identity
        else candidate_model
    )
    traces = tuple(_trace(index, recorded_model) for index in range(12))
    built = build_project(
        TraceNormalizationResult(
            traces=traces,
            issues=(),
            identity_evidence=(
                normalized_model_identity_evidence(traces) if inferred_identity else None
            ),
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
    completed = _build_grounded_artifacts(
        store,
        built,
        resolved_world=runtime.resolve("world"),
        resolved_embedder=runtime.resolve("embedder"),
        top_k=2,
    )
    select_completed_build(store, completed, built.review)
    return store, catalog, state


def _custom_agent_factory() -> ChatAgentRuntime:
    """Return one valid custom runtime used to prove replay factory drift.

    Returns:
        Bounded chat runtime exposed through a custom project factory reference.
    """
    global _CUSTOM_AGENT_CONSTRUCTIONS

    _CUSTOM_AGENT_CONSTRUCTIONS += 1
    return ChatAgentRuntime(maximum_model_calls=1)


def _approve_manual_judge(
    store: ProjectStore,
    catalog: ModelCatalog,
    state: _ProviderState,
) -> None:
    """Persist one explicitly approved real-trace judge calibration.

    Args:
        store: Completed project store.
        catalog: Exact build-time catalog.
        state: Shared provider counters.
    """
    setup_plan = prepare_manual_judge_setup(
        store,
        catalog,
        preview_count=1,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    commit_manual_judge_setup(store, setup_plan, confirmed=True)
    plan = prepare_manual_judge_calibration(store, sample_size=2)
    labels = tuple(
        ManualJudgeLabel(
            trace_id=trace.trace_id,
            dimension_id="task-success",
            score=1,
        )
        for trace in plan.traces
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=32_768,
        maximum_cost_usd=1.0,
    )
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    calibrate_manual_judge(
        store,
        runtime,
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    approved = calibrate_manual_judge(
        store,
        runtime,
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=True,
        accept_insufficient_labels=True,
        created_at=_TIME + timedelta(seconds=1),
        code_revision=_REVISION,
    )
    assert approved.approved_calibration is not None


def _catalog() -> ModelCatalog:
    """Return complete build roles and two unselected router candidate aliases."""
    completion = ModelCapabilities(
        supports_completions=True,
        supports_tools=False,
        supports_structured_output=True,
        context_window_tokens=128_000,
        maximum_output_tokens=32_000,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
        cached_input_cost_per_million_tokens_usd=0.5,
        cache_write_cost_per_million_tokens_usd=1.5,
    )
    embedding = ModelCapabilities(
        supports_embeddings=True,
        input_cost_per_million_tokens_usd=0.1,
    )
    return ModelCatalog(
        connections={
            "provider": ConnectionConfig(provider="openai", api_key_env="FIXTURE_API_KEY")
        },
        models={
            **{
                alias: ModelRecord(
                    connection="provider",
                    model=alias,
                    capabilities=completion,
                )
                for alias in ("candidate-a", "candidate-b", "world", "judge")
            },
            "embedder": ModelRecord(
                connection="provider",
                model="embedder",
                capabilities=embedding,
            ),
        },
        roles=ModelRoles(world_model="world", judge="judge", embedder="embedder"),
    )


def _trace(index: int, model: ModelSnapshot) -> Trace:
    """Return one real two-turn trace attributed to the selected incumbent model.

    Args:
        index: Unique trace and lineage index.
        model: Exact incumbent provider model snapshot.

    Returns:
        Successful canonical trace with one retrievable real transition.
    """
    started = _TIME + timedelta(minutes=index)
    suffix = "".join(chr(ord("a") + value) for value in divmod(index, 26))
    first_attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "content": f"Handle unique support request {suffix}"}]
        ),
        "gen_ai.output.messages": json.dumps(
            [{"role": "assistant", "content": "What account email?"}]
        ),
    }
    second_attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.input.messages": json.dumps(
            [
                {"role": "assistant", "content": "What account email?"},
                {"role": "user", "content": f"customer-{suffix}@example.test"},
            ]
        ),
        "gen_ai.output.messages": json.dumps(
            [{"role": "assistant", "content": "Reset instructions sent."}]
        ),
    }
    return Trace(
        trace_id=f"trace-{index}",
        conversation_id=f"conversation-{index}",
        task=f"Handle unique support request {suffix}",
        spans=(
            TraceSpan(
                span_id=f"span-{index}-a",
                name="agent.model_call",
                started_at=started,
                ended_at=started + timedelta(seconds=1),
                attributes=first_attributes,
                model=model,
            ),
            TraceSpan(
                span_id=f"span-{index}-b",
                name="agent.model_call",
                started_at=started + timedelta(seconds=2),
                ended_at=started + timedelta(seconds=3),
                attributes=second_attributes,
                model=model,
            ),
        ),
        outcome=TraceOutcome(status="success"),
        source=TraceSource(
            identity=SourceIdentity(
                kind="otlp",
                source_id="fixture.otlp.jsonl",
                sha256="a" * 64,
            ),
            semantic_convention_version="1.37.0",
        ),
    )
