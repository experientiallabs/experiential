"""Interactive build-wizard state-machine and safety tests."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from click import unstyle
from rich.console import Console
from typer.testing import CliRunner

import wmo.cli.build_cmd as build_command
import wmo.cli.build_wizard as wizard
import wmo.cli.build_wizard_screens as screens
from wmo.cli.app import app
from wmo.cli.build_cmd_test import _otlp_export
from wmo.cli.provider_setup_test import _FakeLister
from wmo.common.config.settings import set_maximum_command_cost_usd
from wmo.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRequest,
    ModelResponse,
    ModelRoles,
    ModelSnapshot,
    write_model_catalog,
)
from wmo.common.project import ProjectModelConfiguration
from wmo.common.routing import KnnRouterPolicy
from wmo.common.tasks import TaskCase, load_task_set
from wmo.optimize.router.activation import load_project_router
from wmo.optimize.router.automatic.replay import AutomaticRouterReplay
from wmo.optimize.router.automatic.service import AutomaticRouterOptions
from wmo.optimize.router.automatic.service_test import (
    _approve_manual_judge,
    _completed_project,
    _CompletionClient,
    _EmbeddingClient,
    _ProviderState,
)
from wmo.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from wmo.simulation.build import ProjectBuild

_RUNNER = CliRunner()
_REVISION = "a" * 40


def _compact_terminal_text(value: str) -> str:
    """Return ANSI-free terminal text without presentation-only whitespace."""
    return "".join(unstyle(value).split())


class _WizardCompletionClient:
    """Return candidate, world, or judge fixture responses from request-visible protocol text."""

    def __init__(
        self,
        alias: str,
        model: ModelSnapshot,
        state: _ProviderState,
    ) -> None:
        """Bind one catalog alias, exact response identity, and shared call counters.

        Args:
            alias: Catalog alias selected for the request.
            model: Exact response model identity.
            state: Shared provider-call counters.
        """
        self._alias = alias
        self._model = model
        self._state = state

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Dispatch to the deterministic fixture protocol matching the visible request.

        Args:
            request: Exact provider-neutral request.

        Returns:
            Schema-valid candidate, world-model, or judge response.
        """
        visible = "\n".join(message.content or "" for message in request.messages)
        if "RESPONSE_SCHEMA:" in visible:
            protocol_alias = "judge"
        elif "Protocol version: text-world-model-v1" in visible:
            protocol_alias = "world"
        else:
            protocol_alias = self._alias
        return _CompletionClient(protocol_alias, self._model, self._state).complete(request)


class _WizardRuntimeCatalog:
    """Resolve recommended aliases through deterministic local provider clients."""

    def __init__(self, catalog: ModelCatalog, state: _ProviderState) -> None:
        """Bind static catalog metadata and shared paid-call counters.

        Args:
            catalog: Current secret-free model catalog.
            state: Shared provider-call counters.
        """
        self._catalog = catalog
        self._static = RuntimeModelCatalog(catalog, environment={})
        self._state = state

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        """Return secret-free static identity and capabilities for one alias.

        Args:
            alias: Current catalog alias.

        Returns:
            Exact model snapshot and verified capabilities.
        """
        return self._static.snapshot(alias)

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        """Create a deterministic provider-shaped runtime while counting resolution.

        Args:
            alias: Current catalog alias.
            role: Optional catalog role the alias serves.

        Returns:
            Exact resolved model with completion and optional embedding clients.
        """
        self._state.credential_resolutions += 1
        snapshot, capabilities = self.snapshot(alias)
        completion = _WizardCompletionClient(alias, snapshot, self._state)
        embedding: _EmbeddingClient | None = None
        if capabilities.supports_embeddings:
            embedding = _EmbeddingClient(self._state)
        return ResolvedModel(alias, snapshot, capabilities, completion, embedding)

    def preflight(
        self,
        alias: str,
        _requirement: object | None = None,
        *,
        role: CatalogRoleName | None = None,
    ) -> ResolvedModel:
        """Resolve one verified alias for the existing build preflight seam.

        Args:
            alias: Current catalog alias.
            _requirement: Capability requirement already represented in the catalog.
            role: Optional catalog role the alias serves.

        Returns:
            Exact resolved model.
        """
        return self.resolve(alias, role=role)

    def with_catalog(self, catalog: ModelCatalog) -> _WizardRuntimeCatalog:
        """Return an equivalent resolver over a candidate-inclusive catalog.

        Args:
            catalog: Confirmed catalog state.

        Returns:
            Resolver sharing the same provider-call counters.
        """
        return _WizardRuntimeCatalog(catalog, self._state)


def _install_integrated_runtime(
    monkeypatch: pytest.MonkeyPatch,
    state: _ProviderState,
) -> _FakeLister:
    """Install deterministic provider listing, runtime, revision, and router bounds.

    Args:
        monkeypatch: Pytest patch fixture replacing network and release seams.
        state: Shared provider-call counters.

    Returns:
        Provider listing fixture used to verify discovery calls.
    """
    lister = _FakeLister()

    def factory(
        catalog: ModelCatalog,
        **_options: object,
    ) -> _WizardRuntimeCatalog:
        """Return a deterministic runtime while accepting static-planner options."""
        return _WizardRuntimeCatalog(catalog, state)

    monkeypatch.setattr("wmo.cli.provider_setup.HttpProviderModelLister", lambda: lister)
    monkeypatch.setattr("wmo.cli.build_cmd.RuntimeModelCatalog", factory)
    monkeypatch.setattr(wizard, "RuntimeModelCatalog", factory)
    monkeypatch.setattr("wmo.cli.build_cmd.can_prompt", lambda _console: True)
    monkeypatch.setattr("wmo.cli.consent.can_prompt", lambda _console: True)
    monkeypatch.setattr("wmo.cli.build_cmd.capture_build_completed", lambda **_kwargs: None)
    monkeypatch.setattr("wmo.cli.build_cmd.installed_release_revision", lambda: _REVISION)
    monkeypatch.setattr(wizard, "installed_release_revision", lambda: _REVISION)

    def options(*, maximum_provider_cost_usd: float = 25.0) -> AutomaticRouterOptions:
        """Return small but production-shaped automatic-router bounds for integration."""
        return AutomaticRouterOptions(
            maximum_provider_cost_usd=maximum_provider_cost_usd,
            maximum_judgments=30,
            maximum_model_calls=1,
            maximum_router_feature_tokens=4_096,
            maximum_retrieval_query_tokens=4_096,
            router_embedding_maximum_attempts=1,
            completion_maximum_attempts=1,
            simulation_maximum_output_tokens=8_000,
        )

    monkeypatch.setattr(wizard, "AutomaticRouterOptions", options)
    return lister


def _catalog() -> ModelCatalog:
    """Return complete secret-free roles and router candidates for wizard tests."""
    completion = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=128_000,
        maximum_output_tokens=16_000,
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
        connections={"provider": ConnectionConfig(provider="openai")},
        models={
            "world": ModelRecord(connection="provider", model="world", capabilities=completion),
            "judge": ModelRecord(connection="provider", model="judge", capabilities=completion),
            "candidate": ModelRecord(
                connection="provider", model="candidate", capabilities=completion
            ),
            "embedder": ModelRecord(
                connection="provider", model="embedder", capabilities=embedding
            ),
        },
        roles=ModelRoles(
            world_model="world",
            judge="judge",
            embedder="embedder",
            candidates=("world", "candidate"),
            incumbent="world",
        ),
    )


def _plan(catalog: ModelCatalog) -> wizard.WizardBuildPlan:
    """Return one completed grounded-build checkpoint without provider work."""
    return wizard.WizardBuildPlan(
        trace_path=None,
        source="otlp",
        catalog=catalog,
        selected=ProjectModelConfiguration(
            world_model="world",
            judge="judge",
            embedder="embedder",
        ),
        tasks=_task_cases(70),
        completed=None,
        accepted_traces=280,
        invalid_traces=0,
        fit_tasks=50,
        held_out_tasks=20,
        build_estimate_usd=0.0,
        build_reused=True,
    )


def _task_cases(count: int) -> tuple[TaskCase, ...]:
    """Return deterministic wizard-plan tasks for boundary tests.

    Args:
        count: Number of unique task and lineage contracts.

    Returns:
        Mixed fit and held-out tasks with exact source references.
    """
    return tuple(
        TaskCase(
            task_id=f"task-{index}",
            lineage_group_id=f"lineage-{index}",
            partition="fit" if index < max(1, count - 20) else "held_out",
            instruction=f"Resolve request {index}",
            workload_weight=1.0,
            source_trace_ids=(f"trace-{index}",),
        )
        for index in range(count)
    )


def _default_trace_corpus(tmp_path: Path) -> Path:
    """Write a multi-lineage corpus under a likely local trace-export file name.

    Args:
        tmp_path: Isolated current directory receiving the trace corpus.

    Returns:
        The one local OTLP JSONL path named at the explicit trace prompt.
    """
    generated = _otlp_export(tmp_path, count=12)
    default = tmp_path / "traces.otel.jsonl"
    default.write_text(
        generated.read_text(encoding="utf-8").replace("gpt-test", "gpt-5.6-luna"),
        encoding="utf-8",
    )
    generated.unlink()
    return default


def test_fresh_bare_wizard_recommends_builds_and_composes_provisional_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The minimal transcript completes every artifact through the shared router service.

    Args:
        tmp_path: Isolated current directory and WMO root.
        monkeypatch: Pytest patch fixture installing deterministic provider seams.
    """
    _default_trace_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    state = _ProviderState()
    lister = _install_integrated_runtime(monkeypatch, state)
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(4.0, root)

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--root", str(root), "--provider", "openai"],
        input="\ntraces.otel.jsonl\n\ny\n",
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )

    assert result.exit_code == 0, result.output
    printed = unstyle(result.output)
    assert lister.requests == ["openai"]
    assert "Select the providers you want to use" not in printed
    assert printed.count("Workflow") == 1
    assert printed.index("judge rubric") < printed.index("Trace path")
    assert printed.count("Trace path (otlp export)") == 1
    assert printed.count("Use these recommended models?") == 1
    assert printed.count("Authorize wmo build support") == 1
    assert "Judge syllabus" in printed
    assert "Human calibration is optional" not in printed
    assert "next            wmo run support" in printed
    assert "optional        wmo config judge calibrate support" in printed
    for label in (
        "serving RAG",
        "fit RAG",
        "simulation",
        "syllabus",
        "calibration",
        "router",
        "report",
    ):
        assert label in printed
    assert "openai-secret" not in printed
    assert "openai-secret" not in (root / "models.toml").read_text(encoding="utf-8")
    store = wizard.ProjectStore(root, "support")
    selected = store.load_project().build
    assert selected is not None
    artifact_types = {
        store.artifacts.read(artifact_id).manifest.artifact_type
        for artifact_id in store.artifacts.list_ids()
    }
    assert "trace-rag-index" in artifact_types
    assert "grounded-world-model" in artifact_types
    assert "router-policy" in artifact_types
    assert "router-report" in artifact_types
    policy_ids = tuple(
        artifact_id
        for artifact_id in store.artifacts.list_ids()
        if store.artifacts.read(artifact_id).manifest.artifact_type == "router-policy"
    )
    assert len(policy_ids) == 1
    policy = KnnRouterPolicy.model_validate_json(
        store.artifacts.read_bytes(policy_ids[0], "policy.json")
    )
    assert policy.judgment_status == "provisional"
    assert state.embedding_calls
    assert state.completion_calls
    runtime = load_project_router(
        "support",
        root,
        policy_id=policy.policy_id,
        runtime_catalog=cast(
            RuntimeModelCatalog,
            _WizardRuntimeCatalog(wizard.load_model_catalog(root / "models.toml"), state),
        ),
    )
    assert runtime.policy.judgment_status == "provisional"
    provisional_policy_bytes = store.artifacts.read_bytes(policy.policy_id, "policy.json")
    served: list[tuple[str, int]] = []
    with monkeypatch.context() as run_patches:
        run_patches.setattr(
            "wmo.optimize.router.activation.load_project_router",
            lambda *_args, **_kwargs: runtime,
        )
        run_patches.setattr(
            "uvicorn.run",
            lambda _app, *, host, port: served.append((host, port)),
        )
        run_result = _RUNNER.invoke(
            app,
            ["run", "support", "--root", str(root), "--port", "8123"],
        )
    assert run_result.exit_code == 0, run_result.output
    assert "provisional judgment" in run_result.output
    assert "http://127.0.0.1:8123/v1" in run_result.output
    assert served == [("127.0.0.1", 8123)]

    current_catalog = wizard.load_model_catalog(root / "models.toml")
    _approve_manual_judge(
        store,
        current_catalog,
        state,
        runtime_catalog=cast(
            RuntimeModelCatalog,
            _WizardRuntimeCatalog(current_catalog, state),
        ),
    )
    successor = _RUNNER.invoke(
        app,
        ["build", "support", "--root", str(root)],
        input="\ny\n",
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )
    assert successor.exit_code == 0, successor.output
    successor_policies = tuple(
        KnnRouterPolicy.model_validate_json(store.artifacts.read_bytes(item, "policy.json"))
        for item in store.artifacts.list_ids()
        if store.artifacts.read(item).manifest.artifact_type == "router-policy"
    )
    assert {item.judgment_status for item in successor_policies} == {
        "provisional",
        "human_calibrated",
    }
    assert store.artifacts.read_bytes(policy.policy_id, "policy.json") == provisional_policy_bytes
    approved_policy = next(
        item for item in successor_policies if item.judgment_status == "human_calibrated"
    )
    approved_runtime = load_project_router(
        "support",
        root,
        policy_id=approved_policy.policy_id,
        runtime_catalog=cast(
            RuntimeModelCatalog,
            _WizardRuntimeCatalog(wizard.load_model_catalog(root / "models.toml"), state),
        ),
    )
    assert approved_runtime.policy.judgment_status == "human_calibrated"
    completed_credentials = state.credential_resolutions
    completed_embeddings = tuple(state.embedding_calls)
    completed_completions = tuple(state.completion_calls)

    def unexpected_replan(*_args: object, **_kwargs: object) -> None:
        """Fail if immutable replay consults current pricing or cost planning."""
        raise AssertionError("completed replay entered current cost planning")

    monkeypatch.setattr(wizard, "_wizard_cost_plan", unexpected_replan)
    replay = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--root",
            str(root),
            "--max-router-cost-usd",
            "0.01",
        ],
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )
    assert replay.exit_code == 0, replay.output
    assert "reused every verified project artifact" in unstyle(replay.output)
    assert "next    wmo run support" in unstyle(replay.output)
    assert "wmo config judge calibrate support" not in unstyle(replay.output)
    assert lister.requests == ["openai"]
    assert state.credential_resolutions == completed_credentials
    assert tuple(state.embedding_calls) == completed_embeddings
    assert tuple(state.completion_calls) == completed_completions


def test_fresh_wizard_refusal_after_discovery_makes_no_paid_calls_or_selected_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model discovery may run, but refusal leaves paid inference and selection untouched.

    Args:
        tmp_path: Isolated current directory and WMO root.
        monkeypatch: Pytest patch fixture installing deterministic provider seams.
    """
    _default_trace_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    state = _ProviderState()
    lister = _install_integrated_runtime(monkeypatch, state)
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(4.0, root)

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--root", str(root)],
        input="\ntraces.otel.jsonl\n1\n\n\nn\n",
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )

    assert result.exit_code == 0, result.output
    printed = unstyle(result.output)
    assert lister.requests == ["openai"]
    assert printed.count("Authorize wmo build support") == 1
    assert "Stopped before paid build or router work" in printed
    assert "wizard" not in printed.casefold()
    assert "Plan" not in printed
    assert "inspect" not in printed
    assert "Existing catalog already covers" not in printed
    assert state.credential_resolutions == 0
    assert state.embedding_calls == []
    assert state.completion_calls == []
    store = wizard.ProjectStore(root, "support")
    assert store.load_project().build is None


def test_explicit_router_cap_below_required_fails_before_consent_or_paid_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inadmissible explicit cap exits nonzero after discovery but before paid work.

    Args:
        tmp_path: Isolated current directory and WMO root.
        monkeypatch: Pytest patch fixture installing deterministic provider seams.
    """
    _default_trace_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    state = _ProviderState()
    _install_integrated_runtime(monkeypatch, state)
    root = tmp_path / ".wmo"

    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--root",
            str(root),
            "--max-router-cost-usd",
            "0.01",
        ],
        input="\ntraces.otel.jsonl\n1\n\n\n",
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )

    assert result.exit_code == 2
    printed = unstyle(result.output)
    flat = " ".join(printed.replace("│", " ").split())
    assert "router cap $0.01 is below the exact required" in flat
    assert "increase --max-router-cost-usd or omit it" in flat
    assert "Authorize wmo build support" not in printed
    assert state.credential_resolutions == 0
    assert state.embedding_calls == []
    assert state.completion_calls == []
    assert wizard.ProjectStore(root, "support").load_project().build is None


def test_explicit_router_cap_above_required_consents_only_to_exact_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A higher admissibility cap does not become consented spend headroom.

    Args:
        tmp_path: Isolated current directory and WMO root.
        monkeypatch: Pytest patch fixture installing deterministic provider seams.
    """
    _default_trace_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    state = _ProviderState()
    _install_integrated_runtime(monkeypatch, state)
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(4.0, root)
    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--root",
            str(root),
            "--max-router-cost-usd",
            "5000",
        ],
        input="\ntraces.otel.jsonl\n1\n\n\ny\n",
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )

    assert result.exit_code == 0, result.output
    printed = unstyle(result.output)
    assert "$5000.00" not in printed
    assert "$5005.00" not in printed
    assert "Authorize wmo build support" in printed


def test_explicit_and_wizard_paths_select_the_same_grounded_build_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both CLI paths execute the same typed grounded-build service boundary.

    Args:
        tmp_path: Isolated current directory and shared WMO root.
        monkeypatch: Pytest patch fixture installing deterministic provider seams.
    """
    traces = _default_trace_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    state = _ProviderState()
    _install_integrated_runtime(monkeypatch, state)
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(100.0, root)
    completed_projects: list[str] = []
    complete_grounded_build = build_command._complete_grounded_build

    def record_grounded_build(
        store: wizard.ProjectStore,
        completed: ProjectBuild,
        *,
        selected: ProjectModelConfiguration,
        runtime_catalog: RuntimeModelCatalog,
        world_snapshot: ModelSnapshot,
        embedder_snapshot: ModelSnapshot,
        top_k: int,
        estimate: float,
        maximum_build_cost_usd: float,
        provider_spend_authorized: bool,
        progress: build_command.ProgressHook | None = None,
    ) -> build_command.GroundedBuildCompletion:
        """Record both adapters using the shared typed execution seam.

        Args:
            store: Project store passed by the explicit or guided adapter.
            completed: Deterministic trace and task evidence.
            selected: Exact model role aliases.
            runtime_catalog: Provider runtime resolver.
            world_snapshot: Static world-model identity.
            embedder_snapshot: Static embedder identity.
            top_k: Frozen retrieval result count.
            estimate: Conservative embedding estimate.
            maximum_build_cost_usd: Strict embedding ceiling.
            provider_spend_authorized: Whether paid embedding work is authorized.
            progress: Optional stage observer forwarded unchanged.

        Returns:
            The real typed grounded-build completion.
        """
        completed_projects.append(store.paths.project_id)
        return complete_grounded_build(
            store,
            completed,
            selected=selected,
            runtime_catalog=runtime_catalog,
            world_snapshot=world_snapshot,
            embedder_snapshot=embedder_snapshot,
            top_k=top_k,
            estimate=estimate,
            maximum_build_cost_usd=maximum_build_cost_usd,
            provider_spend_authorized=provider_spend_authorized,
            progress=progress,
        )

    monkeypatch.setattr(build_command, "_complete_grounded_build", record_grounded_build)

    explicit = _RUNNER.invoke(
        app,
        ["build", "explicit", "-t", str(traces), "--root", str(root)],
        input="1\n\n1\n\n1\n\n1\n1,2\n\n\n\n1\ny\n",
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )
    assert explicit.exit_code == 0, explicit.output

    guided = _RUNNER.invoke(
        app,
        ["build", "guided", "--root", str(root)],
        input=f"\n{traces}\ny\n",
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )
    assert guided.exit_code == 0, guided.output
    assert completed_projects == ["explicit", "guided"]

    explicit_store = wizard.ProjectStore(root, "explicit")
    guided_store = wizard.ProjectStore(root, "guided")
    explicit_build = explicit_store.load_project().build
    guided_build = guided_store.load_project().build
    assert explicit_build is not None
    assert guided_build is not None
    explicit_traces = wizard.load_trace_dataset(
        explicit_store.artifacts,
        explicit_build.trace_dataset.artifact_id,
    )
    guided_traces = wizard.load_trace_dataset(
        guided_store.artifacts,
        guided_build.trace_dataset.artifact_id,
    )
    explicit_tasks = load_task_set(
        explicit_store.artifacts,
        explicit_build.task_set.artifact_id,
    )
    guided_tasks = load_task_set(
        guided_store.artifacts,
        guided_build.task_set.artifact_id,
    )
    assert explicit_traces.traces == guided_traces.traces
    assert explicit_tasks.tasks == guided_tasks.tasks
    assert explicit_store.load_project().models == guided_store.load_project().models
    for explicit_pointer, guided_pointer in (
        (explicit_build.serving_rag, guided_build.serving_rag),
        (explicit_build.fit_rag, guided_build.fit_rag),
    ):
        explicit_index = build_command.load_rag_index(
            explicit_store.artifacts,
            explicit_pointer.artifact_id,
        ).index
        guided_index = build_command.load_rag_index(
            guided_store.artifacts,
            guided_pointer.artifact_id,
        ).index
        semantic_exclusions = {
            "rag_id": True,
            "created_at": True,
            "inputs": True,
            "sources": {"__all__": {"artifact_input"}},
        }
        assert explicit_index.model_dump(
            mode="json",
            exclude=semantic_exclusions,
        ) == guided_index.model_dump(mode="json", exclude=semantic_exclusions)
        assert explicit_index.created_at != guided_index.created_at
        assert explicit_index.sources[0].artifact_input != guided_index.sources[0].artifact_input


@pytest.mark.parametrize("boundary", ["after_build", "after_calibration"])
def test_interrupted_wizard_resumes_durable_stages_without_duplicate_build_calls(
    boundary: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable build or calibration checkpoint resumes without rebuilding RAG.

    Args:
        boundary: Durable wizard boundary after which the first run is interrupted.
        tmp_path: Isolated current directory and WMO root.
        monkeypatch: Pytest patch fixture installing deterministic provider seams.
    """
    _default_trace_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    state = _ProviderState()
    _install_integrated_runtime(monkeypatch, state)
    root = tmp_path / ".wmo"
    set_maximum_command_cost_usd(100.0, root)
    original_calibration = wizard._ensure_judge_calibration
    original_preflight = wizard.preflight_automatic_router

    def stop_after_build(*_args: object, **_kwargs: object) -> None:
        """Interrupt immediately after the selected grounded build is durable."""
        raise RuntimeError("interrupt after build")

    def stop_after_calibration(*_args: object, **_kwargs: object) -> None:
        """Interrupt after provisional calibration but before router provider work."""
        raise RuntimeError("interrupt after calibration")

    if boundary == "after_build":
        monkeypatch.setattr(wizard, "_ensure_judge_calibration", stop_after_build)
    else:
        monkeypatch.setattr(wizard, "preflight_automatic_router", stop_after_calibration)
    first = _RUNNER.invoke(
        app,
        ["build", "support", "--root", str(root)],
        input="\ntraces.otel.jsonl\n1\n\n\ny\n",
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )
    assert first.exit_code == 1
    assert boundary.replace("_", " ") in str(first.exception)
    assert state.completion_calls == []
    build_embeddings = tuple(state.embedding_calls)
    assert len(build_embeddings) == 2
    monkeypatch.setattr(wizard, "_ensure_judge_calibration", original_calibration)
    monkeypatch.setattr(wizard, "preflight_automatic_router", original_preflight)

    resumed = _RUNNER.invoke(
        app,
        ["build", "support", "--root", str(root)],
        input="\ny\n",
        env={"OPENAI_API_KEY": "openai-secret", "WMO_RELEASE_REVISION": _REVISION},
    )

    assert resumed.exit_code == 0, resumed.output
    assert all(state.embedding_calls.count(item) == 1 for item in build_embeddings)


def test_approved_calibration_resume_builds_human_calibrated_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approved judge resumes through the shared service without provisional bootstrap.

    Args:
        tmp_path: Isolated completed project and WMO root.
        monkeypatch: Pytest patch fixture installing deterministic provider seams.
    """
    store, catalog, state = _completed_project(tmp_path)
    _approve_manual_judge(store, catalog, state)
    _install_integrated_runtime(monkeypatch, state)
    config = store.load_project()
    selected = config.models
    completed = config.build
    assert selected is not None
    assert completed is not None
    existing = wizard.WizardBuildPlan(
        trace_path=None,
        source="otlp",
        catalog=catalog,
        selected=selected,
        tasks=load_task_set(
            store.artifacts,
            completed.task_set.artifact_id,
        ).tasks,
        completed=None,
        accepted_traces=12,
        invalid_traces=0,
        fit_tasks=2,
        held_out_tasks=1,
        build_estimate_usd=0.0,
        build_reused=True,
    )
    monkeypatch.setattr(wizard, "_completed_build_plan", lambda *_args, **_kwargs: existing)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        """Fail if an approved calibration is replaced by provisional evidence."""
        raise AssertionError("approved calibration was re-bootstrapped")

    monkeypatch.setattr(wizard, "bootstrap_provisional_judge", unexpected)
    result = _RUNNER.invoke(
        app,
        ["build", "support", "--root", str(store.paths.root)],
        input="\ny\n",
        env={"FIXTURE_API_KEY": "fixture-secret", "WMO_RELEASE_REVISION": _REVISION},
    )

    assert result.exit_code == 0, result.output
    assert "human_calibrated" in unstyle(result.output)
    policies = []
    for artifact_id in store.artifacts.list_ids():
        if store.artifacts.read(artifact_id).manifest.artifact_type != "router-policy":
            continue
        policies.append(
            KnnRouterPolicy.model_validate_json(
                store.artifacts.read_bytes(artifact_id, "policy.json")
            )
        )
    assert len(policies) == 1
    assert policies[0].judgment_status == "human_calibrated"


def test_bare_build_dispatches_wizard_without_required_trace_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare project syntax reaches the wizard while automation keeps explicit traces.

    Args:
        monkeypatch: Pytest patch fixture replacing terminal and wizard boundaries.
    """
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr("wmo.cli.build_cmd.can_prompt", lambda _console: True)
    monkeypatch.setattr(
        wizard,
        "run_build_wizard",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = _RUNNER.invoke(app, ["build", "support"])

    assert result.exit_code == 0, result.output
    assert calls and calls[0][0] == ("support",)
    help_result = _RUNNER.invoke(app, ["build", "--help"])
    assert help_result.exit_code == 0
    assert "-t" in unstyle(help_result.output)
    help_text = unstyle(help_result.output)
    assert "interactive" in help_text
    assert "wizard" not in help_text.casefold()


def test_invalid_traces_fail_before_provider_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace validation precedes credentials, provider listing, and catalog mutation.

    Args:
        tmp_path: Isolated invalid trace and WMO root.
        monkeypatch: Pytest patch fixture rejecting any provider setup attempt.
    """
    source = tmp_path / "traces.otel.jsonl"
    source.write_text("{}\n")
    monkeypatch.setattr(
        "wmo.cli.build_cmd._load_canonical_traces",
        lambda *_args, **_kwargs: SimpleNamespace(traces=()),
    )

    def unexpected(*_args: object, **_kwargs: object) -> None:
        """Fail if invalid traces reach authenticated provider discovery."""
        raise AssertionError("provider setup ran before trace validation")

    monkeypatch.setattr("wmo.cli.provider_setup.run_provider_setup", unexpected)

    with pytest.raises(ValueError, match="no valid canonical traces"):
        wizard._prepare_new_build(
            "support",
            trace_path=source,
            source="otlp",
            root=tmp_path / ".wmo",
            world_model=None,
            judge=None,
            embedder=None,
            top_k=5,
            maximum_build_cost_usd=5.0,
            code_revision="a" * 40,
            providers=(),
            console=Console(file=StringIO(), force_terminal=False),
        )


def test_existing_catalog_router_defaults_use_ranking_diversity_and_world_incumbent(
    tmp_path: Path,
) -> None:
    """Missing router roles reuse ranked IDs and prefer a diverse verified alternative.

    Args:
        tmp_path: Isolated catalog root receiving selected defaults.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    completion = ModelCapabilities(
        supports_completions=True,
        supports_tools=True,
        supports_structured_output=True,
        context_window_tokens=1_000_000,
        maximum_output_tokens=128_000,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=6.0,
        cached_input_cost_per_million_tokens_usd=0.1,
        cache_write_cost_per_million_tokens_usd=1.25,
    )
    embedding = ModelCapabilities(
        supports_embeddings=True,
        input_cost_per_million_tokens_usd=0.13,
    )
    catalog = ModelCatalog(
        connections={
            "openai": ConnectionConfig(provider="openai"),
            "anthropic": ConnectionConfig(provider="anthropic"),
        },
        models={
            "luna": ModelRecord(connection="openai", model="gpt-5.6-luna", capabilities=completion),
            "terra": ModelRecord(
                connection="openai", model="gpt-5.6-terra", capabilities=completion
            ),
            "sonnet": ModelRecord(
                connection="anthropic", model="claude-sonnet-5", capabilities=completion
            ),
            "embed": ModelRecord(
                connection="openai",
                model="text-embedding-3-large",
                capabilities=embedding,
            ),
        },
        roles=ModelRoles(world_model="terra", judge="luna", embedder="embed"),
    )
    write_model_catalog(root / "models.toml", catalog)

    selected = wizard._ensure_router_defaults(
        root,
        catalog,
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert selected.roles.candidates == ("terra", "sonnet")
    assert selected.roles.incumbent == "terra"


@pytest.mark.parametrize("selected", ["approved_calibration", "provisional_calibration"])
def test_existing_calibration_resume_never_bootstraps(
    selected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approved and provisional successors resume without replacing calibration evidence.

    Args:
        selected: Existing calibration pointer kind returned by review state.
        tmp_path: Isolated project store.
        monkeypatch: Pytest patch fixture replacing judge persistence boundaries.
    """
    store = wizard.ProjectStore(tmp_path, "support")
    calls: list[str] = []
    monkeypatch.setattr(wizard, "prepare_manual_judge_setup", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        wizard,
        "commit_manual_judge_setup",
        lambda *_args, **_kwargs: calls.append("setup"),
    )
    state = SimpleNamespace(
        approved_calibration=object() if selected == "approved_calibration" else None,
        provisional_calibration=object() if selected == "provisional_calibration" else None,
        audit=object() if selected == "approved_calibration" else None,
    )
    monkeypatch.setattr(wizard, "read_review_state", lambda _store: state)
    monkeypatch.setattr(
        wizard,
        "bootstrap_provisional_judge",
        lambda *_args, **_kwargs: calls.append("bootstrap"),
    )

    wizard._ensure_judge_calibration(
        store,
        _catalog(),
        created_at=wizard.datetime.now(wizard.UTC),
        code_revision="a" * 40,
    )

    assert calls == []


def test_refused_named_consent_stops_before_paid_provider_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one exact bounded refusal performs no paid build, calibration, or router work.

    Args:
        tmp_path: Isolated WMO root.
        monkeypatch: Pytest patch fixture replacing state-machine boundaries.
    """
    catalog = _catalog()
    authorizations: list[tuple[str, float]] = []
    provider_stages: list[str] = []
    monkeypatch.setattr(wizard, "_completed_replay", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        screens.Prompt,
        "ask",
        lambda *_args, **kwargs: cast(str, kwargs.get("default", "")),
    )
    monkeypatch.setattr(wizard, "_completed_build_plan", lambda *_args, **_kwargs: _plan(catalog))
    monkeypatch.setattr(wizard, "_ensure_router_defaults", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(
        wizard,
        "_wizard_observed_candidate_aliases",
        lambda *_args: ("world",) * 5 + ("candidate",) * 5,
    )
    monkeypatch.setattr(
        wizard,
        "_wizard_simulation_input_estimate",
        lambda *_args, **_kwargs: 32_768,
    )

    def refuse(*_args: object, **kwargs: object) -> bool:
        """Capture the exact named command and ceiling, then refuse authorization."""
        authorizations.append(
            (
                cast(str, kwargs["command"]),
                cast(float, kwargs["estimated_cost_usd"]),
            )
        )
        return False

    monkeypatch.setattr(wizard, "require_spend_consent", refuse)
    monkeypatch.setattr(
        "wmo.cli.build_cmd.build",
        lambda *_args, **_kwargs: provider_stages.append("build"),
    )
    monkeypatch.setattr(
        wizard,
        "optimize_project_router",
        lambda *_args, **_kwargs: provider_stages.append("router"),
    )

    wizard.run_build_wizard(
        "support",
        source="otlp",
        root=tmp_path,
        world_model=None,
        judge=None,
        embedder=None,
        top_k=5,
        maximum_build_cost_usd=5.0,
        maximum_router_cost_usd=None,
        providers=(),
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert len(authorizations) == 1
    assert authorizations[0][0] == "wmo build support"
    assert authorizations[0][1] > 0
    assert provider_stages == []
    assert not (tmp_path / "projects").exists()


def test_completed_replay_skips_every_prompt_and_provider_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact completed replay prints identities with zero prompts or provider work.

    Args:
        tmp_path: Isolated WMO root.
        monkeypatch: Pytest patch fixture forbidding all incomplete-stage boundaries.
    """
    replay = AutomaticRouterReplay(
        policy_id="policy-a",
        report_id="report-a",
        execution_contract_id="execution-a",
        policy_lock_id="lock-a",
        judgment_status="provisional",
    )
    monkeypatch.setattr(wizard, "_completed_replay", lambda *_args, **_kwargs: replay)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        """Fail if replay enters any interactive or provider-capable stage."""
        raise AssertionError("completed replay entered an incomplete stage")

    catalog = _catalog()
    monkeypatch.setattr(screens.Prompt, "ask", unexpected)
    monkeypatch.setattr(wizard, "_completed_build_plan", lambda *_args, **_kwargs: _plan(catalog))
    monkeypatch.setattr(wizard, "_ensure_router_defaults", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(
        wizard,
        "_wizard_observed_candidate_aliases",
        lambda *_args: ("world",) * 5 + ("candidate",) * 5,
    )
    output = StringIO()

    wizard.run_build_wizard(
        "support",
        source="otlp",
        root=tmp_path,
        world_model=None,
        judge=None,
        embedder=None,
        top_k=5,
        maximum_build_cost_usd=5.0,
        maximum_router_cost_usd=None,
        providers=(),
        console=Console(file=output, force_terminal=False),
    )

    printed = unstyle(output.getvalue())
    assert "reused every verified project artifact" in printed
    assert "policy-a" in printed
    assert "report-a" in printed
    assert "next    wmo run support" in printed
    assert "wmo config judge calibrate support" in printed
    assert "after approval wmo build support" in printed


def test_completed_replay_rejects_role_override_before_replay_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conflicting role flag cannot be silently ignored by zero-I/O replay.

    Args:
        tmp_path: Isolated completed project and WMO root.
        monkeypatch: Pytest patch fixture rejecting entry into immutable replay.
    """
    store, _catalog_value, _state = _completed_project(tmp_path)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        """Fail if a mismatched override reaches immutable replay verification."""
        raise AssertionError("role mismatch reached replay")

    monkeypatch.setattr("wmo.cli.build_cmd.can_prompt", lambda _console: True)
    monkeypatch.setattr(wizard, "_completed_replay", unexpected)
    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--root",
            str(store.paths.root),
            "--world-model",
            "different",
        ],
        env={"WMO_RELEASE_REVISION": _REVISION},
    )

    assert result.exit_code == 2
    assert "world_model='different' (selected 'world')" in unstyle(result.output)
