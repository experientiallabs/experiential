"""End-to-end tests for the named grounded-world-model build command."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import pytest
from click import unstyle
from pydantic import JsonValue
from rich.console import Console
from typer.testing import CliRunner

import wmo.cli.build_cmd as build_command
import wmo.cli.consent as consent_module
import wmo.simulation.build as simulation_build
from wmo.cli.app import app
from wmo.cli.build_wizard import _prepare_new_build
from wmo.cli.provider_setup_test import _FakeLister as _SetupLister
from wmo.common.config.settings import set_maximum_command_cost_usd
from wmo.common.core.artifacts import sha256_json
from wmo.common.models import (
    ConnectionConfig,
    Embedding,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRequest,
    ModelResponse,
    ModelRoles,
    ModelSnapshot,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.project import ArtifactCorruptionError, ProjectStore, ProjectStoreError
from wmo.common.traces import load_trace_dataset
from wmo.runtime.models import ResolvedModel
from wmo.simulation.ingest.dataset import read_trace_model_identity_evidence
from wmo.simulation.retrieval import load_rag_index
from wmo.simulation.world_model import GroundedWorldModelArtifact

_RUNNER = CliRunner()
_RESOLVE_CALLS: list[str] = []


def test_malformed_release_revision_fails_before_build_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed producer identity before catalog, project, or artifact writes.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped invalid release revision override.
    """
    root = tmp_path / ".wmo"
    monkeypatch.setenv("WMO_RELEASE_REVISION", "HEAD")

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(tmp_path / "missing.jsonl"), "--root", str(root)],
    )

    assert result.exit_code == 2
    assert "full lowercase 40-hex" in result.output
    assert not root.exists()


def _attribute(key: str, value: str) -> dict[str, object]:
    """Encode one textual OpenTelemetry attribute for a canonical fixture.

    Args:
        key: Semantic-convention attribute key.
        value: Text stored in the OTLP value envelope.

    Returns:
        Serialized OTLP attribute mapping.
    """
    return {"key": key, "value": {"stringValue": value}}


def _otlp_export(tmp_path: Path, count: int = 1) -> Path:
    """Write real two-turn traces with one observed assistant-to-user transition each.

    Args:
        tmp_path: Temporary directory receiving the trace export.
        count: Number of independent trace lineages to write.

    Returns:
        Path to the completed OTLP JSON export.
    """
    records = []
    for index in range(count):
        trace_id = f"{index + 1:032x}"
        base = 1_760_000_000_000_000_000 + index * 10_000_000_000
        common = [
            _attribute("gen_ai.operation.name", "chat"),
            _attribute("gen_ai.provider.name", "openai"),
            _attribute("gen_ai.request.model", "gpt-test"),
            _attribute("wmo.customer.id", f"customer-{index}"),
            _attribute("wmo.conversation.id", f"conversation-{index}"),
        ]
        records.extend(
            (
                {
                    "traceId": trace_id,
                    "spanId": f"{index * 2 + 1:016x}",
                    "name": "agent.model_call",
                    "startTimeUnixNano": str(base),
                    "endTimeUnixNano": str(base + 1_000_000_000),
                    "attributes": common
                    + [
                        _attribute(
                            "gen_ai.input.messages",
                            json.dumps([{"role": "user", "content": f"Support request {index}"}]),
                        ),
                        _attribute(
                            "gen_ai.output.messages",
                            json.dumps([{"role": "assistant", "content": "What account email?"}]),
                        ),
                    ],
                },
                {
                    "traceId": trace_id,
                    "spanId": f"{index * 2 + 2:016x}",
                    "name": "agent.model_call",
                    "startTimeUnixNano": str(base + 2_000_000_000),
                    "endTimeUnixNano": str(base + 3_000_000_000),
                    "attributes": common
                    + [
                        _attribute(
                            "gen_ai.input.messages",
                            json.dumps(
                                [
                                    {"role": "assistant", "content": "What account email?"},
                                    {"role": "user", "content": f"customer-{index}@example.test"},
                                ]
                            ),
                        ),
                        _attribute(
                            "gen_ai.output.messages",
                            json.dumps(
                                [{"role": "assistant", "content": "Reset instructions sent."}]
                            ),
                        ),
                    ],
                },
            )
        )
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


class _EmbeddingClient:
    """Deterministic semantic-shaped client for no-network build tests."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return stable distinct unit vectors for each input text.

        Args:
            texts: Canonical RAG key texts.

        Returns:
            Deterministic fixture embeddings in input order.
        """
        results = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [float(value + 1) for value in digest[:8]]
            norm = math.sqrt(sum(value * value for value in raw))
            results.append(Embedding(values=tuple(value / norm for value in raw)))
        return tuple(results)


class _CompletionClient:
    """Unused completion client proving build does not call the world model or judge."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Fail if deterministic build dispatches a completion.

        Args:
            request: Unexpected provider completion request.

        Raises:
            AssertionError: Always, because build must not call completion models.
        """
        raise AssertionError(f"build must not dispatch completion: {request}")


class _RuntimeCatalog:
    """Resolve catalog aliases to deterministic no-network test clients."""

    def __init__(self, catalog: ModelCatalog) -> None:
        """Create deterministic embedding and completion clients.

        Args:
            catalog: Parsed fixture catalog used for exact static identities.
        """
        self._catalog = catalog
        self._embedding = _EmbeddingClient()
        self._completion = _CompletionClient()

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        """Return the fixture alias identity without constructing a provider client.

        Args:
            alias: Configured fixture model alias.

        Returns:
            Secret-free model snapshot and static capabilities.
        """
        record = self._catalog.models[alias]
        connection = self._catalog.connections[record.connection]
        capabilities = record.capabilities or ModelCapabilities()
        return (
            ModelSnapshot(
                provider=connection.provider,
                model_id=record.model,
                revision=record.revision,
                capabilities_sha256=sha256_json(capabilities),
                connection_sha256=connection.identity_sha256(),
            ),
            capabilities,
        )

    def preflight(self, alias: str, _requirement: object | None = None) -> ResolvedModel:
        """Return exact static identities with alias-specific capabilities.

        Args:
            alias: Configured fixture model alias.
            _requirement: Unused requirement accepted by the runtime seam.

        Returns:
            Deterministic resolved fixture model.
        """
        _RESOLVE_CALLS.append(alias)
        snapshot, capabilities = self.snapshot(alias)
        embedding = self._embedding if capabilities.supports_embeddings else None
        return ResolvedModel(alias, snapshot, capabilities, self._completion, embedding)

    def resolve(self, alias: str) -> ResolvedModel:
        """Reuse local preflight for roles with no extra capability requirement.

        Args:
            alias: Configured fixture model alias.

        Returns:
            Deterministic resolved fixture model.
        """
        _RESOLVE_CALLS.append(alias)
        snapshot, capabilities = self.snapshot(alias)
        embedding = self._embedding if capabilities.supports_embeddings else None
        return ResolvedModel(alias, snapshot, capabilities, self._completion, embedding)


def _catalog(root: Path, *, embedder_input_usd_per_million: float = 0.0) -> None:
    """Write complete secret-free build roles while leaving router candidates empty.

    Args:
        root: Temporary WMO root receiving ``models.toml``.
        embedder_input_usd_per_million: Explicit fixture embedding input price.
    """
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "fixture": ConnectionConfig(provider="openai", api_key_env="FIXTURE_API_KEY")
            },
            models={
                "world": ModelRecord(
                    connection="fixture",
                    model="world-id",
                    capabilities=ModelCapabilities(maximum_output_tokens=16_000),
                ),
                "judge": ModelRecord(connection="fixture", model="judge-id"),
                "embed": ModelRecord(
                    connection="fixture",
                    model="embed-id",
                    capabilities=ModelCapabilities(
                        supports_embeddings=True,
                        input_cost_per_million_tokens_usd=embedder_input_usd_per_million,
                    ),
                ),
            },
            roles=ModelRoles(world_model="world", judge="judge", embedder="embed"),
        ),
    )


@pytest.fixture(autouse=True)
def _fake_runtime_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep build tests local and prove no provider network is necessary.

    Args:
        monkeypatch: Pytest patch fixture replacing provider-backed seams.
    """
    monkeypatch.setattr("wmo.cli.build_cmd.RuntimeModelCatalog", _RuntimeCatalog)
    monkeypatch.setattr("wmo.cli.build_cmd.capture_build_completed", lambda **_kwargs: None)
    _RESOLVE_CALLS.clear()


def test_first_build_provider_flags_skip_the_opening_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exact --provider values skip the opening list.

    Args:
        monkeypatch: Pytest patch fixture supplying a terminal, credential, and listing seam.
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    lister = _SetupLister()
    monkeypatch.setattr("wmo.cli.build_cmd.can_prompt", lambda _console: True)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setattr("wmo.cli.provider_setup.HttpProviderModelLister", lambda: lister)

    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--traces",
            str(source),
            "--root",
            str(root),
            "--provider",
            "openai",
        ],
        input="1,3\n\n1\n1\n1\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert lister.requests == ["openai"]
    printed = unstyle(result.output)
    assert "Select the providers you want to use" not in printed
    saved = load_model_catalog(root / "models.toml")
    assert saved.roles.world_model == "gpt-5-6-luna"

    replay = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
    )

    assert replay.exit_code == 0, replay.output
    assert lister.requests == ["openai"]
    assert "Select the providers you want to use" not in unstyle(replay.output)
    assert "Model setup is required" not in unstyle(replay.output)


def test_first_build_rejects_bad_provider_flags_before_any_write(tmp_path: Path) -> None:
    """Unsupported or duplicate --provider values fail before project or catalog writes.

    Args:
        tmp_path: Temporary root without model configuration.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"

    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--traces",
            str(source),
            "--root",
            str(root),
            "--no-interactive",
            "--provider",
            "openai",
            "--provider",
            "not-a-provider",
            "--provider",
            "openai",
        ],
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert "unsupported --provider value 'not-a-provider'" in output
    assert "duplicate --provider value 'openai'" in output
    assert not root.exists()


def test_first_build_configures_providers_and_models_through_the_picker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A clean checkout reaches a completed build through provider, model, and role screens.

    Args:
        monkeypatch: Pytest patch fixture supplying a terminal, credential, and listing seam.
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    lister = _SetupLister()
    monkeypatch.setattr("wmo.cli.build_cmd.can_prompt", lambda _console: True)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setattr("wmo.cli.provider_setup.HttpProviderModelLister", lambda: lister)

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        input="1\n\n1,3\n\n1\n1\n1\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert lister.requests == ["openai"]
    printed = unstyle(result.output)
    assert "Select the providers you want to use" in printed
    assert "openai-secret" not in printed
    saved = load_model_catalog(root / "models.toml")
    assert saved.connections["openai"].api_key_env == "OPENAI_API_KEY"
    assert saved.roles.world_model == "gpt-5-6-luna"
    assert saved.roles.embedder == "text-embedding-3-small"

    replay = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
    )

    assert replay.exit_code == 0, replay.output
    assert lister.requests == ["openai"]
    assert "Select the providers you want to use" not in unstyle(replay.output)


def test_build_positional_happy_path_creates_two_rags_and_executable_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One real trace is enough to complete the named project happy path.

    Args:
        monkeypatch: Pytest patch fixture used to forbid replay rebuilds.
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)

    result = _RUNNER.invoke(app, ["build", "support", "--traces", str(source), "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "100 to 1,000 traces is the usual starting range" in result.output
    store = ProjectStore(root, "support")
    config = store.load_project()
    assert config.models is not None
    assert config.models.candidates == ()
    assert config.build is not None
    assert config.build.serving_rag != config.build.fit_rag
    loaded_traces = load_trace_dataset(store.artifacts, config.build.trace_dataset.artifact_id)
    identity_evidence = read_trace_model_identity_evidence(store.artifacts, loaded_traces)
    assert identity_evidence is not None
    assert identity_evidence.records
    assert {(record.capabilities, record.connection) for record in identity_evidence.records} == {
        ("inferred", "inferred")
    }
    serving = load_rag_index(store.artifacts, config.build.serving_rag.artifact_id)
    fit = load_rag_index(store.artifacts, config.build.fit_rag.artifact_id)
    assert serving.index.transition_count == 1
    assert fit.index.transition_count == 1
    world = GroundedWorldModelArtifact.model_validate_json(
        store.artifacts.read_bytes(config.build.world_model.artifact_id, "world-model.json")
    )
    assert world.serving_rag == config.build.serving_rag
    assert world.model_alias == "world"

    def forbid_rebuild(*_args: object, **_kwargs: object) -> None:
        """Fail if exact replay attempts to rebuild immutable RAG artifacts.

        Raises:
            AssertionError: Always, because exact replay must reuse persisted artifacts.
        """
        raise AssertionError("exact replay must not rebuild provider-backed RAG artifacts")

    monkeypatch.setattr("wmo.cli.build_cmd._build_grounded_artifacts", forbid_rebuild)
    replay = _RUNNER.invoke(app, ["build", "support", "--traces", str(source), "--root", str(root)])
    assert replay.exit_code == 0, replay.output
    assert "embedding spend ceiling: $0.000000" in replay.output

    swapped = config.build.model_copy(
        update={
            "serving_rag": config.build.fit_rag,
            "fit_rag": config.build.serving_rag,
        }
    )
    with pytest.raises(ProjectStoreError, match="completed build graph"):
        store.bind_completed_build(swapped)
    assert store.load_project().build == config.build


def test_build_package_upgrade_creates_new_immutable_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical evidence from a new installed revision advances without mutating old artifacts.

    Args:
        tmp_path: Temporary trace, catalog, and project root.
        monkeypatch: Pytest patch fixture used to forbid an exact replay rebuild.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)

    first_result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "a" * 40},
    )
    assert first_result.exit_code == 0, first_result.output
    store = ProjectStore(root, "support")
    first_build = store.load_project().build
    assert first_build is not None
    first_review = store.read_review()
    first_manifests = {
        pointer.artifact_id: store.artifacts.read(pointer.artifact_id).manifest
        for pointer in (
            first_build.trace_dataset,
            first_build.task_set,
            first_build.serving_rag,
            first_build.fit_rag,
            first_build.world_model,
        )
    }

    original_build_grounded = build_command._build_grounded_artifacts

    def fail_after_readiness(*_args: object, **_kwargs: object) -> None:
        """Inject failure after deterministic readiness but before completed-build selection.

        Raises:
            ValueError: Always, to simulate interrupted provider-backed construction.
        """
        raise ValueError("injected grounded build failure")

    monkeypatch.setattr(build_command, "_build_grounded_artifacts", fail_after_readiness)
    failed_result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )
    assert failed_result.exit_code == 2
    assert "injected grounded build failure" in failed_result.output
    assert store.load_project().build == first_build
    assert store.read_review() == first_review

    monkeypatch.setattr(build_command, "_build_grounded_artifacts", original_build_grounded)
    second_result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )

    assert second_result.exit_code == 0, second_result.output
    second_build = store.load_project().build
    assert second_build is not None
    assert second_build.trace_dataset != first_build.trace_dataset
    assert second_build.task_set != first_build.task_set
    assert second_build.serving_rag != first_build.serving_rag
    assert second_build.fit_rag != first_build.fit_rag
    assert second_build.world_model != first_build.world_model
    for artifact_id, manifest in first_manifests.items():
        assert store.artifacts.read(artifact_id).manifest == manifest
    second_manifests = {
        pointer.artifact_id: store.artifacts.read(pointer.artifact_id).manifest
        for pointer in (
            second_build.trace_dataset,
            second_build.task_set,
            second_build.serving_rag,
            second_build.fit_rag,
            second_build.world_model,
        )
    }

    def forbid_rebuild(*_args: object, **_kwargs: object) -> None:
        """Fail if exact upgraded replay attempts another provider-backed build.

        Raises:
            AssertionError: Always, because exact replay must reuse the upgraded graph.
        """
        raise AssertionError("exact upgraded replay must not rebuild provider-backed artifacts")

    monkeypatch.setattr("wmo.cli.build_cmd._build_grounded_artifacts", forbid_rebuild)
    replay_result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )
    assert replay_result.exit_code == 0, replay_result.output
    assert store.load_project().build == second_build
    for artifact_id, manifest in second_manifests.items():
        assert store.artifacts.read(artifact_id).manifest == manifest


@pytest.mark.parametrize("selected_graph", ["old", "new"])
def test_build_package_upgrade_graphs_remain_independently_verified(
    tmp_path: Path,
    selected_graph: str,
) -> None:
    """Corruption in either immutable revision graph is detected recursively.

    Args:
        tmp_path: Temporary trace, catalog, and project root.
        selected_graph: Revision graph whose trace payload is corrupted.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)
    first_result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "a" * 40},
    )
    assert first_result.exit_code == 0, first_result.output
    store = ProjectStore(root, "support")
    first_build = store.load_project().build
    assert first_build is not None
    second_result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )
    assert second_result.exit_code == 0, second_result.output
    second_build = store.load_project().build
    assert second_build is not None
    selected = first_build if selected_graph == "old" else second_build
    trace_directory = store.artifacts.read(selected.trace_dataset.artifact_id).directory
    trace_path = trace_directory / "traces.jsonl"
    trace_path.write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(ArtifactCorruptionError, match="digest mismatch"):
        store.artifacts.read(selected.trace_dataset.artifact_id)


def test_build_package_upgrade_over_ceiling_preserves_selected_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-ceiling replacement leaves the selected build and review handoff together.

    Args:
        tmp_path: Temporary trace, catalog, and project root.
        monkeypatch: Pytest patch fixture forcing a paid-build ceiling failure.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)
    first_result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "a" * 40},
    )
    assert first_result.exit_code == 0, first_result.output
    store = ProjectStore(root, "support")
    first_build = store.load_project().build
    first_review = store.read_review()
    assert first_build is not None
    _RESOLVE_CALLS.clear()
    monkeypatch.setattr(build_command, "_embedding_cost_ceiling", lambda *_args: 6.0)

    blocked = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )

    assert blocked.exit_code == 2
    assert "conservative embedding estimate $6.000000 exceeds" in unstyle(blocked.output)
    assert _RESOLVE_CALLS == []
    assert store.load_project().build == first_build
    assert store.read_review() == first_review


def test_configured_budget_rejects_build_before_provider_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-budget estimate performs zero credential or provider-client work.

    Args:
        tmp_path: Temporary trace, catalog, and settings root.
        monkeypatch: Cost and provider-resolution boundary replacements.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)
    set_maximum_command_cost_usd(0.5, root)
    monkeypatch.setattr(build_command, "_embedding_cost_ceiling", lambda *_args: 1.0)
    provider_resolutions: list[str] = []

    def forbidden_preflight(
        self: _RuntimeCatalog,
        alias: str,
        requirement: object | None = None,
    ) -> ResolvedModel:
        """Fail if hard budget rejection reaches runtime model construction.

        Args:
            self: Runtime catalog instance.
            alias: Unexpected model alias.
            requirement: Unexpected capability requirement.

        Raises:
            AssertionError: Always, because provider resolution must not run.
        """
        del self, requirement
        provider_resolutions.append(alias)
        raise AssertionError("provider resolution must follow command budget authorization")

    monkeypatch.setattr(_RuntimeCatalog, "preflight", forbidden_preflight)

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root), "--yes"],
    )

    assert result.exit_code == 2
    output = unstyle(result.output)
    assert "estimated cost: $1.00" in output
    assert "configured budget: $0.50 per command" in output
    assert "wmo config budget 1.00" in output
    assert "--yes cannot override" in output
    assert provider_resolutions == []
    store = ProjectStore(root, "support")
    assert store.load_project().build is None
    assert store.read_review() is None


def test_noninteractive_build_above_half_requires_yes_before_provider_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-declared noninteractive build receives deterministic remediation.

    Args:
        tmp_path: Temporary trace, catalog, and settings root.
        monkeypatch: Cost and provider-work boundary replacements.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)
    set_maximum_command_cost_usd(1.0, root)
    monkeypatch.setattr(build_command, "_embedding_cost_ceiling", lambda *_args: 0.75)
    provider_calls: list[bool] = []

    def forbidden_build(*_args: object, **_kwargs: object) -> None:
        """Fail if missing confirmation reaches provider-backed artifact construction.

        Raises:
            AssertionError: Always, because ``--yes`` is absent.
        """
        provider_calls.append(True)
        raise AssertionError("provider work requires shared cost authorization")

    monkeypatch.setattr(build_command, "_build_grounded_artifacts", forbidden_build)

    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--traces",
            str(source),
            "--root",
            str(root),
            "--no-interactive",
        ],
    )

    assert result.exit_code == 2
    assert "requires explicit confirmation" in unstyle(result.output)
    assert "re-run with --yes" in unstyle(result.output)
    assert provider_calls == []


def test_noninteractive_build_yes_confirms_an_in_budget_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-complete agent invocation runs after the shared preflight.

    Args:
        tmp_path: Temporary trace, catalog, and settings root.
        monkeypatch: Conservative estimate replacement.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)
    set_maximum_command_cost_usd(1.0, root)
    monkeypatch.setattr(build_command, "_embedding_cost_ceiling", lambda *_args: 0.75)

    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--traces",
            str(source),
            "--root",
            str(root),
            "--no-interactive",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "authorization: confirmed by --yes" in unstyle(result.output)
    assert ProjectStore(root, "support").load_project().build is not None


def test_interactive_build_uses_the_cost_specific_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real CLI prompt names the command, estimate, and configured budget.

    Args:
        tmp_path: Temporary trace, catalog, and settings root.
        monkeypatch: Interactive-session and conservative-estimate replacements.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)
    set_maximum_command_cost_usd(1.0, root)
    monkeypatch.setattr(build_command, "_embedding_cost_ceiling", lambda *_args: 0.75)
    monkeypatch.setattr(consent_module, "can_prompt", lambda _console: True)

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    output = unstyle(result.output)
    assert "Authorize wmo build support" in output
    assert "$0.75" in output
    assert "$1.00 per-command budget" in output
    assert "Proceed?" not in output


@pytest.mark.parametrize("failure_mode", ["cost", "grounded"])
def test_first_build_failure_does_not_publish_review_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    """No first-build readiness is visible until its completed graph is selected.

    Args:
        tmp_path: Temporary trace, catalog, and project root.
        monkeypatch: Pytest patch fixture selecting the pre-selection failure boundary.
        failure_mode: Cost rejection or grounded construction failure.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)
    if failure_mode == "cost":
        monkeypatch.setattr(build_command, "_embedding_cost_ceiling", lambda *_args: 6.0)
    else:

        def fail_grounded_build(*_args: object, **_kwargs: object) -> None:
            """Simulate grounded artifact construction failing before selection.

            Raises:
                ValueError: Always, at the provider-backed construction boundary.
            """
            raise ValueError("injected first grounded build failure")

        monkeypatch.setattr(build_command, "_build_grounded_artifacts", fail_grounded_build)

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "a" * 40},
    )

    assert result.exit_code == 2
    if failure_mode == "cost":
        assert _RESOLVE_CALLS == []
    store = ProjectStore(root, "support")
    assert store.load_project().build is None
    assert store.read_review() is None


def test_build_package_upgrade_recovers_selection_before_review_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart repairs review after completed-build selection without rebuilding providers.

    Args:
        tmp_path: Temporary trace, catalog, and project root.
        monkeypatch: Pytest patch fixture injecting and recovering a final review-write failure.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)
    first_result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "a" * 40},
    )
    assert first_result.exit_code == 0, first_result.output
    store = ProjectStore(root, "support")
    first_build = store.load_project().build
    first_review = store.read_review()
    assert first_build is not None
    assert isinstance(first_review, dict)

    def add_build_scoped_review_state(current: JsonValue | None) -> JsonValue:
        """Attach graph-A judge namespaces and unrelated review state.

        Args:
            current: Current project review value.

        Returns:
            Review state with stale build-scoped fixtures and an unrelated marker.
        """
        assert isinstance(current, dict)
        return {
            **current,
            "manual_judge": {"build": "a"},
            "rubric_review": {"build": "a"},
            "human_score_history": [{"build": "a"}],
            "human_score_submissions": [{"build": "a"}],
            "unrelated_review_state": {"preserve": True},
        }

    first_review = store.update_review(add_build_scoped_review_state)
    original_select_review = simulation_build.select_build_review

    def fail_review_selection(*_args: object, **_kwargs: object) -> None:
        """Inject a crash after completed-build selection and before review advancement.

        Raises:
            ValueError: Always, to simulate interrupted coordination.
        """
        raise ValueError("injected final review failure")

    monkeypatch.setattr(simulation_build, "select_build_review", fail_review_selection)
    interrupted = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )
    assert interrupted.exit_code == 2
    assert "injected final review failure" in interrupted.output
    selected_build = store.load_project().build
    assert selected_build is not None
    assert selected_build != first_build
    assert store.read_review() == first_review

    monkeypatch.setattr(simulation_build, "select_build_review", original_select_review)

    def forbid_rebuild(*_args: object, **_kwargs: object) -> None:
        """Fail if recovery reconstructs already selected provider-backed artifacts.

        Raises:
            AssertionError: Always, because restart must reuse the selected immutable graph.
        """
        raise AssertionError("review recovery must not rebuild provider-backed artifacts")

    monkeypatch.setattr(build_command, "_build_grounded_artifacts", forbid_rebuild)
    recovered = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )

    assert recovered.exit_code == 0, recovered.output
    assert store.load_project().build == selected_build
    recovered_review = store.read_review()
    assert isinstance(recovered_review, dict)
    assert recovered_review["build_review"]["trace_dataset"] == (
        selected_build.trace_dataset.model_dump(mode="json")
    )
    assert "manual_judge" not in recovered_review
    assert "rubric_review" not in recovered_review
    assert "human_score_history" not in recovered_review
    assert "human_score_submissions" not in recovered_review
    assert recovered_review["unrelated_review_state"] == {"preserve": True}


@pytest.mark.parametrize("count", [2, 100, 1_001])
def test_build_accepts_trace_counts_outside_or_inside_guidance(tmp_path: Path, count: int) -> None:
    """The 100 to 1,000 range is guidance and never a validity gate.

    Args:
        tmp_path: Temporary project and trace root.
        count: Parameterized positive trace count.
    """
    source = _otlp_export(tmp_path, count=count)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)

    result = _RUNNER.invoke(app, ["build", "support", "--traces", str(source), "--root", str(root)])

    assert result.exit_code == 0, result.output
    config = ProjectStore(root, "support").load_project()
    assert config.build is not None
    if count == 2:
        store = ProjectStore(root, "support")
        serving = load_rag_index(store.artifacts, config.build.serving_rag.artifact_id)
        fit = load_rag_index(store.artifacts, config.build.fit_rag.artifact_id)
        assert serving.index.included_partitions == ("fit", "held_out")
        assert fit.index.included_partitions == ("fit",)
        assert serving.index.transition_count == 2
        assert fit.index.transition_count == 1
    if count < 100 or count > 1_000:
        assert "usual starting range" in result.output
    else:
        assert "usual starting range" not in result.output


def test_missing_config_noninteractive_fails_before_project_write(tmp_path: Path) -> None:
    """Automation receives complete remediation and no partial project state.

    Args:
        tmp_path: Temporary root without model configuration.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root), "--no-interactive"],
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    for missing in ("models.toml", "provider connections", "world_model", "judge", "embedder"):
        assert missing in output
    assert "wmo config providers" in output
    assert not (root / "projects" / "support").exists()


def test_missing_config_is_reported_before_a_missing_trace_path(tmp_path: Path) -> None:
    """First build handles required setup before inspecting trace-file contents.

    Args:
        tmp_path: Temporary root without configuration or trace input.
    """
    root = tmp_path / ".wmo"

    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--traces",
            str(tmp_path / "missing.jsonl"),
            "--root",
            str(root),
            "--no-interactive",
        ],
    )

    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert result.exit_code == 2
    assert "model configuration is incomplete before build" in output
    assert "trace file not found" not in output
    assert not (root / "projects" / "support").exists()


def test_interactive_first_build_commits_setup_before_trace_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real terminal enters setup first and preserves valid catalog state after trace failure.

    Args:
        monkeypatch: Pytest patch fixture simulating terminal setup.
        tmp_path: Temporary root receiving the configured catalog.
    """
    root = tmp_path / ".wmo"
    configured: list[Path] = []
    catalog = ModelCatalog(
        connections={"fixture": ConnectionConfig(provider="openai", api_key_env="FIXTURE_API_KEY")},
        models={
            "world": ModelRecord(
                connection="fixture",
                model="world-id",
                capabilities=ModelCapabilities(maximum_output_tokens=16_000),
            ),
            "judge": ModelRecord(connection="fixture", model="judge-id"),
            "embed": ModelRecord(
                connection="fixture",
                model="embed-id",
                capabilities=ModelCapabilities(
                    supports_embeddings=True,
                    input_cost_per_million_tokens_usd=0,
                ),
            ),
        },
        roles=ModelRoles(world_model="world", judge="judge", embedder="embed"),
    )

    monkeypatch.setattr("wmo.cli.build_cmd.can_prompt", lambda _console: True)

    def configure(path: Path, *_args: object, **_kwargs: object) -> ModelCatalog:
        """Persist the fixture catalog as the simulated interactive setup result.

        Args:
            path: Local WMO root receiving the shared catalog.

        Returns:
            Complete fixture catalog returned by simulated setup.
        """
        configured.append(path)
        path.mkdir(parents=True, exist_ok=True)
        write_model_catalog(path / "models.toml", catalog)
        return catalog

    monkeypatch.setattr("wmo.cli.build_cmd.run_provider_setup", configure)

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(tmp_path / "missing.jsonl"), "--root", str(root)],
    )

    assert result.exit_code == 2
    assert configured == [root]
    assert (root / "models.toml").exists()
    assert "trace file not found" in result.output
    assert not (root / "projects" / "support").exists()


def test_build_retains_active_positional_trace_consumer_but_rejects_project_option(
    tmp_path: Path,
) -> None:
    """The hidden trace positional remains active while PROJECT stays positional.

    Args:
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    positional = _RUNNER.invoke(app, ["build", "support", str(source)])
    project_option = _RUNNER.invoke(
        app,
        ["build", "--project", "support", "--traces", str(source)],
    )

    assert positional.exit_code == 2
    assert "model configuration is incomplete before build" in unstyle(positional.output).casefold()
    assert project_option.exit_code == 2
    assert "No such option: --project" in unstyle(project_option.output)


def test_build_help_describes_the_completed_grounded_artifact() -> None:
    """CLI guidance names the reusable output of the build happy path.

    The regression asserts the public help without constructing project state.
    """
    result = _RUNNER.invoke(app, ["build", "--help"])

    assert result.exit_code == 0, result.output
    help_text = unstyle(result.output)
    assert "Build a reusable grounded world model from local trace evidence." in help_text
    assert "-t" in help_text
    assert "--traces" in help_text
    assert "--dry-run" in help_text
    assert "--max-build-cost-usd" in help_text
    assert "--yes" in help_text


def test_build_preflight_auto_runs_without_proceed(tmp_path: Path) -> None:
    """An under-ceiling build shows exact preflight and truthful progress without prompting.

    Args:
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root, embedder_input_usd_per_million=1.0)

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
    )
    assert result.exit_code == 0, result.output
    output = unstyle(result.output)
    assert "Build preflight" in output
    assert "traces       1 accepted, 0 invalid" in output
    assert "split        1 fit, 0 held out" in output
    assert "world model  world (world-id)" in output
    assert "embedder     embed (embed-id)" in output
    assert "embedding    at most $" in output
    assert "ceiling      $5.000000" in output
    assert "Proceed?" not in output
    assert "Build serving index with embed-id" in output
    assert "Build fit-only index" in output
    assert "Ground world model world-id" in output
    assert "next: wmo optimize router support" in output
    assert _RESOLVE_CALLS == ["embed"]


def test_over_ceiling_build_fails_before_provider_construction(tmp_path: Path) -> None:
    """An over-ceiling estimate fails before credentials or provider clients.

    Args:
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root, embedder_input_usd_per_million=1_000_000.0)

    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--traces",
            str(source),
            "--root",
            str(root),
            "--max-build-cost-usd",
            "0.01",
        ],
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert "conservative embedding estimate $" in output
    assert "exceeds --max-build-cost-usd $0.010000" in output
    assert "wmo build support --traces" in output
    assert "Proceed?" not in output
    assert _RESOLVE_CALLS == []
    store = ProjectStore(root, "support")
    assert store.load_project().build is None
    assert store.read_review() is None


def test_dry_run_has_zero_calls_and_no_completed_selection(tmp_path: Path) -> None:
    """``--dry-run`` shows preflight without provider construction or selection.

    Args:
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root, embedder_input_usd_per_million=1.0)

    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--traces",
            str(source),
            "--root",
            str(root),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    output = unstyle(result.output)
    assert "Build preflight" in output
    assert "embedding    at most $" in output
    assert "dry run complete" in output
    assert "Proceed?" not in output
    assert _RESOLVE_CALLS == []
    store = ProjectStore(root, "support")
    assert store.load_project().build is None
    assert store.read_review() is None


def test_wizard_preconsent_plan_persists_only_provider_free_unselected_evidence(
    tmp_path: Path,
) -> None:
    """Planning may checkpoint deterministic evidence but never paid outputs or selection.

    Args:
        tmp_path: Temporary trace, project, and artifact root.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)

    plan = _prepare_new_build(
        "support",
        trace_path=source,
        source="otlp",
        root=root,
        world_model=None,
        judge=None,
        embedder=None,
        top_k=5,
        maximum_build_cost_usd=5.0,
        code_revision="a" * 40,
        providers=(),
        console=Console(file=StringIO(), force_terminal=False),
    )

    store = ProjectStore(root, "support")
    artifact_types = {
        store.artifacts.read(artifact_id).manifest.artifact_type
        for artifact_id in store.artifacts.list_ids()
    }
    assert plan.build_reused is False
    assert store.load_project().build is None
    assert _RESOLVE_CALLS == []
    assert "rag-index" not in artifact_types
    assert "grounded-world-model" not in artifact_types


def test_exact_replay_has_zero_calls_and_no_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exact replay verifies and reuses grounded artifacts without provider construction.

    Args:
        monkeypatch: Pytest patch fixture forbidding grounded reconstruction.
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root, embedder_input_usd_per_million=1.0)
    arguments = ["build", "support", "--traces", str(source), "--root", str(root)]
    first = _RUNNER.invoke(app, arguments)
    assert first.exit_code == 0, first.output
    first_build = ProjectStore(root, "support").load_project().build
    assert first_build is not None
    _RESOLVE_CALLS.clear()

    def forbid_rebuild(*_args: object, **_kwargs: object) -> None:
        """Fail if an exact replay attempts provider-backed reconstruction.

        Raises:
            AssertionError: Always, because exact replay must reuse artifacts.
        """
        raise AssertionError("exact replay must not rebuild grounded artifacts")

    monkeypatch.setattr(build_command, "_build_grounded_artifacts", forbid_rebuild)
    replay = _RUNNER.invoke(app, arguments)

    assert replay.exit_code == 0, replay.output
    output = unstyle(replay.output)
    assert "reuse exact completed indexes, $0.000000 new spend" in output
    assert "embedding spend ceiling: $0.000000" in output
    assert "Proceed?" not in output
    assert _RESOLVE_CALLS == []
    assert ProjectStore(root, "support").load_project().build == first_build


def test_provider_failure_does_not_publish_build_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A provider embedding failure leaves the completed-build pointer empty.

    Args:
        monkeypatch: Pytest patch fixture injecting an embedding failure.
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root, embedder_input_usd_per_million=1.0)

    def fail_embed(self: _EmbeddingClient, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Fail at the first provider embedding dispatch.

        Args:
            texts: Canonical RAG key texts that would have been embedded.

        Raises:
            ValueError: Always, to simulate a provider failure.
        """
        del self, texts
        raise ValueError("injected provider embedding failure")

    monkeypatch.setattr(_EmbeddingClient, "embed", fail_embed)
    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(source), "--root", str(root)],
    )

    assert result.exit_code == 2
    assert "injected provider embedding failure" in result.output
    assert _RESOLVE_CALLS == ["embed"]
    store = ProjectStore(root, "support")
    assert store.load_project().build is None
    assert store.read_review() is None


def test_other_spend_consent_commands_still_require_yes() -> None:
    """Build authorization does not change other command spend gates."""
    model_help = unstyle(_RUNNER.invoke(app, ["optimize", "model", "--help"]).output)
    router_help = unstyle(_RUNNER.invoke(app, ["optimize", "router", "--help"]).output)

    assert "--yes" in model_help
    assert "--yes" in router_help
    assert "--yes" in unstyle(_RUNNER.invoke(app, ["build", "--help"]).output)


_TURNS = (
    ("Support request", "What account email?"),
    ("customer@example.test", "Reset instructions sent."),
)


def _chat_json_export(tmp_path: Path) -> Path:
    """Write one two-turn chat JSON conversation export.

    Args:
        tmp_path: Temporary directory receiving the trace export.

    Returns:
        Path to the completed chat JSON export.
    """
    messages: list[JsonValue] = []
    for request, completion in _TURNS:
        messages.append({"role": "user", "content": request})
        messages.append({"role": "assistant", "content": completion})
    path = tmp_path / "chat.json"
    path.write_text(
        json.dumps(
            [
                {
                    "trace_id": "conversation-1",
                    "provider": "openai",
                    "model": "gpt-test",
                    "messages": messages,
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_build_accepts_a_declared_vendor_source_through_the_source_flag(tmp_path: Path) -> None:
    """One declared vendor export completes the positional build path via --source.

    Args:
        tmp_path: Temporary project and trace root.
    """
    export = _chat_json_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)

    result = _RUNNER.invoke(
        app,
        [
            "build",
            "support",
            "--traces",
            str(export),
            "--source",
            "chat-json",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "built 1 accepted, 0 invalid" in result.output
    store = ProjectStore(root, "support")
    config = store.load_project()
    assert config.build is not None
    traces = load_trace_dataset(store.artifacts, config.build.trace_dataset.artifact_id)
    assert len(traces.traces) == 1
    assert traces.traces[0].source.identity.source_id.startswith("chat-json:")


def test_build_rejects_an_undeclared_trace_source(tmp_path: Path) -> None:
    """An unknown source names every supported canonical format.

    Args:
        tmp_path: Temporary project and trace root.
    """
    export = _chat_json_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)

    result = _RUNNER.invoke(
        app,
        ["build", "support", "--traces", str(export), "--source", "helicone", "--root", str(root)],
    )

    assert result.exit_code == 2
    assert "unsupported trace source 'helicone'" in unstyle(result.output)
    assert "posthog" in unstyle(result.output)
