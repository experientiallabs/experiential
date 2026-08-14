"""End-to-end tests for the named grounded-world-model build command."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

import wmo.cli.build_cmd as build_command
import wmo.simulation.build as simulation_build
from wmo.cli.app import app
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
    write_model_catalog,
)
from wmo.common.project import ArtifactCorruptionError, ProjectStore, ProjectStoreError
from wmo.runtime.models import ResolvedModel
from wmo.simulation.retrieval import load_rag_index
from wmo.simulation.world_model import GroundedWorldModelArtifact

_RUNNER = CliRunner()


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
        ["build", "support", str(tmp_path / "missing.jsonl"), "--root", str(root)],
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

    def __init__(self, _catalog: ModelCatalog) -> None:
        """Create deterministic embedding and completion clients.

        Args:
            _catalog: Unused catalog accepted by the production constructor seam.
        """
        self._embedding = _EmbeddingClient()
        self._completion = _CompletionClient()

    def preflight(self, alias: str, _requirement: object | None = None) -> ResolvedModel:
        """Return exact static identities with alias-specific capabilities.

        Args:
            alias: Configured fixture model alias.
            _requirement: Unused requirement accepted by the runtime seam.

        Returns:
            Deterministic resolved fixture model.
        """
        if alias == "embed":
            capabilities = ModelCapabilities(
                supports_embeddings=True,
                input_cost_per_million_tokens_usd=0,
            )
            embedding = self._embedding
        else:
            capabilities = ModelCapabilities(maximum_output_tokens=16_000)
            embedding = None
        snapshot = ModelSnapshot(
            provider="fixture",
            model_id=f"fixture-{alias}",
            capabilities_sha256=sha256_json(capabilities),
            connection_sha256=sha256_json({"connection": alias}),
        )
        return ResolvedModel(alias, snapshot, capabilities, self._completion, embedding)

    def resolve(self, alias: str) -> ResolvedModel:
        """Reuse local preflight for roles with no extra capability requirement.

        Args:
            alias: Configured fixture model alias.

        Returns:
            Deterministic resolved fixture model.
        """
        return self.preflight(alias)


def _catalog(root: Path) -> None:
    """Write complete secret-free build roles while leaving router candidates empty.

    Args:
        root: Temporary WMO root receiving ``models.toml``.
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
                        input_cost_per_million_tokens_usd=0,
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

    result = _RUNNER.invoke(app, ["build", "support", str(source), "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "100 to 1,000 traces is the usual starting range" in result.output
    store = ProjectStore(root, "support")
    config = store.load_project()
    assert config.models is not None
    assert config.models.candidates == ()
    assert config.build is not None
    assert config.build.serving_rag != config.build.fit_rag
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
    replay = _RUNNER.invoke(app, ["build", "support", str(source), "--root", str(root)])
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
        ["build", "support", str(source), "--root", str(root)],
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
        ["build", "support", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )
    assert failed_result.exit_code == 2
    assert "injected grounded build failure" in failed_result.output
    assert store.load_project().build == first_build
    assert store.read_review() == first_review

    monkeypatch.setattr(build_command, "_build_grounded_artifacts", original_build_grounded)
    second_result = _RUNNER.invoke(
        app,
        ["build", "support", str(source), "--root", str(root)],
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
        ["build", "support", str(source), "--root", str(root)],
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
        ["build", "support", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "a" * 40},
    )
    assert first_result.exit_code == 0, first_result.output
    store = ProjectStore(root, "support")
    first_build = store.load_project().build
    assert first_build is not None
    second_result = _RUNNER.invoke(
        app,
        ["build", "support", str(source), "--root", str(root)],
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


def test_build_package_upgrade_decline_preserves_selected_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining replacement spend leaves the selected build and review handoff together.

    Args:
        tmp_path: Temporary trace, catalog, and project root.
        monkeypatch: Pytest patch fixture forcing a paid-build decline.
    """
    source = _otlp_export(tmp_path)
    root = tmp_path / ".wmo"
    root.mkdir()
    _catalog(root)
    first_result = _RUNNER.invoke(
        app,
        ["build", "support", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "a" * 40},
    )
    assert first_result.exit_code == 0, first_result.output
    store = ProjectStore(root, "support")
    first_build = store.load_project().build
    first_review = store.read_review()
    assert first_build is not None
    monkeypatch.setattr(build_command, "_embedding_cost_ceiling", lambda *_args: 1.0)
    monkeypatch.setattr(build_command, "require_spend_consent", lambda *_args, **_kwargs: False)

    declined = _RUNNER.invoke(
        app,
        ["build", "support", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )

    assert declined.exit_code == 0, declined.output
    assert store.load_project().build == first_build
    assert store.read_review() == first_review


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
        ["build", "support", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "a" * 40},
    )
    assert first_result.exit_code == 0, first_result.output
    store = ProjectStore(root, "support")
    first_build = store.load_project().build
    first_review = store.read_review()
    assert first_build is not None
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
        ["build", "support", str(source), "--root", str(root)],
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
        ["build", "support", str(source), "--root", str(root)],
        env={"WMO_RELEASE_REVISION": "b" * 40},
    )

    assert recovered.exit_code == 0, recovered.output
    assert store.load_project().build == selected_build
    recovered_review = store.read_review()
    assert isinstance(recovered_review, dict)
    assert recovered_review["build_review"]["trace_dataset"] == (
        selected_build.trace_dataset.model_dump(mode="json")
    )


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

    result = _RUNNER.invoke(app, ["build", "support", str(source), "--root", str(root)])

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
        ["build", "support", str(source), "--root", str(root), "--no-interactive"],
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
        ["build", "support", str(tmp_path / "missing.jsonl"), "--root", str(root)],
    )

    assert result.exit_code == 2
    assert configured == [root]
    assert (root / "models.toml").exists()
    assert "trace file not found" in result.output
    assert not (root / "projects" / "support").exists()


def test_build_rejects_old_project_option_shape(tmp_path: Path) -> None:
    """The command has exactly PROJECT then TRACES as its positional happy path.

    Args:
        tmp_path: Temporary project and trace root.
    """
    source = _otlp_export(tmp_path)
    result = _RUNNER.invoke(app, ["build", str(source), "--project", "support"])

    assert result.exit_code == 2
    assert "No such option: --project" in unstyle(result.output)


def test_build_help_describes_the_completed_grounded_artifact() -> None:
    """CLI guidance names the reusable output of the build happy path.

    The regression asserts the public help without constructing project state.
    """
    result = _RUNNER.invoke(app, ["build", "--help"])

    assert result.exit_code == 0, result.output
    assert "Build a reusable grounded world model from local trace evidence." in unstyle(
        result.output
    )
