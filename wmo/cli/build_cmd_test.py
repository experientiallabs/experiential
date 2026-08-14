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
from wmo.common.project import ProjectStore, ProjectStoreError
from wmo.runtime.models import ResolvedModel
from wmo.simulation.retrieval import load_rag_index
from wmo.simulation.world_model import GroundedWorldModelArtifact

_RUNNER = CliRunner()


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
