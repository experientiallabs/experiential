"""CLI surface tests for runtime retrieval refresh."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.core.artifacts import ArtifactInput
from wmo.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    write_model_catalog,
)
from wmo.common.project import ProjectBuildArtifacts, ProjectConfig, ProjectStore


def _pointer(artifact_id: str) -> ArtifactInput:
    """Return one secret-free artifact pointer for a fixture completed build.

    Args:
        artifact_id: Valid local artifact identity.

    Returns:
        Manifest pointer with a placeholder digest.
    """
    return ArtifactInput(artifact_id=artifact_id, sha256="0" * 64)


def test_rag_refresh_renders_as_a_nested_config_command() -> None:
    """Refresh is discoverable under config without expanding the root surface."""
    result = CliRunner().invoke(app, ["config", "rag", "refresh", "--help"])

    assert result.exit_code == 0, result.output
    output = unstyle(result.output)
    assert "--yes" in output
    assert "--non-interactive" in output
    assert "--maximum-cost-usd" in output


def test_rag_refresh_requires_a_completed_build(tmp_path: Path) -> None:
    """A project without a selected build fails before journal or embedding work.

    Args:
        tmp_path: Isolated root containing only an initialized project.
    """
    root = tmp_path / ".wmo"
    store = ProjectStore(root, "demo")
    store.initialize(ProjectConfig(project_id="demo"))

    result = CliRunner().invoke(
        app,
        ["config", "rag", "refresh", "demo", "--root", str(root), "--yes"],
    )

    assert result.exit_code == 2
    assert "completed build" in unstyle(result.output)
    assert "wmo build" in unstyle(result.output)


def test_rag_refresh_requires_durable_routed_traffic(tmp_path: Path) -> None:
    """An empty journal fails before catalog resolution or spend consent.

    Args:
        tmp_path: Isolated root containing a completed-build pointer and no journal.
    """
    root = tmp_path / ".wmo"
    store = ProjectStore(root, "demo")
    store.initialize(
        ProjectConfig(
            project_id="demo",
            build=ProjectBuildArtifacts(
                trace_dataset=_pointer("trace-dataset"),
                task_set=_pointer("task-set"),
                serving_rag=_pointer("serving-rag"),
                fit_rag=_pointer("fit-rag"),
                world_model=_pointer("world-model"),
            ),
        )
    )

    result = CliRunner().invoke(
        app,
        ["config", "rag", "refresh", "demo", "--root", str(root), "--yes"],
    )

    assert result.exit_code == 2
    output = unstyle(result.output)
    assert "durable routed traffic" in output
    assert "without --ghost" in output


def _write_priced_embedder_catalog(root: Path) -> None:
    """Persist one secret-free catalog with an explicitly priced embedder.

    Args:
        root: Local WMO root receiving ``models.toml``.
    """
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "fixture": ConnectionConfig(provider="openai", api_key_env="FIXTURE_API_KEY")
            },
            models={
                "embed": ModelRecord(
                    connection="fixture",
                    model="embed-id",
                    capabilities=ModelCapabilities(
                        supports_embeddings=True,
                        input_cost_per_million_tokens_usd=0.1,
                        context_window_tokens=8_192,
                    ),
                )
            },
            roles=ModelRoles(embedder="embed"),
        ),
    )


def test_rag_refresh_replay_does_not_construct_an_embedder_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed exact replay reprints the receipt without resolving a live client.

    Args:
        tmp_path: Isolated root with a completed-build pointer and priced embedder.
        monkeypatch: Scoped replacements for journal, lookup, and client construction.
    """
    root = tmp_path / ".wmo"
    store = ProjectStore(root, "demo")
    store.initialize(
        ProjectConfig(
            project_id="demo",
            build=ProjectBuildArtifacts(
                trace_dataset=_pointer("trace-dataset"),
                task_set=_pointer("task-set"),
                serving_rag=_pointer("serving-rag"),
                fit_rag=_pointer("fit-rag"),
                world_model=_pointer("world-model"),
            ),
        )
    )
    _write_priced_embedder_catalog(root)
    receipt = SimpleNamespace(
        refresh_id="runtime-rag-refresh-replay",
        snapshot=_pointer("runtime-snapshot"),
        runtime_trace_dataset=_pointer("runtime-traces"),
        combined_trace_dataset=_pointer("combined-traces"),
        retrieval_index=_pointer("retrieval-index"),
        imported_trace_datasets=(),
        reserved_embedding_cost_usd=0.0025,
        maximum_embedding_cost_usd=5.0,
    )
    fake = SimpleNamespace(
        refresh=receipt,
        snapshot_export=SimpleNamespace(
            snapshot=SimpleNamespace(last_ordinal=3, completed_target_count=1)
        ),
    )

    monkeypatch.setattr(
        "wmo.cli.rag_config.RuntimeInteractionJournal.read_events",
        lambda self: ("event",),
    )
    monkeypatch.setattr(
        "wmo.cli.rag_config.load_completed_build_rag_lineage_bindings",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        "wmo.cli.rag_config.find_completed_runtime_rag_refresh",
        lambda *args, **kwargs: fake,
    )

    def fail_preflight(_self: object, _alias: str, _requirement: object = None) -> None:
        """Replay must not construct a credential-backed embedder client."""
        raise AssertionError("replay must not construct an embedder client")

    monkeypatch.setattr("wmo.cli.rag_config.RuntimeModelCatalog.preflight", fail_preflight)
    monkeypatch.setattr(
        "wmo.cli.rag_config.refresh_runtime_trace_rag",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("replay must not dispatch a new refresh")
        ),
    )

    result = CliRunner().invoke(
        app,
        ["config", "rag", "refresh", "demo", "--root", str(root), "--yes"],
    )

    assert result.exit_code == 0, result.output
    output = unstyle(result.output)
    assert "runtime-rag-refresh-replay" in output
    assert "serving-rag" in output
    assert "fit-rag" in output
