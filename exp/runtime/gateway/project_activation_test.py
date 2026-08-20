"""Tests for immutable project activations and local artifact loading."""

from __future__ import annotations

from pathlib import Path

from exp.common.models import ModelMessage, ModelRequest
from exp.common.project import ProjectConfig, ProjectStore
from exp.runtime.gateway.project_activation import (
    LocalArtifactProjectActivationRepository,
    ProjectActivation,
    load_project_activation,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.router.runtime import RouterRuntime
from exp.runtime.router.runtime_test import _persist_runtime_fixture


class _StaticActivationRepository:
    """Return one caller-owned activation without consulting local project state."""

    def __init__(self, activation: ProjectActivation) -> None:
        """Store the exact immutable activation returned by this repository."""
        self.activation = activation
        self.loads = 0

    def load(
        self,
        project_ref: str,
        activation_ref: str | None,
        *,
        runtime_catalog: RuntimeModelCatalog,
    ) -> ProjectActivation:
        """Return the activation after checking its requested authority."""
        del runtime_catalog
        assert project_ref == self.activation.project_ref
        assert activation_ref == self.activation.activation_ref
        self.loads += 1
        return self.activation


def test_caller_supplied_activation_selects_without_a_local_project_directory(
    tmp_path: Path,
) -> None:
    """A stable activation can be loaded and used after its project directory is absent."""
    store, policy, catalog, client = _persist_runtime_fixture(tmp_path)
    activation = load_project_activation(
        store,
        project_ref="platform-project",
        activation_ref=policy.policy_id,
    )
    repository = _StaticActivationRepository(activation)

    supplied = repository.load(
        "platform-project",
        policy.policy_id,
        runtime_catalog=catalog,
    )
    runtime = RouterRuntime.from_activation(supplied, catalog)

    assert repository.loads == 1
    assert not (tmp_path / "projects" / "platform-project").exists()
    assert client.embed_calls == 0
    decision = runtime.select(
        ModelRequest(messages=(ModelMessage(role="user", content="route this"),)),
        episode_id="episode-one",
    )
    assert decision.selected_alias == "cheap"
    assert client.embed_calls == 1
    assert client.complete_calls == 0


def test_local_artifact_repository_returns_the_same_immutable_activation(
    tmp_path: Path,
) -> None:
    """The default local adapter preserves the direct activation contract exactly."""
    store, policy, catalog, client = _persist_runtime_fixture(tmp_path)
    ProjectStore(tmp_path, "project-a").initialize(ProjectConfig(project_id="project-a"))
    direct = load_project_activation(
        store,
        project_ref="project-a",
        activation_ref=policy.policy_id,
    )

    local = LocalArtifactProjectActivationRepository(tmp_path).load(
        "project-a",
        policy.policy_id,
        runtime_catalog=catalog,
    )

    assert local == direct
    assert local.candidate_aliases == ("baseline", "cheap")
    assert client.embed_calls == 0
    assert client.complete_calls == 0
