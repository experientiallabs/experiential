"""Tests for optimizer-owned automatic router activation."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from openai import OpenAI

import wmo.optimize.router.activation as activation
from wmo.common.core.artifacts import ArtifactId
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.application import RouterPolicyVerifier
from wmo.runtime.router.runtime import DecisionSink, RouterRuntime


def test_public_activation_surfaces_inject_one_optimizer_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project and OpenAI-client loading share the automatic policy verifier.

    Args:
        tmp_path: Pytest-owned root passed through both wrappers.
        monkeypatch: Patch fixture replacing the low-level runtime loaders.
    """
    runtime = cast(RouterRuntime, object())
    client = cast(OpenAI, object())
    project_verifiers: list[RouterPolicyVerifier | None] = []
    client_verifiers: list[RouterPolicyVerifier | None] = []
    client_ghost_modes: list[bool] = []

    def load_project(
        project: str,
        root: Path,
        *,
        policy_id: ArtifactId | None = None,
        environment: Mapping[str, str] | None = None,
        runtime_catalog: RuntimeModelCatalog | None = None,
        decision_sink: DecisionSink | None = None,
        policy_verifier: RouterPolicyVerifier | None = None,
    ) -> RouterRuntime:
        """Capture the project-loader verifier while preserving the public call shape."""
        assert project == "support"
        assert root == tmp_path
        assert policy_id is None
        assert environment is None
        assert runtime_catalog is None
        assert decision_sink is None
        project_verifiers.append(policy_verifier)
        return runtime

    def load_client(
        project: str,
        root: Path = Path(".wmo"),
        *,
        policy_id: ArtifactId | None = None,
        environment: Mapping[str, str] | None = None,
        runtime_catalog: RuntimeModelCatalog | None = None,
        decision_sink: DecisionSink | None = None,
        policy_verifier: RouterPolicyVerifier | None = None,
        ghost: bool = False,
    ) -> OpenAI:
        """Capture the client-loader verifier while preserving the public call shape."""
        assert project == "support"
        assert root == tmp_path
        assert policy_id is None
        assert environment is None
        assert runtime_catalog is None
        assert decision_sink is None
        client_verifiers.append(policy_verifier)
        client_ghost_modes.append(ghost)
        return client

    monkeypatch.setattr(activation, "load_runtime_project_router", load_project)
    monkeypatch.setattr(activation, "load_runtime_router", load_client)

    assert activation.load_project_router("support", tmp_path) is runtime
    assert activation.load_router("support", tmp_path) is client
    assert activation.load_router("support", tmp_path, ghost=True) is client
    assert project_verifiers == [activation.verify_automatic_router_policy]
    assert client_verifiers == [
        activation.verify_automatic_router_policy,
        activation.verify_automatic_router_policy,
    ]
    assert client_ghost_modes == [False, True]


def test_public_project_loader_has_no_provisional_bypass_parameter() -> None:
    """Production activation exposes no flag or alternate public provisional verifier."""
    parameters = inspect.signature(activation.load_project_router).parameters

    assert "allow_provisional" not in parameters
    assert "policy_verifier" not in parameters
    assert activation._load_project_router_for_composition.__name__.startswith("_")
