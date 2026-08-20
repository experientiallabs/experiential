"""Tests for project selection and gateway-backed Python compatibility."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from openai import OpenAI

from exp.common.routing import RoutingDecision
from exp.runtime.gateway.project_alias import ProjectGatewayAlias
from exp.runtime.router.application import RouterApplicationError, load_router


def test_load_router_uses_the_normal_gateway_application_and_revokes_its_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python compatibility client is an authenticated normal-gateway client."""
    application = FastAPI()

    @application.get("/v1/models")
    async def models() -> dict[str, object]:
        """Return the smallest official model-list response."""
        return {"object": "list", "data": []}

    prepared: list[tuple[str, Path, str | None]] = []
    loaded: list[tuple[Path, object, frozenset[str] | None]] = []
    issued: list[tuple[str, str]] = []
    revoked: list[str] = []

    def decision_sink(_decision: RoutingDecision) -> None:
        """Accept one served project selection."""

    class Management:
        """Capture the virtual-key lifecycle owned by the compatibility client."""

        def __init__(self, root: Path) -> None:
            """Require the requested EXP root."""
            assert root == tmp_path

        def issue_key(self, *, identity_id: str, key_id: str) -> object:
            """Return one synthetic raw key and record its owner."""
            issued.append((identity_id, key_id))
            return SimpleNamespace(raw_key="exp_test_key")

        def revoke_key(self, *, key_id: str) -> bool:
            """Record exact key revocation."""
            revoked.append(key_id)
            return True

    def prepare(
        project: str,
        root: Path,
        *,
        policy_id: str | None,
        project_repository: object,
        environment: object,
        runtime_catalog: object,
    ) -> ProjectGatewayAlias:
        """Return one already activated project-backed alias."""
        del environment, project_repository, runtime_catalog
        prepared.append((project, root, policy_id))
        return ProjectGatewayAlias(
            alias=project,
            alias_revision_id="revision-a",
            identity_id="project-identity",
            policy_id="policy-a",
            changed=True,
        )

    def load_gateway(
        root: Path,
        *,
        graceful_timeout_seconds: float,
        environment: object,
        project_repository: object,
        decision_sink: object,
        only_aliases: frozenset[str] | None,
    ) -> object:
        """Return the same gateway application used by the CLI launch path."""
        del graceful_timeout_seconds, environment, project_repository
        loaded.append((root, decision_sink, only_aliases))
        return SimpleNamespace(app=application)

    monkeypatch.setattr("exp.runtime.router.application.GatewayManagement", Management)
    monkeypatch.setattr(
        "exp.runtime.router.application.prepare_project_gateway_alias",
        prepare,
    )
    monkeypatch.setattr("exp.runtime.router.application.load_local_gateway", load_gateway)

    client = load_router(
        "support",
        root=tmp_path,
        policy_id="policy-a",
        decision_sink=decision_sink,
    )
    assert isinstance(client, OpenAI)
    assert client.models.list().data == []
    client.close()

    assert prepared == [("support", tmp_path, "policy-a")]
    assert loaded == [(tmp_path, decision_sink, frozenset({"support"}))]
    assert issued[0][0] == "project-identity"
    assert revoked == [issued[0][1]]


def test_ghost_compatibility_rejects_a_persistent_project_decision_sink(tmp_path: Path) -> None:
    """Ghost compatibility cannot silently persist project-selection decisions."""

    def decision_sink(_decision: object) -> None:
        """Accept one decision for the rejected option combination."""

    with pytest.raises(RouterApplicationError, match="ghost mode cannot use"):
        load_router("support", root=tmp_path, ghost=True, decision_sink=decision_sink)
