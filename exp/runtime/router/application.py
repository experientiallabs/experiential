"""Project selection loading and gateway-backed Python compatibility client."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import OpenAI

from exp.common.core.artifacts import ArtifactId
from exp.common.models import load_model_catalog
from exp.runtime.gateway.lifecycle import load_local_gateway
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.project_activation import (
    LocalArtifactProjectActivationRepository,
    ProjectActivationVerifier,
)
from exp.runtime.gateway.project_alias import prepare_project_gateway_alias
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.router.errors import RouterApplicationError
from exp.runtime.router.runtime import DecisionSink, RouterRuntime

RouterPolicyVerifier = ProjectActivationVerifier


class _RevokingGatewayClient(TestClient):
    """In-process gateway transport that revokes its private compatibility key on close."""

    def __init__(self, app: FastAPI, *, manager: GatewayManagement, key_id: str) -> None:
        """Bind one transport to the exact ephemeral key it owns."""
        super().__init__(app, raise_server_exceptions=False)
        self._manager = manager
        self._key_id = key_id

    def close(self) -> None:
        """Close the transport and revoke its key even if client teardown fails."""
        try:
            super().close()
        finally:
            self._manager.revoke_key(key_id=self._key_id)


def load_project_router(
    project: str,
    root: Path,
    *,
    policy_id: ArtifactId | None = None,
    environment: Mapping[str, str] | None = None,
    runtime_catalog: RuntimeModelCatalog | None = None,
    decision_sink: DecisionSink | None = None,
    policy_verifier: RouterPolicyVerifier | None = None,
) -> RouterRuntime:
    """Load one project's frozen router without selecting or calling a provider.

    Args:
        project: Canonical local project identifier.
        root: Local ``.exp`` root containing project artifacts and ``models.toml``.
        policy_id: Optional exact policy identity when the project contains more than one.
        environment: Optional credential mapping used only to construct runtime clients.
        runtime_catalog: Optional explicit runtime catalog for deterministic local tests.
        decision_sink: Optional aggregate-safe routing-decision recorder.
        policy_verifier: Optimizer-owned verification for automatic policy inputs.

    Returns:
        Activated immutable router runtime. No model request is issued during loading.

    Raises:
        RouterApplicationError: Project, policy, catalog, or frozen identity is invalid.
    """
    try:
        catalog = runtime_catalog
        if catalog is None:
            catalog = RuntimeModelCatalog(
                load_model_catalog(root / "models.toml"),
                environment=environment,
            )
        repository = LocalArtifactProjectActivationRepository(
            root,
            verifier=policy_verifier,
        )
        activation = repository.load(
            project,
            policy_id,
            runtime_catalog=catalog,
        )
        return RouterRuntime.from_activation(
            activation,
            catalog,
            decision_sink=decision_sink,
        )
    except RouterApplicationError:
        raise
    except (OSError, ValueError) as exc:
        raise RouterApplicationError(str(exc)) from exc


def load_router(
    project: str,
    root: Path = Path(".exp"),
    *,
    policy_id: ArtifactId | None = None,
    environment: Mapping[str, str] | None = None,
    runtime_catalog: RuntimeModelCatalog | None = None,
    decision_sink: DecisionSink | None = None,
    policy_verifier: RouterPolicyVerifier | None = None,
    ghost: bool = False,
) -> OpenAI:
    """Load one project alias through the normal authenticated gateway application.

    Args:
        project: Canonical project identifier and public OpenAI model name.
        root: Local ``.exp`` root. Defaults to the happy-path project location.
        policy_id: Optional exact policy identity when the project contains several.
        environment: Optional credential mapping for runtime client construction.
        runtime_catalog: Optional explicit catalog for deterministic applications and tests.
        decision_sink: Optional aggregate-safe routing-decision recorder.
        policy_verifier: Optimizer-owned verification for automatic policy inputs.
        ghost: Compatibility flag that leaves project journals disabled. Gateway accounting,
            authentication, idempotency, and replay always remain enabled.

    Returns:
        Official OpenAI client whose Chat Completions and Responses resources call the same
        authenticated gateway application as ``exp run``. Close it or use it as a context manager
        to revoke its ephemeral virtual key.
    """
    if ghost and decision_sink is not None:
        raise RouterApplicationError("ghost mode cannot use a routing decision sink")
    del ghost

    repository = LocalArtifactProjectActivationRepository(
        root,
        verifier=policy_verifier,
    )

    alias = prepare_project_gateway_alias(
        project,
        root,
        policy_id=policy_id,
        project_repository=repository,
        environment=environment,
        runtime_catalog=runtime_catalog,
    )
    manager = GatewayManagement(root)
    key_id = f"python-{uuid4().hex}"
    issued = manager.issue_key(identity_id=alias.identity_id, key_id=key_id)
    try:
        runtime = load_local_gateway(
            root,
            graceful_timeout_seconds=10,
            environment=environment,
            project_repository=repository,
            decision_sink=decision_sink,
            only_aliases=frozenset({alias.alias}),
        )
    except BaseException:
        manager.revoke_key(key_id=key_id)
        raise
    transport = _RevokingGatewayClient(
        runtime.app,
        manager=manager,
        key_id=key_id,
    )
    return OpenAI(
        api_key=issued.raw_key,
        base_url="http://exp.local/v1",
        http_client=transport,
    )
