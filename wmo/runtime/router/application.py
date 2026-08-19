"""Project selection loading and gateway-backed Python compatibility client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import OpenAI

from wmo.common.core.artifacts import ArtifactId
from wmo.common.evaluations.evidence import read_evaluation_plan
from wmo.common.models import load_model_catalog
from wmo.common.project import (
    ArtifactStore,
    ArtifactStoreError,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from wmo.common.routing import KnnRouterPolicy
from wmo.runtime.gateway.lifecycle import load_local_gateway
from wmo.runtime.gateway.management import GatewayManagement
from wmo.runtime.gateway.project_alias import prepare_project_gateway_alias
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.errors import RouterApplicationError
from wmo.runtime.router.runtime import DecisionSink, RouterRuntime

RouterPolicyVerifier = Callable[[ArtifactStore, KnnRouterPolicy, RuntimeModelCatalog], None]

_AUTOMATIC_POLICY_INPUT_TYPES = frozenset(
    {"router-execution-contract", "router-runtime-capabilities"}
)


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
        root: Local ``.wmo`` root containing project artifacts and ``models.toml``.
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
    project_store = ProjectStore(root, project)
    try:
        project_store.load_project()
        resolved_policy_id = policy_id or _only_policy(project_store.artifacts)
        policy = _load_policy(project_store.artifacts, resolved_policy_id)
        catalog = runtime_catalog
        if catalog is None:
            catalog = RuntimeModelCatalog(
                load_model_catalog(project_store.model_catalog_path),
                environment=environment,
            )
        _verify_policy_activation(
            project_store.artifacts,
            policy,
            catalog,
            policy_verifier=policy_verifier,
        )
        return RouterRuntime.load(
            project_store.artifacts,
            resolved_policy_id,
            catalog,
            pricing_snapshot_id=policy.pricing_snapshot_id,
            decision_sink=decision_sink,
        )
    except RouterApplicationError:
        raise
    except (ArtifactStoreError, OSError, ProjectStoreError, ValueError) as exc:
        raise RouterApplicationError(str(exc)) from exc


def load_router(
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
    """Load one project alias through the normal authenticated gateway application.

    Args:
        project: Canonical project identifier and public OpenAI model name.
        root: Local ``.wmo`` root. Defaults to the happy-path project location.
        policy_id: Optional exact policy identity when the project contains several.
        environment: Optional credential mapping for runtime client construction.
        runtime_catalog: Optional explicit catalog for deterministic applications and tests.
        decision_sink: Optional aggregate-safe routing-decision recorder.
        policy_verifier: Optimizer-owned verification for automatic policy inputs.
        ghost: Compatibility flag that leaves project journals disabled. Gateway accounting,
            authentication, idempotency, and replay always remain enabled.

    Returns:
        Official OpenAI client whose Chat Completions and Responses resources call the same
        authenticated gateway application as ``wmo run``. Close it or use it as a context manager
        to revoke its ephemeral virtual key.
    """
    if ghost and decision_sink is not None:
        raise RouterApplicationError("ghost mode cannot use a routing decision sink")
    del ghost

    def project_loader(
        project: str,
        root: Path,
        *,
        policy_id: ArtifactId | None = None,
        environment: Mapping[str, str] | None = None,
        runtime_catalog: RuntimeModelCatalog | None = None,
    ) -> RouterRuntime:
        """Load the caller-selected immutable policy for gateway selection only."""
        return load_project_router(
            project,
            root,
            policy_id=policy_id,
            environment=environment,
            runtime_catalog=runtime_catalog,
            decision_sink=decision_sink,
            policy_verifier=policy_verifier,
        )

    alias = prepare_project_gateway_alias(
        project,
        root,
        policy_id=policy_id,
        project_loader=project_loader,
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
            project_loader=project_loader,
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
        base_url="http://wmo.local/v1",
        http_client=transport,
    )


def _only_policy(store: ArtifactStore) -> ArtifactId:
    """Resolve the only completed router policy, failing on absent or ambiguous state."""
    policies = tuple(
        artifact_id
        for artifact_id in store.list_ids()
        if store.read(artifact_id).manifest.artifact_type == "router-policy"
    )
    if not policies:
        raise RouterApplicationError("project has no frozen router policy; run wmo optimize router")
    if len(policies) > 1:
        raise RouterApplicationError(
            "project has multiple frozen router policies; pass --policy with one of: "
            + ", ".join(policies)
        )
    return policies[0]


def _load_policy(store: ArtifactStore, policy_id: ArtifactId) -> KnnRouterPolicy:
    """Load one manifest-verified policy envelope for runtime activation."""
    stored = store.read(policy_id)
    if stored.manifest.artifact_type != "router-policy":
        raise RouterApplicationError(f"artifact {policy_id} is not a frozen router policy")
    policy = KnnRouterPolicy.model_validate_json(store.read_bytes(policy_id, "policy.json"))
    if policy.policy_id != policy_id:
        raise RouterApplicationError("router policy identity differs from its artifact")
    return policy


def _verify_policy_activation(
    store: ArtifactStore,
    policy: KnnRouterPolicy,
    catalog: RuntimeModelCatalog,
    *,
    policy_verifier: RouterPolicyVerifier | None,
) -> None:
    """Require optimizer verification whenever a plan contains automatic artifacts.

    Runtime owns provider activation but does not depend on offline optimization. Automatic plans
    therefore require their optimizer owner to inject the complete artifact verifier. Plans with
    no automatic inputs retain the provider-free runtime loading path.

    Args:
        store: Project-local immutable artifact store.
        policy: Selected frozen router policy.
        catalog: Current credential-free catalog resolver.
        policy_verifier: Optional optimizer-owned automatic artifact verifier.

    Raises:
        RouterApplicationError: Plan inputs drift or automatic verification is unavailable.
    """
    plan, _plan_input = read_evaluation_plan(store, policy.evaluation_plan_id)
    automatic_inputs = []
    for item in plan.inputs:
        stored = store.read(item.artifact_id)
        if artifact_input(stored.manifest) != item:
            raise RouterApplicationError(
                f"router plan input {item.artifact_id!r} differs from its manifest"
            )
        if stored.manifest.artifact_type in _AUTOMATIC_POLICY_INPUT_TYPES:
            automatic_inputs.append(item)
    if policy_verifier is not None:
        policy_verifier(store, policy, catalog)
        return
    if automatic_inputs:
        raise RouterApplicationError(
            "automatic router policy requires optimizer-owned activation verification"
        )
