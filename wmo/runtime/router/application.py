"""Project loading and loopback-only application composition for a frozen router."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import OpenAI

from wmo.common.core.artifacts import ArtifactId
from wmo.common.models import load_model_catalog
from wmo.common.project import (
    ArtifactStore,
    ArtifactStoreError,
    ProjectStore,
    ProjectStoreError,
)
from wmo.common.routing import KnnRouterPolicy
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.endpoint import create_router_endpoint
from wmo.runtime.router.runtime import DecisionSink, RouterRuntime


class RouterApplicationError(ValueError):
    """A project cannot select one verified frozen router for local execution."""


def load_project_router(
    project: str,
    root: Path,
    *,
    policy_id: ArtifactId | None = None,
    environment: Mapping[str, str] | None = None,
    runtime_catalog: RuntimeModelCatalog | None = None,
    decision_sink: DecisionSink | None = None,
) -> RouterRuntime:
    """Load one project's frozen router without selecting or calling a provider.

    Args:
        project: Canonical local project identifier.
        root: Local ``.wmo`` root containing project artifacts and ``models.toml``.
        policy_id: Optional exact policy identity when the project contains more than one.
        environment: Optional credential mapping used only to construct runtime clients.
        runtime_catalog: Optional explicit runtime catalog for deterministic local tests.
        decision_sink: Optional aggregate-safe routing-decision recorder.

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


def create_project_router_app(project: str, runtime: RouterRuntime) -> FastAPI:
    """Create the dev-only loopback HTTP adapter for one loaded project router.

    Args:
        project: Public local model name accepted by the endpoint.
        runtime: Already verified frozen router runtime.

    Returns:
        FastAPI application exposing OpenAI Chat Completions and Responses routes.
    """
    application = FastAPI(
        title="WMO local router",
        description="Development-only loopback adapter over one frozen RouterRuntime.",
    )
    application.include_router(create_router_endpoint({project: runtime}))
    return application


def load_router(
    project: str,
    root: Path = Path(".wmo"),
    *,
    policy_id: ArtifactId | None = None,
    environment: Mapping[str, str] | None = None,
    runtime_catalog: RuntimeModelCatalog | None = None,
    decision_sink: DecisionSink | None = None,
) -> OpenAI:
    """Load one local project as an official synchronous OpenAI client.

    Args:
        project: Canonical project identifier and public OpenAI model name.
        root: Local ``.wmo`` root. Defaults to the happy-path project location.
        policy_id: Optional exact policy identity when the project contains several.
        environment: Optional credential mapping for runtime client construction.
        runtime_catalog: Optional explicit catalog for deterministic applications and tests.
        decision_sink: Optional aggregate-safe routing-decision recorder.

    Returns:
        Official OpenAI client whose Chat Completions and Responses resources call the local
        verified router in process. Close it or use it as a context manager when finished.
    """
    runtime = load_project_router(
        project,
        root,
        policy_id=policy_id,
        environment=environment,
        runtime_catalog=runtime_catalog,
        decision_sink=decision_sink,
    )
    application = create_project_router_app(project, runtime)
    transport = TestClient(application, raise_server_exceptions=False)
    return OpenAI(
        api_key="wmo-local-runtime",
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
