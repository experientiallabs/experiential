"""Project loading and loopback-only application composition for a frozen router."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from openai import OpenAI

from wmo.common.core.artifacts import ArtifactId
from wmo.common.models import ModelRequest, load_model_catalog
from wmo.common.project import (
    ArtifactStore,
    ArtifactStoreError,
    ProjectStore,
    ProjectStoreError,
)
from wmo.common.routing import KnnRouterPolicy
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.capability import (
    load_router_runtime_capability_contract,
    verify_router_runtime_capabilities,
)
from wmo.runtime.router.completion import (
    JournaledRouterCompletionService,
    RouterCompletionService,
)
from wmo.runtime.router.endpoint import create_router_endpoint
from wmo.runtime.router.journal import JournaledRouterRuntime, RuntimeInteractionJournal
from wmo.runtime.router.runtime import DecisionSink, RoutedModelResponse, RouterRuntime


class RouterApplicationError(ValueError):
    """A project cannot select one verified frozen router for local execution."""


class _GhostRouterCompletionService:
    """Route requests without creating durable interaction state."""

    def __init__(self, runtime: RouterRuntime) -> None:
        """Bind the stateless completion adapter to one verified runtime."""
        self._runtime = runtime

    def complete(
        self,
        request: ModelRequest,
        *,
        idempotency_key: str,
        conversation_id: str | None = None,
    ) -> RoutedModelResponse:
        """Dispatch directly while accepting but not persisting caller identity.

        Args:
            request: Provider-neutral request to route and execute.
            idempotency_key: Caller key intentionally ignored in ghost mode.
            conversation_id: Optional in-memory Responses affinity identity.

        Returns:
            Newly routed provider response with no durable replay record.
        """
        del idempotency_key
        return self._runtime.complete(request, episode_id=conversation_id)


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
        _verify_automatic_runtime_capabilities(project_store.artifacts, policy, catalog)
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


def create_project_router_app(
    project: str,
    runtime: RouterRuntime,
    *,
    completion_service: RouterCompletionService | None = None,
) -> FastAPI:
    """Create the dev-only loopback HTTP adapter for one loaded project router.

    Args:
        project: Public local model name accepted by the endpoint.
        runtime: Already verified frozen router runtime.
        completion_service: Optional durable service for standard idempotent requests.

    Returns:
        FastAPI application exposing OpenAI Chat Completions and Responses routes.
    """
    application = FastAPI(
        title="WMO local router",
        description="Development-only loopback adapter over one frozen RouterRuntime.",
    )

    @application.exception_handler(RequestValidationError)
    async def openai_validation_error(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        """Return the OpenAI error envelope for public request validation failures."""
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid OpenAI request",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_request",
                }
            },
        )

    services = {project: completion_service} if completion_service is not None else None
    application.include_router(
        create_router_endpoint({project: runtime}, completion_services=services)
    )
    return application


def create_project_completion_service(
    store: ProjectStore,
    runtime: RouterRuntime,
    *,
    ghost: bool = False,
) -> RouterCompletionService:
    """Compose one project's selected durable or ghost completion boundary.

    Args:
        store: Initialized project that owns the selected runtime policy.
        runtime: Loaded frozen router wrapped by the selected traffic service.
        ghost: Whether requests must bypass all durable interaction state.

    Returns:
        Neutral completion service shared by every endpoint request in one application. Ghost
        services accept caller keys but never provide durable idempotent replay.

    Raises:
        RouterApplicationError: The store does not own the runtime's exact verified policy.
    """
    try:
        stored_policy = _load_policy(store.artifacts, runtime.policy.policy_id)
    except (ArtifactStoreError, OSError, ValueError) as exc:
        raise RouterApplicationError(
            "project cannot bind runtime journaling without its verified router policy"
        ) from exc
    if stored_policy != runtime.policy:
        raise RouterApplicationError(
            "project router policy content differs from the runtime selected for completion"
        )
    if ghost:
        if runtime.records_decisions:
            raise RouterApplicationError("ghost mode cannot use a routing decision sink")
        return _GhostRouterCompletionService(runtime)
    return JournaledRouterCompletionService(
        JournaledRouterRuntime(runtime, RuntimeInteractionJournal(store.paths))
    )


def load_router(
    project: str,
    root: Path = Path(".wmo"),
    *,
    policy_id: ArtifactId | None = None,
    environment: Mapping[str, str] | None = None,
    runtime_catalog: RuntimeModelCatalog | None = None,
    decision_sink: DecisionSink | None = None,
    ghost: bool = False,
) -> OpenAI:
    """Load one local project as an official synchronous OpenAI client.

    Args:
        project: Canonical project identifier and public OpenAI model name.
        root: Local ``.wmo`` root. Defaults to the happy-path project location.
        policy_id: Optional exact policy identity when the project contains several.
        environment: Optional credential mapping for runtime client construction.
        runtime_catalog: Optional explicit catalog for deterministic applications and tests.
        decision_sink: Optional aggregate-safe routing-decision recorder.
        ghost: Whether completed traffic must bypass durable journal and replay state.

    Returns:
        Official OpenAI client whose Chat Completions and Responses resources call the local
        verified router in process. By default every completion is journaled; ghost mode keeps no
        durable traffic or replay state. Close it or use it as a context manager when finished.
    """
    if ghost and decision_sink is not None:
        raise RouterApplicationError("ghost mode cannot use a routing decision sink")
    runtime = load_project_router(
        project,
        root,
        policy_id=policy_id,
        environment=environment,
        runtime_catalog=runtime_catalog,
        decision_sink=decision_sink,
    )
    store = ProjectStore(root, project)
    completion_service = create_project_completion_service(store, runtime, ghost=ghost)
    application = create_project_router_app(
        project,
        runtime,
        completion_service=completion_service,
    )
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


def _verify_automatic_runtime_capabilities(
    store: ArtifactStore,
    policy: KnnRouterPolicy,
    catalog: RuntimeModelCatalog,
) -> None:
    """Verify an automatic plan's separate capability contract before credential access.

    Legacy plans contain neither automatic artifact and retain their existing activation path.
    Automatic plans must freeze exactly one capability contract and one execution contract, with
    the execution manifest recursively binding the capability pointer.

    Args:
        store: Project-local immutable artifact store.
        policy: Selected frozen router policy.
        catalog: Current credential-free catalog resolver.

    Raises:
        RouterApplicationError: Automatic artifacts are missing, ambiguous, or drifted.
    """
    from wmo.common.evaluations.evidence import read_evaluation_plan
    from wmo.common.project import artifact_input
    from wmo.workflow.router_execution_contract import load_router_execution_contract

    plan, _plan_input = read_evaluation_plan(store, policy.evaluation_plan_id)
    capability_inputs = []
    execution_inputs = []
    for item in plan.inputs:
        stored = store.read(item.artifact_id)
        if artifact_input(stored.manifest) != item:
            raise RouterApplicationError(
                f"router plan input {item.artifact_id!r} differs from its manifest"
            )
        if stored.manifest.artifact_type == "router-runtime-capabilities":
            capability_inputs.append(item)
        elif stored.manifest.artifact_type == "router-execution-contract":
            execution_inputs.append(item)
    if not capability_inputs and not execution_inputs:
        return
    if len(capability_inputs) != 1 or len(execution_inputs) != 1:
        raise RouterApplicationError(
            "automatic router plan must bind one capability and one execution contract"
        )
    capability_input = capability_inputs[0]
    execution = load_router_execution_contract(store, execution_inputs[0].artifact_id)
    if execution.runtime_capability_input != capability_input:
        raise RouterApplicationError(
            "automatic router execution contract differs from its runtime capability binding"
        )
    execution_candidates = tuple(
        (item.candidate_alias, item.model) for item in execution.candidates
    )
    policy_candidates = tuple((item.alias, item.model) for item in policy.candidates)
    if (
        execution_candidates != policy_candidates
        or execution.incumbent_alias != policy.baseline_alias
    ):
        raise RouterApplicationError(
            "automatic router execution candidates or incumbent differ from the policy"
        )
    contract = load_router_runtime_capability_contract(store, capability_input.artifact_id)
    verify_router_runtime_capabilities(contract, policy.candidates, catalog)
