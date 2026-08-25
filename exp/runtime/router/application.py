"""Project selection loading and the gateway-backed Python compatibility client."""

from __future__ import annotations

import importlib
import socket
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import httpx
import httpx2
from openai import OpenAI

from exp.common.core.artifacts import ArtifactId
from exp.common.models import load_model_catalog
from exp.runtime.gateway.guardrails.config import load_guardrail_engine
from exp.runtime.gateway.lifecycle import LocalGatewayComponents, load_gateway_components
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_execution import native_serving_blockers
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.project_activation import (
    LocalArtifactProjectActivationRepository,
    ProjectActivationVerifier,
)
from exp.runtime.gateway.project_alias import prepare_project_gateway_alias
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.router.errors import RouterApplicationError
from exp.runtime.router.runtime import DecisionSink, RouterRuntime

RouterPolicyVerifier = ProjectActivationVerifier

_LOOPBACK_HOST = "127.0.0.1"
_STARTUP_TIMEOUT_SECONDS = 30.0
_SHUTDOWN_JOIN_SECONDS = 15.0


class _OwnedGateway:
    """One private native gateway serving a single project alias.

    The compatibility client owns the whole serving stack: the loaded
    components, the native data plane on a loopback port in a background
    thread, the embedder stop handle, and the ephemeral virtual key. Closing
    the transport stops the plane gracefully, drains the shared ledger
    writer, and revokes the key.
    """

    def __init__(
        self,
        *,
        components: LocalGatewayComponents,
        manager: GatewayManagement,
        key_id: str,
        port: int,
    ) -> None:
        """Bind the owned serving resources before the plane starts.

        Args:
            components: Loaded engine-neutral gateway components.
            manager: Gateway management owning the ephemeral key.
            key_id: Exact ephemeral key to revoke on close.
            port: Loopback port the native plane will bind.
        """
        self.components = components
        self.manager = manager
        self.key_id = key_id
        self.port = port
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        native = importlib.import_module("exp_gateway_native")
        self.shutdown = native.shutdown_handle()
        self.control_plane = NativeControlPlane(
            components,
            data_plane_metrics=native.metrics_snapshot_json,
            guardrails=load_guardrail_engine(manager.root),
        )

    def start(self) -> None:
        """Serve the native plane on a background thread and await liveness.

        Raises:
            RouterApplicationError: The plane did not report live within the
                startup bound, or serving failed outright.
        """

        def serve() -> None:
            """Run the blocking native server, retaining any failure."""
            try:
                serve_native_gateway(
                    self.control_plane,
                    host=_LOOPBACK_HOST,
                    port=self.port,
                    shutdown=self.shutdown,
                )
            except BaseException as error:  # noqa: BLE001 - surfaced to the starter.
                self.error = error

        self.thread = threading.Thread(target=serve, name="exp-router-gateway", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.error is not None or not self.thread.is_alive():
                error = self.error
                self.stop()
                raise RouterApplicationError(f"the compatibility gateway failed to start: {error}")
            try:
                live = httpx.get(
                    f"http://{_LOOPBACK_HOST}:{self.port}/health/live",
                    timeout=1.0,
                )
            except httpx.HTTPError:
                time.sleep(0.05)
                continue
            if live.status_code == 200:
                return
            time.sleep(0.05)
        self.stop()
        raise RouterApplicationError(
            "the compatibility gateway did not become live within the startup bound"
        )

    def stop(self) -> None:
        """Stop the plane, drain the writer, and revoke the ephemeral key."""
        try:
            self.shutdown.request_shutdown()
            if self.thread is not None:
                self.thread.join(timeout=_SHUTDOWN_JOIN_SECONDS)
        finally:
            try:
                self.components.selection_workers.shutdown()
                self.components.write_ledger.close()
            finally:
                self.manager.revoke_key(key_id=self.key_id)


class _RevokingGatewayClient(httpx2.Client):
    """Loopback gateway transport that stops its owned plane on close."""

    def __init__(self, gateway: _OwnedGateway) -> None:
        """Bind one transport to the exact serving stack it owns."""
        super().__init__()
        self._gateway = gateway

    def close(self) -> None:
        """Close the transport and stop the owned gateway even on failure."""
        try:
            super().close()
        finally:
            self._gateway.stop()


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
    """Load one project alias through the normal authenticated native gateway.

    The alias is served by the native data plane on a private loopback port
    owned by the returned client: authorization, accounting, idempotent
    replay, and learned selection run through exactly the serving stack the
    default ``exp`` flow uses.

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
        Official OpenAI client whose Chat Completions and Responses resources call the
        authenticated native gateway. Close it or use it as a context manager to stop the
        owned gateway and revoke its ephemeral virtual key.

    Raises:
        RouterApplicationError: Ghost mode was combined with a decision sink, an alias
            is not natively servable, or the owned gateway failed to start.
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
    gateway: _OwnedGateway | None = None
    try:
        components = load_gateway_components(
            root,
            environment=environment,
            project_repository=repository,
            decision_sink=decision_sink,
            only_aliases=frozenset({alias.alias}),
        )
        blockers = native_serving_blockers(components)
        if blockers:
            raise RouterApplicationError(
                "the native engine cannot serve the project alias: " + "; ".join(blockers)
            )
        gateway = _OwnedGateway(
            components=components,
            manager=manager,
            key_id=key_id,
            port=_free_loopback_port(),
        )
        gateway.start()
    except BaseException:
        if gateway is None:
            manager.revoke_key(key_id=key_id)
        raise
    return OpenAI(
        api_key=issued.raw_key,
        base_url=f"http://{_LOOPBACK_HOST}:{gateway.port}/v1",
        http_client=_RevokingGatewayClient(gateway),
    )


def _free_loopback_port() -> int:
    """Reserve one currently free loopback port for the owned plane to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((_LOOPBACK_HOST, 0))
        return probe.getsockname()[1]
