"""Loopback launch command for the single local gateway runtime."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.cli.shared.theme import EXP_THEME

_LOOPBACK_HOST = "127.0.0.1"
_console = Console(theme=EXP_THEME)
_EXP_WORDMARK = "\n".join(
    (
        "███████╗██╗  ██╗██████╗ ",
        "██╔════╝╚██╗██╔╝██╔══██╗",
        "█████╗   ╚███╔╝ ██████╔╝",
        "██╔══╝   ██╔██╗ ██╔═══╝ ",
        "███████╗██╔╝ ██╗██║     ",
        "╚══════╝╚═╝  ╚═╝╚═╝     ",
    )
)
_POLICY_OPTION = typer.Option(
    None,
    "--policy",
    help="Exact frozen policy ID for the optional project-backed alias.",
)
_PORT_OPTION = typer.Option(8000, "--port", min=1, max=65_535)
_GHOST_OPTION = typer.Option(
    False,
    "--ghost",
    help="Compatibility flag: project journals stay disabled while gateway accounting remains on.",
)
_NON_INTERACTIVE_OPTION = typer.Option(
    False,
    "--non-interactive",
    help="Never open first-run prompts.",
)
_JSON_OPTION = typer.Option(False, "--json", help="Write a versioned launch receipt.")
_CHECK_OPTION = typer.Option(False, "--check", help="Validate readiness and exit without binding.")
_GRACEFUL_TIMEOUT_OPTION = typer.Option(
    10.0,
    "--graceful-timeout",
    min=0.1,
    help="Seconds to drain admitted gateway work during shutdown.",
)
_ENGINE_OPTION = typer.Option(
    "auto",
    "--engine",
    help=(
        "Data-plane engine: 'auto' (rust when built, otherwise python), 'rust' "
        "(native data plane with an embedded python engine for Responses, "
        "replay, and project aliases), or 'python' (uvicorn only)."
    ),
)
_MAX_ACTIVE_REQUESTS_DEFAULT = 1024
_MAX_ACTIVE_REQUESTS_OPTION = typer.Option(
    _MAX_ACTIVE_REQUESTS_DEFAULT,
    "--max-active-requests",
    min=1,
    help="Rust engine only: maximum concurrently admitted requests.",
)


def run(
    project: str | None = typer.Argument(
        None,
        help="Optional project to expose as one project-backed gateway alias.",
    ),
    root: Path = ROOT_OPTION,
    policy: str | None = _POLICY_OPTION,
    port: int = _PORT_OPTION,
    ghost: bool = _GHOST_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
    check: bool = _CHECK_OPTION,
    graceful_timeout: float = _GRACEFUL_TIMEOUT_OPTION,
    engine: str = _ENGINE_OPTION,
    max_active_requests: int = _MAX_ACTIVE_REQUESTS_OPTION,
) -> None:
    """Start the local gateway, optionally materializing one project-backed alias.

    Args:
        project: Optional project identifier and endpoint alias.
        root: Local artifact and model-catalog root.
        policy: Exact policy for an ambiguous legacy project.
        port: Local loopback TCP port.
        ghost: Compatibility marker for project traffic, which always uses gateway accounting.
        non_interactive: Whether first-run gateway prompts are forbidden.
        json_output: Whether startup output is one versioned JSON receipt.
        check: Whether to validate gateway readiness without binding.
        graceful_timeout: Gateway shutdown drain bound in seconds.
        engine: Data-plane engine: ``auto``, ``rust``, or ``python``.
        max_active_requests: Rust engine concurrent-admission bound.

    Raises:
        typer.BadParameter: The selected form or activation is invalid.
    """
    if engine not in {"auto", "rust", "python"}:
        raise typer.BadParameter("--engine must be 'auto', 'rust', or 'python'")
    if policy is not None or ghost:
        if project is None:
            raise typer.BadParameter("--policy and --ghost require the 'exp run PROJECT' form")
    if engine == "rust" and project is not None:
        raise typer.BadParameter(
            "--engine rust serves direct aliases only; project-backed serving "
            "requires the python engine"
        )
    if project is None and engine != "python":
        blocker = _rust_engine_blocker(root)
        if blocker is None:
            _run_rust_gateway(
                root=root,
                port=port,
                json_output=json_output,
                check=check,
                max_active_requests=max_active_requests,
                graceful_timeout=graceful_timeout,
            )
            return
        if engine == "rust":
            if blocker == _NOT_INITIALIZED_BLOCKER:
                _gateway_not_initialized(json_output=json_output)
            raise typer.BadParameter(blocker)
        if not json_output:
            _console.print(f"[yellow]rust engine unavailable ({blocker}); using python[/yellow]")
    if max_active_requests != _MAX_ACTIVE_REQUESTS_DEFAULT:
        typer.echo(
            "--max-active-requests applies only to the rust engine; the python "
            "engine keeps its own executor bound",
            err=True,
        )
    _run_gateway(
        project=project,
        root=root,
        policy=policy,
        port=port,
        ghost=ghost,
        non_interactive=non_interactive,
        json_output=json_output,
        check=check,
        graceful_timeout=graceful_timeout,
    )


def _run_gateway(
    *,
    project: str | None,
    root: Path,
    policy: str | None,
    port: int,
    ghost: bool,
    non_interactive: bool,
    json_output: bool,
    check: bool,
    graceful_timeout: float,
) -> None:
    """Validate and optionally serve the normal gateway application."""
    if not json_output:
        _emit_exp_wordmark()

    import uvicorn

    from exp.cli.gateway.compatibility import prepare_project_gateway
    from exp.cli.gateway.setup import interactive_gateway_setup
    from exp.optimize.router.activation import verify_automatic_router_policy
    from exp.runtime.gateway.lifecycle import (
        gateway_instance_lock,
        load_local_gateway,
    )
    from exp.runtime.gateway.management import GatewayManagement
    from exp.runtime.gateway.project_activation import LocalArtifactProjectActivationRepository

    setup = None
    manager = GatewayManagement(root)
    compatibility = None
    if project is not None:
        with usage_error(ValueError):
            compatibility = prepare_project_gateway(project, root, policy_id=policy)
    elif not manager.initialized:
        if non_interactive or json_output or not sys.stdin.isatty() or not sys.stdout.isatty():
            _gateway_not_initialized(json_output=json_output)
        with usage_error(ValueError):
            setup = interactive_gateway_setup(root)
        if setup is not None:
            _emit_setup_credentials(port=port, setup=setup)

    try:
        with usage_error(ValueError):
            with gateway_instance_lock(root, port=port):
                project_repository = LocalArtifactProjectActivationRepository(
                    root,
                    verifier=verify_automatic_router_policy,
                )
                runtime = load_local_gateway(
                    root,
                    graceful_timeout_seconds=graceful_timeout,
                    project_repository=project_repository,
                    only_aliases=(
                        None if compatibility is None else frozenset({compatibility.alias})
                    ),
                )
                asyncio.run(runtime.service.preflight())
                receipt = {
                    "schema_version": 1,
                    "operation": "gateway.check" if check else "gateway.run",
                    "status": "ready",
                    "base_url": f"http://{_LOOPBACK_HOST}:{port}/v1",
                    "usage_url": f"http://{_LOOPBACK_HOST}:{port}/usage",
                    "reconciled_expired_requests": runtime.reconciled_expired_requests,
                    "reconciled_unknown_attempts": runtime.reconciled_unknown_attempts,
                    "launch_mode": "gateway" if compatibility is None else "project_alias",
                }
                if compatibility is not None:
                    receipt.update(
                        {
                            "project_alias": compatibility.alias,
                            "alias_revision_id": compatibility.alias_revision_id,
                            "policy_id": compatibility.policy_id,
                            "key_file": str(compatibility.key_file),
                            "project_journal": "disabled",
                            "gateway_accounting": "enabled",
                        }
                    )
                if json_output:
                    typer.echo(json.dumps(receipt, separators=(",", ":")))
                else:
                    _emit_gateway_ready(
                        port=port,
                        compatibility=compatibility,
                        ghost=ghost,
                    )
                if check:
                    return
                uvicorn.run(runtime.app, host=_LOOPBACK_HOST, port=port)
    except typer.BadParameter:
        if setup is not None:
            _emit_setup_recovery(setup=setup)
        raise


_NOT_INITIALIZED_BLOCKER = "the gateway is not initialized yet"


def _rust_engine_blocker(root: Path) -> str | None:
    """Return why the rust data plane cannot serve this root, or ``None``.

    Args:
        root: Local artifact and model-catalog root.

    Returns:
        A display-safe reason to use the python engine, or ``None`` when the
        rust engine can serve every granted active alias.
    """
    import importlib.util

    from exp.runtime.gateway.management import GatewayManagement

    if importlib.util.find_spec("exp_gateway_native") is None:
        return (
            "the exp_gateway_native extension is not built; run 'just native' "
            "(uv run maturin develop --uv --release "
            "--manifest-path exp/runtime/gateway/native/Cargo.toml)"
        )
    manager = GatewayManagement(root)
    if not manager.initialized:
        return _NOT_INITIALIZED_BLOCKER
    return None


def _run_rust_gateway(
    *,
    root: Path,
    port: int,
    json_output: bool,
    check: bool,
    max_active_requests: int,
    graceful_timeout: float,
) -> None:
    """Serve the rust data plane with an embedded python fallback engine.

    The rust engine owns the public socket and the anonymous Chat Completions
    fast path; a python engine over the same authority, ledger, and routes
    runs on an internal loopback port and serves Responses, replay-keyed
    chat, project-backed aliases, and the usage page.

    Args:
        root: Local artifact and model-catalog root.
        port: Local loopback TCP port.
        json_output: Whether startup output is one versioned JSON receipt.
        check: Whether to validate gateway readiness without binding.
        max_active_requests: Concurrent-admission bound for the data plane.
        graceful_timeout: Gateway shutdown drain bound in seconds.

    Raises:
        typer.BadParameter: The extension module is missing or the gateway
            configuration cannot form one ready route.
    """
    if not json_output:
        _emit_exp_wordmark()

    import importlib
    import socket
    import threading
    import time

    import uvicorn

    from exp.optimize.router.activation import verify_automatic_router_policy
    from exp.runtime.gateway.lifecycle import (
        compose_local_gateway,
        gateway_instance_lock,
        load_gateway_components,
    )
    from exp.runtime.gateway.management import GatewayManagement
    from exp.runtime.gateway.native_bridge import NativeControlPlane
    from exp.runtime.gateway.project_activation import LocalArtifactProjectActivationRepository

    try:
        exp_gateway_native = importlib.import_module("exp_gateway_native")
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "the Rust gateway engine is not built; run 'just native' "
            "(uv run maturin develop --uv --release "
            "--manifest-path exp/runtime/gateway/native/Cargo.toml) and retry"
        ) from exc

    manager = GatewayManagement(root)
    if not manager.initialized:
        _gateway_not_initialized(json_output=json_output)
    with usage_error(ValueError):
        with gateway_instance_lock(root, port=port):
            project_repository = LocalArtifactProjectActivationRepository(
                root,
                verifier=verify_automatic_router_policy,
            )
            components = load_gateway_components(root, project_repository=project_repository)
            control_plane = NativeControlPlane(components)
            runtime = compose_local_gateway(
                components,
                graceful_timeout_seconds=graceful_timeout,
            )
            asyncio.run(runtime.service.preflight())
            receipt = {
                "schema_version": 1,
                "operation": "gateway.check" if check else "gateway.run",
                "status": "ready",
                "engine": "rust",
                "base_url": f"http://{_LOOPBACK_HOST}:{port}/v1",
                "usage_url": f"http://{_LOOPBACK_HOST}:{port}/usage",
                "reconciled_expired_requests": control_plane.reconciled_expired_requests,
                "reconciled_unknown_attempts": control_plane.reconciled_unknown_attempts,
                "launch_mode": "gateway",
            }
            if json_output:
                typer.echo(json.dumps(receipt, separators=(",", ":")))
            elif not check:
                _console.print(
                    f"[green]Gateway ready (rust engine)[/green] http://{_LOOPBACK_HOST}:{port}/v1",
                    markup=True,
                )
            if check:
                return

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind((_LOOPBACK_HOST, port))
                except OSError as exc:
                    raise typer.BadParameter(
                        f"port {port} is unavailable on {_LOOPBACK_HOST}: {exc}"
                    ) from exc

            # The fallback socket stays bound from allocation to serving so
            # the ephemeral port cannot be lost to another process.
            fallback_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            fallback_socket.bind((_LOOPBACK_HOST, 0))
            fallback_port = fallback_socket.getsockname()[1]
            fallback = uvicorn.Server(
                uvicorn.Config(
                    runtime.app,
                    host=_LOOPBACK_HOST,
                    port=fallback_port,
                    log_level="warning",
                )
            )
            fallback_thread = threading.Thread(
                target=lambda: fallback.run(sockets=[fallback_socket]),
                name="exp-fallback-engine",
                daemon=True,
            )
            fallback_thread.start()
            deadline = time.monotonic() + 30
            while not fallback.started:
                if not fallback_thread.is_alive() or time.monotonic() > deadline:
                    raise typer.BadParameter(
                        "the embedded python fallback engine failed to start; "
                        "inspect the gateway configuration with 'exp run --engine python'"
                    )
                time.sleep(0.05)
            try:
                config = json.dumps(
                    {
                        "host": _LOOPBACK_HOST,
                        "port": port,
                        "max_active_requests": max_active_requests,
                        "request_timeout_seconds": control_plane.request_timeout_seconds,
                        "fallback_port": fallback_port,
                        "graceful_timeout_seconds": graceful_timeout,
                    }
                )
                try:
                    exp_gateway_native.serve(control_plane, config)
                except RuntimeError as exc:
                    raise typer.BadParameter(f"the gateway engine failed: {exc}") from exc
            finally:
                fallback.should_exit = True
                fallback_thread.join(timeout=graceful_timeout + 5)
                fallback_socket.close()


def _emit_exp_wordmark() -> None:
    """Render the block-letter EXP wordmark for human-facing gateway startup output."""
    _console.print(Text(_EXP_WORDMARK, style="bold green"))


def _gateway_not_initialized(*, json_output: bool) -> None:
    """Return a stable empty-state error and exact non-interactive next commands."""
    commands = (
        "exp config gateway init --non-interactive --json",
        "exp config gateway provider add NAME --provider PROVIDER "
        "--credential-env ENV --non-interactive --json",
        "exp config gateway alias create ALIAS --deployment CONNECTION:MODEL "
        "--exact-model EXACT --non-interactive --json",
        "exp config gateway identity create default --non-interactive --json",
        "exp config gateway grant add default ALIAS --non-interactive --json",
        "exp config gateway key issue default --key-id KEY --json",
    )
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "error": {
                        "code": "gateway_not_initialized",
                        "message": "The local gateway has no explicit configuration.",
                        "next_commands": commands,
                    },
                },
                separators=(",", ":"),
            )
        )
    else:
        typer.echo("gateway_not_initialized: run these commands:", err=True)
        for command in commands:
            typer.echo(f"  {command}", err=True)
    raise typer.Exit(2)


def _emit_setup_credentials(*, port: int, setup: object) -> None:
    """Deliver first-run gateway credentials before independent readiness checks run.

    Args:
        port: Loopback port that the gateway will serve when ready.
        setup: First-run setup result containing the one-time raw gateway key.

    Raises:
        TypeError: The setup result is not the gateway setup contract.
    """
    from exp.cli.gateway.setup import InteractiveSetupResult

    if not isinstance(setup, InteractiveSetupResult):
        raise TypeError("interactive setup returned an invalid result")
    _console.print(f"export EXP_GATEWAY_URL=http://{_LOOPBACK_HOST}:{port}/v1", markup=False)
    _console.print(f"export EXP_GATEWAY_KEY={setup.raw_key}", markup=False)


def _emit_setup_recovery(*, setup: object) -> None:
    """Print recovery steps when first-run setup outlives a failed readiness check.

    Args:
        setup: First-run setup result identifying the identity that owns the key.

    Raises:
        TypeError: The setup result is not the gateway setup contract.
    """
    from exp.cli.gateway.setup import InteractiveSetupResult

    if not isinstance(setup, InteractiveSetupResult):
        raise TypeError("interactive setup returned an invalid result")
    _console.print(
        "[yellow]First-run gateway setup completed, but the gateway is not ready.[/yellow]",
        markup=True,
    )
    _console.print(
        "Keep the gateway credentials printed above, fix the listed provider configuration, "
        "and rerun `exp run`.",
        markup=False,
    )
    _console.print(
        "If the key was not saved, issue a replacement with:",
        markup=False,
    )
    _console.print(
        f"  exp config gateway key issue {setup.identity_id} --key-id RECOVERY_KEY --json",
        markup=False,
    )


def _emit_gateway_ready(
    *,
    port: int,
    compatibility: object | None,
    ghost: bool,
) -> None:
    """Print the green startup result and project compatibility credentials."""
    _console.print(
        f"[green]✓ Gateway ready[/green] http://{_LOOPBACK_HOST}:{port}/v1",
        markup=True,
    )
    if compatibility is not None:
        from exp.cli.gateway.compatibility import ProjectGatewayCompatibility

        if not isinstance(compatibility, ProjectGatewayCompatibility):
            raise TypeError("project gateway compatibility returned an invalid result")
        _console.print(
            f"export EXP_GATEWAY_URL=http://{_LOOPBACK_HOST}:{port}/v1",
            markup=False,
        )
        _console.print(
            f"export EXP_GATEWAY_KEY=\"$(tr -d '\\n' < {compatibility.key_file})\"",
            markup=False,
        )
