"""Loopback launch command for the single local gateway runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError

LOOPBACK_HOST = "127.0.0.1"
_LOOPBACK_HOST = LOOPBACK_HOST
DEFAULT_GATEWAY_PORT = 8000
DEFAULT_GRACEFUL_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_ACTIVE_REQUESTS = 1024
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


def run(
    project: str | None = typer.Argument(
        None,
        help="Optional project to expose as one project-backed gateway alias.",
    ),
    root: Path = ROOT_OPTION,
    policy: str | None = typer.Option(
        None,
        "--policy",
        help="Exact frozen policy ID for the optional project-backed alias.",
    ),
    port: int = typer.Option(DEFAULT_GATEWAY_PORT, "--port", min=1, max=65_535),
    ghost: bool = typer.Option(
        False,
        "--ghost",
        help=(
            "Compatibility flag: project journals stay disabled while gateway accounting "
            "remains on."
        ),
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Never open first-run prompts.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Write a versioned launch receipt."),
    check: bool = typer.Option(
        False,
        "--check",
        help="Validate readiness and exit without binding.",
    ),
    graceful_timeout: float = typer.Option(
        DEFAULT_GRACEFUL_TIMEOUT_SECONDS,
        "--graceful-timeout",
        min=0.1,
        help="Seconds to drain admitted gateway work during shutdown.",
    ),
    max_active_requests: int = typer.Option(
        DEFAULT_MAX_ACTIVE_REQUESTS,
        "--max-active-requests",
        min=1,
        help="Maximum concurrently admitted requests.",
    ),
) -> None:
    """Start the local gateway directly, optionally with one project-backed alias.

    Args:
        project: Optional project identifier and endpoint alias.
        root: Local artifact and model-catalog root.
        policy: Exact policy for an ambiguous project.
        port: Local loopback TCP port.
        ghost: Compatibility marker for project traffic, which always uses gateway accounting.
        non_interactive: Whether first-run gateway prompts are forbidden.
        json_output: Whether startup output is one versioned JSON receipt.
        check: Whether to validate gateway readiness without binding.
        graceful_timeout: Gateway shutdown drain bound in seconds.
        max_active_requests: Concurrent-admission bound for the data plane.

    Raises:
        typer.BadParameter: The selected project form or activation is invalid.
    """
    start_gateway(
        project=project,
        root=root,
        policy=policy,
        port=port,
        ghost=ghost,
        non_interactive=non_interactive,
        json_output=json_output,
        check=check,
        graceful_timeout=graceful_timeout,
        max_active_requests=max_active_requests,
    )


def start_gateway(
    *,
    project: str | None = None,
    root: Path,
    policy: str | None = None,
    port: int = DEFAULT_GATEWAY_PORT,
    ghost: bool = False,
    non_interactive: bool = False,
    json_output: bool = False,
    check: bool = False,
    graceful_timeout: float = DEFAULT_GRACEFUL_TIMEOUT_SECONDS,
    max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS,
) -> None:
    """Start the local gateway, optionally materializing one project-backed alias.

    Args:
        project: Optional project identifier and endpoint alias.
        root: Local artifact and model-catalog root.
        policy: Exact policy for an ambiguous project.
        port: Local loopback TCP port.
        ghost: Compatibility marker for project traffic, which always uses gateway accounting.
        non_interactive: Whether first-run gateway prompts are forbidden.
        json_output: Whether startup output is one versioned JSON receipt.
        check: Whether to validate gateway readiness without binding.
        graceful_timeout: Gateway shutdown drain bound in seconds.
        max_active_requests: Concurrent-admission bound for the data plane.

    Raises:
        typer.BadParameter: The selected project form or activation is invalid,
            or the native gateway extension is not installed.
    """
    if (policy is not None or ghost) and project is None:
        raise typer.BadParameter("--policy and --ghost require --project")
    blocker = _native_engine_blocker()
    if blocker is not None:
        raise typer.BadParameter(blocker)
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
        max_active_requests=max_active_requests,
    )


def _native_engine_blocker() -> str | None:
    """Return why the native data plane cannot run at all, or ``None``.

    Returns:
        A display-safe reason with the exact build command when the compiled
        extension is missing, or ``None`` when the gateway can serve.
    """
    import importlib.util

    if importlib.util.find_spec("exp_gateway_native") is None:
        return (
            "the exp_gateway_native extension is not built; run 'just native' "
            "(uv run maturin develop --uv --release "
            "--manifest-path exp/runtime/gateway/native/Cargo.toml)"
        )
    return None


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
    max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS,
) -> None:
    """Validate, optionally set up, and serve the native gateway.

    The composition proves at startup exactly what serving requires: loading
    the granted aliases builds each alias's credential-free route proof and
    fails with actionable per-alias reasons when none is available, and
    native-servability validation confirms every pool deployment resolves to
    a provider client with a native wire dialect.

    Args:
        project: Optional project identifier and endpoint alias.
        root: Local artifact and model-catalog root.
        policy: Exact policy for an ambiguous project.
        port: Local loopback TCP port.
        ghost: Compatibility marker for project traffic, which always uses gateway accounting.
        non_interactive: Whether first-run gateway prompts are forbidden.
        json_output: Whether startup output is one versioned JSON receipt.
        check: Whether to validate gateway readiness without binding.
        graceful_timeout: Gateway shutdown drain bound in seconds.
        max_active_requests: Concurrent-admission bound for the data plane.

    Raises:
        typer.BadParameter: Setup, alias loading, native servability, or the
            loopback bind failed.
    """
    if not json_output:
        _emit_exp_wordmark()

    import importlib
    import socket

    from exp.cli.gateway.compatibility import prepare_project_gateway
    from exp.cli.gateway.setup import interactive_gateway_setup
    from exp.optimize.router.activation import verify_automatic_router_policy
    from exp.runtime.gateway.guardrails.config import load_guardrail_engine
    from exp.runtime.gateway.lifecycle import (
        gateway_instance_lock,
        load_gateway_components,
    )
    from exp.runtime.gateway.management import GatewayManagement
    from exp.runtime.gateway.native_bridge import NativeControlPlane
    from exp.runtime.gateway.native_execution import native_serving_blockers
    from exp.runtime.gateway.native_server import (
        NativeGatewayServerError,
        serve_native_gateway,
    )
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
        try:
            setup = interactive_gateway_setup(root)
        except AliasActivationOutcomeUnknownError as exc:
            _emit_setup_unknown_recovery(port=port, root=root, error=exc)
            raise typer.BadParameter(str(exc)) from None
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from None
        if setup is not None:
            _emit_setup_credentials(port=port, setup=setup)

    try:
        with usage_error(ValueError):
            with gateway_instance_lock(root, port=port):
                project_repository = LocalArtifactProjectActivationRepository(
                    root,
                    verifier=verify_automatic_router_policy,
                )
                components = load_gateway_components(
                    root,
                    project_repository=project_repository,
                    only_aliases=(
                        None if compatibility is None else frozenset({compatibility.alias})
                    ),
                )
                blockers = native_serving_blockers(components)
                if blockers:
                    details = "; ".join(blockers)
                    raise typer.BadParameter(
                        "the native engine cannot serve every granted alias: "
                        f"{details}. Fix the provider configuration and rerun 'exp'."
                    )
                # Loaded directly (rather than through native_server's own
                # import) so this composition can wire the content-free
                # metrics snapshot into the control plane before the process
                # host ever starts.
                exp_gateway_native = importlib.import_module("exp_gateway_native")
                guardrails = load_guardrail_engine(root)
                control_plane = NativeControlPlane(
                    components,
                    data_plane_metrics=exp_gateway_native.metrics_snapshot_json,
                    guardrails=guardrails,
                )
                # The bind is proved before the ready receipt is emitted, so
                # a launch that cannot own its port never announces a usable
                # gateway. The probe closes with SO_REUSEADDR set, leaving
                # the port immediately rebindable by the native server.
                if not check:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        try:
                            probe.bind((_LOOPBACK_HOST, port))
                        except OSError as exc:
                            raise typer.BadParameter(
                                f"port {port} is unavailable on {_LOOPBACK_HOST}: {exc}"
                            ) from exc
                receipt = {
                    "schema_version": 1,
                    "operation": "gateway.check" if check else "gateway.run",
                    "status": "ready",
                    "base_url": f"http://{_LOOPBACK_HOST}:{port}/v1",
                    "usage_url": f"http://{_LOOPBACK_HOST}:{port}/usage",
                    "reconciled_expired_requests": control_plane.reconciled_expired_requests,
                    "reconciled_unknown_attempts": control_plane.reconciled_unknown_attempts,
                    "launch_mode": "gateway" if compatibility is None else "project_alias",
                    "unavailable_aliases": _unavailable_alias_entries(
                        components.unavailable_aliases
                    ),
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
                    _emit_unavailable_aliases(components.unavailable_aliases)
                if check:
                    return

                try:
                    serve_native_gateway(
                        control_plane,
                        host=_LOOPBACK_HOST,
                        port=port,
                        max_active_requests=max_active_requests,
                        graceful_timeout_seconds=graceful_timeout,
                    )
                except NativeGatewayServerError as exc:
                    raise typer.BadParameter(str(exc)) from exc
    except typer.BadParameter:
        if setup is not None:
            _emit_setup_recovery(setup=setup)
        raise


def _unavailable_alias_entries(
    unavailable_aliases: tuple[tuple[str, str], ...],
) -> list[dict[str, str]]:
    """Shape failed aliases and their reasons for the versioned startup receipt.

    Args:
        unavailable_aliases: Alias names paired with their exact load-failure reasons.

    Returns:
        JSON-ready entries naming each alias that will answer 503 unavailable_route.
    """
    return [{"alias": alias_name, "reason": reason} for alias_name, reason in unavailable_aliases]


def _emit_unavailable_aliases(unavailable_aliases: tuple[tuple[str, str], ...]) -> None:
    """Warn about granted aliases that failed to load and will answer 503.

    Args:
        unavailable_aliases: Alias names paired with their exact load-failure reasons.
    """
    for alias_name, reason in unavailable_aliases:
        _console.print(
            f"! alias {alias_name!r} is unavailable and will answer "
            f"503 unavailable_route: {reason}",
            style="yellow",
            markup=False,
        )


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


def _emit_setup_credentials(
    *,
    port: int,
    setup: object,
    console: Console | None = None,
) -> None:
    """Deliver first-run gateway credentials before independent readiness checks run.

    Args:
        port: Loopback port that the gateway will serve when ready.
        setup: First-run setup result containing the one-time raw gateway key.
        console: Optional console receiving the credentials.

    Raises:
        TypeError: The setup result is not the gateway setup contract.
    """
    from exp.cli.gateway.setup import InteractiveSetupResult

    if not isinstance(setup, InteractiveSetupResult):
        raise TypeError("interactive setup returned an invalid result")
    output = console or _console
    output.print(f"export EXP_GATEWAY_URL=http://{_LOOPBACK_HOST}:{port}/v1", markup=False)
    output.print(f"export EXP_GATEWAY_KEY={setup.raw_key}", markup=False)
    output.print(f"Guardrails: {setup.guardrails}", markup=False)


def emit_setup_credentials(*, port: int, setup: object, console: Console | None = None) -> None:
    """Print the one-time credentials created by interactive gateway setup.

    Args:
        port: Loopback port used by the gateway.
        setup: Result returned by the first-run setup flow.
        console: Optional console receiving the user-facing exports.
    """
    _emit_setup_credentials(port=port, setup=setup, console=console)


def _emit_setup_unknown_recovery(
    *,
    port: int,
    root: Path,
    error: AliasActivationOutcomeUnknownError,
) -> None:
    """Deliver the only raw key when first-run setup has an unknown commit outcome.

    Args:
        port: Loopback port that the gateway will serve if the setup committed.
        root: EXP root whose selected guardrail file may already have landed.
        error: Typed activation error carrying the one-time key material when available.
    """
    if error.issued is None:
        return
    from exp.cli.gateway.guardrail_setup import GUARDRAILS_CUSTOM, inspect_setup_guardrails
    from exp.cli.gateway.setup import InteractiveSetupResult

    try:
        guardrails = inspect_setup_guardrails(
            root,
            error.issued.organization_id,
            error.issued.identity_id,
        ).display
    except ValueError:
        guardrails = GUARDRAILS_CUSTOM
    setup = InteractiveSetupResult(
        identity_id=error.issued.identity_id,
        alias=error.alias_id,
        raw_key=error.issued.raw_key,
        guardrails=guardrails,
    )
    _emit_setup_credentials(port=port, setup=setup)
    _console.print(
        "[yellow]First-run gateway setup outcome is unknown; inspect gateway status before "
        "retrying.[/yellow]",
        markup=True,
    )
    _console.print(
        f"Preserve this one-time gateway key: {setup.raw_key}",
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
        "and rerun `exp`.",
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
