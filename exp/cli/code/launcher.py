"""Launch exp-code, the Node.js agent, against a local or hosted Experiential gateway.

``exp code`` runs the exact exp-code release this CLI was released with, through
``npx --yes @experiential-labs/exp-code@<version>``. ``EXP_CODE_DIR`` points at a built
checkout of the ``agent/`` package instead, for development. A local ``exp`` gateway is
used when its virtual key is in ``EXP_GATEWAY_KEY`` and it answers on loopback; otherwise the hosted
gateway is used with ``EXP_API_KEY``, ``EXPLABS_API_KEY``, or the ``exp login`` credential.
The key reaches exp-code only through its environment.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from rich.console import Console

from exp.cli.gateway.caller import DEFAULT_GATEWAY_URL
from exp.cli.providers.experiential_cloud import (
    HOSTED_GATEWAY_API_KEY_ENV,
    HOSTED_GATEWAY_DEFAULT_BASE_URL,
    HOSTED_GATEWAY_URL_ENV,
    hosted_connection,
    hosted_credential_binding,
)
from exp.cli.shared.theme import EXP_THEME
from exp.common.auth import ProviderAuthStore, ProviderAuthStoreError

# The exp-code release this CLI launches. It must equal "version" in agent/package.json;
# launcher_test.py enforces that, so a release bumps both in one change.
EXP_CODE_VERSION = "0.1.7"
EXP_CODE_PACKAGE = "@experiential-labs/exp-code"
MINIMUM_NODE_VERSION = (22, 19, 0)
LOCAL_PROBE_TIMEOUT_SECONDS = 1.5
NODE_ENV = "EXP_CODE_NODE"
CHECKOUT_ENV = "EXP_CODE_DIR"
HOSTED_KEY_ENV = "EXP_API_KEY"
LOCAL_KEY_ENV = "EXP_GATEWAY_KEY"

NodeVersion = tuple[int, int, int]


class CodeLaunchError(RuntimeError):
    """exp-code cannot be launched as asked; the message says what to do."""


class NoGatewayConfigured(CodeLaunchError):
    """Neither a local gateway key nor a hosted credential exists.

    exp-code still starts (its help, version, and other providers work) and explains how to
    configure a gateway itself, so the launcher does not repeat the message.
    """


@dataclass(frozen=True)
class NodeRuntime:
    """A Node.js binary new enough for exp-code.

    Attributes:
        path: Absolute path of the ``node`` executable.
        version: Its ``major.minor.patch`` version.
    """

    path: Path
    version: NodeVersion


@dataclass(frozen=True)
class GatewayTarget:
    """The gateway exp-code will call.

    Attributes:
        kind: ``local`` for a loopback ``exp`` gateway, ``hosted`` for Experiential's.
        base_url: OpenAI-compatible root ending in ``/v1``.
        api_key: Key for that gateway; never shown in ``repr``.
    """

    kind: Literal["hosted", "local"]
    base_url: str
    api_key: str = field(repr=False)


def parse_node_version(text: str) -> NodeVersion | None:
    """Parse ``node --version`` output such as ``v22.19.0``.

    Args:
        text: Raw command output.

    Returns:
        The version triple, or ``None`` when the text is not a Node version.
    """
    match = re.fullmatch(r"\s*v?(\d+)\.(\d+)\.(\d+)\S*\s*", text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _read_node_version(path: str) -> str:
    """Return the raw ``--version`` output of one node binary."""
    completed = subprocess.run(
        [path, "--version"], capture_output=True, text=True, timeout=15, check=True
    )
    return completed.stdout


def find_node(
    environment: Mapping[str, str],
    *,
    which: Callable[[str], str | None] = shutil.which,
    read_version: Callable[[str], str] = _read_node_version,
) -> NodeRuntime:
    """Find a Node.js 22.19+ binary from ``EXP_CODE_NODE`` or ``PATH``.

    Args:
        environment: Process environment.
        which: Executable lookup, injectable for tests.
        read_version: Version reader, injectable for tests.

    Returns:
        The usable runtime.

    Raises:
        CodeLaunchError: No node binary exists, or it is older than 22.19.
    """
    wanted = ".".join(str(part) for part in MINIMUM_NODE_VERSION)
    configured = environment.get(NODE_ENV, "").strip()
    path = configured or which("node")
    install = (
        f"Install Node.js {wanted} or newer from https://nodejs.org or with a version "
        f"manager (for example `nvm install 22`), or set {NODE_ENV} to a newer node binary."
    )
    if not path:
        raise CodeLaunchError(
            f"Node.js {wanted} or newer is required, and none was found. {install}"
        )
    try:
        version = parse_node_version(read_version(path))
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodeLaunchError(f"Could not run {path} --version ({exc}). {install}") from None
    if version is None or version < MINIMUM_NODE_VERSION:
        found = "an unknown version" if version is None else "v" + ".".join(map(str, version))
        raise CodeLaunchError(
            f"Node.js {wanted} or newer is required; {path} is {found}. {install}"
        )
    return NodeRuntime(path=Path(path).resolve(), version=version)


def _is_loopback(url: str) -> bool:
    """Return whether a URL points at this machine."""
    host = urlparse(url).hostname or ""
    return host in {"127.0.0.1", "localhost", "::1"}


def local_gateway_available(base_url: str, api_key: str) -> bool:
    """Return whether a local gateway answers its model listing for this key.

    Args:
        base_url: Loopback OpenAI-compatible root.
        api_key: Virtual gateway key.

    Returns:
        ``True`` only for a successful listing; refusals and connection errors are ``False``.
    """
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=LOCAL_PROBE_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _stored_hosted_key(environment: Mapping[str, str]) -> str | None:
    """Read the ``exp login`` credential for the hosted gateway, if one is saved."""
    connection = hosted_connection(environment)
    try:
        return ProviderAuthStore().get(
            connection.name, binding=hosted_credential_binding(environment)
        )
    except ProviderAuthStoreError as exc:
        raise CodeLaunchError(f"Your saved Experiential login cannot be used: {exc}") from None


def resolve_gateway_target(
    environment: Mapping[str, str],
    *,
    probe: Callable[[str, str], bool] = local_gateway_available,
    stored_key: Callable[[Mapping[str, str]], str | None] = _stored_hosted_key,
) -> GatewayTarget:
    """Choose the gateway: a running local gateway first, else the hosted one.

    Args:
        environment: Process environment.
        probe: Local gateway check, injectable for tests.
        stored_key: ``exp login`` credential reader, injectable for tests.

    Returns:
        The gateway and its key.

    Raises:
        CodeLaunchError: No gateway can be reached with a key.
    """
    configured_url = environment.get(HOSTED_GATEWAY_URL_ENV, "").strip()
    local_url = (
        configured_url if configured_url and _is_loopback(configured_url) else DEFAULT_GATEWAY_URL
    )
    local_key = environment.get(LOCAL_KEY_ENV, "").strip()
    if local_key and probe(local_url, local_key):
        return GatewayTarget(kind="local", base_url=local_url.rstrip("/"), api_key=local_key)
    hosted_url = (
        configured_url
        if configured_url and not _is_loopback(configured_url)
        else HOSTED_GATEWAY_DEFAULT_BASE_URL
    )
    hosted_key = (
        environment.get(HOSTED_KEY_ENV, "").strip()
        or environment.get(HOSTED_GATEWAY_API_KEY_ENV, "").strip()
        or stored_key(environment)
    )
    if hosted_key:
        return GatewayTarget(kind="hosted", base_url=hosted_url.rstrip("/"), api_key=hosted_key)
    if local_key:
        raise CodeLaunchError(
            f"EXP_GATEWAY_KEY is set, but no local gateway answered at {local_url}. Start it with "
            "`exp` in another terminal, or run `exp login` to use the hosted gateway."
        )
    raise NoGatewayConfigured(
        "No Experiential gateway is configured. Run `exp login` to use the hosted gateway, or "
        "start a local gateway with `exp` and export the EXP_GATEWAY_KEY it prints."
    )


def exp_code_command(
    node: NodeRuntime,
    arguments: Sequence[str],
    environment: Mapping[str, str],
    *,
    which: Callable[..., str | None] = shutil.which,
) -> list[str]:
    """Build the command that runs exp-code.

    Args:
        node: The Node.js runtime.
        arguments: Arguments passed through to exp-code.
        environment: Process environment (``EXP_CODE_DIR`` selects a checkout).
        which: Executable lookup, injectable for tests.

    Returns:
        The argv to execute.

    Raises:
        CodeLaunchError: The checkout is not built, or ``npx`` is missing.
    """
    checkout = environment.get(CHECKOUT_ENV, "").strip()
    if checkout:
        cli = Path(checkout).expanduser() / "dist" / "cli.js"
        if not cli.is_file():
            raise CodeLaunchError(
                f"{cli} does not exist. Build the checkout first: "
                f"cd {checkout} && npm ci && npm run build"
            )
        return [str(node.path), str(cli), *arguments]
    npx = which("npx", path=str(node.path.parent)) or which("npx")
    if not npx:
        raise CodeLaunchError(
            f"npx was not found next to {node.path} or on PATH. Install npm with Node.js, "
            f"or set {CHECKOUT_ENV} to a built exp-code checkout."
        )
    return [npx, "--yes", f"{EXP_CODE_PACKAGE}@{EXP_CODE_VERSION}", *arguments]


def child_environment(
    environment: Mapping[str, str], node: NodeRuntime, target: GatewayTarget | None
) -> dict[str, str]:
    """Build exp-code's environment: the chosen gateway, its key, and this node first on PATH.

    Args:
        environment: Process environment.
        node: The Node.js runtime, whose directory leads ``PATH`` so ``npx`` uses it.
        target: The chosen gateway, or ``None`` to let exp-code report that none is configured.

    Returns:
        The complete child environment.
    """
    child = dict(environment)
    child["PATH"] = os.pathsep.join(
        filter(None, [str(node.path.parent), environment.get("PATH", "")])
    )
    if target is None:
        return child
    child["EXP_CODE_GATEWAY"] = target.kind
    child["EXP_CODE_BASE_URL"] = target.base_url
    child[HOSTED_KEY_ENV if target.kind == "hosted" else LOCAL_KEY_ENV] = target.api_key
    return child


def _run(command: list[str], environment: dict[str, str]) -> int:
    """Run exp-code in the foreground; Ctrl+C belongs to it, not to this process."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return subprocess.run(command, env=environment, check=False).returncode
    finally:
        signal.signal(signal.SIGINT, previous)


def launch(
    arguments: Sequence[str],
    environment: Mapping[str, str],
    *,
    run: Callable[[list[str], dict[str, str]], int] = _run,
    probe: Callable[[str, str], bool] = local_gateway_available,
    stored_key: Callable[[Mapping[str, str]], str | None] = _stored_hosted_key,
    which: Callable[..., str | None] = shutil.which,
    read_version: Callable[[str], str] = _read_node_version,
    warn: Callable[[str], None] = lambda message: None,
) -> int:
    """Resolve Node.js and the gateway, then run exp-code with the given arguments.

    A gateway problem does not stop the launch: exp-code still runs its help, version, and
    other configured providers. Problems worth reading (a local gateway that is down, a saved
    login that cannot be used) go to ``warn``.

    Args:
        arguments: Arguments passed through to exp-code.
        environment: Process environment.
        run: Process runner, injectable for tests.
        probe: Local gateway check, injectable for tests.
        stored_key: ``exp login`` credential reader, injectable for tests.
        which: Executable lookup, injectable for tests.
        read_version: Node version reader, injectable for tests.
        warn: Receives gateway problems shown before exp-code starts.

    Returns:
        exp-code's exit status.

    Raises:
        CodeLaunchError: Node.js or exp-code itself is unavailable.
    """
    node = find_node(environment, which=which, read_version=read_version)
    target: GatewayTarget | None
    try:
        target = resolve_gateway_target(environment, probe=probe, stored_key=stored_key)
    except NoGatewayConfigured:
        target = None
    except CodeLaunchError as exc:
        warn(str(exc))
        target = None
    command = exp_code_command(node, arguments, environment, which=which)
    return run(command, child_environment(environment, node, target))


def launch_from_cli(arguments: Sequence[str]) -> int:
    """Launch exp-code for the CLI, printing launch problems instead of raising them.

    Args:
        arguments: Arguments passed through to exp-code.

    Returns:
        exp-code's exit status, or 1 when it could not be launched.
    """
    console = Console(theme=EXP_THEME, stderr=True, soft_wrap=True)

    def warn(message: str) -> None:
        """Show a gateway problem before exp-code starts."""
        console.print(f"exp code: {message}", markup=False)

    try:
        return launch(arguments, os.environ, warn=warn)
    except (CodeLaunchError, OSError) as exc:
        console.print(f"exp code: {exc}", markup=False)
        return 1
