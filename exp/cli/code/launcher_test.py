"""Tests for launching exp-code: Node.js discovery, gateway choice, command, and environment."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from exp.cli.code.launcher import (
    EXP_CODE_PACKAGE,
    EXP_CODE_VERSION,
    CodeLaunchError,
    GatewayTarget,
    NodeRuntime,
    NoGatewayConfigured,
    child_environment,
    exp_code_command,
    find_node,
    launch,
    parse_node_version,
    resolve_gateway_target,
)

REPOSITORY = Path(__file__).resolve().parents[3]
NODE = NodeRuntime(path=Path("/opt/node/bin/node"), version=(22, 19, 0))


def _which(found: Mapping[str, str]) -> Callable[..., str | None]:
    """Build an executable lookup answering from a fixed table."""

    def which(name: str, path: str | None = None) -> str | None:
        """Return the fixed path for a name, honoring an explicit search directory."""
        del path
        return found.get(name)

    return which


def _no_stored_key(environment: Mapping[str, str]) -> str | None:
    """Report no saved login."""
    del environment
    return None


def test_pinned_version_matches_the_agent_package() -> None:
    """The launched exp-code release is the one in agent/package.json."""
    package = json.loads((REPOSITORY / "agent" / "package.json").read_text(encoding="utf-8"))
    assert package["name"] == EXP_CODE_PACKAGE
    assert package["version"] == EXP_CODE_VERSION


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("v22.19.0\n", (22, 19, 0)),
        ("v26.8.1", (26, 8, 1)),
        ("v23.0.0-nightly", (23, 0, 0)),
        ("x", None),
    ],
)
def test_parse_node_version(text: str, expected: tuple[int, int, int] | None) -> None:
    """Node version output parses to a comparable triple."""
    assert parse_node_version(text) == expected


def test_find_node_accepts_new_enough_node_and_prefers_the_override() -> None:
    """A 22.19+ node is accepted; EXP_CODE_NODE wins over PATH."""
    found = find_node(
        {}, which=_which({"node": "/usr/bin/node"}), read_version=lambda _: "v22.19.0"
    )
    assert found.version == (22, 19, 0)
    seen: list[str] = []

    def read(path: str) -> str:
        """Record which binary was asked."""
        seen.append(path)
        return "v24.0.0"

    find_node(
        {"EXP_CODE_NODE": "/custom/node"},
        which=_which({"node": "/usr/bin/node"}),
        read_version=read,
    )
    assert seen == ["/custom/node"]


def test_find_node_explains_missing_or_old_node() -> None:
    """Missing and old Node.js both say how to fix it."""
    with pytest.raises(CodeLaunchError, match=r"none was found.*nvm install 22"):
        find_node({}, which=_which({}), read_version=lambda _: "")
    with pytest.raises(CodeLaunchError, match=r"/usr/bin/node is v22\.18\.0.*EXP_CODE_NODE"):
        find_node({}, which=_which({"node": "/usr/bin/node"}), read_version=lambda _: "v22.18.0")


def test_running_local_gateway_wins() -> None:
    """A local key whose gateway answers selects the local gateway."""
    probed: list[tuple[str, str]] = []

    def probe(url: str, key: str) -> bool:
        """Record the probe and report the gateway as running."""
        probed.append((url, key))
        return True

    target = resolve_gateway_target(
        {"EXP_GATEWAY_KEY": "local-secret", "EXP_API_KEY": "xpl_h"},
        probe=probe,
        stored_key=_no_stored_key,
    )
    assert target == GatewayTarget(
        kind="local", base_url="http://127.0.0.1:8000/v1", api_key="local-secret"
    )
    assert probed == [("http://127.0.0.1:8000/v1", "local-secret")]
    assert "local-secret" not in repr(target)


def test_loopback_gateway_url_is_the_local_gateway_not_a_hosted_override() -> None:
    """EXP_GATEWAY_URL on loopback (as `exp` prints it) points the local probe, not hosted."""
    target = resolve_gateway_target(
        {"EXP_GATEWAY_URL": "http://127.0.0.1:9000/v1", "EXP_GATEWAY_KEY": "k"},
        probe=lambda url, key: url == "http://127.0.0.1:9000/v1",
        stored_key=_no_stored_key,
    )
    assert target.base_url == "http://127.0.0.1:9000/v1"
    hosted = resolve_gateway_target(
        {"EXP_GATEWAY_URL": "http://127.0.0.1:9000/v1", "EXP_API_KEY": "xpl_h"},
        probe=lambda url, key: False,
        stored_key=_no_stored_key,
    )
    assert hosted.base_url == "https://api.experientiallabs.ai/v1"


def test_hosted_gateway_uses_env_keys_then_the_login_credential() -> None:
    """Hosted keys come from EXP_API_KEY, EXPLABS_API_KEY, then `exp login`."""
    down = lambda url, key: False  # noqa: E731 - a one-line probe stub
    assert (
        resolve_gateway_target(
            {"EXP_API_KEY": "a", "EXPLABS_API_KEY": "b"}, probe=down, stored_key=_no_stored_key
        ).api_key
        == "a"
    )
    assert (
        resolve_gateway_target(
            {"EXPLABS_API_KEY": "b"}, probe=down, stored_key=_no_stored_key
        ).api_key
        == "b"
    )
    saved = resolve_gateway_target({}, probe=down, stored_key=lambda _: "xpl_saved")
    assert saved == GatewayTarget(
        kind="hosted", base_url="https://api.experientiallabs.ai/v1", api_key="xpl_saved"
    )
    staging = resolve_gateway_target(
        {"EXP_GATEWAY_URL": "https://gw.example/v1"}, probe=down, stored_key=lambda _: "k"
    )
    assert staging.base_url == "https://gw.example/v1"


def test_no_gateway_explains_both_paths() -> None:
    """Without any key, the error names `exp login` and the local gateway."""
    with pytest.raises(NoGatewayConfigured, match=r"exp login.*EXP_GATEWAY_KEY"):
        resolve_gateway_target({}, probe=lambda url, key: False, stored_key=_no_stored_key)
    with pytest.raises(CodeLaunchError, match=r"no local gateway answered"):
        resolve_gateway_target(
            {"EXP_GATEWAY_KEY": "k"}, probe=lambda url, key: False, stored_key=_no_stored_key
        )


def test_command_runs_the_pinned_release_through_npx() -> None:
    """The default command is `npx --yes @experiential-labs/exp-code@<version>` plus arguments."""
    command = exp_code_command(
        NODE, ["-p", "hi", "--mode", "json"], {}, which=_which({"npx": "/opt/node/bin/npx"})
    )
    assert command == [
        "/opt/node/bin/npx",
        "--yes",
        f"@experiential-labs/exp-code@{EXP_CODE_VERSION}",
        "-p",
        "hi",
        "--mode",
        "json",
    ]
    with pytest.raises(CodeLaunchError, match="npx was not found"):
        exp_code_command(NODE, [], {}, which=_which({}))


def test_command_uses_a_built_checkout(tmp_path: Path) -> None:
    """EXP_CODE_DIR runs the checkout's built CLI with this node, or explains how to build it."""
    with pytest.raises(CodeLaunchError, match="npm ci && npm run build"):
        exp_code_command(NODE, [], {"EXP_CODE_DIR": str(tmp_path)})
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "cli.js").write_text("", encoding="utf-8")
    command = exp_code_command(NODE, ["--help"], {"EXP_CODE_DIR": str(tmp_path)})
    assert command == [str(NODE.path), str(tmp_path / "dist" / "cli.js"), "--help"]


def test_child_environment_carries_the_gateway_and_puts_node_first() -> None:
    """The child gets the gateway kind, URL and key, and this node leads PATH."""
    hosted = child_environment(
        {"PATH": "/usr/bin", "OTHER": "1"}, NODE, GatewayTarget("hosted", "https://h/v1", "xpl_k")
    )
    assert hosted["PATH"] == os.pathsep.join(["/opt/node/bin", "/usr/bin"])
    assert hosted["EXP_CODE_GATEWAY"] == "hosted"
    assert hosted["EXP_CODE_BASE_URL"] == "https://h/v1"
    assert hosted["EXP_API_KEY"] == "xpl_k"
    assert hosted["OTHER"] == "1"
    local = child_environment({}, NODE, GatewayTarget("local", "http://127.0.0.1:8000/v1", "v"))
    assert local["EXP_GATEWAY_KEY"] == "v"
    assert local["EXP_CODE_GATEWAY"] == "local"


def test_launch_runs_exp_code_and_returns_its_status() -> None:
    """launch composes discovery, gateway choice and command, and returns the exit status."""
    ran: list[tuple[list[str], dict[str, str]]] = []

    def run(command: list[str], environment: dict[str, str]) -> int:
        """Record the launch and exit with status 3."""
        ran.append((command, environment))
        return 3

    status = launch(
        ["-p", "hi"],
        {"EXP_API_KEY": "xpl_k", "PATH": "/usr/bin"},
        run=run,
        probe=lambda url, key: False,
        stored_key=_no_stored_key,
        which=_which({"node": "/opt/node/bin/node", "npx": "/opt/node/bin/npx"}),
        read_version=lambda _: "v22.19.0",
    )
    assert status == 3
    command, environment = ran[0]
    assert command[-2:] == ["-p", "hi"]
    assert environment["EXP_CODE_GATEWAY"] == "hosted"
    assert "xpl_k" not in command


def test_launch_without_a_gateway_still_runs_exp_code() -> None:
    """exp-code starts without gateway variables; a down local gateway is reported first."""
    ran: list[dict[str, str]] = []
    warnings: list[str] = []

    def run(command: list[str], environment: dict[str, str]) -> int:
        """Record the environment and succeed."""
        del command
        ran.append(environment)
        return 0

    tools = _which({"node": "/opt/node/bin/node", "npx": "/opt/node/bin/npx"})
    for environment in ({}, {"EXP_GATEWAY_KEY": "k"}):
        status = launch(
            ["--version"],
            environment,
            run=run,
            probe=lambda url, key: False,
            stored_key=_no_stored_key,
            which=tools,
            read_version=lambda _: "v22.19.0",
            warn=warnings.append,
        )
        assert status == 0
    assert all("EXP_CODE_GATEWAY" not in environment for environment in ran)
    assert len(warnings) == 1
    assert "no local gateway answered" in warnings[0]
