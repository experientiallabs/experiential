"""Tests for the wmo.simulation.context public package surface."""

from __future__ import annotations

import wmo.simulation.context as connect

_PUBLIC_SYMBOLS = (
    # types
    "ConnectError",
    "ConnectorAuth",
    "ContextItem",
    "ItemKind",
    "PullQuery",
    # connector seam + registry
    "ConnectUI",
    "ContextConnector",
    "get_connector",
    "list_connectors",
    "register_connector",
    # oauth building blocks
    "OAuthApp",
    "ensure_fresh",
    "pkce_challenge",
    "refresh_auth",
    "run_device_flow",
    "run_loopback_flow",
    # app registry
    "EMBEDDED_APPS",
    "get_app",
    # credentials
    "connectors_path",
    "delete_connector_auth",
    "list_connected",
    "load_connector_auth",
    "save_connector_auth",
    "token_env_var",
    # bundle store
    "BundleManifest",
    "ContextStore",
    "render_markdown",
)


def test_public_api_reexports_the_connector_author_surface() -> None:
    for symbol in _PUBLIC_SYMBOLS:
        assert symbol in connect.__all__, f"{symbol} missing from wmo.simulation.context.__all__"
        assert getattr(connect, symbol) is not None


def test_builtin_connectors_register_on_package_import() -> None:
    """Importing the context package registers every built-in without the optional MCP extra."""
    registered = set(connect.list_connectors())
    assert {"github", "google-calendar", "google-drive", "gmail", "slack", "notion"} <= registered
