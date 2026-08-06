"""Context connectors: fetch content from a service into normalized `ContextItem`s.

This is a library, not a CLI. A host (the platform's connector tools) supplies a per-service
access token and calls a connector's `pull`:

    from wmo.simulation.context import ConnectorAuth, PullQuery, get_connector

    items = get_connector("github").pull(
        ConnectorAuth(kind="token", access_token=token),
        PullQuery(query="repo:me/proj is:open", limit=20),
    )

A `ContextConnector` owns one service: `pull` (normalized `ContextItem`s) plus `verify` (a cheap
identity check) and `connect` (an interactive auth flow, for callers that want to acquire tokens
themselves). Token/OAuth acquisition is the caller's responsibility. Connectors register
themselves on import and are looked up by name (`get_connector`) or listed (`list_connectors`),
mirroring `wmo.simulation.ingest`.
"""

# Import each built-in connector module for its registration side effect so `get_connector(...)`
# works on package import, mirroring wmo/simulation/ingest/__init__.py. All are SDK-free at import
# time: they talk httpx directly, and notion imports its optional `mcp` SDK (the `connectors`
# extra) lazily inside the MCP code paths, so a bare install still imports this package.
from wmo.simulation.context import brave as brave  # noqa: F401
from wmo.simulation.context import github as github  # noqa: F401
from wmo.simulation.context import google as google  # noqa: F401
from wmo.simulation.context import notion as notion  # noqa: F401
from wmo.simulation.context import slack as slack  # noqa: F401
from wmo.simulation.context.apps import EMBEDDED_APPS, get_app
from wmo.simulation.context.connector import (
    ConnectUI,
    ContextConnector,
    get_connector,
    list_connectors,
    register_connector,
)
from wmo.simulation.context.credentials import (
    connectors_path,
    delete_connector_auth,
    list_connected,
    load_connector_auth,
    resolve_env_token,
    save_connector_auth,
    token_env_var,
    token_env_vars,
)
from wmo.simulation.context.oauth import (
    OAuthApp,
    ensure_fresh,
    pkce_challenge,
    refresh_auth,
    run_device_flow,
    run_loopback_flow,
)
from wmo.simulation.context.store import BundleManifest, ContextStore, render_markdown
from wmo.simulation.context.types import (
    ConnectError,
    ConnectorAuth,
    ContextItem,
    ItemKind,
    PullQuery,
)

__all__ = [
    "EMBEDDED_APPS",
    "BundleManifest",
    "ConnectError",
    "ConnectUI",
    "ConnectorAuth",
    "ContextConnector",
    "ContextItem",
    "ContextStore",
    "ItemKind",
    "OAuthApp",
    "PullQuery",
    "connectors_path",
    "delete_connector_auth",
    "ensure_fresh",
    "get_app",
    "get_connector",
    "list_connected",
    "list_connectors",
    "load_connector_auth",
    "pkce_challenge",
    "refresh_auth",
    "register_connector",
    "render_markdown",
    "resolve_env_token",
    "run_device_flow",
    "run_loopback_flow",
    "save_connector_auth",
    "token_env_var",
    "token_env_vars",
]
