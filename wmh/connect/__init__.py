"""Context connectors: OAuth to a service, pull content into normalized, replayable bundles.

`wmh connect <service>` runs a connector's interactive auth flow (browser OAuth via
`run_loopback_flow`, headless via `run_device_flow`, or a pasted token) and stores the credential
in the user-global `~/.wmh/connectors.toml`. `wmh context pull <service>` calls the connector's
`pull` and persists the result as a bundle (`manifest.json` + `items.jsonl`) under
`.wmh/context/`, which `render_markdown` turns into the markdown document attached to a model's
knowledge base.

A `ContextConnector` owns one service: `connect` (auth), `verify` (cheap identity check), and
`pull` (normalized `ContextItem`s). Connectors register themselves on import and are looked up by
name (`get_connector`) or listed (`list_connectors`), mirroring `wmh.ingest`.
"""

# Import each built-in connector module for its registration side effect so `get_connector(...)`
# works on package import, mirroring wmh/ingest/__init__.py. All are SDK-free at import
# time: they talk httpx directly, and notion imports its optional `mcp` SDK (the `connectors`
# extra) lazily inside the MCP code paths, so a bare install still imports this package.
from wmh.connect import brave as brave  # noqa: F401
from wmh.connect import github as github  # noqa: F401
from wmh.connect import google as google  # noqa: F401
from wmh.connect import notion as notion  # noqa: F401
from wmh.connect import slack as slack  # noqa: F401
from wmh.connect.apps import EMBEDDED_APPS, get_app
from wmh.connect.connector import (
    ConnectUI,
    ContextConnector,
    get_connector,
    list_connectors,
    register_connector,
)
from wmh.connect.credentials import (
    connectors_path,
    delete_connector_auth,
    list_connected,
    load_connector_auth,
    resolve_env_token,
    save_connector_auth,
    token_env_var,
    token_env_vars,
)
from wmh.connect.oauth import (
    OAuthApp,
    ensure_fresh,
    pkce_challenge,
    refresh_auth,
    run_device_flow,
    run_loopback_flow,
)
from wmh.connect.store import BundleManifest, ContextStore, render_markdown
from wmh.connect.types import ConnectError, ConnectorAuth, ContextItem, ItemKind, PullQuery

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
