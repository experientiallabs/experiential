"""The shared OAuth app registry: embedded endpoint defaults + env-var credential overrides.

Endpoint/scope configuration per provider is embedded here; client credentials are not (the
embedded ids are placeholders until the maintainer's registered apps land). `get_app` layers
`WMH_<NAME>_CLIENT_ID` / `WMH_<NAME>_CLIENT_SECRET` env overrides over the embedded defaults, so
users can always bring their own OAuth app.
"""

from __future__ import annotations

import os

from wmh.connect.oauth import OAuthApp
from wmh.connect.types import ConnectError

# Full endpoint/scope config per provider; client ids stay empty until the maintainer's
# registered apps are embedded (a follow-up change fills them in).
EMBEDDED_APPS: dict[str, OAuthApp] = {
    "github": OAuthApp(
        name="github",
        client_id="",
        auth_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        device_url="https://github.com/login/device/code",
        scopes=["repo", "read:org"],
    ),
    "google": OAuthApp(
        name="google",
        client_id="",
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        extra_auth_params={"access_type": "offline", "prompt": "consent"},
    ),
    "slack": OAuthApp(
        name="slack",
        client_id="",
        auth_url="https://slack.com/oauth/v2/authorize",
        token_url="https://slack.com/api/oauth.v2.access",
    ),
}


def get_app(name: str) -> OAuthApp:
    """Resolve the OAuth app for `name`, layering env overrides over the embedded defaults.

    Env vars (name upper-cased, hyphens to underscores): `WMH_<NAME>_CLIENT_ID` and
    `WMH_<NAME>_CLIENT_SECRET`. Set-but-empty values are treated as unset.

    Raises:
        ConnectError: When `name` is unknown, or when neither layer provides a client id.
    """
    if name not in EMBEDDED_APPS:
        known = ", ".join(sorted(EMBEDDED_APPS))
        raise ConnectError(f"no OAuth app configuration for {name!r}; known apps: {known}")
    prefix = f"WMH_{name.upper().replace('-', '_')}"
    app = EMBEDDED_APPS[name]
    updates: dict[str, str] = {}
    client_id = os.environ.get(f"{prefix}_CLIENT_ID")
    if client_id:
        updates["client_id"] = client_id
    client_secret = os.environ.get(f"{prefix}_CLIENT_SECRET")
    if client_secret:
        updates["client_secret"] = client_secret
    if updates:
        app = app.model_copy(update=updates)
    if not app.client_id:
        raise ConnectError(
            f"no OAuth client id available for {name!r}: set ${prefix}_CLIENT_ID (and "
            f"${prefix}_CLIENT_SECRET if the provider requires one) to use your own OAuth "
            "app; see docs/connectors.md"
        )
    return app
