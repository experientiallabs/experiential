"""The shared OAuth app registry: embedded endpoint defaults, env-var client credentials.

Endpoint configuration per provider is embedded here; client credentials never are. The repo
policy is all-or-nothing: since some providers' credentials cannot ship (Google and Slack
token exchanges demand a client secret), none do, and every OAuth connector requires its
`WMH_<NAME>_CLIENT_ID` (plus `WMH_<NAME>_CLIENT_SECRET` where the provider demands one) when,
and only when, that connector is used. `get_app` resolves the env credentials over the
embedded endpoints. A planned hosted path will supply registered apps platform-side instead.
"""

from __future__ import annotations

import os

from wmh.connect.oauth import OAuthApp
from wmh.connect.types import ConnectError

# Full endpoint config per provider; client ids/secrets come from the env, never the repo.
EMBEDDED_APPS: dict[str, OAuthApp] = {
    "github": OAuthApp(
        name="github",
        # Expects a GitHub App client id ($WMH_GITHUB_CLIENT_ID): GitHub Apps carry their
        # permissions in the app registration, repo-scoped at install time, so no scopes are
        # sent in the flow, and the device grant needs no client secret.
        client_id="",
        auth_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        device_url="https://github.com/login/device/code",
    ),
    "google": OAuthApp(
        name="google",
        # Google's Desktop-client token exchange requires the client secret even with PKCE
        # ($WMH_GOOGLE_CLIENT_ID + $WMH_GOOGLE_CLIENT_SECRET). The planned zero-setup path
        # points token_url at a hosted stateless exchange holding a secret platform-side.
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
            "app; see docs/reference/connect-library.md"
        )
    return app
