"""Consumer-plan connection vocabulary and its secret-free shape checks.

A plan connection (``ConnectionConfig.subscription``) signs in with a consumer subscription
instead of an API key. The kinds, the one catalog provider each kind signs in for, and the
plan-specific connection fields (the Antigravity Code Assist project and user agent) are
validated here so the connection model stays one coherent contract.
"""

from __future__ import annotations

import re
from typing import Literal

SubscriptionKind = Literal["chatgpt", "anthropic", "antigravity"]
"""Consumer plan a connection signs in with instead of an API key.

``chatgpt`` is a ChatGPT plan reaching the Codex Responses backend; ``anthropic`` is a Claude
plan reaching the Messages API through the OAuth application Anthropic issued to the operator;
``antigravity`` is a Google Antigravity plan reaching Gemini models through the Cloud Code
Assist backend on the account's Code Assist project. The connection stores no credential NAME:
the sign-in lives under the connection ID and the gateway mints a fresh bearer per dispatch.
"""

SUBSCRIPTION_PROVIDERS: dict[SubscriptionKind, str] = {
    "chatgpt": "openai",
    "anthropic": "anthropic",
    "antigravity": "gemini",
}
"""The one catalog provider each plan kind is a sign-in for."""

MAXIMUM_USER_AGENT_CHARACTERS = 512
"""Upper bound on a connection's Code Assist ``User-Agent`` header."""

# Google Cloud project ids (and the managed Code Assist ids, which share the shape): lowercase
# letters, digits and hyphens, starting with a letter. Kept a little wider than the documented
# 6-30 characters so a legacy or domain-scoped id ("example.com:project") still validates.
_CODE_ASSIST_PROJECT = re.compile(r"[a-z][a-z0-9.:-]{2,126}[a-z0-9]")


def require_plan_connection_shape(
    *,
    subscription: SubscriptionKind | None,
    provider: str,
    code_assist_project: str | None,
    code_assist_user_agent: str | None,
) -> None:
    """Reject plan-specific fields that do not fit the connection's plan kind.

    Args:
        subscription: The connection's plan kind, or ``None`` for an API-key connection.
        provider: The connection's catalog provider.
        code_assist_project: The Antigravity account's Cloud Code Assist project id.
        code_assist_user_agent: The full ``User-Agent`` Antigravity dispatches send, or
            ``None`` for the operator default.

    Raises:
        ValueError: The plan names another provider, an Antigravity plan has no valid project
            or carries a malformed user agent, or another connection carries these fields.
    """
    if subscription is not None:
        expected = SUBSCRIPTION_PROVIDERS[subscription]
        if provider != expected:
            raise ValueError(
                f"subscription {subscription!r} is a sign-in for provider {expected!r}, "
                f"not {provider!r}"
            )
    if subscription != "antigravity":
        if code_assist_project is not None or code_assist_user_agent is not None:
            raise ValueError(
                "code_assist_project and code_assist_user_agent are only accepted for "
                "subscription='antigravity'"
            )
        return
    if code_assist_project is None or not code_assist_project.strip():
        raise ValueError(
            "an antigravity connection requires code_assist_project, the Cloud Code Assist "
            "project id the account's loadCodeAssist or onboardUser call returned"
        )
    if not _CODE_ASSIST_PROJECT.fullmatch(code_assist_project):
        raise ValueError(
            "code_assist_project must be a Google Cloud project id such as 'proud-sprite-a1b2c3'"
        )
    if code_assist_user_agent is not None and (
        not code_assist_user_agent.strip()
        or len(code_assist_user_agent) > MAXIMUM_USER_AGENT_CHARACTERS
        or any(not character.isprintable() for character in code_assist_user_agent)
    ):
        raise ValueError(
            "code_assist_user_agent must be one printable line of at most "
            f"{MAXIMUM_USER_AGENT_CHARACTERS} characters"
        )
