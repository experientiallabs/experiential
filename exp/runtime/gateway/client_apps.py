"""Content-free classification of the calling application from request headers.

The gateway attributes each accepted request to the coding agent or application that sent it
(Claude Code, Codex, OpenCode, Hermes Agent, ...) so usage can be reported per agent. The
classification reads only request headers the caller already sends: ``User-Agent``, Codex's
``originator``, and the OpenRouter-style ``X-Title`` and ``HTTP-Referer`` app identity. It never
reads the request body, and it never treats a caller-chosen label as a credential or an
authorization input: an app id is reporting metadata only.

The vocabulary is closed. A header that matches no known application classifies as ``None``
(reported as an unidentified app), never as a guessed name, so a new agent appears in reports
only after its rule is added here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import Field

from exp.common.core.artifacts import ContractModel

# Longest stored ``User-Agent`` prefix; longer values are truncated, never rejected.
MAXIMUM_USER_AGENT_CHARS = 256

_MAXIMUM_INSPECTED_CHARS = 512


class ClientApp(StrEnum):
    """Closed vocabulary of calling applications the gateway can attribute."""

    CLAUDE_CODE = "claude_code"
    CODEX = "codex"
    OPENCODE = "opencode"
    HERMES = "hermes"
    KILO_CODE = "kilo_code"
    CLINE = "cline"
    ROO_CODE = "roo_code"
    CURSOR = "cursor"
    QWEN_CODE = "qwen_code"
    GEMINI_CLI = "gemini_cli"
    GITHUB_COPILOT = "github_copilot"
    ZED = "zed"
    PI = "pi"
    OPENCLAW = "openclaw"
    CHERRY_STUDIO = "cherry_studio"
    CHATBOX = "chatbox"
    N8N = "n8n"
    LITELLM = "litellm"


# Display label of every application, the one place a report reads its name from.
CLIENT_APP_LABELS: dict[ClientApp, str] = {
    ClientApp.CLAUDE_CODE: "Claude Code",
    ClientApp.CODEX: "Codex",
    ClientApp.OPENCODE: "OpenCode",
    ClientApp.HERMES: "Hermes Agent",
    ClientApp.KILO_CODE: "Kilo Code",
    ClientApp.CLINE: "Cline",
    ClientApp.ROO_CODE: "Roo Code",
    ClientApp.CURSOR: "Cursor",
    ClientApp.QWEN_CODE: "Qwen Code",
    ClientApp.GEMINI_CLI: "Gemini CLI",
    ClientApp.GITHUB_COPILOT: "GitHub Copilot",
    ClientApp.ZED: "Zed",
    ClientApp.PI: "Pi",
    ClientApp.OPENCLAW: "OpenClaw",
    ClientApp.CHERRY_STUDIO: "Cherry Studio",
    ClientApp.CHATBOX: "Chatbox",
    ClientApp.N8N: "n8n",
    ClientApp.LITELLM: "LiteLLM",
}

# Ordered: the first matching rule wins. Hermes precedes Claude Code because Hermes's gateway
# mode identifies as ``claude-code/1.0.0 (Hermes Gateway)``.
_USER_AGENT_RULES: tuple[tuple[ClientApp, re.Pattern[str]], ...] = (
    (ClientApp.HERMES, re.compile(r"hermes[-_ ]?agent/|\(hermes gateway\)")),
    (ClientApp.CLAUDE_CODE, re.compile(r"^claude-(?:cli|code)/")),
    (ClientApp.CODEX, re.compile(r"^codex(?:_[a-z_]+)?/|^codex desktop/|^codex$")),
    (ClientApp.OPENCODE, re.compile(r"^opencode/")),
    (ClientApp.KILO_CODE, re.compile(r"^kilo-?code/")),
    (ClientApp.CLINE, re.compile(r"^cline/")),
    (ClientApp.ROO_CODE, re.compile(r"^roo-?code/")),
    (ClientApp.CURSOR, re.compile(r"^cursor/")),
    (ClientApp.QWEN_CODE, re.compile(r"^qwen-?code/")),
    (ClientApp.GEMINI_CLI, re.compile(r"^gemini-?cli/")),
    (ClientApp.GITHUB_COPILOT, re.compile(r"^githubcopilotchat/")),
    (ClientApp.ZED, re.compile(r"^zed/")),
    (ClientApp.PI, re.compile(r"^pi(?:/|$)")),
    (ClientApp.OPENCLAW, re.compile(r"\bopenclaw\b")),
    (ClientApp.CHERRY_STUDIO, re.compile(r"\bcherrystudio/")),
    (ClientApp.CHATBOX, re.compile(r"\bchatboxapp\b")),
    (ClientApp.N8N, re.compile(r"^n8n\b")),
    (ClientApp.LITELLM, re.compile(r"^litellm/")),
)

_TITLE_APPS: dict[str, ClientApp] = {
    "claude code": ClientApp.CLAUDE_CODE,
    "codex": ClientApp.CODEX,
    "opencode": ClientApp.OPENCODE,
    "hermes agent": ClientApp.HERMES,
    "hermes": ClientApp.HERMES,
    "kilo code": ClientApp.KILO_CODE,
    "cline": ClientApp.CLINE,
    "roo code": ClientApp.ROO_CODE,
    "cursor": ClientApp.CURSOR,
    "qwen code": ClientApp.QWEN_CODE,
    "zed": ClientApp.ZED,
    "openclaw": ClientApp.OPENCLAW,
    "cherry studio": ClientApp.CHERRY_STUDIO,
    "chatbox": ClientApp.CHATBOX,
}

_REFERER_HOSTS: dict[str, ClientApp] = {
    "opencode.ai": ClientApp.OPENCODE,
    "hermes-agent.nousresearch.com": ClientApp.HERMES,
    "kilocode.ai": ClientApp.KILO_CODE,
    "kilo.ai": ClientApp.KILO_CODE,
    "cline.bot": ClientApp.CLINE,
    "roocode.com": ClientApp.ROO_CODE,
    "cursor.com": ClientApp.CURSOR,
    "zed.dev": ClientApp.ZED,
    "openclaw.ai": ClientApp.OPENCLAW,
    "cherry-ai.com": ClientApp.CHERRY_STUDIO,
    "chatboxai.app": ClientApp.CHATBOX,
}


def _inspected(value: str | None) -> str | None:
    """Return a bounded, lower-cased, stripped header value, or ``None`` when blank."""
    if value is None:
        return None
    text = value[:_MAXIMUM_INSPECTED_CHARS].strip().lower()
    return text or None


def bounded_user_agent(user_agent: str | None) -> str | None:
    """Return the storable ``User-Agent`` prefix, or ``None`` when absent or blank.

    Args:
        user_agent: Raw header value decoded as latin-1.

    Returns:
        At most ``MAXIMUM_USER_AGENT_CHARS`` characters with surrounding whitespace removed.
    """
    if user_agent is None:
        return None
    text = user_agent.strip()[:MAXIMUM_USER_AGENT_CHARS]
    return text or None


def _referer_app(referer: str) -> ClientApp | None:
    """Match an ``HTTP-Referer`` host (or one of its parent domains) to an application."""
    try:
        host = urlsplit(referer if "//" in referer else f"//{referer}").hostname
    except ValueError:
        return None
    while host:
        if host in _REFERER_HOSTS:
            return _REFERER_HOSTS[host]
        _, _, host = host.partition(".")
        if "." not in host:
            return None
    return None


def classify_client_app(
    *,
    user_agent: str | None,
    originator: str | None = None,
    app_title: str | None = None,
    app_referer: str | None = None,
) -> ClientApp | None:
    """Classify the calling application from content-free request headers.

    ``User-Agent`` is the most specific signal, then Codex's ``originator``, then the
    OpenRouter-style ``X-Title`` and ``HTTP-Referer`` app identity.

    Args:
        user_agent: ``User-Agent`` header value.
        originator: ``originator`` header value (sent by Codex clients).
        app_title: ``X-Title`` header value.
        app_referer: ``HTTP-Referer`` header value.

    Returns:
        The matched application, or ``None`` when no rule recognizes the caller.
    """
    agent = _inspected(user_agent)
    if agent is not None:
        for app, pattern in _USER_AGENT_RULES:
            if pattern.search(agent):
                return app
    origin = _inspected(originator)
    if origin is not None and origin.startswith("codex"):
        return ClientApp.CODEX
    title = _inspected(app_title)
    if title is not None and title in _TITLE_APPS:
        return _TITLE_APPS[title]
    referer = _inspected(app_referer)
    if referer is not None:
        return _referer_app(referer)
    return None


class ClientAttribution(ContractModel):
    """Content-free calling-application facts frozen onto an authorization snapshot.

    Attributes:
        client_app: Classified calling application, or ``None`` when no rule matched.
        user_agent: Bounded ``User-Agent`` prefix the classification read, kept so an
            unrecognized agent can be identified and added to the vocabulary later.
    """

    client_app: ClientApp | None = None
    user_agent: str | None = Field(default=None, max_length=MAXIMUM_USER_AGENT_CHARS)


def _header_text(data: Mapping[str, object], name: str) -> str | None:
    """Return one forwarded header value when it is a string, otherwise ``None``."""
    value = data.get(name)
    return value if isinstance(value, str) else None


def with_client_identity[T: ClientAttribution](authorization: T, data: Mapping[str, object]) -> T:
    """Freeze the caller's application identity onto an authorized snapshot.

    Args:
        authorization: Snapshot returned by the control store.
        data: Admission argument carrying the forwarded ``user_agent``, ``originator``,
            ``app_title`` and ``app_referer`` header values.

    Returns:
        The same snapshot type with ``client_app`` and ``user_agent`` set.
    """
    user_agent = _header_text(data, "user_agent")
    return authorization.model_copy(
        update={
            "client_app": classify_client_app(
                user_agent=user_agent,
                originator=_header_text(data, "originator"),
                app_title=_header_text(data, "app_title"),
                app_referer=_header_text(data, "app_referer"),
            ),
            "user_agent": bounded_user_agent(user_agent),
        }
    )
