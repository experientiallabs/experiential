"""Normalized types for context connectors.

Connectors (github, google, slack, notion, ...) authenticate against a service and pull content
into these vendor-agnostic shapes. Everything downstream (the bundle store, markdown rendering,
knowledge attachment) operates on `ContextItem`, never on raw vendor payloads. The `opt_str`,
`capped`, and `strip_html` helpers are the shared coercions for raw vendor JSON and fetched
content bodies.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from wmh.core.types import JsonObject, JsonValue

# Fetched content is capped so one huge document or page cannot blow up a bundle.
CONTENT_CAP_CHARS = 200_000
TRUNCATION_MARKER = f"\n[content truncated at {CONTENT_CAP_CHARS} characters]"


def opt_str(value: JsonValue | None) -> str | None:
    """`value` when it is a non-empty string, else None (vendor JSON field coercion)."""
    return value if isinstance(value, str) and value else None


def capped(text: str) -> str:
    """Cap fetched content, appending a loud truncation marker when anything was cut."""
    if len(text) <= CONTENT_CAP_CHARS:
        return text
    return text[:CONTENT_CAP_CHARS] + TRUNCATION_MARKER


def strip_html(markup: str) -> str:
    """A small HTML-to-text fallback: drop script/style, break on block ends, strip tags."""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", markup)
    text = re.sub(r"(?is)<br\s*/?>|</p>|</div>", "\n", text)
    text = re.sub(r"(?s)<[^>]*>", " ", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


class ConnectError(RuntimeError):
    """A connector operation failed; messages say what went wrong and what to do about it."""


@contextmanager
def transport_errors(host: str) -> Iterator[None]:
    """Turn httpx transport failures inside the block into actionable ConnectErrors.

    Connectors wrap their HTTP calls with this so network-level failures (DNS errors, refused
    connections, timeouts) honor the ConnectError contract instead of escaping to callers as
    raw httpx tracebacks.

    Args:
        host: The host the block talks to, named in the error message.

    Raises:
        ConnectError: For any `httpx.HTTPError` raised inside the block.
    """
    try:
        yield
    except httpx.HTTPError as exc:
        detail = str(exc) or type(exc).__name__
        raise ConnectError(
            f"could not reach {host} ({detail}); check your network connection and retry"
        ) from exc


class ItemKind(StrEnum):
    """The normalized kind of one pulled content item."""

    DOCUMENT = "document"
    PAGE = "page"
    ISSUE = "issue"
    PULL_REQUEST = "pull_request"
    MESSAGE = "message"
    THREAD = "thread"
    EMAIL = "email"
    EVENT = "event"
    FILE = "file"


class ContextItem(BaseModel):
    """One normalized piece of pulled content (an issue, a page, a message, ...).

    Attributes:
        id: Stable identifier within the source service (issue number, page id, message ts).
        source: The connector name that produced the item (e.g. "github").
        kind: What the item is, from the normalized `ItemKind` vocabulary.
        title: Short human title; shown as the item's markdown section heading.
        body: The content itself, plain text or markdown.
        url: Canonical link back to the item, when the service has one.
        created_at: ISO-8601 creation timestamp, when known.
        updated_at: ISO-8601 last-modified timestamp, when known.
        metadata: Connector-specific extras (labels, authors, channel ids) as arbitrary JSON.
    """

    id: str
    source: str
    kind: ItemKind
    title: str
    body: str
    url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class PullQuery(BaseModel):
    """What to pull: the parameters every connector's `pull` accepts.

    Attributes:
        target: Service-specific container: a repo "owner/name", a channel name, a calendar id,
            a drive folder.
        query: Free-text or service search-syntax filter.
        since: ISO-8601 date or datetime lower bound on item time.
        until: ISO-8601 date or datetime upper bound on item time.
        limit: Maximum number of items a connector may fetch (connectors must cap at this).
    """

    target: str | None = None
    query: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int = 100


class ConnectorAuth(BaseModel):
    """A stored credential for one connector.

    Attributes:
        kind: "oauth" for browser/device OAuth grants, "token" for pasted or env-injected tokens.
        access_token: The bearer credential API calls send.
        refresh_token: OAuth refresh token, when the provider issued one.
        expires_at: ISO-8601 absolute expiry of `access_token`, when known.
        scopes: The granted OAuth scopes.
        account: Human-readable identity captured at connect time (e.g. "octocat").
        extra: Connector-specific extras (e.g. a slack team id) as arbitrary JSON.
    """

    kind: Literal["oauth", "token"]
    access_token: str
    refresh_token: str | None = None
    expires_at: str | None = None
    scopes: list[str] = Field(default_factory=list)
    account: str | None = None
    extra: JsonObject = Field(default_factory=dict)
