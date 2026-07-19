"""ContextConnector protocol + a small registry (mirrors `wmh.ingest.adapter`).

A connector owns one service end to end: the interactive auth flow, a cheap credential check,
and pulling content normalized into `ContextItem`s. Connectors register themselves on import and
are looked up by name (`get_connector`) or listed (`list_connectors`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from wmh.connect.types import ConnectorAuth, ContextItem, PullQuery


@dataclass
class ConnectUI:
    """Presentation callbacks a connector uses during interactive auth.

    Connectors never print: the CLI layer builds one of these over its rich Console; tests pass
    recording lambdas.

    Attributes:
        open_url: Show and/or open the browser authorization URL.
        present_code: Show a device-flow verification URI and user code.
        prompt_secret: Ask the user for a secret (e.g. a pasted API token); returns the value.
        info: Show a short status message.
    """

    open_url: Callable[[str], None]
    present_code: Callable[[str, str], None]
    prompt_secret: Callable[[str], str]
    info: Callable[[str], None]


@runtime_checkable
class ContextConnector(Protocol):
    """One service's auth + pull logic, producing normalized `ContextItem`s.

    Attributes:
        name: Registry key and `ContextItem.source` value (e.g. "github").
        label: Human name shown in pickers (e.g. "GitHub").
    """

    name: str
    label: str

    def connect(self, ui: ConnectUI) -> ConnectorAuth:
        """Run the interactive auth flow (OAuth or token prompt) and return the credential."""
        ...

    def verify(self, auth: ConnectorAuth) -> str:
        """Cheap API call returning a human identity string; raises ConnectError on bad auth."""
        ...

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Fetch content matching `query`, normalized and capped at `query.limit` items."""
        ...


_CONNECTORS: dict[str, ContextConnector] = {}


def register_connector(connector: ContextConnector) -> None:
    """Register a connector under its `name` (typically at module import time)."""
    _CONNECTORS[connector.name] = connector


def get_connector(name: str) -> ContextConnector:
    """Look up a registered connector by name."""
    if name not in _CONNECTORS:
        raise ValueError(f"no context connector registered for {name!r}; have {list(_CONNECTORS)}")
    return _CONNECTORS[name]


def list_connectors() -> list[str]:
    """Names of all registered connectors, sorted (what the CLI picker shows)."""
    return sorted(_CONNECTORS)
