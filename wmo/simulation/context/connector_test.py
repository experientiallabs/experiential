"""Tests for the ContextConnector protocol, ConnectUI, and the registry."""

from __future__ import annotations

import pytest

import wmo.simulation.context as connect
from wmo.simulation.context.connector import (
    ConnectUI,
    ContextConnector,
    get_connector,
    list_connectors,
    register_connector,
)
from wmo.simulation.context.types import ConnectorAuth, ContextItem, ItemKind, PullQuery


class FakeConnector:
    """A minimal in-memory connector used to exercise the seam."""

    name = "fake"
    label = "Fake Service"

    def connect(self, ui: ConnectUI) -> ConnectorAuth:
        ui.info("starting fake auth")
        ui.open_url("https://fake.test/authorize")
        ui.present_code("https://fake.test/activate", "ABCD-1234")
        secret = ui.prompt_secret("paste the token")
        return ConnectorAuth(kind="token", access_token=secret)

    def verify(self, auth: ConnectorAuth) -> str:
        return f"fake-user ({auth.access_token})"

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        item = ContextItem(
            id="1", source=self.name, kind=ItemKind.DOCUMENT, title="Doc", body="hello"
        )
        return [item][: query.limit]


def test_fake_connector_satisfies_the_protocol() -> None:
    assert isinstance(FakeConnector(), ContextConnector)


def test_connect_ui_dispatches_to_the_injected_callbacks() -> None:
    calls: list[tuple[str, ...]] = []
    ui = ConnectUI(
        open_url=lambda url: calls.append(("open", url)),
        present_code=lambda uri, code: calls.append(("code", uri, code)),
        prompt_secret=lambda label: "s3cret",
        info=lambda message: calls.append(("info", message)),
    )

    auth = FakeConnector().connect(ui)

    assert auth.access_token == "s3cret"
    assert ("info", "starting fake auth") in calls
    assert ("open", "https://fake.test/authorize") in calls
    assert ("code", "https://fake.test/activate", "ABCD-1234") in calls


def test_registry_round_trip_and_sorted_listing() -> None:
    first = FakeConnector()
    second = FakeConnector()
    second.name = "another-fake"

    register_connector(first)
    register_connector(second)

    assert get_connector("fake") is first
    assert get_connector("another-fake") is second
    listed = list_connectors()
    assert listed == sorted(listed)
    assert {"fake", "another-fake"} <= set(listed)


def test_get_connector_unknown_name_is_actionable() -> None:
    with pytest.raises(ValueError, match="no context connector registered"):
        get_connector("nope")


# --- The package surface ------------------------------------------------------------------------
# `wmo/simulation/context/__init__.py` has no suite of its own (AGENTS.md rule 2 exempts it), and
# both of these are about the package rather than any one module in it, so they live beside the
# registry they are mostly about.


def test_every_name_the_package_promises_is_importable() -> None:
    """A connector author imports from `wmo.simulation.context`, so `__all__` must resolve.

    Listing the expected symbols here would only restate `__all__`; what can actually break is a
    name left in `__all__` after the thing it points at moved, which turns a documented import
    into an AttributeError while `import wmo.simulation.context` still succeeds.
    """
    missing = [name for name in connect.__all__ if not hasattr(connect, name)]

    assert not missing, f"names in wmo.simulation.context.__all__ that no longer resolve: {missing}"


def test_builtin_connectors_register_on_package_import() -> None:
    """Importing the package registers every built-in, without the optional MCP extra installed."""
    registered = set(connect.list_connectors())

    assert {"github", "google-calendar", "google-drive", "gmail", "slack", "notion"} <= registered
