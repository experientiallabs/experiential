"""Tests for the TraceAdapter seam: the protocol, the registry, and the vendor-pull params."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import wmo.simulation.ingest.adapter as adapter_module
from wmo.simulation.ingest import list_adapters as list_bundled_adapters
from wmo.simulation.ingest.adapter import (
    SourceCredentialError,
    TraceAdapter,
    VendorPull,
    get_adapter,
    list_adapters,
    register_adapter,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from wmo.common.core.types import Trace


class _StubAdapter:
    """A registrable adapter that answers both transports with nothing."""

    def __init__(self, name: str) -> None:
        self.name = name

    def from_file(self, path: str) -> list[Trace]:
        return []

    def from_vendor(self, pull: VendorPull) -> list[Trace]:
        return []


@pytest.fixture
def isolated_registry() -> Iterator[None]:
    """Register into the real registry, then restore it, so bundled adapters stay untouched."""
    saved = dict(adapter_module._ADAPTERS)
    yield
    adapter_module._ADAPTERS.clear()
    adapter_module._ADAPTERS.update(saved)


def test_the_protocol_is_a_name_plus_both_transports() -> None:
    declared = sorted(
        member
        for member in (*vars(TraceAdapter), *TraceAdapter.__annotations__)
        if not member.startswith("_")
    )

    assert declared == ["from_file", "from_vendor", "name"]
    assert isinstance(_StubAdapter("stub"), TraceAdapter)


def test_register_then_get_returns_the_same_adapter(isolated_registry: None) -> None:
    adapter = _StubAdapter("stub-source")

    register_adapter(adapter)

    assert get_adapter("stub-source") is adapter


def test_registering_the_same_name_replaces_the_earlier_adapter(isolated_registry: None) -> None:
    register_adapter(_StubAdapter("stub-source"))
    second = _StubAdapter("stub-source")

    register_adapter(second)

    assert get_adapter("stub-source") is second


def test_an_unknown_name_names_what_is_registered(isolated_registry: None) -> None:
    # The error is the interface here: `wmo ingest --source typo` must show the real choices.
    register_adapter(_StubAdapter("stub-source"))

    with pytest.raises(ValueError, match="no trace adapter registered for 'nope'") as excinfo:
        get_adapter("nope")

    assert "stub-source" in str(excinfo.value)


def test_list_adapters_is_sorted(isolated_registry: None) -> None:
    register_adapter(_StubAdapter("zzz"))
    register_adapter(_StubAdapter("aaa"))

    names = list_adapters()

    assert names == sorted(names)
    assert names.index("aaa") < names.index("zzz")


def test_importing_the_package_registers_the_bundled_adapters() -> None:
    # Registration is an import side effect of `wmo.simulation.ingest`, so a bundled adapter that
    # stopped registering would only fail when someone selected that source.
    assert {"otel-genai", "chat-json", "postgres"} <= set(list_bundled_adapters())


def test_vendor_pull_defaults_to_env_and_source_defaults() -> None:
    # Every field is optional: a bare `VendorPull()` means "use the vendor env var and the
    # adapter's own column/table defaults", which is what the CLI passes when given no flags.
    pull = VendorPull()

    assert pull.api_key is None
    assert pull.project is None
    assert pull.since is None
    assert pull.limit is None
    assert pull.dsn is None
    assert pull.table is None
    assert pull.trace_id_column is None
    assert pull.payload_column is None
    assert pull.order_column is None


def test_source_credential_error_is_a_permission_error() -> None:
    # Subclassing PermissionError keeps it catchable without a driver import, and is what lets the
    # streaming ingest map it to `bad_credentials` without misreading an unreadable local file.
    error = SourceCredentialError("bad key")

    assert isinstance(error, PermissionError)
    assert not isinstance(PermissionError("unreadable file"), SourceCredentialError)
