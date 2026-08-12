"""Tests for the TraceAdapter registry: what `--source` resolves to, and what it says when wrong.

The `TraceAdapter` protocol gets no test of its own, and neither does `VendorPull`: listing the
members one declares, or the all-optional fields of the other, only restates the declaration. What
bites is registry behavior (a bundled source silently not registering, a typo'd `--source` failing
without naming the real choices), so that is what is exercised here, against the real registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import wmo.simulation.ingest.adapter as adapter_module
from wmo.simulation.ingest import list_adapters as list_bundled_adapters
from wmo.simulation.ingest.adapter import (
    SourceCredentialError,
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


def test_a_bare_vendor_pull_asks_every_adapter_for_its_own_defaults() -> None:
    # This is what the CLI passes when the user gives no pull flags, so nothing here may carry a
    # value of its own: a default table or column set here would silently override the adapter's.
    assert not VendorPull().model_dump(exclude_none=True)


def test_source_credential_error_is_a_permission_error() -> None:
    # Subclassing PermissionError keeps it catchable without a driver import, and is what lets the
    # streaming ingest map it to `bad_credentials` without misreading an unreadable local file.
    error = SourceCredentialError("bad key")

    assert isinstance(error, PermissionError)
    assert not isinstance(PermissionError("unreadable file"), SourceCredentialError)
