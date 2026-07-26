"""TraceAdapter protocol + a small registry.

Sources differ in two ways: *transport* (file vs. vendor SDK) and *schema* (which OTel semantic
convention the spans follow). An adapter owns both: it pulls/reads raw spans and normalizes them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from wmh.core.types import Trace


class SourceCredentialError(PermissionError):
    """A source rejected the supplied credentials (API key, DSN password).

    Adapters raise this instead of bare `PermissionError` so the streaming ingest can map it
    to the `bad_credentials` wire code without misclassifying OS-level permission failures
    (e.g. an unreadable local file) as credential problems. Subclassing `PermissionError`
    keeps it a stdlib type: callers never need a driver import to catch it.
    """


class VendorPull(BaseModel):
    """Parameters for pulling traces from an observability vendor's API or a database."""

    api_key: str | None = None  # falls back to the vendor's env var when None
    project: str | None = None  # vendor project / workspace to pull from
    since: str | None = None  # ISO-8601 lower bound on trace start time
    limit: int | None = None  # max traces to pull

    # Database-source transport params (the `postgres` adapter); API-vendor adapters ignore them.
    dsn: str | None = None  # connection string; falls back to $WMH_POSTGRES_DSN
    table: str | None = None  # table holding the trace rows
    trace_id_column: str | None = None  # override; default "trace_id" (absent -> row-per-trace)
    payload_column: str | None = None  # override; default "payload"
    order_column: str | None = None  # override; default "created_at" when the table has it


@runtime_checkable
class TraceAdapter(Protocol):
    """Turns one source's raw telemetry into normalized `Trace` objects."""

    name: str

    def from_file(self, path: str) -> list[Trace]:
        """Read traces from an exported file (OTLP-JSON / vendor JSONL)."""
        ...

    def from_vendor(self, pull: VendorPull) -> list[Trace]:
        """Pull traces via a vendor SDK/API."""
        ...


_ADAPTERS: dict[str, TraceAdapter] = {}


def register_adapter(adapter: TraceAdapter) -> None:
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> TraceAdapter:
    if name not in _ADAPTERS:
        raise ValueError(f"no trace adapter registered for {name!r}; have {list(_ADAPTERS)}")
    return _ADAPTERS[name]


def list_adapters() -> list[str]:
    """Names of all registered trace adapters, sorted (what the build source picker shows)."""
    return sorted(_ADAPTERS)
