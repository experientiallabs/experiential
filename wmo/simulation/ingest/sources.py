"""Resolve one declared trace source name to its canonical loader.

Ingestion owns the set of supported sources, so a caller names a source and passes a local path
instead of importing one loader per vendor. The mapping is an explicit table in this module: there
is no import-time registration, no plugin discovery, and no format detection. A name the table does
not declare fails closed.

    result = load_trace_source("langfuse", Path("export.jsonl"))

Every loader returns the same ``TraceNormalizationResult``, and every failure that is specific to a
source (unreadable bytes, malformed payloads, an invalid Postgres declaration) is raised as
``TraceSourceError`` so a caller does not have to know which vendor errors exist.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from wmo.simulation.ingest.braintrust import load_braintrust_file
from wmo.simulation.ingest.chat_json import load_chat_json_file
from wmo.simulation.ingest.langfuse import load_langfuse_file
from wmo.simulation.ingest.langsmith import load_langsmith_file
from wmo.simulation.ingest.mastra import load_mastra_file
from wmo.simulation.ingest.otel_genai import load_otel_genai_file
from wmo.simulation.ingest.otlp import (
    OtlpTraceFormatError,
    TraceNormalizationResult,
    load_otlp_file,
)
from wmo.simulation.ingest.phoenix import load_phoenix_file
from wmo.simulation.ingest.postgres import (
    PostgresSourceError,
    load_postgres_source,
    read_postgres_source_file,
)
from wmo.simulation.ingest.posthog import PostHogPullError, load_posthog_file
from wmo.simulation.ingest.vendor_records import VendorTraceFormatError

POSTGRES_SOURCE = "postgres"


class TraceSourceError(ValueError):
    """Raised when a source name is unsupported or its declared corpus cannot be normalized."""


def _load_postgres_declaration(path: Path) -> TraceNormalizationResult:
    """Read one Postgres table declared by a local JSON file.

    Args:
        path: Local Postgres source declaration.

    Returns:
        Canonical traces and every retained validation exclusion.
    """
    return load_postgres_source(read_postgres_source_file(path))


_LOADERS: dict[str, Callable[[Path], TraceNormalizationResult]] = {
    "braintrust": load_braintrust_file,
    "chat-json": load_chat_json_file,
    "langfuse": load_langfuse_file,
    "langsmith": load_langsmith_file,
    "mastra": load_mastra_file,
    "otel-genai": load_otel_genai_file,
    "otlp": load_otlp_file,
    "phoenix": load_phoenix_file,
    POSTGRES_SOURCE: _load_postgres_declaration,
    "posthog": load_posthog_file,
}
CANONICAL_TRACE_SOURCES: tuple[str, ...] = tuple(sorted(_LOADERS))


def load_trace_source(source: str, path: Path) -> TraceNormalizationResult:
    """Normalize one local corpus through the loader of its declared source.

    Args:
        source: Declared source name, matched case-insensitively after trimming.
        path: Local trace export, or the JSON table declaration of the Postgres source.

    Returns:
        Canonical traces and every retained validation exclusion.

    Raises:
        TraceSourceError: The source is unsupported or its corpus cannot be normalized.
    """
    loader = _LOADERS.get(source.strip().casefold())
    if loader is None:
        choices = ", ".join(CANONICAL_TRACE_SOURCES)
        raise TraceSourceError(f"unsupported trace source {source!r}; choose one of: {choices}")
    try:
        return loader(path)
    except (
        OtlpTraceFormatError,
        PostHogPullError,
        PostgresSourceError,
        VendorTraceFormatError,
        ValueError,
    ) as exc:
        raise TraceSourceError(f"{source.strip().casefold()} normalization failed: {exc}") from None
