"""Trace ingestion: file uploads and vendor pulls -> normalized `Trace` objects.

A `TraceAdapter` turns one source's telemetry into the generic `Trace` schema. Adapters register
themselves on import and are looked up by name (`get_adapter`) or listed (`list_adapters`). The
span-based adapters share one normalizer (`wmh.ingest.normalize`) and the `BaseTraceAdapter`
scaffolding, so adding a new source is transport + attribute-mapping, not a rewrite. See
`docs/ingest.md`.

Bundled adapters:
  - `otel-genai`  : OTLP-JSON spans following the OTel GenAI semantic conventions (file or pull).
  - `chat-json`   : recorded OpenAI-style chat/tool-call conversations (file).
Provider adapters (Braintrust, Phoenix/Arize, Langfuse, LangSmith) register when their module is
imported; their heavy SDKs are optional extras, imported lazily inside the adapter.
"""

# Import for the registration side effect so `get_adapter(...)` works on package import. The
# provider adapters are SDK-free (they parse exports as JSON and pull over httpx), so importing them
# here is cheap and brings no heavy dependency — their optional extras only matter if a user drives
# the provider's own SDK alongside `wmh ingest`.
from wmh.ingest import braintrust as braintrust  # noqa: F401
from wmh.ingest import langfuse as langfuse  # noqa: F401
from wmh.ingest import langsmith as langsmith  # noqa: F401
from wmh.ingest import mastra as mastra  # noqa: F401
from wmh.ingest import messages as messages  # noqa: F401
from wmh.ingest import otel_genai as otel_genai  # noqa: F401
from wmh.ingest import phoenix as phoenix  # noqa: F401
from wmh.ingest import posthog as posthog  # noqa: F401
from wmh.ingest.adapter import (
    TraceAdapter,
    VendorPull,
    get_adapter,
    list_adapters,
    register_adapter,
)
from wmh.ingest.base import BaseTraceAdapter

__all__ = [
    "BaseTraceAdapter",
    "TraceAdapter",
    "VendorPull",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
