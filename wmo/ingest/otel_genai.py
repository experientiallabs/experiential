"""Official OpenTelemetry GenAI semantic-convention adapter.

This is the reference adapter and the harness's own persistence format (`wmo.ingest.otel_writer`
emits it): OTLP-JSON spans following the `gen_ai.*` semantic conventions, mapped to normalized
`Trace`s by the shared normalizer (`wmo.ingest.normalize`), which owns span classification,
action/observation pairing, and the optional `wmo.*` enrichments (`wmo.state.*`,
`wmo.trace.metadata`, `wmo.attribution`).

Supported file formats (via `BaseTraceAdapter`):
  - OTLP-JSON: a single object with `resourceSpans` -> `scopeSpans` -> `spans`
  - JSON array of the above, or of bare span objects
  - JSONL: one OTLP-JSON object or one bare span object per line

The vendor pull is a best-effort generic OTLP-JSON fetch: OTLP itself is push-only, so "pulling"
traces is inherently vendor-specific (Grafana Tempo, Jaeger, Honeycomb, ... each expose their own
query API and auth).
"""

from __future__ import annotations

import os

import httpx
from pydantic import JsonValue

from wmo.ingest.adapter import VendorPull, register_adapter
from wmo.ingest.base import BaseTraceAdapter

# Env vars the (placeholder) vendor pull reads. The real query semantics are vendor-specific; see
# the TODO in `_pull_payloads`.
VENDOR_ENDPOINT_ENV = "WMO_OTLP_QUERY_ENDPOINT"
VENDOR_API_KEY_ENV = "WMO_OTLP_API_KEY"


class OtelGenAIAdapter(BaseTraceAdapter):
    """OTLP-JSON GenAI spans, file or generic OTLP query backend. Pure shared-normalizer."""

    name = "otel-genai"

    def _pull_payloads(self, pull: VendorPull) -> list[JsonValue]:
        """Fetch OTLP-JSON from an OTLP-compatible query backend.

        TODO: implement the vendor-specific query protocol and auth. Today we:
          - read the query URL from ``$WMO_OTLP_QUERY_ENDPOINT``;
          - send ``pull.api_key`` (or ``$WMO_OTLP_API_KEY``) as a Bearer token;
          - pass project/since/limit as query params (real backends name these differently).
        """
        endpoint = os.environ.get(VENDOR_ENDPOINT_ENV)
        if not endpoint:
            raise ValueError(
                f"set ${VENDOR_ENDPOINT_ENV} to an OTLP-compatible query URL to pull traces "
                "(vendor-specific query support is not yet implemented; use from_file meanwhile)"
            )
        api_key = pull.api_key or os.environ.get(VENDOR_API_KEY_ENV)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # TODO: these param names are placeholders; map to the target backend's query schema.
        params: dict[str, str] = {}
        if pull.project is not None:
            params["project"] = pull.project
        if pull.since is not None:
            params["since"] = pull.since
        if pull.limit is not None:
            params["limit"] = str(pull.limit)
        response = httpx.get(endpoint, headers=headers, params=params, timeout=30.0)
        response.raise_for_status()
        return [response.json()]


register_adapter(OtelGenAIAdapter())
