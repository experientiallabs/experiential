"""`BaseTraceAdapter` — shared scaffolding so a new source is transport + mapping, not a rewrite.

A span-based provider adapter only needs to answer two questions:
  1. how do I get raw bytes/objects? (a file path, or a vendor API/SDK pull)
  2. how do I turn one raw payload into `SpanRecord`s? (the provider's export shape)
Everything after that — JSON/JSONL loading, grouping spans into `Trace`s, honoring `wmh.*`
enrichments — is shared (`wmh.ingest.normalize`). `BaseTraceAdapter` wires (1) and (2) together so
a concrete adapter is typically ~30 lines: set `name`, implement `spans_from_payload`, and (if it
supports live pulls) `_pull_payloads`.

Subclasses override:
  - `name`: the registry key (e.g. "phoenix").
  - `spans_from_payload(payload) -> list[SpanRecord]`: map ONE decoded JSON payload to spans. The
    default delegates to `wmh.ingest.normalize.collect_spans` (OTLP/OpenInference-JSON); override it
    when the provider's export is not OTLP-shaped.
  - `_pull_payloads(pull) -> list[JsonValue]`: yield raw payloads from the vendor API/SDK. The
    default raises a friendly "not implemented" — so `from_file` works with no override, and
    `from_vendor` is opt-in.

`from_file` accepts a single JSON document (object or array) OR JSONL (one payload per line), and
skips a single corrupt JSONL line rather than aborting the whole ingest.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import JsonValue

from wmh.core.types import Trace
from wmh.ingest.adapter import VendorPull
from wmh.ingest.normalize import SpanRecord, collect_spans, spans_to_traces


class BaseTraceAdapter:
    """Default file+vendor plumbing around the shared span normalizer."""

    name: str = "base"

    # --- mapping hook (override for non-OTLP export shapes) -----------------------------------

    def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
        """Map ONE decoded payload to `SpanRecord`s. Default: OTLP/OpenInference-JSON collection."""
        return collect_spans(payload)

    # --- transport hooks ----------------------------------------------------------------------

    def _pull_payloads(self, pull: VendorPull) -> list[JsonValue]:
        """Fetch raw payloads from the vendor API/SDK. Override to support live pulls."""
        raise ValueError(
            f"{self.name!r} does not support live vendor pulls yet; export traces to a file and "
            f"use `from_file` (or `wmh ingest run --source {self.name} --file <export>`)"
        )

    # --- public API (rarely overridden) -------------------------------------------------------

    def _load_payloads(self, text: str) -> list[JsonValue]:
        """Parse a whole-document JSON payload, or per-line JSONL on a top-level decode failure."""
        try:
            return [json.loads(text)]
        except json.JSONDecodeError:
            payloads: list[JsonValue] = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payloads.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue  # tolerate a truncated/corrupt line; keep the rest
            return payloads

    def _collect_all(self, payloads: list[JsonValue]) -> list[SpanRecord]:
        """Map every payload to spans and re-stamp `span_id` globally unique in emission order.

        Adapters assign `span_id`/`start_nano` per payload, so a trace split across payloads (e.g.
        one row/observation/run per JSONL line) would otherwise emit colliding ids and identical
        `start_nano`. `spans_to_traces` sorts by `(start_nano, span_id)`, so the collision scrambles
        the action/observation pairing. Stamping a globally monotonic `span_id` makes equal-time
        spans (the row-adapter case, all `start_nano=0`) order by emission, while real timestamps
        (e.g. Phoenix) still dominate the sort. Uniqueness is owned here so adapters can't get it
        wrong individually.
        """
        spans: list[SpanRecord] = []
        for payload in payloads:
            for span in self.spans_from_payload(payload):
                span.span_id = f"{len(spans):012d}-{span.span_id}"
                spans.append(span)
        return spans

    def from_file(self, path: str) -> list[Trace]:
        text = Path(path).read_text(encoding="utf-8")
        spans = self._collect_all(self._load_payloads(text))
        return spans_to_traces(spans, source=f"{self.name}:{path}")

    def from_vendor(self, pull: VendorPull) -> list[Trace]:
        spans = self._collect_all(self._pull_payloads(pull))
        traces = spans_to_traces(spans, source=f"{self.name}:{pull.project or 'vendor'}")
        if pull.limit is not None:
            traces = traces[: pull.limit]
        return traces
