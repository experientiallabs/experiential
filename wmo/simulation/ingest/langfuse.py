"""Langfuse adapter: turn a Langfuse trace export (observation tree) into `Trace`s.

Langfuse does NOT export OTLP spans. Its public API (`GET /api/public/traces/{id}`) and SDK return
a **trace** object with a flat list of nested **observations**, each typed
`SPAN | GENERATION | EVENT | TOOL`:

    {"id": "<traceId>", "name": "...", "input": <task>, "output": ...,
     "observations": [
        {"id": "o1", "type": "GENERATION", "name": "llm",
         "input": [{"role": "user", "content": "..."}], "output": {...},
         "startTime": "2026-01-01T00:00:00.000Z", "model": "gpt-4o",
         # a GENERATION that issues a tool call carries it in output.tool_calls (OpenAI shape)
         "output": {"role": "assistant", "tool_calls": [{"id": "c1",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}},
        {"id": "o2", "type": "TOOL", "name": "get_weather",
         "input": {"city": "Paris"}, "output": "18C and sunny",
         "startTime": "...", "level": "DEFAULT"},
        {"id": "o3", "type": "SPAN", "name": "...", "level": "ERROR", ...}
     ]}

Because this is not an OTLP/OpenInference span shape, the adapter overrides `spans_from_payload`
(like `wmo.simulation.ingest.messages`) and emits `SpanRecord`s in the **OTel-GenAI vocabulary**
so the shared classifier and normalizer (`wmo.simulation.ingest.normalize`) does the mapping work:

  - A tool-producing observation (a GENERATION whose `output` carries `tool_calls`, or any
    TOOL/SPAN named like a tool) becomes a `chat` action span with
    `{"gen_ai.tool.name", "gen_ai.tool.call.arguments"}`.
  - A tool result (a TOOL/SPAN observation's `output`) becomes an `execute_tool` span with
    `{"gen_ai.tool.message": <output text>}`; a GENERATION's own `tool_calls` are paired with a
    synthesized result span from the call id (when the result is a sibling TOOL observation, the
    normalizer pairs the nearest following execute_tool span).
  - A GENERATION with no tool call becomes a plain `chat` message span (`gen_ai.completion`).
  - `level == "ERROR"` sets `status_error=True` (ObservationLevel = DEBUG|DEFAULT|WARNING|ERROR).

Observations are ordered by `startTime` (ISO-8601 -> a monotonic ordinal); when absent, list index
is used. The trace `input` is carried as `gen_ai.prompt` on the first emitted span (the task), and
the trace-level `metadata` is carried as `wmo.trace.metadata` so it round-trips.

Export the FULL trace: `GET /api/public/traces/{traceId}` (`TraceWithFullDetails`) returns
`observations` as full objects. The LIST endpoint `GET /api/public/traces` returns each trace's
`observations` as ID *strings* only, so such a page yields no steps — fetch each trace by id (or use
Langfuse's native OTLP endpoint `POST /api/public/otel/v1/traces` and the `otel-genai` source, which
is the better route for framework traces where tool calls are separate child observations).

Pull: `from_vendor` fetches traces live over the public REST API with plain httpx (no SDK): it
pages the list endpoint, then re-fetches each trace by id for full observations. See
`_pull_payloads`. File exports remain fully supported via `from_file`.
"""

from __future__ import annotations

import json
import os

import httpx
from pydantic import JsonValue

from wmo.common.core.types import JsonObject
from wmo.simulation.ingest.adapter import VendorPull, register_adapter
from wmo.simulation.ingest.base import BaseTraceAdapter
from wmo.simulation.ingest.normalize import (
    SpanEmitter,
    SpanRecord,
    as_str,
    as_text,
    first_str,
    iso_to_ordinal,
    openai_call_name_args,
    openai_tool_calls,
    payload_digest_id,
)


def _start_ordinal(observation: JsonObject, fallback: int) -> int:
    """Monotonic ordering key from the observation's `startTime` (shared helper; UTC-safe)."""
    return iso_to_ordinal(observation.get("startTime"), fallback)


def _is_error(observation: JsonObject) -> bool:
    """An observation errored iff its `level` is ERROR.

    Langfuse's `ObservationLevel` is DEBUG | DEFAULT | WARNING | ERROR. `statusMessage` is NOT an
    error signal — it is generic context Langfuse sets on any level — so we must not treat its
    presence as an error (that misclassified successful observations).
    """
    return as_str(observation.get("level")).upper() == "ERROR"


def _observation_tool_name(observation: JsonObject) -> str:
    """A tool name for a TOOL/SPAN observation (explicit field, else the observation name)."""
    for key in ("toolName", "tool_name", "name"):
        value = observation.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
_HOST_ENV = "LANGFUSE_HOST"
_DEFAULT_HOST = "https://cloud.langfuse.com"
_PAGE_SIZE = 50
# Backstop on unbounded pulls (no --limit): every listed trace is re-fetched individually, so a
# forever-pagination against a huge project would turn into thousands of requests. Mirrors the
# LangSmith adapter's _MAX_RUNS backstop.
_MAX_PAGES = 40


class LangfuseAdapter(BaseTraceAdapter):
    """Map a Langfuse trace export (observation tree) into normalized `Trace`s. No SDK."""

    name = "langfuse"

    def _pull_payloads(self, pull: VendorPull) -> list[JsonValue]:
        """Fetch traces live from the Langfuse public API (plain httpx, no SDK).

        Credentials are a public/secret key pair: `pull.api_key` as `"pk-...:sk-..."`, else
        `$LANGFUSE_PUBLIC_KEY`/`$LANGFUSE_SECRET_KEY`. Host: `pull.project` (the keys already pin
        the Langfuse project, so the field carries the host for self-hosted instances), else
        `$LANGFUSE_HOST`, else Langfuse Cloud. The LIST endpoint returns observation-id strings
        only, so each listed trace is re-fetched by id for its full observation objects.
        """
        api_key = pull.api_key
        if not api_key:
            public = os.environ.get(_PUBLIC_KEY_ENV)
            secret = os.environ.get(_SECRET_KEY_ENV)
            api_key = f"{public}:{secret}" if public and secret else None
        if not api_key or ":" not in api_key:
            raise ValueError(
                "langfuse pull needs a key pair: pass --api-key 'pk-...:sk-...' or set "
                f"${_PUBLIC_KEY_ENV} and ${_SECRET_KEY_ENV}"
            )
        public, secret = api_key.split(":", 1)
        auth = (public, secret)
        host = (pull.project or os.environ.get(_HOST_ENV) or _DEFAULT_HOST).rstrip("/")

        trace_ids: list[str] = []
        page = 1
        while True:
            params = {"page": str(page), "limit": str(_PAGE_SIZE)}
            if pull.since is not None:
                params["fromTimestamp"] = pull.since
            resp = httpx.get(f"{host}/api/public/traces", auth=auth, params=params, timeout=60.0)
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, list) or not data:
                break
            for trace in data:
                if isinstance(trace, dict) and isinstance(trace.get("id"), str):
                    trace_ids.append(trace["id"])
            if pull.limit is not None and len(trace_ids) >= pull.limit:
                trace_ids = trace_ids[: pull.limit]
                break
            if len(data) < _PAGE_SIZE or page >= _MAX_PAGES:
                break
            page += 1

        payloads: list[JsonValue] = []
        for trace_id in trace_ids:
            resp = httpx.get(f"{host}/api/public/traces/{trace_id}", auth=auth, timeout=60.0)
            resp.raise_for_status()
            payloads.append(resp.json())
        return payloads

    def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
        """Map one Langfuse trace (or a list/`{data:[...]}` page of them) to `SpanRecord`s."""
        spans: list[SpanRecord] = []
        for trace in self._traces(payload):
            spans.extend(self._spans_for_trace(trace))
        return spans

    def _traces(self, payload: JsonValue) -> list[JsonObject]:
        """Normalize a payload into a list of Langfuse trace objects.

        Accepts a single trace object, a bare list of traces, or an API list page (`{"data": [...]}`
        as returned by `GET /api/public/traces`).
        """
        if isinstance(payload, list):
            out: list[JsonObject] = []
            for item in payload:
                out.extend(self._traces(item))
            return out
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if isinstance(data, list) and "observations" not in payload:
            out = []
            for item in data:
                out.extend(self._traces(item))
            return out
        if "observations" in payload or "id" in payload:
            return [payload]
        return []

    def _spans_for_trace(self, trace: JsonObject) -> list[SpanRecord]:
        trace_id = self._trace_id(trace)
        metadata = trace.get("metadata")
        meta_obj: JsonObject = metadata if isinstance(metadata, dict) else {}
        task = trace.get("input")

        observations = trace.get("observations")
        obs_list: list[JsonObject] = (
            [o for o in observations if isinstance(o, dict)]
            if isinstance(observations, list)
            else []
        )
        # Order by startTime; ties (or absent timestamps) keep input order via the index fallback.
        indexed = list(enumerate(obs_list))
        indexed.sort(key=lambda pair: (_start_ordinal(pair[1], pair[0]), pair[0]))

        first_attributes: JsonObject = {}
        if task is not None:
            first_attributes["gen_ai.prompt"] = as_text(task)
        if meta_obj:
            first_attributes["wmo.trace.metadata"] = json.dumps(meta_obj)
        emitter = SpanEmitter(trace_id, first_attributes)

        for _, obs in indexed:
            otype = as_str(obs.get("type")).upper()
            error = _is_error(obs)
            calls = openai_tool_calls(obs.get("output")) if otype == "GENERATION" else []
            if calls:
                # A GENERATION that issued tool calls: emit a `chat` action span per call. The tool
                # RESULT comes from the sibling TOOL/SPAN observation below (the normalizer pairs
                # the nearest following execute_tool span), so we do NOT synthesize a result here.
                for tool_call in calls:
                    name, args = openai_call_name_args(tool_call)
                    emitter.emit(
                        {"gen_ai.tool.name": name, "gen_ai.tool.call.arguments": args},
                        tool=False,
                        error=error,
                    )
            elif otype == "TOOL" or (otype == "SPAN" and self._span_is_tool(obs)):
                # A TOOL/SPAN observation = a tool execution: an `execute_tool` result span. Its
                # `output` is the observation; the normalizer pairs it with the preceding action
                # span (the GENERATION's tool call), backfilling name/args from here if the action
                # lacked them. We carry name/args too so a standalone TOOL (no GENERATION) pairs.
                emitter.emit(
                    {
                        "gen_ai.tool.name": _observation_tool_name(obs),
                        "gen_ai.tool.call.arguments": as_text(obs.get("input")),
                        "gen_ai.tool.message": as_text(obs.get("output")),
                    },
                    tool=True,
                    error=error,
                )
            elif otype == "GENERATION":
                # A plain LLM turn (no tool call): a message action, no observation.
                emitter.emit(
                    {"gen_ai.completion": as_text(obs.get("output"))}, tool=False, error=error
                )
            # EVENT (and other non-actionable) observations are ignored.
        return emitter.spans

    def _span_is_tool(self, observation: JsonObject) -> bool:
        """A SPAN observation is a tool execution when it has output or a tool name/field."""
        if observation.get("output") is not None:
            return True
        for key in ("toolName", "tool_name", "input"):
            if observation.get(key) is not None:
                return True
        return False

    def _trace_id(self, trace: JsonObject) -> str:
        """A stable grouping key. Langfuse ids are not 32-hex; that's fine (it's just a key)."""
        return first_str(trace, "id") or payload_digest_id(trace)


register_adapter(LangfuseAdapter())
