"""OpenAI-compatible `/v1/chat/completions` serving over a routing policy.

The endpoint face of the pivot: a customer points their OpenAI client at wmh, `model` in the
request names an ENDPOINT (world model + policy), and the learned inference policy picks which
pool model actually serves each call. Responses stay OpenAI-pure and name only the endpoint;
the mechanism (routed model, cluster, reason) goes to the request log and the
`x-wmh-routed-model` debug header, never customer-facing copy.

Conversation affinity: provider prompt caches are per-model, so switching mid-conversation
forfeits warm cache reads. The runtime fingerprints each finished exchange (full transcript
including the assistant reply) and, when the next request arrives with that transcript as its
prefix, `select_model` sees the incumbent and sticks to it by default.

Compression stage (D-COMPRESS): when the policy carries a compression config, the pipeline is
request -> [compress] -> [route] -> provider call. Only user-message content is compressed;
system prompts and the model's own prior replies pass through verbatim. The affinity state
decides segment boundaries: an incumbent conversation's compressed prefix is stored alongside
its fingerprint and REUSED, never recompressed, so the provider-visible prefix stays
byte-identical across turns (the prompt cache survives by construction). Routing embeds the
compressed text (the router sees what the model sees) while stickiness keys on the raw
transcript the client resends. Compression fields go to the request log only, never response
bodies or headers.

Request log: one JSONL row per call with the D-SERVING-LOG fields (id, ts, endpoint, routed
model, cluster, tokens incl. cached, cache-adjusted cost, latency, ttfb, status, reason).
Provider cache CONTROLS (breakpoint placement, TTL) are not exposed yet; they land with the
cache-aware routing model.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, JsonValue, field_validator
from starlette.background import BackgroundTask

from wmh.optimize.compression import (
    CompressionStats,
    Compressor,
    estimate_tokens,
    get_compressor,
)
from wmh.optimize.policy import Embedder, RoutingDecision, RoutingPolicy, select_model
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Message,
    Provider,
    StreamingProvider,
    TokenUsage,
)
from wmh.providers.pool import PoolEntry, pool_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

# Finished-exchange fingerprints remembered per endpoint for conversation affinity. Bounded so
# a long-running server cannot grow without limit; least-recently-used conversations re-route.
_AFFINITY_CAPACITY = 4096

ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One chat turn, normalized from the OpenAI request shapes clients actually send.

    `developer` (OpenAI's system replacement on gpt-5-class models) maps to `system`, and
    multi-part text content (`[{"type": "text", "text": ...}]`, emitted by LangChain and the
    Vercel AI SDK) is joined; a non-text part is a hard error, not a silent drop.
    """

    role: ChatRole
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def _developer_is_system(cls, value: object) -> object:
        return "system" if value == "developer" else value

    @field_validator("content", mode="before")
    @classmethod
    def _join_text_parts(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        parts: list[str] = []
        for part in value:
            if not (isinstance(part, dict) and part.get("type") == "text"):
                raise ValueError(
                    "only text content parts are supported; images and other modalities "
                    "are not available on this endpoint yet"
                )
            parts.append(str(part.get("text", "")))
        return "".join(parts)


class StreamOptions(BaseModel):
    """OpenAI `stream_options`: opt-in for the trailing usage chunk on streamed responses."""

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """The OpenAI request subset the endpoint serves (text chat; tools are future work).

    Unsupported FUNCTIONAL parameters are declared here so they can be rejected explicitly
    (see `unsupported_features`); genuinely unknown extra fields are ignored like OpenAI does.
    """

    model: str  # the ENDPOINT name
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    # Declared only to be REJECTED with a clear 400 (silently ignoring tools would make a
    # tool-calling client read prose where it expects tool_calls).
    tools: list[JsonValue] | None = None
    tool_choice: JsonValue | None = None
    response_format: JsonValue | None = None
    n: int = 1
    logprobs: bool | None = None

    def output_budget(self) -> int:
        return self.max_completion_tokens or self.max_tokens or DEFAULT_MAX_TOKENS

    def wants_stream_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage

    def unsupported_features(self) -> str:
        """Name the requested-but-unsupported params, '' when the request is serveable."""
        used = [
            name
            for name, value in (
                ("tools", self.tools),
                ("tool_choice", self.tool_choice),
                ("response_format", self.response_format),
                ("logprobs", self.logprobs),
            )
            if value
        ]
        if self.n != 1:
            used.append("n != 1")
        return ", ".join(used)


def _error_response(status_code: int, message: str, *, err_type: str, code: str) -> Response:
    """An OpenAI-shaped error body: real OpenAI clients read `body["error"]["message"]`.

    FastAPI's default `{"detail": ...}` shape parses as a generic APIStatusError in the openai
    SDK but loses the message; this shape surfaces it exactly like the upstream API does.
    """
    return Response(
        content=json.dumps(
            {"error": {"message": message, "type": err_type, "param": None, "code": code}},
            ensure_ascii=False,
        ),
        status_code=status_code,
        media_type="application/json",
    )


def install_openai_error_shapes(app: FastAPI) -> None:
    """Convert request-validation failures to OpenAI's 400 + error-body shape.

    Without this a malformed request gets FastAPI's 422 `{"detail": [...]}`, which OpenAI
    clients surface as an empty error. App-level because exception handlers cannot attach to
    a router; every app that mounts `create_chat_router` should call this.
    """

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> Response:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        message = f"invalid request: {location}: {first.get('msg', 'validation failed')}"
        return _error_response(
            400, message, err_type="invalid_request_error", code="invalid_request"
        )


class RequestLogRecord(BaseModel):
    """One metered call, as the request log persists it (D-METERING / D-SERVING-LOG shape).

    This is the wmh half of the metering contract: the platform wrap adds tenancy
    (org_id, api_key_id) when it persists these rows. `cached_tokens` mirrors
    `TokenUsage.cached_input_tokens` (cache-read prompt tokens, a subset of `input_tokens`);
    `cost_usd` is cache-adjusted via `PoolEntry.cost_usd`. `router_cost_usd` is the policy's
    OWN inference cost per call, 0 for the free hashing policy and real once a trained router
    serves.
    """

    id: str
    ts: str
    endpoint: str
    leg: Literal["serving", "optimization", "eval", "overhead"] = "serving"
    model: str  # routed pool entry name
    provider_model: str  # the provider runtime id behind it
    cluster_id: int | None = None
    cluster_label: str = ""
    routing_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # cache-read prompt tokens (subset of input_tokens)
    cost_usd: float = 0.0  # effective cost: cached tokens billed at the cache-read rate
    router_cost_usd: float = 0.0  # the routing decision's own inference cost, passed through
    # D-COMPRESS fields: stored and OPAQUE like the routing fields above (log only, never in
    # response bodies or headers). 0/"" defaults = the request served uncompressed. Token
    # counts are the compressor's deterministic proxy totals (see wmh.optimize.compression);
    # billable truth stays in input_tokens/cost_usd from the provider-reported usage.
    tokens_in_raw: int = 0
    tokens_in_compressed: int = 0
    compressor_id: str = ""
    compressor_version: str = ""
    aggressiveness: float = 0.0
    latency_ms: float = 0.0
    ttfb_ms: float | None = None
    status: Literal["ok", "error"] = "ok"
    error_message: str | None = None


class RequestLog:
    """Append-only JSONL request log plus a bounded in-memory tail."""

    def __init__(self, path: Path | None, *, keep: int = 200) -> None:
        self._path = path
        self._recent: deque[RequestLogRecord] = deque(maxlen=keep)
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RequestLogRecord) -> None:
        with self._lock:
            self._recent.append(record)
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json() + "\n")

    def recent(self) -> list[RequestLogRecord]:
        with self._lock:
            return list(self._recent)


class EndpointRuntime:
    """One served endpoint: its policy, its providers, its affinity memory, its log."""

    def __init__(
        self,
        name: str,
        policy: RoutingPolicy,
        *,
        provider_factory: Callable[[PoolEntry], Provider] = pool_provider,
        log: RequestLog,
    ) -> None:
        self.name = name
        self.policy = policy
        self.log = log
        self._provider_factory = provider_factory
        self._providers: dict[str, Provider] = {}
        self._policy_embedder: Embedder | None = None
        self._affinity: OrderedDict[str, str] = OrderedDict()
        # Compressed provider-visible transcripts, keyed by the SAME raw-transcript fingerprint
        # as _affinity: the affinity state decides compression segment boundaries. Only
        # populated when the policy carries a compression config.
        self._compressed: OrderedDict[str, list[ChatMessage]] = OrderedDict()
        # Resolved once at mount (policy validation already proved the id is known), mirroring
        # the embedder-once pattern: no per-request registry lookups.
        self._compressor: Compressor | None = (
            get_compressor(policy.compression.compressor_id)
            if policy.compression is not None
            else None
        )
        self._lock = threading.Lock()

    def decide(
        self, messages: list[ChatMessage], *, route_text: str | None = None
    ) -> RoutingDecision:
        """Route the request. Stickiness keys on the RAW transcript the client resends;
        `route_text` (the compressed routable text when compression is on) is what gets
        embedded, so the router scores exactly what the model will see."""
        incumbent = None
        if len(messages) > 1:
            with self._lock:
                incumbent = self._affinity.get(_fingerprint(messages[:-1]))
        text = route_text if route_text is not None else _routable_text(messages)
        return select_model(self.policy, text, incumbent=incumbent, embedder=self._embedder())

    def compress(
        self, messages: list[ChatMessage]
    ) -> tuple[list[ChatMessage], CompressionStats | None]:
        """The [compress] stage: raw request messages -> provider-visible messages + stats.

        Cache safety by construction: when the conversation's previous exchange is known
        (affinity hit on the raw prefix), the stored compressed prefix is returned verbatim
        and only the new final user message passes through the compressor. On a miss (new
        conversation, or affinity evicted) every user message is compressed fresh; per-segment
        determinism makes that reproduce the same bytes, so the provider-visible prefix stays
        append-only either way. Returns the input list untouched when compression is off.
        """
        config = self.policy.compression
        if config is None or self._compressor is None:
            return messages, None
        started = time.monotonic()
        cost_usd = 0.0
        prefix: list[ChatMessage] | None = None
        if len(messages) > 1:
            with self._lock:
                prefix = self._compressed.get(_fingerprint(messages[:-1]))
        if prefix is not None:
            last = messages[-1]
            if last.role == "user":
                result = self._compressor.compress([last.content], config)
                cost_usd = result.cost_usd
                last = ChatMessage(role="user", content=result.segments[0])
            compressed = [*prefix, last]
        else:
            user_segments = [m.content for m in messages if m.role == "user"]
            result = self._compressor.compress(user_segments, config)
            cost_usd = result.cost_usd
            replacements = iter(result.segments)
            compressed = [
                ChatMessage(role="user", content=next(replacements)) if m.role == "user" else m
                for m in messages
            ]
        stats = CompressionStats(
            compressor_id=self._compressor.id,
            compressor_version=self._compressor.version,
            aggressiveness=config.aggressiveness,
            tokens_in_raw=sum(estimate_tokens(m.content) for m in messages),
            tokens_in_compressed=sum(estimate_tokens(m.content) for m in compressed),
            latency_s=time.monotonic() - started,
            cost_usd=cost_usd,
        )
        return compressed, stats

    def _embedder(self) -> Embedder | None:
        """Build the policy's embedder once per runtime, not once per request.

        Static policies never embed, so their (possibly unbuildable-in-this-environment)
        embedder spec must not be constructed at all: a static route has to keep serving
        even when an azure spec's credentials are absent here.

        An azure spec otherwise constructs a fresh SDK client (TLS handshake and all) inside
        every request's latency budget. Double-checked locking mirrors provider_for.
        """
        if self.policy.kind == "static" or self.policy.embedder is None:
            return None
        with self._lock:
            embedder = self._policy_embedder
        if embedder is None:
            embedder = self.policy.embedder.build()
            with self._lock:
                if self._policy_embedder is None:
                    self._policy_embedder = embedder
                embedder = self._policy_embedder
        return embedder

    def remember(
        self,
        messages: list[ChatMessage],
        assistant_text: str,
        model: str,
        *,
        compressed: list[ChatMessage] | None = None,
    ) -> None:
        """Record the finished exchange so the conversation's next request finds its incumbent.

        `messages` must be the RAW request messages (the client resends that transcript, so
        the fingerprint must match it). `compressed` is the provider-visible transcript when
        compression ran; stored under the same key so the next turn reuses the exact bytes
        the provider's prompt cache was written with.
        """
        transcript = [*messages, ChatMessage(role="assistant", content=assistant_text)]
        key = _fingerprint(transcript)
        with self._lock:
            self._affinity[key] = model
            self._affinity.move_to_end(key)
            while len(self._affinity) > _AFFINITY_CAPACITY:
                self._affinity.popitem(last=False)
            if compressed is not None:
                reply = ChatMessage(role="assistant", content=assistant_text)
                self._compressed[key] = [*compressed, reply]
                self._compressed.move_to_end(key)
                while len(self._compressed) > _AFFINITY_CAPACITY:
                    self._compressed.popitem(last=False)

    def provider_for(self, pool_name: str) -> tuple[PoolEntry, Provider]:
        entry = next(e for e in self.policy.pool if e.name == pool_name)
        with self._lock:
            provider = self._providers.get(pool_name)
        if provider is None:
            # Construct OUTSIDE the lock: a slow client build (TLS handshake, credential
            # resolution) must not head-of-line-block every other request's affinity lookup.
            # A racing duplicate build is harmless; first insert wins.
            provider = self._provider_factory(entry)
            with self._lock:
                provider = self._providers.setdefault(pool_name, provider)
        return entry, provider


def _fingerprint(messages: list[ChatMessage]) -> str:
    canonical = json.dumps([(m.role, m.content) for m in messages], ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _routable_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content


def _split_for_provider(messages: list[ChatMessage]) -> tuple[str, list[Message]]:
    """Fold system turns into the provider's system string; keep user/assistant order."""
    system_parts = [m.content for m in messages if m.role == "system"]
    turns = [Message(role=m.role, content=m.content) for m in messages if m.role != "system"]
    return "\n\n".join(system_parts), turns


def _usage_dict(usage: TokenUsage) -> dict[str, object]:
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        # OpenAI's cached-prompt reporting shape; 0 when the upstream provider reported none.
        "prompt_tokens_details": {"cached_tokens": usage.cached_input_tokens},
    }


def _sse(payload: object) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_payload(
    completion_id: str,
    created: int,
    endpoint: str,
    delta: dict[str, str],
    *,
    finish_reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": endpoint,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def create_chat_router(endpoints: Mapping[str, EndpointRuntime]) -> APIRouter:
    """Mount `/v1/models` + `/v1/chat/completions` over the given endpoints."""
    router = APIRouter()

    def _endpoint_or_none(name: str) -> EndpointRuntime | None:
        return endpoints.get(name)

    @router.get("/v1/models")
    def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "wmh"}
                for name in sorted(endpoints)
            ],
        }

    @router.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> Response:
        runtime = _endpoint_or_none(request.model)
        if runtime is None:
            available = ", ".join(sorted(endpoints)) or "(none)"
            return _error_response(
                404,
                f"no endpoint {request.model!r}; have: {available}",
                err_type="invalid_request_error",
                code="model_not_found",
            )
        unsupported = request.unsupported_features()
        if unsupported:
            # Silently dropping tools/n/response_format would make a tool-calling client read
            # plain text where it expects tool_calls: a compatibility gap disguised as a
            # model-quality problem. Reject loudly instead.
            return _error_response(
                400,
                f"this endpoint does not support {unsupported} yet",
                err_type="invalid_request_error",
                code="unsupported_parameter",
            )
        if not any(m.role != "system" for m in request.messages):
            return _error_response(
                400,
                "at least one user or assistant message is required",
                err_type="invalid_request_error",
                code="invalid_messages",
            )
        try:
            # request -> [compress] -> [route]: the router embeds the compressed text below.
            provider_messages, compression = runtime.compress(request.messages)
            decision = runtime.decide(
                request.messages, route_text=_routable_text(provider_messages)
            )
            entry, provider = runtime.provider_for(decision.model)
        except Exception as exc:  # noqa: BLE001 - reported as an OpenAI-shaped 502 + log row
            # The likeliest production failure: an unset api_key_env or a failing embed call.
            # Without this guard it surfaces as a bare text/plain 500 with no log row.
            logger.error("routing/provider setup for %s failed: %s", runtime.name, exc)
            runtime.log.append(
                RequestLogRecord(
                    id=f"chatcmpl-{uuid.uuid4().hex}",
                    ts=datetime.now(tz=UTC).isoformat(),
                    endpoint=runtime.name,
                    model="",
                    provider_model="",
                    routing_reason="error-before-routing",
                    status="error",
                    error_message=str(exc),
                )
            )
            return _error_response(
                502,
                f"endpoint setup failed ({type(exc).__name__})",
                err_type="api_error",
                code="routing_error",
            )
        system, turns = _split_for_provider(provider_messages)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        started = time.monotonic()

        def _record(
            usage: TokenUsage,
            *,
            ttfb_ms: float | None,
            status: Literal["ok", "error"] = "ok",
            error_message: str | None = None,
        ) -> None:
            runtime.log.append(
                RequestLogRecord(
                    id=completion_id,
                    ts=datetime.now(tz=UTC).isoformat(),
                    endpoint=runtime.name,
                    model=entry.name,
                    provider_model=entry.model,
                    cluster_id=decision.cluster_id,
                    cluster_label=decision.cluster_label,
                    routing_reason=decision.reason,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_tokens=usage.cached_input_tokens,
                    cost_usd=entry.cost_usd(usage),
                    tokens_in_raw=compression.tokens_in_raw if compression else 0,
                    tokens_in_compressed=compression.tokens_in_compressed if compression else 0,
                    compressor_id=compression.compressor_id if compression else "",
                    compressor_version=compression.compressor_version if compression else "",
                    aggressiveness=compression.aggressiveness if compression else 0.0,
                    latency_ms=(time.monotonic() - started) * 1000,
                    ttfb_ms=ttfb_ms,
                    status=status,
                    error_message=error_message,
                )
            )

        headers = {"x-wmh-routed-model": decision.model}

        if not request.stream:
            try:
                completion = provider.complete(
                    system,
                    turns,
                    temperature=request.temperature if request.temperature is not None else 1.0,
                    max_tokens=request.output_budget(),
                )
            except Exception as exc:  # noqa: BLE001 - reported as an OpenAI-shaped 502
                # Full detail goes to the request log and server log only: upstream exception
                # text can carry internal endpoints/stack info (CodeQL: information exposure).
                _record(TokenUsage(), ttfb_ms=None, status="error", error_message=str(exc))
                logger.error("upstream call for %s failed: %s", entry.name, exc)
                return _error_response(
                    502,
                    f"upstream model call failed ({type(exc).__name__})",
                    err_type="api_error",
                    code="upstream_error",
                )
            runtime.remember(
                request.messages,
                completion.text,
                decision.model,
                compressed=provider_messages if compression else None,
            )
            _record(completion.usage, ttfb_ms=None)
            return Response(
                content=json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": runtime.name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": completion.text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": _usage_dict(completion.usage),
                    },
                    ensure_ascii=False,
                ),
                media_type="application/json",
                headers=headers,
            )

        if not isinstance(provider, StreamingProvider):
            # A real endpoint-level failure: keep one-record-per-request intact.
            _record(
                TokenUsage(),
                ttfb_ms=None,
                status="error",
                error_message=f"pool model '{entry.name}' has no native streaming backend",
            )
            return _error_response(
                501,
                f"pool model '{entry.name}' has no native streaming backend",
                err_type="api_error",
                code="streaming_unsupported",
            )
        try:
            upstream = provider.stream(
                system,
                turns,
                temperature=request.temperature if request.temperature is not None else 1.0,
                max_tokens=request.output_budget(),
            )
            first = next(upstream, None)
        except Exception as exc:  # noqa: BLE001 - reported as an OpenAI-shaped 502
            _record(TokenUsage(), ttfb_ms=None, status="error", error_message=str(exc))
            logger.error("stream start for %s failed: %s", entry.name, exc)
            return _error_response(
                502,
                f"upstream model call failed ({type(exc).__name__})",
                err_type="api_error",
                code="upstream_error",
            )
        ttfb_ms = (time.monotonic() - started) * 1000

        # Shared with _finalize: a disconnecting client makes starlette CANCEL the stream
        # without ever closing the sync generator, so cleanup inside the generator (finally,
        # GeneratorExit) never runs on that path. The BackgroundTask below is the only hook
        # that fires on both normal completion and disconnect.
        stream_state = {"recorded": False}
        partial_usage = TokenUsage()

        def _events() -> Iterator[str]:
            yield _sse(
                _chunk_payload(
                    completion_id, created, runtime.name, {"role": "assistant", "content": ""}
                )
            )
            parts: list[str] = []
            usage = partial_usage
            try:
                chunk = first
                while chunk is not None:
                    if chunk.done:
                        if chunk.usage is not None:
                            usage = chunk.usage
                    elif chunk.delta:
                        parts.append(chunk.delta)
                        yield _sse(
                            _chunk_payload(
                                completion_id,
                                created,
                                runtime.name,
                                {"content": chunk.delta},
                            )
                        )
                    chunk = next(upstream, None)
            except Exception as exc:  # noqa: BLE001 - response already started; log and end
                stream_state["recorded"] = True
                _record(usage, ttfb_ms=ttfb_ms, status="error", error_message=str(exc))
                logger.error("stream from %s failed mid-response: %s", entry.name, exc)
                yield "data: [DONE]\n\n"
                return
            yield _sse(
                _chunk_payload(completion_id, created, runtime.name, {}, finish_reason="stop")
            )
            if request.wants_stream_usage():
                # OpenAI's include_usage framing: one extra chunk with NO choices and the
                # final usage, after the finish_reason chunk and before [DONE].
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": runtime.name,
                        "choices": [],
                        "usage": _usage_dict(usage),
                    }
                )
            yield "data: [DONE]\n\n"
            runtime.remember(
                request.messages,
                "".join(parts),
                decision.model,
                compressed=provider_messages if compression else None,
            )
            stream_state["recorded"] = True
            _record(usage, ttfb_ms=ttfb_ms)

        def _finalize() -> None:
            """Runs after the response ends, HOWEVER it ends (starlette BackgroundTask).

            An abandoned stream still consumed upstream tokens: leaving it unrecorded would
            be silent usage loss (D-METERING). Also closes the upstream iterator, which the
            cancelled threadpool iteration otherwise leaks.
            """
            close = getattr(upstream, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            if not stream_state["recorded"]:
                stream_state["recorded"] = True
                _record(
                    partial_usage,
                    ttfb_ms=ttfb_ms,
                    status="error",
                    error_message="client disconnected mid-stream",
                )

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache"},
            background=BackgroundTask(_finalize),
        )

    return router
