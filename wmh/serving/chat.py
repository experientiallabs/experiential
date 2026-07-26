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

Request log: one JSONL row per call with the D-SERVING-LOG fields (id, ts, endpoint, routed
model, cluster, tokens incl. cached, cache-adjusted cost, latency, ttfb, status, reason).
Provider cache CONTROLS (breakpoint placement, TTL) are not exposed yet; they land with the
cache-aware routing model.

The endpoint's one operator control lives here too: `GET`/`PUT /v1/endpoints/{name}/config`
reads and moves the cost/quality dial (`wmh.optimize.knn.apply_cost_quality`) on the live
runtime, no restart and no refit. It is deliberately outside the OpenAI surface: a customer's
OpenAI client sees exactly the routes it saw before.
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

from wmh.optimize.knn import (
    COST_QUALITY_ANCHORS,
    CostQualityAnchor,
    CostQualityKnobs,
    apply_cost_quality,
    cost_quality_knobs,
    cost_quality_named_point,
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
from wmh.serving.endpoint_config import EndpointConfig

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
    """One served endpoint: its policy, its providers, its affinity memory, its log.

    `cost_quality` sets the endpoint's cost/quality dial at mount time (see
    `wmh.optimize.knn.apply_cost_quality`); None serves the policy exactly as fitted, so
    mounting never silently re-tunes an artifact. `config_path` is the `endpoint.toml` a live
    dial change is persisted to, so the setting survives a restart; None keeps changes in memory
    (injected-policy tests, and any caller that owns persistence itself).
    """

    def __init__(
        self,
        name: str,
        policy: RoutingPolicy,
        *,
        provider_factory: Callable[[PoolEntry], Provider] = pool_provider,
        log: RequestLog,
        cost_quality: float | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.name = name
        self.policy = policy
        self.log = log
        self._base_policy = policy
        self._config_path = config_path
        self._provider_factory = provider_factory
        self._providers: dict[str, Provider] = {}
        self._policy_embedder: Embedder | None = None
        self._affinity: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        if cost_quality is not None:
            self._swap_policy(cost_quality)

    @property
    def cost_quality(self) -> float | None:
        """The dial the served policy is currently on (None: served as fitted)."""
        return self.policy.cost_quality

    def set_cost_quality(self, cost_quality: float) -> None:
        """Move the dial on the live endpoint, and persist it when there is a file to persist to.

        In-flight requests keep the policy they started on: the swap replaces the whole policy
        object, so no request can ever read half of one dial position and half of another. The
        pool, baseline, and evidence bank are identical across positions, so a conversation that
        spans a swap is still served by a model this endpoint knows.
        """
        self._swap_policy(cost_quality)
        if self._config_path is not None:
            EndpointConfig(cost_quality=cost_quality).save(self._config_path)

    def _swap_policy(self, cost_quality: float) -> None:
        adjusted = apply_cost_quality(self._base_policy, cost_quality)
        with self._lock:
            self.policy = adjusted

    def decide(self, messages: list[ChatMessage]) -> RoutingDecision:
        incumbent = None
        if len(messages) > 1:
            with self._lock:
                incumbent = self._affinity.get(_fingerprint(messages[:-1]))
        text = _routable_text(messages)
        return select_model(self.policy, text, incumbent=incumbent, embedder=self._embedder())

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

    def remember(self, messages: list[ChatMessage], assistant_text: str, model: str) -> None:
        """Record the finished exchange so the conversation's next request finds its incumbent."""
        transcript = [*messages, ChatMessage(role="assistant", content=assistant_text)]
        key = _fingerprint(transcript)
        with self._lock:
            self._affinity[key] = model
            self._affinity.move_to_end(key)
            while len(self._affinity) > _AFFINITY_CAPACITY:
                self._affinity.popitem(last=False)

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


class EndpointConfigResponse(BaseModel):
    """The endpoint's cost/quality dial, everything needed to render it, and where it stands.

    `cost_quality` is null when the endpoint serves its policy exactly as fitted (nobody has set
    the dial); `named_point` is "as-fitted" then. `knobs` is what the served policy is actually
    running, so a client can see the effect of the dial and not just its label. `anchors` are the
    MEASURED points behind the mapping (routerbench-ours9, 5 held-out splits, quality and cost
    both against the best single pool model), which is what a slider should show a human instead
    of an unlabeled 0-to-1 track. `dialable` is false for policy kinds with no dial (static and
    rank endpoints), and then the dial fields are null and PUT returns 409.
    """

    endpoint: str
    dialable: bool
    cost_quality: float | None
    named_point: str
    knobs: CostQualityKnobs | None
    anchors: list[CostQualityAnchor]


class EndpointConfigUpdate(BaseModel):
    """A live dial change: the one field the platform's slider sends."""

    cost_quality: float = Field(ge=0.0, le=1.0)


def _config_response(runtime: EndpointRuntime) -> EndpointConfigResponse:
    dialable = runtime.policy.kind == "knn"
    dial = runtime.cost_quality if dialable else None
    return EndpointConfigResponse(
        endpoint=runtime.name,
        dialable=dialable,
        cost_quality=dial,
        named_point=cost_quality_named_point(dial) if dial is not None else "as-fitted",
        knobs=(
            CostQualityKnobs(
                knn_z=runtime.policy.knn_z,
                # The served floor is a similarity threshold; the dial's floor_q is the quantile
                # it came from, and only the mapping knows which quantile that was.
                floor_q=cost_quality_knobs(dial).floor_q if dial is not None else 0.0,
                pick_lam=runtime.policy.pick_lam,
                guard_mode=runtime.policy.guard_mode,
            )
            if dialable
            else None
        ),
        anchors=list(COST_QUALITY_ANCHORS),
    )


def create_chat_router(endpoints: Mapping[str, EndpointRuntime]) -> APIRouter:
    """Mount `/v1/models`, `/v1/chat/completions`, and the per-endpoint dial over `endpoints`."""
    router = APIRouter()

    def _endpoint_or_none(name: str) -> EndpointRuntime | None:
        return endpoints.get(name)

    def _endpoint_or_error(name: str) -> EndpointRuntime | Response:
        runtime = endpoints.get(name)
        if runtime is not None:
            return runtime
        available = ", ".join(sorted(endpoints)) or "(none)"
        return _error_response(
            404,
            f"no endpoint {name!r}; have: {available}",
            err_type="invalid_request_error",
            code="model_not_found",
        )

    @router.get("/v1/endpoints/{name}/config", response_model=EndpointConfigResponse)
    def get_endpoint_config(name: str) -> EndpointConfigResponse | Response:
        found = _endpoint_or_error(name)
        if isinstance(found, Response):
            return found
        return _config_response(found)

    @router.put("/v1/endpoints/{name}/config", response_model=EndpointConfigResponse)
    def put_endpoint_config(
        name: str, update: EndpointConfigUpdate
    ) -> EndpointConfigResponse | Response:
        found = _endpoint_or_error(name)
        if isinstance(found, Response):
            return found
        try:
            found.set_cost_quality(update.cost_quality)
        except ValueError as exc:
            # A dial the policy cannot honor: a non-knn kind, or a savings position on a policy
            # fitted without cost evidence. Both are configuration, not transport, so say which.
            return _error_response(
                409, str(exc), err_type="invalid_request_error", code="dial_unavailable"
            )
        return _config_response(found)

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
            decision = runtime.decide(request.messages)
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
        system, turns = _split_for_provider(request.messages)
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
            runtime.remember(request.messages, completion.text, decision.model)
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
            runtime.remember(request.messages, "".join(parts), decision.model)
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
