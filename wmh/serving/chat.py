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
model, cluster, tokens, cost, latency, ttfb, status, reason). Cached-token counts and
provider cache controls are not captured yet; they land with the cache-aware cost model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from wmh.optimize.policy import RoutingDecision, RoutingPolicy, select_model
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
    role: ChatRole
    content: str


class ChatCompletionRequest(BaseModel):
    """The OpenAI request subset the endpoint serves (text chat; tools are future work)."""

    model: str  # the ENDPOINT name
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None

    def output_budget(self) -> int:
        return self.max_completion_tokens or self.max_tokens or DEFAULT_MAX_TOKENS


class RequestLogRecord(BaseModel):
    """One metered call, as the request log persists it (D-METERING / D-SERVING-LOG shape).

    This is the wmh half of the metering contract: the platform wrap adds tenancy
    (org_id, api_key_id) when it persists these rows. `cached_tokens` is carried but always 0
    until providers surface cache-read counts; `router_cost_usd` is the policy's OWN inference
    cost per call, 0 for the free hashing policy and real once a trained router serves.
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
    cached_tokens: int = 0  # cache-read tokens; 0 until providers surface them
    cost_usd: float = 0.0  # cache-adjusted once cached_tokens are real; list-priced until then
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
        self._affinity: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def decide(self, messages: list[ChatMessage]) -> RoutingDecision:
        incumbent = None
        if len(messages) > 1:
            with self._lock:
                incumbent = self._affinity.get(_fingerprint(messages[:-1]))
        text = _routable_text(messages)
        return select_model(self.policy, text, incumbent=incumbent)

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
                provider = self._provider_factory(entry)
                self._providers[pool_name] = provider
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


def _usage_dict(usage: TokenUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
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
    usage: TokenUsage | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": endpoint,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = _usage_dict(usage)
    return payload


def create_chat_router(endpoints: Mapping[str, EndpointRuntime]) -> APIRouter:
    """Mount `/v1/models` + `/v1/chat/completions` over the given endpoints."""
    router = APIRouter()

    def _endpoint_or_404(name: str) -> EndpointRuntime:
        runtime = endpoints.get(name)
        if runtime is None:
            available = ", ".join(sorted(endpoints)) or "(none)"
            raise HTTPException(status_code=404, detail=f"no endpoint {name!r}; have: {available}")
        return runtime

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
        runtime = _endpoint_or_404(request.model)
        decision = runtime.decide(request.messages)
        entry, provider = runtime.provider_for(decision.model)
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
                    temperature=request.temperature if request.temperature is not None else 0.7,
                    max_tokens=request.output_budget(),
                )
            except Exception as exc:
                _record(TokenUsage(), ttfb_ms=None, status="error", error_message=str(exc))
                raise HTTPException(
                    status_code=502, detail=f"upstream model call failed: {exc}"
                ) from exc
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
            raise HTTPException(
                status_code=501,
                detail=f"pool model '{entry.name}' has no native streaming backend",
            )
        try:
            upstream = provider.stream(
                system,
                turns,
                temperature=request.temperature if request.temperature is not None else 0.7,
                max_tokens=request.output_budget(),
            )
            first = next(upstream, None)
        except Exception as exc:
            _record(TokenUsage(), ttfb_ms=None, status="error", error_message=str(exc))
            raise HTTPException(
                status_code=502, detail=f"upstream model call failed: {exc}"
            ) from exc
        ttfb_ms = (time.monotonic() - started) * 1000

        def _events() -> Iterator[str]:
            yield _sse(
                _chunk_payload(
                    completion_id, created, runtime.name, {"role": "assistant", "content": ""}
                )
            )
            parts: list[str] = []
            usage = TokenUsage()
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
                _record(usage, ttfb_ms=ttfb_ms, status="error", error_message=str(exc))
                logger.error("stream from %s failed mid-response: %s", entry.name, exc)
                yield "data: [DONE]\n\n"
                return
            yield _sse(
                _chunk_payload(
                    completion_id,
                    created,
                    runtime.name,
                    {},
                    finish_reason="stop",
                    usage=usage,
                )
            )
            yield "data: [DONE]\n\n"
            runtime.remember(request.messages, "".join(parts), decision.model)
            _record(usage, ttfb_ms=ttfb_ms)

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache"},
        )

    return router
