"""Local OpenAI-compatible endpoint serving per-episode Tinker providers.

The tau2 rollout source (`wmo.distill.tau2`) runs Sierra's real tau2-bench
harness as subprocesses, and tau2's agent reaches its LLM through litellm
against an OpenAI-compatible `api_base`. This proxy is that base: it binds a
loopback port, and every registered episode gets a MODEL ALIAS that routes to
that episode's own `TinkerChatProvider`.

Why per-episode providers rather than one shared one: the provider owns the
episode's token-exactness. Its `TokenRecorder` is single-episode by contract
(`call_index` restarts at 0 per sink), and its prompt state splices the next
prompt as `prompt(N) + sampled(N) + suffix`, so consecutive turns must come
from ONE conversation. The alias is what carries episode identity across the
HTTP boundary; concurrent episodes hit different providers and never share
state. Within an episode tau2's orchestrator is sequential, which is the
recorder's threading contract.

The request/response mapping is deliberately thin: tau2 sends OpenAI
chat-completion JSON, `ChatRequest`/`ChatResponse` (`llm_waterfall.types`)
ARE that wire shape, and `ToolCallingProvider.complete_chat` consumes and
produces them directly. Tool schemas render into the prompt and sampled tool
calls parse back through the provider's own renderer, the same path every
other structured consumer uses (see `wmo.serving.chat` for the full serving
surface; this proxy serves exactly one caller on loopback and meters nothing,
which is why it does not mount that router).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from llm_waterfall.types import ChatRequest, ChatResponse, ChatTool
from pydantic import JsonValue

from wmo.core.types import JsonObject
from wmo.providers.base import ToolCallingProvider

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT_S = 15.0
"""How long `start()` waits for uvicorn to bind before giving up."""


def _losslessly_aligned(value: JsonValue, expected: str) -> JsonValue:
    """`value` converted to the schema's scalar type when that loses nothing.

    Only conversions whose text round-trips exactly are applied; anything else
    (including a value that already matches, or a type this cannot align
    losslessly) comes back unchanged, so a genuinely malformed argument still
    reaches the environment and fails there, which is real behavioral signal.
    """
    if expected == "string" and isinstance(value, int | float) and not isinstance(value, bool):
        # json.dumps, not str(): it renders floats exactly as JSON parsed them,
        # so int(19122) -> "19122" and 3.5 -> "3.5" with no repr drift.
        return json.dumps(value)
    if expected == "integer" and isinstance(value, str):
        stripped = value.strip()
        try:
            converted = int(stripped)
        except ValueError:
            return value
        return converted if str(converted) == stripped else value
    if expected == "number" and isinstance(value, str):
        stripped = value.strip()
        try:
            converted = float(stripped)
        except ValueError:
            return value
        return converted if json.dumps(converted) == stripped else value
    if expected == "boolean" and isinstance(value, str) and value.strip() in ("true", "false"):
        return value.strip() == "true"
    return value


def realign_tool_argument_types(response: ChatResponse, tools: list[ChatTool] | None) -> None:
    """Re-align each sampled tool call's argument types with its declared schema.

    The cookbook's Qwen3.5 XML tool parser JSON-decodes every parameter value
    with no schema in hand, so a string-typed id that happens to be numeric
    comes out an integer: `<parameter=zip>19122</parameter>` parses to
    `{"zip": 19122}` where the tool schema says `"zip": {"type": "string"}`.
    tau2's DB keys are strings, so every such lookup fails "not found". Measured
    before this existed: retail (all-numeric product/item ids) sat at a uniform
    0.00 while airline (alphanumeric ids) scored normally, for the TEACHER as
    well as the student.

    The schema is authoritative and the proxy is the one place that holds both
    the parsed call and the schema, so alignment happens here, in place, on the
    top-level properties (tau2's tools are flat). Conversions are strictly
    lossless (`_losslessly_aligned`); a call whose tool is unknown or whose
    arguments do not parse as a JSON object is left untouched.
    """
    if not tools:
        return
    properties_by_tool: dict[str, JsonObject] = {}
    for tool in tools:
        parameters = tool.function.parameters
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            properties_by_tool[tool.function.name] = properties
    for choice in response.choices:
        for call in choice.message.tool_calls or []:
            properties = properties_by_tool.get(call.function.name)
            if properties is None:
                continue
            try:
                arguments = json.loads(call.function.arguments)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(arguments, dict):
                continue
            changed = False
            for name, value in arguments.items():
                declared = properties.get(name)
                expected = declared.get("type") if isinstance(declared, dict) else None
                if not isinstance(expected, str):
                    continue
                aligned = _losslessly_aligned(value, expected)
                if aligned is not value:
                    arguments[name] = aligned
                    changed = True
            if changed:
                call.function.arguments = json.dumps(arguments)


@dataclass
class EpisodeProxy:
    """One loopback OpenAI-compatible server multiplexing per-episode providers.

    Lifecycle: `start()` binds an ephemeral loopback port on a daemon thread;
    `register()`/`release()` scope a provider to an episode; `stop()` shuts the
    server down. The collector holds one proxy per rollout batch.
    """

    host: str = "127.0.0.1"
    _providers: dict[str, ToolCallingProvider] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _server: uvicorn.Server | None = None
    _thread: threading.Thread | None = None
    _port: int | None = None

    def register(self, alias: str, provider: ToolCallingProvider) -> None:
        """Route requests for model `alias` to `provider`.

        Args:
            alias: The episode's model alias (sent by tau2 as the request `model`).
            provider: The episode's span-recording provider.

        Raises:
            ValueError: If the alias is empty or already registered; alias reuse
                would silently splice two episodes' spans into one recorder.
        """
        if not alias:
            raise ValueError("an episode alias must be a nonempty string")
        with self._lock:
            if alias in self._providers:
                raise ValueError(
                    f"episode alias {alias!r} is already registered; each episode "
                    "needs its own alias so its spans land on its own recorder"
                )
            self._providers[alias] = provider

    def release(self, alias: str) -> None:
        """Drop the provider routed as `alias` (idempotent)."""
        with self._lock:
            self._providers.pop(alias, None)

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible base URL (`http://host:port/v1`) once started."""
        if self._port is None:
            raise RuntimeError("EpisodeProxy.base_url read before start()")
        return f"http://{self.host}:{self._port}/v1"

    def _app(self) -> FastAPI:
        """Build the FastAPI app serving `/v1/chat/completions`."""
        app = FastAPI()

        # A `def` route (not `async def`) so FastAPI runs it on its threadpool:
        # `complete_chat` blocks on the Tinker SDK, and concurrent episodes must
        # not serialize behind one event loop.
        @app.post("/v1/chat/completions")
        def chat_completions(payload: JsonObject) -> JSONResponse:
            alias = payload.get("model")
            with self._lock:
                provider = self._providers.get(alias) if isinstance(alias, str) else None
            if provider is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": {
                            "message": (
                                f"unknown episode alias {alias!r}; the proxy serves only "
                                "aliases registered by the tau2 rollout collector"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )
            try:
                chat_request = ChatRequest.model_validate(payload)
                response = provider.complete_chat(chat_request)
                realign_tool_argument_types(response, chat_request.tools)
            except Exception as exc:  # noqa: BLE001 - reported as an OpenAI-shaped 502
                logger.error("proxy completion for %s failed: %s", alias, exc)
                # Same split as wmo.serving.chat: full detail to the log above,
                # exception class name to the wire (CodeQL: stack-trace exposure).
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": (
                                f"upstream sampling failed ({type(exc).__name__}); see the "
                                "rollout collector's log for details"
                            ),
                            "type": "api_error",
                        }
                    },
                )
            body = response.model_dump(mode="json", exclude_none=True)
            # The OpenAI client requires the envelope fields the provider response
            # does not carry; usage totals are re-derived so litellm's accounting
            # never reads a missing field as zero.
            body.setdefault("id", f"chatcmpl-{alias}-{int(time.time() * 1000)}")
            body.setdefault("object", "chat.completion")
            body.setdefault("created", int(time.time()))
            usage = body.get("usage")
            if isinstance(usage, dict):
                usage.setdefault(
                    "total_tokens",
                    int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0)),
                )
            return JSONResponse(content=body)

        return app

    def start(self) -> None:
        """Bind an ephemeral loopback port and serve on a daemon thread.

        Raises:
            RuntimeError: If the server does not report startup within
                `_STARTUP_TIMEOUT_S`, or `start()` is called twice.
        """
        if self._server is not None:
            raise RuntimeError("EpisodeProxy.start() called twice")
        config = uvicorn.Config(
            self._app(),
            host=self.host,
            port=0,
            log_level="warning",
            # One loopback client; the default worker pool handles concurrent
            # episodes because the route is sync and threadpooled.
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="wmo-tau2-proxy", daemon=True)
        thread.start()
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if server.started and server.servers:
                sockets = server.servers[0].sockets
                if sockets:
                    self._port = sockets[0].getsockname()[1]
                    break
            if not thread.is_alive():
                raise RuntimeError("the tau2 proxy server thread died during startup")
            time.sleep(0.05)
        else:
            raise RuntimeError(f"the tau2 proxy did not start within {_STARTUP_TIMEOUT_S:.0f}s")
        self._server = server
        self._thread = thread
        logger.info("tau2 episode proxy serving on %s", self.base_url)

    def stop(self) -> None:
        """Shut the server down and join its thread (idempotent)."""
        server = self._server
        thread = self._thread
        if server is None or thread is None:
            return
        server.should_exit = True
        thread.join(timeout=10.0)
        if thread.is_alive():  # pragma: no cover - defensive; uvicorn honors should_exit
            logger.warning("tau2 proxy thread did not exit within 10s; abandoning it")
        self._server = None
        self._thread = None
        self._port = None
