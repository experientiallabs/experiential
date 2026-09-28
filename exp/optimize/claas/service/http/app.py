"""A private learning API independent of production gateways and deployments."""

from __future__ import annotations

import hashlib
import hmac
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

from exp.common.claas.learning import FeedbackSubmission
from exp.common.core.artifacts import JsonObject
from exp.optimize.claas.service.controller import LearningController
from exp.optimize.claas.service.http.inputs import ProtocolName, parse_generation
from exp.optimize.claas.service.http.outputs import generation_response
from exp.optimize.claas.service.http.parsing import parse_object
from exp.optimize.claas.training_contracts import ClaasTrainingError, TrainingExample

_EXAMPLES = TypeAdapter(tuple[TrainingExample, ...])


def create_app(
    controller: LearningController,
    *,
    api_key: str,
    maximum_request_bytes: int = 2_097_152,
    manage_lifecycle: bool = True,
) -> FastAPI:
    """Expose one scope and fixed student behind an explicit bearer credential."""
    if len(api_key) < 16:
        raise ValueError("learner API key must contain at least 16 characters")
    if not 1024 <= maximum_request_bytes <= 268_435_456:
        raise ValueError("maximum_request_bytes must be between 1024 and 268435456")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Hold one resident run across all requests and preserve state on shutdown."""
        if manage_lifecycle:
            await controller.start()
        try:
            yield
        finally:
            if manage_lifecycle:
                await controller.close(reason="shutdown")

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        """Authenticate before reading content or calling any service operation."""
        expected = f"Bearer {api_key}"
        if authorization is None or not hmac.compare_digest(
            authorization.encode(), expected.encode()
        ):
            raise HTTPException(status_code=401, detail="valid bearer authorization is required")

    app = FastAPI(
        lifespan=lifespan,
        dependencies=[Depends(authorize)],
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.exception_handler(ValueError)
    async def invalid_request(request: Request, error: ValueError) -> JSONResponse:
        """Return an actionable request error without hiding it as a server failure."""
        message = str(error)
        if isinstance(error, ValidationError):
            message = "; ".join(
                ".".join(str(part) for part in item["loc"]) + ": " + item["msg"]
                for item in error.errors(include_input=False, include_url=False)
            )
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "invalid_request_error", "message": message}},
        )

    @app.exception_handler(ClaasTrainingError)
    @app.exception_handler(TimeoutError)
    async def runtime_unavailable(request: Request, error: Exception) -> JSONResponse:
        """Surface uncertain execution without advertising a completed update."""
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "learner_unavailable",
                    "message": (
                        "learner operation failed; inspect /v1/status and its retained "
                        "checkpoint before retrying"
                    ),
                }
            },
        )

    async def read_body(request: Request) -> JsonObject:
        """Enforce the payload ceiling for both fixed-length and chunked requests."""
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > maximum_request_bytes:
                raise HTTPException(
                    status_code=413, detail="learner request exceeds its byte limit"
                )
            data.extend(chunk)
        return parse_object(data)

    async def generate(request: Request, protocol: ProtocolName) -> JSONResponse:
        """Bind an optional caller retry key to a persisted generation request."""
        body = await read_body(request)
        key = request.headers.get("idempotency-key")
        if key is not None and (not key.strip() or len(key.encode()) > 512):
            raise ValueError("Idempotency-Key must be nonblank and at most 512 bytes")
        prefixes = {"chat": "chatcmpl_", "responses": "resp_", "completions": "cmpl_"}
        identity = uuid.uuid4().hex if key is None else hashlib.sha256(key.encode()).hexdigest()
        response_id = prefixes[protocol] + identity
        generation = parse_generation(body, protocol, response_id)
        if generation.model not in {controller.spec.adapter_id, controller.spec.base_model}:
            raise ValueError("model must name this run's adapter_id or base_model")
        result = await controller.generate(generation)
        return JSONResponse(generation_response(result, generation, protocol))

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        """Generate one retained assistant message with optional function calls."""
        return await generate(request, "chat")

    @app.post("/v1/responses")
    async def responses(request: Request) -> JSONResponse:
        """Generate one retained stateless Responses output."""
        return await generate(request, "responses")

    @app.post("/v1/completions")
    async def completions(request: Request) -> JSONResponse:
        """Generate one retained raw-text completion."""
        return await generate(request, "completions")

    @app.post("/v1/feedback")
    async def feedback(request: Request) -> JSONResponse:
        """Attach sparse scalar, binary, or text feedback to one original response."""
        submission = FeedbackSubmission.model_validate(await read_body(request))
        status = await controller.submit_feedback(
            submission.response_id,
            scalar_reward=submission.training_reward,
            text_feedback=submission.text,
        )
        return JSONResponse(status.model_dump(mode="json"))

    @app.post("/v1/experiences")
    async def experiences(request: Request) -> JSONResponse:
        """Import already exact student examples without treating arbitrary traces as RL."""
        body = await read_body(request)
        if set(body) != {"examples"}:
            raise ValueError("provide exactly one examples array")
        status = await controller.import_examples(_EXAMPLES.validate_python(body["examples"]))
        return JSONResponse(status.model_dump(mode="json"))

    @app.post("/v1/train", status_code=202)
    async def train() -> JSONResponse:
        """Request an asynchronous update, including a ready partial batch."""
        status = await controller.trigger_train()
        return JSONResponse(status.model_dump(mode="json"), status_code=202)

    @app.post("/v1/drain")
    async def drain() -> JSONResponse:
        """Drain a finite snapshot, closing burst runs and retaining full-run engines."""
        report = await controller.drain()
        return JSONResponse(report.model_dump(mode="json"))

    @app.post("/v1/stop")
    async def stop() -> JSONResponse:
        """Drain available work within run limits and close resident compute."""
        await controller.drain()
        report = await controller.close(reason="requested")
        return JSONResponse(report.model_dump(mode="json"))

    @app.get("/v1/status")
    async def status() -> JSONResponse:
        """Report pending, ready, consumed, and rejected records without raw content."""
        value = await controller.status()
        return JSONResponse(value.model_dump(mode="json"))

    @app.get("/v1/checkpoint")
    async def checkpoint() -> JSONResponse:
        """Return only the last acknowledged checkpoint, without triggering training."""
        value = controller.buffer.checkpoint()
        return JSONResponse(
            {"checkpoint": None if value is None else value.model_dump(mode="json")}
        )

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        """Advertise the one application adapter owned by this service."""
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": controller.spec.adapter_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "learner",
                    }
                ],
            }
        )

    return app
