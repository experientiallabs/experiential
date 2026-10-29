"""Provider batch API clients for the v1 batch lane: OpenAI, Anthropic, OpenRouter.

Each client speaks one provider's asynchronous batch product directly and
declares the public surfaces it serves. OpenAI and OpenRouter accept a line's
body as the caller shaped it; Anthropic Message Batches speak only the
Messages wire, so Chat Completions and Responses lines are carried across by
``line_wire`` (the synchronous lane's own decoders and payload builder) and
their results rendered back in the caller's surface. Result parsing is
strict, and any URL a provider hands back is exact-host validated before it
is fetched, so a compromised response cannot redirect credentials.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import Field, JsonValue

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.batch.contracts import (
    COMPLETION_WINDOW,
    BatchJob,
    BatchLine,
    BatchLineResult,
    BatchStatus,
    BatchSubmitError,
    BatchSurface,
    provider_error_message,
)
from exp.runtime.gateway.batch.line_wire import (
    anthropic_line_params,
    anthropic_result_body,
    line_usage,
)
from exp.runtime.gateway.contracts import GatewayFailureClass, GatewayUsage
from exp.runtime.models.providers.errors import (
    ProviderResponseError,
    normalized_provider_failure,
)

_LOGGER = logging.getLogger(__name__)


class AmbiguousProviderResponse(Exception):
    """A provider accepted the call but its response could not be parsed.

    The provider-side outcome is unknown, so callers must treat the operation
    as possibly performed and never as a definitive rejection.
    """


_REQUEST_TIMEOUT_SECONDS = 120.0
_ANTHROPIC_VERSION = "2023-06-01"

OPENAI_HOST = "api.openai.com"
ANTHROPIC_HOST = "api.anthropic.com"
OPENROUTER_HOST = "openrouter.ai"


class ProviderBatchSnapshot(ContractModel):
    """One poll of a provider batch job: status plus progress counts.

    ``cancelled_lines`` counts the lines the provider cut short on a
    cancellation request; it is included in ``failed``. A provider that ends
    a cancelled batch under its ordinary completed status reports the cut
    here, and the engine, which holds the caller's persisted cancellation
    intent, decides whether the job ends CANCELLED or COMPLETED.
    """

    status: BatchStatus
    completed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    cancelled_lines: int = Field(default=0, ge=0)
    results_ready: bool = False
    failure_message: str | None = None


class ProviderBatchClient(Protocol):
    """One provider's asynchronous batch product behind a uniform seam.

    ``surfaces`` is the engine's truth of which public surfaces this client
    can carry to its provider; the host catalog may narrow it per model but
    never widen it. ``line_request`` shapes one line's body for the wire and
    is the submit-time proof that a line is expressible on it.
    """

    provider: str
    supports_cancel: bool
    requires_uniform_model: bool
    surfaces: tuple[BatchSurface, ...]

    def line_request(self, line: BatchLine) -> JsonObject:
        """Return the provider-shaped request body for one line.

        Raises:
            BatchSubmitError: The line's surface body cannot be expressed on
                this provider's wire; reported per line at submit.
        """
        ...

    async def submit(self, *, job: BatchJob, api_key: str) -> str:
        """Submit one whole job; returns the provider's batch id."""
        ...

    async def poll(self, *, job: BatchJob, api_key: str) -> ProviderBatchSnapshot:
        """Return the provider's current view of one submitted job."""
        ...

    async def results(self, *, job: BatchJob, api_key: str) -> list[BatchLineResult]:
        """Fetch and parse every per-line result of one finished job."""
        ...

    async def cancel(self, *, job: BatchJob, api_key: str) -> None:
        """Request provider-side cancellation of one submitted job."""
        ...


def require_exact_host(url: str, host: str) -> str:
    """Return ``url`` unchanged after proving its authority is exactly ``host``.

    Raises:
        BatchSubmitError: When the scheme is not https, the hostname differs,
            userinfo is present, or a non-default port is set.
    """
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname != host
        or parts.username is not None
        or parts.password is not None
        or (parts.port is not None and parts.port != 443)
    ):
        raise BatchSubmitError(f"provider returned a URL outside https://{host}/; refusing it")
    return url


def _line_result(
    custom_id: str,
    status_code: int,
    *,
    usage: GatewayUsage,
    response: JsonObject | None = None,
    error: JsonObject | None = None,
) -> BatchLineResult:
    """Build one line result carrying the provider's reported usage in full.

    ``usage`` is read from the provider's own body (``line_usage``), which for
    a translated result differs from the rendered response: the Message keeps
    the cache legs the rendered surface may not carry. An error result carries
    usage only when the provider served the line and its result could not be
    rendered; every other error result reports zero.
    """
    return BatchLineResult(
        custom_id=custom_id,
        status_code=status_code,
        response=response,
        error=error,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cached_input_tokens=usage.cached_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )


def provider_error_detail(response: httpx.Response) -> str | None:
    """Return the provider's own error message from one failed response.

    OpenAI, Anthropic, and OpenRouter all answer failures with a JSON body
    whose ``error.message`` names the rejection (an unknown model, a
    malformed line, an exhausted quota). Only that field is read, through the
    shared :func:`provider_error_message` walker (nested envelopes, printable
    characters, whitespace-normalized, bounded): a body that is not JSON, or
    that carries no message string, yields None, so an HTML error page or an
    unexpected shape never reaches the caller.
    """
    try:
        parsed = response.json()
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return provider_error_message(parsed.get("error"))


def _unrenderable_line(
    custom_id: str, exc: ProviderResponseError | BatchSubmitError, *, usage: GatewayUsage
) -> BatchLineResult:
    """Fail one served line whose result the engine could not read or render.

    The caller-visible error object carries only the synchronous lane's
    sanitized failure message for a malformed provider response, never text
    derived from the provider body; a stored line body that no longer decodes
    (an engine-authored message) is reported as it is. The provider served
    the line, so its reported usage rides the result and still bills.
    """
    if isinstance(exc, ProviderResponseError):
        failure = normalized_provider_failure(exc)
        error: JsonObject = {
            "type": GatewayFailureClass.MALFORMED_RESPONSE.value,
            "message": failure.safe_message,
        }
    else:
        error = {"type": "invalid_request", "message": exc.message}
    return _line_result(custom_id, 502, usage=usage, error=error)


async def _checked(response: httpx.Response, *, action: str) -> JsonObject:
    """Return the JSON object body of one successful provider response.

    Raises:
        BatchSubmitError: When the call failed (status plus the provider's
            own ``error.message`` when it carries one) or the body is not a
            JSON object.
    """
    if response.status_code >= 400:
        summary = f"provider {action} failed with status {response.status_code}"
        detail = provider_error_detail(response)
        raise BatchSubmitError(
            summary if detail is None else f"{summary}: {detail}",
            code="provider_error",
        )
    try:
        parsed = response.json()
    except json.JSONDecodeError as exc:
        raise AmbiguousProviderResponse(f"provider {action} returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise AmbiguousProviderResponse(f"provider {action} returned a non-object body")
    return parsed


def _client(transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    """Build the short-lived HTTP client for one provider call.

    Args:
        transport: Optional injected transport, used by tests to serve
            wire-level fixtures without network access.
    """
    return httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_SECONDS, follow_redirects=False, transport=transport
    )


def _openai_status(raw: str) -> BatchStatus:
    """Map one OpenAI-vocabulary status string onto the shared enum."""
    try:
        return BatchStatus(raw)
    except ValueError:
        _LOGGER.warning("unknown provider batch status %r treated as in_progress", raw)
        return BatchStatus.IN_PROGRESS


def _int_count(counts: JsonObject, key: str) -> int:
    """Read one progress count field tolerantly, defaulting to zero.

    Progress counts (``request_counts``) only drive the public batch object
    and the poller's terminal decision; money settles from the per-line
    results, which are read strictly. A malformed count must therefore never
    wedge polling, so a bool, negative, or non-integer value reads as zero.
    """
    value = counts.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _parse_output_line(raw: JsonObject) -> BatchLineResult | None:
    """Parse one OpenAI-shaped output JSONL object into a line result."""
    custom_id = raw.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        return None
    error = raw.get("error")
    response = raw.get("response")
    if isinstance(response, dict) and response.get("body") is not None:
        body = response.get("body")
        status_code = response.get("status_code")
        code = status_code if isinstance(status_code, int) else 200
        if not isinstance(body, dict):
            return None
        if code >= 400:
            return BatchLineResult(custom_id=custom_id, status_code=code, error=body)
        try:
            usage = line_usage(body)
        except ProviderResponseError as exc:
            # A malformed usage value fails the line as the synchronous lane
            # fails a malformed response; it never settles at zero tokens.
            return _unrenderable_line(
                custom_id, exc, usage=GatewayUsage(input_tokens=0, output_tokens=0)
            )
        return _line_result(custom_id, code, usage=usage, response=body)
    if isinstance(error, dict):
        return BatchLineResult(custom_id=custom_id, status_code=500, error=error)
    return None


class OpenAIBatchClient:
    """OpenAI Batch API: file upload, batch creation, polling, output files."""

    provider = "openai"
    supports_cancel = True
    requires_uniform_model = False
    surfaces: tuple[BatchSurface, ...] = ("/v1/chat/completions", "/v1/responses")
    _base = f"https://{OPENAI_HOST}/v1"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Bind an optional injected transport for tests."""
        self._transport = transport

    def line_request(self, line: BatchLine) -> JsonObject:
        """The caller's body under the provider model id; the wire is native."""
        return {**line.body, "model": line.provider_model}

    async def submit(self, *, job: BatchJob, api_key: str) -> str:
        """Upload the input JSONL then create the provider batch."""
        payload = "\n".join(
            json.dumps(
                {
                    "custom_id": line.custom_id,
                    "method": "POST",
                    "url": line.surface,
                    "body": self.line_request(line),
                }
            )
            for line in job.lines
        ).encode("utf-8")
        headers = {"Authorization": f"Bearer {api_key}"}
        async with _client(self._transport) as client:
            upload = await client.post(
                f"{self._base}/files",
                headers=headers,
                data={"purpose": "batch"},
                files={"file": ("batch.jsonl", payload, "application/jsonl")},
            )
            uploaded = await _checked(upload, action="file upload")
            file_id = uploaded.get("id")
            if not isinstance(file_id, str) or not file_id:
                raise BatchSubmitError("provider file upload returned no file id")
            created = await client.post(
                f"{self._base}/batches",
                headers=headers,
                json={
                    "input_file_id": file_id,
                    "endpoint": job.surface,
                    "completion_window": COMPLETION_WINDOW,
                },
            )
            body = await _checked(created, action="batch create")
        batch_id = body.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise BatchSubmitError("provider batch create returned no batch id")
        return batch_id

    async def poll(self, *, job: BatchJob, api_key: str) -> ProviderBatchSnapshot:
        """Read the provider batch object and map its status and counts."""
        headers = {"Authorization": f"Bearer {api_key}"}
        async with _client(self._transport) as client:
            response = await client.get(
                f"{self._base}/batches/{job.provider_batch_id}", headers=headers
            )
            body = await _checked(response, action="batch poll")
        counts = body.get("request_counts")
        counts = counts if isinstance(counts, dict) else {}
        status = _openai_status(str(body.get("status", "in_progress")))
        errors = body.get("errors")
        failure: str | None = None
        if status is BatchStatus.FAILED and isinstance(errors, dict):
            failure = json.dumps(errors.get("data", []))[:2_000]
        return ProviderBatchSnapshot(
            status=status,
            completed=_int_count(counts, "completed"),
            failed=_int_count(counts, "failed"),
            results_ready=status is BatchStatus.COMPLETED,
            failure_message=failure,
        )

    async def results(self, *, job: BatchJob, api_key: str) -> list[BatchLineResult]:
        """Download and parse the output and error files of one finished job."""
        headers = {"Authorization": f"Bearer {api_key}"}
        parsed: list[BatchLineResult] = []
        async with _client(self._transport) as client:
            poll = await client.get(
                f"{self._base}/batches/{job.provider_batch_id}", headers=headers
            )
            body = await _checked(poll, action="batch poll")
            for key in ("output_file_id", "error_file_id"):
                file_id = body.get(key)
                if not isinstance(file_id, str) or not file_id:
                    continue
                content = await client.get(f"{self._base}/files/{file_id}/content", headers=headers)
                if content.status_code >= 400:
                    raise BatchSubmitError(
                        f"provider result download failed with status {content.status_code}",
                        code="provider_error",
                    )
                for raw_line in content.text.splitlines():
                    text = raw_line.strip()
                    if not text:
                        continue
                    raw = json.loads(text)
                    if isinstance(raw, dict):
                        result = _parse_output_line(raw)
                        if result is not None:
                            parsed.append(result)
        return parsed

    async def cancel(self, *, job: BatchJob, api_key: str) -> None:
        """Request cancellation of one provider batch."""
        headers = {"Authorization": f"Bearer {api_key}"}
        async with _client(self._transport) as client:
            response = await client.post(
                f"{self._base}/batches/{job.provider_batch_id}/cancel", headers=headers
            )
            await _checked(response, action="batch cancel")


_ANTHROPIC_UNSERVED_REASONS: dict[str, str] = {
    "canceled": "the provider canceled this request before it completed",
    "expired": "the provider expired this request before it completed",
}


def _anthropic_line_error(kind: JsonValue | None, error: JsonValue | None) -> JsonObject:
    """Shape one non-succeeded Anthropic result as a line error with a reason.

    An ``errored`` result carries the provider's own error object (its
    ``message`` is the reason). ``canceled`` and ``expired`` results carry no
    error object at all, so the reason is written here: the host ledgers the
    line as failed WITH a cause, never as completed with zero tokens. A
    result type the provider never documented is named as ``unknown``.
    """
    if isinstance(error, dict):
        return error
    name = kind if isinstance(kind, str) and kind else "unknown"
    return {
        "type": name,
        "message": _ANTHROPIC_UNSERVED_REASONS.get(
            name, f"the provider reported this request as {name}"
        ),
    }


class AnthropicBatchClient:
    """Anthropic Message Batches: inline requests, results via a returned URL."""

    provider = "anthropic"
    supports_cancel = True
    requires_uniform_model = False
    surfaces: tuple[BatchSurface, ...] = ("/v1/messages", "/v1/chat/completions", "/v1/responses")
    _base = f"https://{ANTHROPIC_HOST}/v1"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Bind an optional injected transport for tests."""
        self._transport = transport

    def _headers(self, api_key: str) -> dict[str, str]:
        """Build the versioned Anthropic auth headers."""
        return {"x-api-key": api_key, "anthropic-version": _ANTHROPIC_VERSION}

    def line_request(self, line: BatchLine) -> JsonObject:
        """Messages params for one line, translated from its surface when needed."""
        return anthropic_line_params(line)

    async def submit(self, *, job: BatchJob, api_key: str) -> str:
        """Create one message batch with every line inline."""
        requests = [
            {"custom_id": line.custom_id, "params": self.line_request(line)} for line in job.lines
        ]
        async with _client(self._transport) as client:
            response = await client.post(
                f"{self._base}/messages/batches",
                headers=self._headers(api_key),
                json={"requests": requests},
            )
            body = await _checked(response, action="batch create")
        batch_id = body.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise BatchSubmitError("provider batch create returned no batch id")
        return batch_id

    async def poll(self, *, job: BatchJob, api_key: str) -> ProviderBatchSnapshot:
        """Map Anthropic processing status onto the shared vocabulary."""
        async with _client(self._transport) as client:
            response = await client.get(
                f"{self._base}/messages/batches/{job.provider_batch_id}",
                headers=self._headers(api_key),
            )
            body = await _checked(response, action="batch poll")
        counts = body.get("request_counts")
        counts = counts if isinstance(counts, dict) else {}
        processing = str(body.get("processing_status", "in_progress"))
        succeeded = _int_count(counts, "succeeded")
        canceled = _int_count(counts, "canceled")
        errored = _int_count(counts, "errored") + canceled + _int_count(counts, "expired")
        if processing == "ended":
            # Anthropic ends a canceled batch with the same "ended" status as a
            # completed one; only the per-line counts say what happened, so
            # the cut is reported as cancelled_lines and the engine, which
            # holds the caller's persisted cancellation intent, picks the
            # terminal status.
            status = BatchStatus.COMPLETED
        elif processing == "canceling":
            status = BatchStatus.CANCELLING
        else:
            status = BatchStatus.IN_PROGRESS
        return ProviderBatchSnapshot(
            status=status,
            completed=succeeded,
            failed=errored,
            cancelled_lines=canceled,
            results_ready=processing == "ended",
        )

    async def results(self, *, job: BatchJob, api_key: str) -> list[BatchLineResult]:
        """Fetch the results JSONL from the batch's exact-host-checked URL."""
        async with _client(self._transport) as client:
            response = await client.get(
                f"{self._base}/messages/batches/{job.provider_batch_id}",
                headers=self._headers(api_key),
            )
            body = await _checked(response, action="batch poll")
            results_url = body.get("results_url")
            if not isinstance(results_url, str) or not results_url:
                return []
            content = await client.get(
                require_exact_host(results_url, ANTHROPIC_HOST),
                headers=self._headers(api_key),
            )
            if content.status_code >= 400:
                raise BatchSubmitError(
                    f"provider result download failed with status {content.status_code}",
                    code="provider_error",
                )
        parsed: list[BatchLineResult] = []
        lines_by_id = {line.custom_id: line for line in job.lines}
        created_at = time.time()
        for raw_line in content.text.splitlines():
            text = raw_line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, dict):
                continue
            custom_id = raw.get("custom_id")
            result = raw.get("result")
            if not isinstance(custom_id, str) or not isinstance(result, dict):
                continue
            kind = result.get("type")
            if kind != "succeeded":
                parsed.append(
                    BatchLineResult(
                        custom_id=custom_id,
                        status_code=500,
                        error=_anthropic_line_error(kind, result.get("error")),
                    )
                )
                continue
            message = result.get("message")
            if not isinstance(message, dict):
                parsed.append(
                    BatchLineResult(
                        custom_id=custom_id,
                        status_code=502,
                        error={
                            "type": "malformed_response",
                            "message": "the provider reported success without a message object",
                        },
                    )
                )
                continue
            try:
                usage = line_usage(message)
            except ProviderResponseError as exc:
                parsed.append(
                    _unrenderable_line(
                        custom_id, exc, usage=GatewayUsage(input_tokens=0, output_tokens=0)
                    )
                )
                continue
            line = lines_by_id.get(custom_id)
            if line is None:
                # A result for a line this job never submitted cannot be
                # rendered in a surface; the Message stays as the provider
                # sent it and the engine drops the unmatched custom id.
                parsed.append(_line_result(custom_id, 200, usage=usage, response=message))
                continue
            try:
                body = anthropic_result_body(
                    line,
                    message,
                    request_id=f"{job.batch_id}:{custom_id}",
                    created_at=created_at,
                )
            except (ProviderResponseError, BatchSubmitError) as exc:
                # The provider served the line, so its reported usage rides
                # the error result and still bills; the caller learns the
                # result could not be rendered in their surface.
                parsed.append(_unrenderable_line(custom_id, exc, usage=usage))
                continue
            parsed.append(_line_result(custom_id, 200, usage=usage, response=body))
        return parsed

    async def cancel(self, *, job: BatchJob, api_key: str) -> None:
        """Request cancellation of one message batch."""
        async with _client(self._transport) as client:
            response = await client.post(
                f"{self._base}/messages/batches/{job.provider_batch_id}/cancel",
                headers=self._headers(api_key),
            )
            await _checked(response, action="batch cancel")


class OpenRouterBatchClient:
    """OpenRouter beta batches: inline requests, inline results, no cancel."""

    provider = "openrouter"
    supports_cancel = False
    requires_uniform_model = True
    surfaces: tuple[BatchSurface, ...] = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
    _base = f"https://{OPENROUTER_HOST}/api/beta"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Bind an optional injected transport for tests."""
        self._transport = transport

    def line_request(self, line: BatchLine) -> JsonObject:
        """The caller's body as shaped; the model rides the batch, not the line."""
        return dict(line.body)

    async def submit(self, *, job: BatchJob, api_key: str) -> str:
        """Create one batch; endpoint and model serialize before requests."""
        model = job.lines[0].provider_model
        # The API stream-parses the body, so endpoint and model must appear
        # before the requests array; dict insertion order carries that.
        payload = {
            "endpoint": job.surface,
            "model": model,
            "requests": [
                {"custom_id": line.custom_id, "body": self.line_request(line)} for line in job.lines
            ],
        }
        async with _client(self._transport) as client:
            response = await client.post(
                f"{self._base}/batches",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            body = await _checked(response, action="batch create")
        batch_id = body.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise BatchSubmitError("provider batch create returned no batch id")
        return batch_id

    async def poll(self, *, job: BatchJob, api_key: str) -> ProviderBatchSnapshot:
        """Read the batch object; the status vocabulary mirrors OpenAI's."""
        async with _client(self._transport) as client:
            response = await client.get(
                f"{self._base}/batches/{job.provider_batch_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            body = await _checked(response, action="batch poll")
        counts = body.get("request_counts")
        counts = counts if isinstance(counts, dict) else {}
        status = _openai_status(str(body.get("status", "in_progress")))
        error = body.get("error")
        failure = json.dumps(error)[:2_000] if isinstance(error, dict) else None
        return ProviderBatchSnapshot(
            status=status,
            completed=_int_count(counts, "completed"),
            failed=_int_count(counts, "failed"),
            results_ready=status is BatchStatus.COMPLETED,
            failure_message=failure,
        )

    async def results(self, *, job: BatchJob, api_key: str) -> list[BatchLineResult]:
        """Parse the inline results array of one completed batch."""
        async with _client(self._transport) as client:
            response = await client.get(
                f"{self._base}/batches/{job.provider_batch_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            body = await _checked(response, action="batch poll")
        raw_results = body.get("results")
        parsed: list[BatchLineResult] = []
        if isinstance(raw_results, list):
            for raw in raw_results:
                if isinstance(raw, dict):
                    result = _parse_output_line(raw)
                    if result is not None:
                        parsed.append(result)
        return parsed

    async def cancel(self, *, job: BatchJob, api_key: str) -> None:
        """Refuse cancellation: the provider exposes no cancel endpoint."""
        raise BatchSubmitError(
            "openrouter batches cannot be cancelled; the job runs to completion",
            code="cancel_unsupported",
        )


PROVIDER_CLIENTS: dict[str, ProviderBatchClient] = {
    OpenAIBatchClient.provider: OpenAIBatchClient(),
    AnthropicBatchClient.provider: AnthropicBatchClient(),
    OpenRouterBatchClient.provider: OpenRouterBatchClient(),
}
