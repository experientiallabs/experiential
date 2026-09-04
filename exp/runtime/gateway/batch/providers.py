"""Provider batch API clients for the v1 batch lane: OpenAI, Anthropic, OpenRouter.

Each client speaks one provider's asynchronous batch product directly, with
no dialect translation: a line's body is already shaped for the surface it
names, and the provider must natively serve that surface. Result parsing is
strict, and any URL a provider hands back is exact-host validated before it
is fetched, so a compromised response cannot redirect credentials.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import Field, JsonValue

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.batch.contracts import (
    COMPLETION_WINDOW,
    BatchJob,
    BatchLineResult,
    BatchStatus,
    BatchSubmitError,
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
    """One provider's asynchronous batch product behind a uniform seam."""

    provider: str
    supports_cancel: bool
    requires_uniform_model: bool

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


class _LineUsage(ContractModel):
    """Provider-reported usage of one served line, in the synchronous lane's shape.

    ``input_tokens`` and ``output_tokens`` are the billable totals; the three
    optional counts are subsets of them (cached and cache-creation legs of the
    input, reasoning inside the output) that the host prices separately.
    """

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


def _count(value: object) -> int | None:
    """Read one non-negative integer count, or None for anything else."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _detail_count(usage: JsonObject, detail_keys: tuple[str, ...], key: str) -> int | None:
    """Read ``key`` from the first present ``*_details`` object among ``detail_keys``."""
    for detail_key in detail_keys:
        details = usage.get(detail_key)
        if isinstance(details, dict):
            return _count(details.get(key))
    return None


def _line_usage(body: JsonObject | None) -> _LineUsage:
    """Extract one served line's usage across the three provider wire shapes.

    Chat Completions bodies report ``prompt_tokens``/``completion_tokens`` with
    ``prompt_tokens_details.cached_tokens`` and
    ``completion_tokens_details.reasoning_tokens``; Responses bodies report
    ``input_tokens``/``output_tokens`` with the ``input_tokens_details`` and
    ``output_tokens_details`` equivalents; Anthropic messages report
    ``input_tokens`` EXCLUDING the ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens`` legs, which fold into the input total here
    exactly as the synchronous Anthropic normalizer folds them. Reasoning
    follows the synchronous rule for OpenAI-shaped usage: the provider's
    ``total_tokens`` decides whether reasoning was reported additively (then
    it folds into the output total) or as a subset (then the output total
    passes through); without a decisive total, a reasoning count above the
    output total is additive. A body without a usage object yields zeros.
    """
    if not isinstance(body, dict):
        return _LineUsage()
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return _LineUsage()
    reported_input = _count(usage.get("prompt_tokens", usage.get("input_tokens"))) or 0
    reported_output = _count(usage.get("completion_tokens", usage.get("output_tokens"))) or 0
    cache_read = _count(usage.get("cache_read_input_tokens"))
    cache_creation = _count(usage.get("cache_creation_input_tokens"))
    cached = _detail_count(
        usage, ("prompt_tokens_details", "input_tokens_details"), "cached_tokens"
    )
    reasoning = _detail_count(
        usage, ("completion_tokens_details", "output_tokens_details"), "reasoning_tokens"
    )
    input_tokens = reported_input + (cache_read or 0) + (cache_creation or 0)
    if cached is None and (cache_read is not None or cache_creation is not None):
        cached = cache_read or 0
    output_tokens = reported_output
    if reasoning:
        total = _count(usage.get("total_tokens"))
        if total == reported_input + reported_output:
            additive = False
        elif total == reported_input + reported_output + reasoning:
            additive = True
        else:
            additive = reasoning > reported_output
        if additive:
            output_tokens = reported_output + reasoning
    return _LineUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        cache_creation_input_tokens=cache_creation if cache_creation else None,
        reasoning_tokens=reasoning,
    )


def _served_line(custom_id: str, status_code: int, body: JsonObject) -> BatchLineResult:
    """Build one served line result carrying the body's usage in full."""
    return BatchLineResult(
        custom_id=custom_id,
        status_code=status_code,
        response=body,
        **_line_usage(body).model_dump(),
    )


_PROVIDER_ERROR_DETAIL_LIMIT = 400


def provider_error_detail(response: httpx.Response) -> str | None:
    """Return the provider's own error message from one failed response.

    OpenAI, Anthropic, and OpenRouter all answer failures with a JSON body
    whose ``error.message`` names the rejection (an unknown model, a
    malformed line, an exhausted quota). Only that field is read: a body that
    is not JSON, or that carries no message string, yields None, so an HTML
    error page or an unexpected shape never reaches the caller. The message
    is reduced to printable characters, whitespace-normalized, and bounded.
    """
    try:
        parsed = response.json()
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error")
    message = error.get("message") if isinstance(error, dict) else error
    if not isinstance(message, str):
        return None
    # Printable characters only: the detail reaches the caller's batch
    # object and the worker log, so a control character (ESC, NUL, a
    # terminal escape) in a malformed upstream body must never pass through.
    printable = "".join(char for char in message if char.isprintable() or char.isspace())
    detail = " ".join(printable.split())
    if not detail:
        return None
    return detail[:_PROVIDER_ERROR_DETAIL_LIMIT]


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
    """Read one non-negative integer count field, defaulting to zero."""
    value = counts.get(key, 0)
    return value if isinstance(value, int) and value >= 0 else 0


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
        return _served_line(custom_id, code, body)
    if isinstance(error, dict):
        return BatchLineResult(custom_id=custom_id, status_code=500, error=error)
    return None


class OpenAIBatchClient:
    """OpenAI Batch API: file upload, batch creation, polling, output files."""

    provider = "openai"
    supports_cancel = True
    requires_uniform_model = False
    _base = f"https://{OPENAI_HOST}/v1"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Bind an optional injected transport for tests."""
        self._transport = transport

    async def submit(self, *, job: BatchJob, api_key: str) -> str:
        """Upload the input JSONL then create the provider batch."""
        payload = "\n".join(
            json.dumps(
                {
                    "custom_id": line.custom_id,
                    "method": "POST",
                    "url": line.surface,
                    "body": {**line.body, "model": line.provider_model},
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


def _anthropic_line_error(kind: str, error: JsonValue | None) -> JsonObject:
    """Shape one non-succeeded Anthropic result as a line error with a reason.

    An ``errored`` result carries the provider's own error object (its
    ``message`` is the reason). ``canceled`` and ``expired`` results carry no
    error object at all, so the reason is written here: the host ledgers the
    line as failed WITH a cause, never as completed with zero tokens.
    """
    if isinstance(error, dict):
        return error
    return {
        "type": kind,
        "message": _ANTHROPIC_UNSERVED_REASONS.get(
            kind, f"the provider reported this request as {kind}"
        ),
    }


class AnthropicBatchClient:
    """Anthropic Message Batches: inline requests, results via a returned URL."""

    provider = "anthropic"
    supports_cancel = True
    requires_uniform_model = False
    _base = f"https://{ANTHROPIC_HOST}/v1"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Bind an optional injected transport for tests."""
        self._transport = transport

    def _headers(self, api_key: str) -> dict[str, str]:
        """Build the versioned Anthropic auth headers."""
        return {"x-api-key": api_key, "anthropic-version": _ANTHROPIC_VERSION}

    async def submit(self, *, job: BatchJob, api_key: str) -> str:
        """Create one message batch with every line inline."""
        requests = [
            {
                "custom_id": line.custom_id,
                "params": {**line.body, "model": line.provider_model},
            }
            for line in job.lines
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
            if kind == "succeeded" and isinstance(result.get("message"), dict):
                parsed.append(_served_line(custom_id, 200, result["message"]))
            else:
                parsed.append(
                    BatchLineResult(
                        custom_id=custom_id,
                        status_code=500,
                        error=_anthropic_line_error(str(kind), result.get("error")),
                    )
                )
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
    _base = f"https://{OPENROUTER_HOST}/api/beta"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Bind an optional injected transport for tests."""
        self._transport = transport

    async def submit(self, *, job: BatchJob, api_key: str) -> str:
        """Create one batch; endpoint and model serialize before requests."""
        model = job.lines[0].provider_model
        # The API stream-parses the body, so endpoint and model must appear
        # before the requests array; dict insertion order carries that.
        payload = {
            "endpoint": job.surface,
            "model": model,
            "requests": [{"custom_id": line.custom_id, "body": line.body} for line in job.lines],
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
