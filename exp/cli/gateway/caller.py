"""Caller-side commands that drive one live authenticated gateway over HTTP.

These commands cover the agent core loop against a running ``exp run`` gateway: discover
the aliases a virtual key can use, validate a key, and make one chat completion. They are
pure HTTP callers: spend authority stays entirely with the gateway's key grants and
monthly budgets, no provider client or provider credential is ever constructed, and the
presented virtual key is used for the one request and never stored or echoed.
"""

from __future__ import annotations

import json
from typing import cast

import httpx
import typer

from exp.common.core.artifacts import JsonObject

DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000/v1"

_URL_OPTION = typer.Option(
    DEFAULT_GATEWAY_URL,
    "--url",
    envvar=["EXP_GATEWAY_URL", "OPENAI_BASE_URL"],
    help="Base URL of a live gateway, normally ending in /v1.",
)
_KEY_OPTION = typer.Option(
    None,
    "--key",
    envvar=["EXP_GATEWAY_KEY", "OPENAI_API_KEY"],
    help="Virtual gateway key; read from the environment when omitted.",
)
_TIMEOUT_OPTION = typer.Option(
    120.0,
    "--timeout",
    min=0.1,
    help="Total HTTP timeout in seconds for the gateway request.",
)
_JSON_OPTION = typer.Option(False, "--json", help="Write the raw gateway JSON to stdout.")


def caller_call(
    alias: str = typer.Argument(..., help="Granted public model alias to call."),
    prompt: str = typer.Argument(..., help="One user message sent to the alias."),
    url: str = _URL_OPTION,
    key: str | None = _KEY_OPTION,
    timeout: float = _TIMEOUT_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Send one chat completion to a live gateway and stream the text to stdout.

    Args:
        alias: Granted public model alias to call.
        prompt: One user message sent to the alias.
        url: Live gateway base URL.
        key: Virtual gateway key, defaulting to the documented environment variables.
        timeout: Total HTTP timeout in seconds.
        json_output: Whether to make one non-streaming call and print the raw envelope.
    """
    raw_key = _require_key(key)
    body: JsonObject = {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "stream": not json_output,
    }
    with _gateway_client(url, timeout=timeout) as client:
        if json_output:
            response = _gateway_request(
                client, "POST", "/chat/completions", raw_key=raw_key, url=url, body=body
            )
            _require_success(response)
            typer.echo(response.text)
            return
        _stream_completion(client, raw_key=raw_key, url=url, body=body)


def caller_models(
    url: str = _URL_OPTION,
    key: str | None = _KEY_OPTION,
    timeout: float = _TIMEOUT_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """List the model aliases a live gateway grants to the presented key.

    This is the caller view served by ``GET /v1/models``, distinct from the operator-side
    ``exp config gateway alias list`` which reads local authority state directly.

    Args:
        url: Live gateway base URL.
        key: Virtual gateway key, defaulting to the documented environment variables.
        timeout: Total HTTP timeout in seconds.
        json_output: Whether to print the raw OpenAI-shaped list envelope.
    """
    raw_key = _require_key(key)
    with _gateway_client(url, timeout=timeout) as client:
        response = _gateway_request(client, "GET", "/models", raw_key=raw_key, url=url)
    _require_success(response)
    if json_output:
        typer.echo(response.text)
        return
    for model in _listed_models(response):
        identity = str(model.get("id", ""))
        authority = model.get("wmo")
        if isinstance(authority, dict) and "alias_revision_id" in authority:
            typer.echo(f"{identity} (revision {authority['alias_revision_id']})")
        else:
            typer.echo(identity)


def caller_key_check(
    url: str = _URL_OPTION,
    key: str | None = _KEY_OPTION,
    timeout: float = _TIMEOUT_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Validate one raw virtual key against a live gateway without storing it.

    The key is sent once as the Bearer credential of ``GET /v1/models``; a valid key
    prints its granted aliases and an invalid key exits nonzero with the gateway's
    remediation message. The raw key never appears in stdout, stderr, or any file.

    Args:
        url: Live gateway base URL.
        key: Virtual gateway key, defaulting to the documented environment variables.
        timeout: Total HTTP timeout in seconds.
        json_output: Whether to print one versioned key-check JSON document.
    """
    raw_key = _require_key(key)
    with _gateway_client(url, timeout=timeout) as client:
        response = _gateway_request(client, "GET", "/models", raw_key=raw_key, url=url)
    if response.status_code == 200:
        models = _listed_authority_models(response)
        if models is not None:
            aliases = [str(model["id"]) for model in models]
            if json_output:
                typer.echo(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "operation": "key.check",
                            "valid": True,
                            "granted_aliases": aliases,
                        },
                        separators=(",", ":"),
                    )
                )
                return
            typer.echo(f"key valid; granted aliases: {', '.join(aliases)}")
            return
        code = "invalid_gateway_response"
        message = "HTTP 200 did not contain the EXP gateway model-authority response shape."
    else:
        code, message = _error_detail(response)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "operation": "key.check",
                    "valid": False,
                    "error_code": code,
                    "message": message,
                },
                separators=(",", ":"),
            )
        )
    else:
        typer.echo(f"key invalid ({code}): {message}", err=True)
    raise typer.Exit(code=1)


def _stream_completion(
    client: httpx.Client,
    *,
    raw_key: str,
    url: str,
    body: JsonObject,
) -> None:
    """Stream one chat completion, echoing text deltas as the gateway sends them.

    Args:
        client: Open gateway HTTP client.
        raw_key: Presented virtual key.
        url: Gateway base URL, used only in connection diagnostics.
        body: Complete Chat Completions request payload with ``stream`` enabled.
    """
    try:
        with client.stream(
            "POST",
            "/chat/completions",
            json=body,
            headers=_authorization(raw_key),
        ) as response:
            if response.status_code != 200:
                response.read()
                _require_success(response)
            emitted = False
            saw_done = False
            for line in response.iter_lines():
                if _stream_done(line):
                    saw_done = True
                    break
                stream_error = _stream_error_detail(line)
                if stream_error is not None:
                    if emitted:
                        typer.echo("")
                    code, message = stream_error
                    typer.echo(f"gateway error {code}: {message}", err=True)
                    raise typer.Exit(code=1)
                delta = _stream_text_delta(line)
                if delta:
                    typer.echo(delta, nl=False)
                    emitted = True
            if emitted:
                typer.echo("")
            if not saw_done:
                typer.echo(
                    "gateway error incomplete_stream: The gateway stream ended before the "
                    "[DONE] terminal marker.",
                    err=True,
                )
                raise typer.Exit(code=1)
    except httpx.HTTPError:
        raise _unreachable_gateway(url) from None


def _stream_done(line: str) -> bool:
    """Return whether one SSE line carries the protocol terminal marker.

    Args:
        line: One raw server-sent-event line.

    Returns:
        ``True`` only for the exact OpenAI-compatible ``[DONE]`` data frame.
    """
    return line.startswith("data: ") and line[len("data: ") :].strip() == "[DONE]"


def _stream_error_detail(line: str) -> tuple[str, str] | None:
    """Return the code and message from one terminal streamed error envelope.

    Args:
        line: One raw server-sent-event line.

    Returns:
        Stream error details, or ``None`` when the line is not a gateway error.
    """
    chunk = _stream_payload(line)
    if chunk is None or not isinstance(chunk.get("error"), dict):
        return None
    error = cast(JsonObject, chunk["error"])
    return str(error.get("code", "unknown")), str(error.get("message", ""))


def _stream_text_delta(line: str) -> str:
    """Return the visible text carried by one Chat Completions SSE line.

    Args:
        line: One raw server-sent-event line.

    Returns:
        Text delta content, or an empty string for control frames and other events.
    """
    chunk = _stream_payload(line)
    if chunk is None:
        return ""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _stream_payload(line: str) -> JsonObject | None:
    """Decode one JSON server-sent-event data line into an object.

    Args:
        line: One raw server-sent-event line.

    Returns:
        JSON object payload, or ``None`` for control, malformed, and non-data lines.
    """
    if not line.startswith("data: "):
        return None
    payload = line[len("data: ") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return cast(JsonObject, chunk) if isinstance(chunk, dict) else None


def _gateway_client(url: str, *, timeout: float) -> httpx.Client:
    """Return the HTTP client used for one live-gateway command invocation.

    Args:
        url: Gateway base URL, normally ending in ``/v1``.
        timeout: Total HTTP timeout in seconds.

    Returns:
        Configured client whose base URL prefixes every relative request path.
    """
    return httpx.Client(base_url=url.rstrip("/"), timeout=timeout)


def _gateway_request(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    raw_key: str,
    url: str,
    body: JsonObject | None = None,
) -> httpx.Response:
    """Perform one authenticated gateway request with actionable connection errors.

    Args:
        client: Open gateway HTTP client.
        method: HTTP method.
        path: Path relative to the gateway base URL.
        raw_key: Presented virtual key.
        url: Gateway base URL, used only in connection diagnostics.
        body: Optional JSON request payload.

    Returns:
        The gateway HTTP response, whatever its status code.

    Raises:
        typer.BadParameter: No gateway answered at the configured URL.
    """
    try:
        return client.request(method, path, json=body, headers=_authorization(raw_key))
    except httpx.HTTPError:
        raise _unreachable_gateway(url) from None


def _require_success(response: httpx.Response) -> None:
    """Exit with the gateway's own remediation message for a non-200 response.

    Args:
        response: Completed gateway HTTP response.

    Raises:
        typer.Exit: The gateway answered with an error envelope.
    """
    if response.status_code == 200:
        return
    code, message = _error_detail(response)
    retry_after = response.headers.get("Retry-After")
    suffix = f" (Retry-After: {retry_after}s)" if retry_after else ""
    typer.echo(f"gateway error {code}: {message}{suffix}", err=True)
    raise typer.Exit(code=1)


def _error_detail(response: httpx.Response) -> tuple[str, str]:
    """Extract the stable code and display-safe message from one error response.

    Args:
        response: Completed non-200 gateway HTTP response.

    Returns:
        Machine-readable code and human-readable message, with HTTP-level fallbacks.
    """
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return (f"http_{response.status_code}", "The gateway returned a non-JSON error response.")
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        error = cast(JsonObject, payload["error"])
        return (str(error.get("code", "unknown")), str(error.get("message", "")))
    return (f"http_{response.status_code}", "The gateway returned an unrecognized error shape.")


def _listed_models(response: httpx.Response) -> list[JsonObject]:
    """Return the model objects of one ``GET /v1/models`` response.

    Args:
        response: Successful models-list response.

    Returns:
        Model objects in gateway order; malformed entries are dropped.
    """
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return []
    return [cast(JsonObject, item) for item in payload["data"] if isinstance(item, dict)]


def _listed_authority_models(response: httpx.Response) -> list[JsonObject] | None:
    """Return a validated EXP gateway model-authority list.

    Args:
        response: Successful response from the caller's ``GET /v1/models`` request.

    Returns:
        Validated non-empty model objects, or ``None`` when the response is not the EXP
        gateway authority shape. An empty list has no authority evidence and fails closed.
    """
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("object") != "list":
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    models: list[JsonObject] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        model = cast(JsonObject, item)
        if not _is_authority_model(model):
            return None
        models.append(model)
    return models


def _is_authority_model(model: JsonObject) -> bool:
    """Check the stable wire fields that identify an EXP authority model object.

    Args:
        model: One decoded model object from a models-list response.

    Returns:
        ``True`` only when the object carries the gateway authority metadata.
    """
    if (
        model.get("object") != "model"
        or model.get("created") != 0
        or model.get("owned_by") != "wmo"
    ):
        return False
    model_id = model.get("id")
    authority = model.get("wmo")
    if not isinstance(model_id, str) or not model_id or not isinstance(authority, dict):
        return False
    revision = authority.get("alias_revision_id")
    digest = authority.get("catalog_sha256")
    return isinstance(revision, str) and bool(revision) and isinstance(digest, str) and bool(digest)


def _authorization(raw_key: str) -> dict[str, str]:
    """Return the Bearer header for one presented virtual key.

    Args:
        raw_key: Presented virtual key.

    Returns:
        Header map used for exactly one request.
    """
    return {"Authorization": f"Bearer {raw_key}"}


def _require_key(key: str | None) -> str:
    """Return the presented key or fail with the exact way to provide one.

    Args:
        key: Optional key from the ``--key`` option or its environment variables.

    Returns:
        Non-empty raw virtual key.

    Raises:
        typer.BadParameter: No key was provided.
    """
    if key is not None and key.strip():
        return key.strip()
    raise typer.BadParameter(
        "a virtual gateway key is required; pass --key, set EXP_GATEWAY_KEY or "
        "OPENAI_API_KEY, or issue one with 'exp config gateway key issue'"
    )


def _unreachable_gateway(url: str) -> typer.BadParameter:
    """Build the actionable error for a gateway that did not answer.

    Args:
        url: Gateway base URL that failed to respond.

    Returns:
        Usage error naming the URL and the exact way to start or select a gateway.
    """
    return typer.BadParameter(
        f"no gateway answered at {url}; start one with 'exp run' or point "
        "--url (or EXP_GATEWAY_URL) at a live gateway"
    )
