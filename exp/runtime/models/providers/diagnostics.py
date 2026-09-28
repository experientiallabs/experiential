"""Opt-in, bounded observation of SDK HTTP failures without changing their execution."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal, cast

from openai import APIStatusError
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

_BODY_LIMIT = 65_536
_MESSAGE_LIMIT = 2048
_LABEL = re.compile(r"[A-Za-z0-9_.-]{1,128}")


class APIErrorEvidence(BaseModel):
    """A bounded diagnostic receipt, never a request, response transcript, or billing receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recorded_at: AwareDatetime
    """UTC observation time, independent of any provider timestamp."""
    operation: str = Field(min_length=1, max_length=128)
    """Caller-selected operation label without URLs or request content."""
    endpoint_role: str = Field(min_length=1, max_length=128)
    """Caller-selected endpoint role, rather than a credential-bearing URL."""
    exception_type: str = Field(min_length=1, max_length=128)
    """SDK exception class name, without its potentially sensitive string representation."""
    status_code: int = Field(ge=100, le=599)
    """HTTP status reported by the SDK; it does not establish whether execution was billed."""
    request_id: str | None = Field(max_length=128)
    """Bounded provider correlation token, excluding arbitrary header content."""
    body_kind: Literal["structured", "omitted_nonobject_or_oversized"]
    """Whether the SDK supplied a bounded JSON object whose allowlisted fields were inspected."""
    error_type: str | None = Field(max_length=256)
    """Bounded, credential-filtered structured error type."""
    error_code: str | None = Field(max_length=256)
    """Bounded, credential-filtered structured error code."""
    error_param: str | None = Field(max_length=256)
    """Bounded, credential-filtered structured parameter name."""
    error_message: str | None = Field(default=None, max_length=_MESSAGE_LIMIT)
    """Explicitly opted-in message text; provider messages may still contain sensitive content."""


def _text(value: object, secrets: tuple[str, ...], limit: int) -> str | None:
    """Filter known credentials, bearer tokens, API keys, URLs and control characters."""
    if not isinstance(value, str):
        return None
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[redacted]")
    value = re.sub(r"(?i)bearer\s+[^\s\"'<>]+", "Bearer [redacted]", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", value)
    value = re.sub(r"(?i)https?://[^\s\"'<>]+", "[redacted-url]", value)
    return "".join(character if character.isprintable() else " " for character in value)[:limit]


def _bounded_body(error: APIStatusError) -> dict[str, object] | None:
    """Inspect bounded decoded or buffered JSON without reading a response stream."""
    body = error.body
    if body is None:
        try:
            content = error.response.content
            if len(content) > _BODY_LIMIT:
                return None
            body = json.loads(content)
        except (ValueError, RuntimeError):
            return None
    if not isinstance(body, dict):
        return None
    size = 0
    try:
        for chunk in json.JSONEncoder(ensure_ascii=True).iterencode(body):
            size += len(chunk)
            if size > _BODY_LIMIT:
                return None
    except (TypeError, ValueError, RecursionError):
        return None
    return cast(dict[str, object], body)


def api_error_evidence(
    error: APIStatusError,
    *,
    operation: str,
    endpoint_role: str,
    secrets: tuple[str, ...] = (),
    include_message: bool = False,
) -> APIErrorEvidence:
    """Extract allowlisted SDK fields without headers, URLs, raw bodies or tracebacks.

    Args:
        error: The original official SDK HTTP exception, including compatible subclasses.
        operation: A stable, content-free operation label.
        endpoint_role: A stable role label identifying the failed boundary.
        secrets: Known credential values to redact from every retained provider string.
        include_message: Opt in to bounded structured message text. Redaction is not a
            general content sanitizer; the caller must protect this potentially sensitive
            diagnostic destination. The default omits message text entirely.

    Returns:
        Structural error evidence without an inference about underlying cause or cost.
    """
    if not _LABEL.fullmatch(operation) or not _LABEL.fullmatch(endpoint_role):
        raise ValueError("diagnostic operation and endpoint_role require content-free labels")
    body = _bounded_body(error)
    details = body.get("error", body) if body is not None else {}
    if not isinstance(details, dict):
        details = {}
    request_id = error.request_id
    if not isinstance(request_id, str) or not _LABEL.fullmatch(request_id):
        request_id = None
    return APIErrorEvidence(
        recorded_at=datetime.now(UTC),
        operation=operation,
        endpoint_role=endpoint_role,
        exception_type=type(error).__name__[:128],
        status_code=error.status_code,
        request_id=_text(request_id, secrets, 128),
        body_kind="structured" if body is not None else "omitted_nonobject_or_oversized",
        error_type=_text(details.get("type"), secrets, 256),
        error_code=_text(details.get("code"), secrets, 256),
        error_param=_text(details.get("param"), secrets, 256),
        error_message=_text(details.get("message"), secrets, _MESSAGE_LIMIT)
        if include_message
        else None,
    )


@contextmanager
def observe_openai_errors(
    observer: Callable[[APIErrorEvidence], None],
    *,
    operation: str,
    endpoint_role: str,
    secrets: tuple[str, ...] = (),
    include_message: bool = False,
) -> Iterator[None]:
    """Observe an SDK HTTP failure once and rethrow the same exception without retries.

    Place the context around a synchronous call or an awaited call. Successful calls and
    other exception types are untouched. Ordinary diagnostic construction or callback
    failures add a type-only note to the original SDK error; process-control exceptions
    from the observer still propagate. No storage or logging destination is implicit.

    Args:
        observer: Explicit synchronous destination for the bounded receipt.
        operation: Content-free label for the enclosed operation.
        endpoint_role: Content-free endpoint role, not its URL.
        secrets: Known credential values to redact from retained provider strings.
        include_message: Explicitly retain potentially sensitive bounded message text.
    """
    try:
        yield
    except APIStatusError as error:
        try:
            observer(
                api_error_evidence(
                    error,
                    operation=operation,
                    endpoint_role=endpoint_role,
                    secrets=secrets,
                    include_message=include_message,
                )
            )
        except Exception as diagnostic_error:  # noqa: BLE001 - preserve the operation's failure
            error.add_note("HTTP error observation failed: " + type(diagnostic_error).__name__)
        raise
