"""Join canonical calls to physical receipts without altering provider payloads."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json

_INTENT: ContextVar[ArtifactInput | None] = ContextVar("canonical_call_intent", default=None)
Consumer = Literal["simulator", "actor", "selector", "investigator", "analyst", "unknown"]
_CONSUMER: ContextVar[Consumer] = ContextVar("call_consumer", default="unknown")


def current_consumer() -> Consumer:
    """Return an execution-owner label, never a guess based on model or prompt text."""
    return _CONSUMER.get()


@contextmanager
def call_consumer(consumer: Consumer) -> Iterator[None]:
    """Label a bounded call at the execution owner without modifying its payload."""
    token = _CONSUMER.set(consumer)
    try:
        yield
    finally:
        _CONSUMER.reset(token)


def current_call_intent() -> ArtifactInput | None:
    """Return the current task-local canonical request reference, if the caller records one."""
    return _INTENT.get()


@contextmanager
def canonical_call(reference: ArtifactInput) -> Iterator[None]:
    """Attach each nested physical attempt to its canonical intent, restoring context on exit."""
    token = _INTENT.set(reference)
    try:
        yield
    finally:
        _INTENT.reset(token)


def intent_idempotency_key(url: str, payload: JsonObject) -> str | None:
    """Derive a stable ``Idempotency-Key`` for one physical request of the current intent.

    The key depends only on the canonical intent's identity, the endpoint and the exact request
    body, so a resend of the same intent after an ambiguous failure, in this process or a
    resumed one, carries the same key and an endpoint that honours it cannot execute twice.
    Different requests made under one intent get different keys. Credentials never enter it:
    the URL is hashed and headers are not read.

    Args:
        url: Absolute provider endpoint URL.
        payload: Exact JSON request body.

    Returns:
        ``exp-intent-`` plus 48 hex digits, or None outside a recorded canonical intent.
    """
    intent = _INTENT.get()
    if intent is None:
        return None
    digest = sha256_json(
        {
            "intent": intent.artifact_id,
            "endpoint_sha256": sha256_json({"url": url}),
            "request_sha256": sha256_json(payload),
        }
    )
    return "exp-intent-" + digest[:48]
