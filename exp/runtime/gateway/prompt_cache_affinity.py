"""Tenant-namespaced provider cache-affinity keys.

Prefix caches at OpenAI-family providers are keyed per cache node behind a load
balancer: an identical prompt lands on the warm node, but the same stem with a
different tail is routed by the whole prompt and usually misses (Tencent
TokenHub, measured 2026-09-05: 2 of 8 shared-stem turns hit without a routing
hint, 7 of 8 with one). Agent loops append a tool result every turn, so
without a stable hint their corrected cache-read rate never applies.

``prompt_cache_key`` is the OpenAI-spec routing hint. The caller's own value
never leaves the gateway: house-funded rungs share one provider account across
every tenant, so the dispatched key is a digest namespaced by organization and
identity. When the caller sends no key, the conversation stem stands in for
it: the leading system/developer messages, which every turn of a session (and
every request sharing that system prompt) repeats verbatim, so the requests
that can share a cached prefix land on the node that holds it with no client
change. A conversation with no system prompt keys on its first user turn, the
next most stable thing an agent loop repeats.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from exp.common.core.artifacts import canonical_json_bytes
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest

_KEY_PREFIX = "xpl-"
_SESSION_PREFIX = "xpls-"
_DIGEST_HEX_LENGTH = 48
_FIELD_SEPARATOR = "\x1f"
_STEM_ROLES = frozenset({"system", "developer"})


def provider_prompt_cache_key(
    request: GatewayRequest, *, organization_id: str, identity_id: str
) -> str | None:
    """Return the cache-affinity key to dispatch for one tenant's request.

    Args:
        request: Canonical request after admission coercions.
        organization_id: Frozen organization authority for the request.
        identity_id: Frozen identity authority for the request.

    Returns:
        A stable ``xpl-`` prefixed digest, or ``None`` when the caller sent no
        ``prompt_cache_key`` and the conversation has no derivable stem (no
        user turn), in which case no hint is dispatched.
    """
    if request.prompt_cache_key is not None:
        material = ("caller", request.prompt_cache_key)
    else:
        stem = conversation_stem(request.messages)
        if stem is None:
            return None
        material = ("stem", stem)
    digest = hashlib.sha256(
        _FIELD_SEPARATOR.join((organization_id, identity_id, *material)).encode()
    ).hexdigest()
    return f"{_KEY_PREFIX}{digest[:_DIGEST_HEX_LENGTH]}"


def conversation_stem(messages: Sequence[GatewayMessage]) -> str | None:
    """Return the cache-stable stem of a conversation, or ``None`` without one.

    The stem is every leading system/developer message, joined with their
    roles: it is the prefix a provider can actually cache across requests, and
    it is what every turn of one session, and every request sharing one system
    prompt, repeats verbatim. Later turns vary per request and are excluded.
    A conversation that opens with no system prompt keys on its first user
    message instead; one that opens with an assistant or tool turn has no
    stable opening to pin on.

    Args:
        messages: The request's canonical messages in order.

    Returns:
        The joined stem text, or ``None`` when nothing stable opens it.
    """
    stem: list[str] = []
    for message in messages:
        if message.role in _STEM_ROLES:
            stem.append(f"{message.role}{_FIELD_SEPARATOR}{message.content or ''}")
            continue
        if stem:
            return _FIELD_SEPARATOR.join(stem)
        if message.role == "user":
            return f"user{_FIELD_SEPARATOR}{message.content or ''}"
        return None
    return _FIELD_SEPARATOR.join(stem) if stem else None


def provider_session_id(
    request: GatewayRequest, *, organization_id: str, identity_id: str
) -> str | None:
    """Return the per-conversation session id for providers that pin by session.

    Novita keeps a conversation on the server holding its prefix cache when
    each request names the conversation in ``x-session-id`` (enabled on the
    house account 2026-10-21; before it, a long DeepSeek conversation crossing
    a context-length pool boundary or a failover landed on a cold server and
    reprocessed up to 520,000 prompt tokens). Unlike
    :func:`provider_prompt_cache_key`, the key must separate conversations: the
    stem alone is shared by every session of one agent (one system prompt per
    API key), and pinning all of them to one server would hot-spot it. So the
    caller's ``prompt_cache_key`` wins, else the conversation's opening (the
    leading system/developer messages and the first user turn), which every
    turn of a growing conversation repeats verbatim.

    Args:
        request: Canonical request after admission coercions.
        organization_id: Frozen organization authority for the request.
        identity_id: Frozen identity authority for the request.

    Returns:
        A stable ``xpls-`` prefixed digest, or ``None`` when the caller sent no
        ``prompt_cache_key`` and the conversation has no user turn to open on.
    """
    if request.prompt_cache_key is not None:
        material = ("caller", request.prompt_cache_key)
    else:
        opening = conversation_opening(request.messages)
        if opening is None:
            return None
        material = ("opening", opening)
    digest = hashlib.sha256(
        _FIELD_SEPARATOR.join((organization_id, identity_id, *material)).encode()
    ).hexdigest()
    return f"{_SESSION_PREFIX}{digest[:_DIGEST_HEX_LENGTH]}"


def conversation_opening(messages: Sequence[GatewayMessage]) -> str | None:
    """Return a digest of a conversation's opening, or ``None`` without a user turn.

    The opening is every message up to and including the first user turn,
    serialized whole (content parts and images included), so two
    conversations that share a system prompt but start differently differ.

    Args:
        messages: The request's canonical messages in order.

    Returns:
        The opening's sha256 hex digest, or ``None`` when no user turn exists.
    """
    for index, message in enumerate(messages):
        if message.role == "user":
            opening = [item.model_dump(mode="json") for item in messages[: index + 1]]
            return hashlib.sha256(canonical_json_bytes(opening)).hexdigest()
    return None
