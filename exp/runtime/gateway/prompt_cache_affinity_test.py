"""Contracts for tenant-namespaced provider cache-affinity keys."""

from __future__ import annotations

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.prompt_cache_affinity import (
    conversation_stem,
    provider_prompt_cache_key,
)

_TENANT = {"organization_id": "org-a", "identity_id": "identity-a"}
_SYSTEM = GatewayMessage(role="system", content="You are Terminus. " * 40)


def _request(*messages: GatewayMessage, prompt_cache_key: str | None = None) -> GatewayRequest:
    """Build a Chat-surface request from ordered messages and an optional caller key."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=tuple(messages),
        prompt_cache_key=prompt_cache_key,
    )


def test_caller_key_is_namespaced_never_forwarded_verbatim() -> None:
    """A caller key becomes a tenant digest; two tenants never share one."""
    request = _request(GatewayMessage(role="user", content="q1"), prompt_cache_key="session-7")
    key = provider_prompt_cache_key(request, **_TENANT)
    assert key is not None
    assert key.startswith("xpl-")
    assert "session-7" not in key
    assert key == provider_prompt_cache_key(request, **_TENANT), "derivation is deterministic"
    other_tenant = provider_prompt_cache_key(
        request, organization_id="org-b", identity_id="identity-a"
    )
    assert other_tenant != key
    other_identity = provider_prompt_cache_key(
        request, organization_id="org-a", identity_id="identity-b"
    )
    assert other_identity != key


def test_stem_key_is_stable_across_a_growing_agent_loop() -> None:
    """Every turn of one Terminus-style session derives the same key.

    The stem (system prompt + first user task) repeats verbatim; the tool
    results and assistant replies appended each turn must not perturb it.
    """
    turn_1 = _request(_SYSTEM, GatewayMessage(role="user", content="Task: list the files."))
    turn_2 = _request(
        _SYSTEM,
        GatewayMessage(role="user", content="Task: list the files."),
        GatewayMessage(role="assistant", content='{"command": "ls"}'),
        GatewayMessage(role="user", content="Output: a.txt b.txt"),
    )
    turn_3 = _request(
        *turn_2.messages,
        GatewayMessage(role="assistant", content='{"command": "cat a.txt"}'),
        GatewayMessage(role="user", content="Output: hello"),
    )
    keys = {provider_prompt_cache_key(turn, **_TENANT) for turn in (turn_1, turn_2, turn_3)}
    assert len(keys) == 1
    assert next(iter(keys)) is not None


def test_requests_sharing_a_system_prompt_share_the_key_and_different_prompts_do_not() -> None:
    """The system stem IS the cacheable prefix, so it alone decides the node.

    Two trials with one system prompt and different opening tasks share the
    cached stem and therefore the key (Akhara's repro: same system stem, user
    ``q1`` then ``q2``); a different system prompt derives a different key.
    """
    a = _request(_SYSTEM, GatewayMessage(role="user", content="Task: list the files."))
    b = _request(_SYSTEM, GatewayMessage(role="user", content="Task: count the lines."))
    assert provider_prompt_cache_key(a, **_TENANT) == provider_prompt_cache_key(b, **_TENANT)
    other = _request(
        GatewayMessage(role="system", content="You are Harbor."),
        GatewayMessage(role="user", content="Task: list the files."),
    )
    assert provider_prompt_cache_key(a, **_TENANT) != provider_prompt_cache_key(other, **_TENANT)


def test_a_conversation_without_a_system_prompt_keys_on_its_first_user_turn() -> None:
    """No system stem: the opening user message is the next most stable prefix."""
    turn_1 = _request(GatewayMessage(role="user", content="Task: list the files."))
    turn_2 = _request(
        GatewayMessage(role="user", content="Task: list the files."),
        GatewayMessage(role="assistant", content='{"command": "ls"}'),
        GatewayMessage(role="user", content="Output: a.txt"),
    )
    assert provider_prompt_cache_key(turn_1, **_TENANT) == provider_prompt_cache_key(
        turn_2, **_TENANT
    )
    other_task = _request(GatewayMessage(role="user", content="Task: count the lines."))
    assert provider_prompt_cache_key(turn_1, **_TENANT) != provider_prompt_cache_key(
        other_task, **_TENANT
    )


def test_caller_key_outranks_the_stem() -> None:
    """An explicit caller key pins the session even when the stem differs."""
    a = _request(_SYSTEM, GatewayMessage(role="user", content="Task: A"), prompt_cache_key="sess")
    b = _request(_SYSTEM, GatewayMessage(role="user", content="Task: B"), prompt_cache_key="sess")
    assert provider_prompt_cache_key(a, **_TENANT) == provider_prompt_cache_key(b, **_TENANT)


def test_an_assistant_opening_means_no_hint() -> None:
    """Without a caller key, a system stem, or an opening user turn: no hint."""
    assistant_first = _request(GatewayMessage(role="assistant", content="hello"))
    assert provider_prompt_cache_key(assistant_first, **_TENANT) is None
    # A system-only request still has its cacheable stem.
    assert provider_prompt_cache_key(_request(_SYSTEM), **_TENANT) is not None


def test_conversation_stem_is_the_leading_system_and_developer_messages() -> None:
    """Developer messages count as stem; every later turn is excluded."""
    stem = conversation_stem(
        (
            GatewayMessage(role="system", content="S"),
            GatewayMessage(role="developer", content="D"),
            GatewayMessage(role="user", content="U1"),
            GatewayMessage(role="assistant", content="A"),
            GatewayMessage(role="user", content="U2"),
        )
    )
    assert stem is not None
    assert "S" in stem and "D" in stem
    assert "U1" not in stem and "U2" not in stem and "A" not in stem
    # The key length is bounded regardless of stem size.
    long = _request(
        GatewayMessage(role="system", content="x" * 200_000),
        GatewayMessage(role="user", content="y"),
    )
    key = provider_prompt_cache_key(long, **_TENANT)
    assert key is not None and len(key) == len("xpl-") + 48
