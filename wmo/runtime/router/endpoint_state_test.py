"""Adversarial bounds for local Responses continuation state."""

import pytest

from wmo.runtime.router.endpoint import HttpMessage, _OpenAIRequestState, _ResponseState


def _state(*, expires_at: float, size_bytes: int) -> _ResponseState:
    """Build retained response state with controlled expiry and size.

    Args:
        expires_at: Monotonic deadline for the test state.
        size_bytes: Serialized size charged to the continuation capacity.

    Returns:
        Deterministic response state for continuation-boundary tests.
    """
    return _ResponseState(
        episode_id="episode-a",
        messages=(HttpMessage(role="user", content="x"),),
        expires_at=expires_at,
        size_bytes=size_bytes,
    )


def test_response_state_expires_before_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove expired response identities cannot recover retained transcript content.

    The test advances the monotonic clock beyond the stored deadline and verifies both lookup
    rejection and byte-accounting cleanup.
    """
    now = [10.0]
    monkeypatch.setattr("wmo.runtime.router.endpoint.time.monotonic", lambda: now[0])
    state = _OpenAIRequestState()
    state.remember_response("resp_a", _state(expires_at=11.0, size_bytes=10))

    now[0] = 12.0

    with pytest.raises(ValueError, match="live local response"):
        state.response_context("resp_a")
    assert state._response_bytes == 0  # noqa: SLF001


def test_response_state_evicts_to_the_byte_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove byte pressure evicts history before the item-count ceiling.

    The test stores two individually valid histories whose combined serialized size exceeds the
    configured capacity, then verifies the oldest identity is unavailable.
    """
    monkeypatch.setattr("wmo.runtime.router.endpoint._RESPONSE_CAPACITY_BYTES", 15)
    state = _OpenAIRequestState()

    state.remember_response("resp_a", _state(expires_at=1e99, size_bytes=10))
    state.remember_response("resp_b", _state(expires_at=1e99, size_bytes=10))

    with pytest.raises(ValueError, match="live local response"):
        state.response_context("resp_a")
    assert state.response_context("resp_b").episode_id == "episode-a"
    assert state._response_bytes == 10  # noqa: SLF001
